#!/usr/bin/env python3
"""
Threaded tile renderer with rotation augmentation and multiple-cache support.

Example:
    python mill_tiles.py \
        --cache ../data/02_osm_cache/noord-brabant \
                ../data/02_osm_cache/limburg \
        --angles 0,22.5,45,67.5 \
        --workers 8 \
        --output ../data/augmented_tiles

Caches are processed sequentially; geometry is not merged between caches.

Angles rotate image contents counterclockwise around each tile center.
Each rotated tile must fit entirely within one administrative boundary.

--max-tiles limits attempted tiles across ALL caches and angles, including
tiles subsequently rejected by the black-pixel check.

Omit --max-tiles to render all eligible tile/angle combinations.

CSV coverage columns:
    {tag}_perc:
        Fraction of visible pixels belonging to a tag.

    class_{name}_perc:
        Fraction of visible pixels belonging to a feature class.

Coverage values range from 0 to 1 and use the entire image as the
denominator. Outlines belong to their feature class. Later layers
overwrite earlier layers. Background pixels have no class or tag.

Class ownership is tracked independently from RGB colors, so classes
sharing colors remain distinguishable.

Grid coordinates:
    tile_x increases eastward.
    tile_y increases northward.

Coordinates are local to each cache and anchored to its selected
boundaries' bounding-box minimum X/Y.

Adjacent grid coordinates are separated by --tile-move metres.
All angles at a grid position share the same tile_x and tile_y.

Some positions may be absent due to boundary filtering or black-pixel
rejection. Use (cache, tile_x, tile_y, angle) to identify a tile variant.

Requires:
    geopandas, pyarrow, numpy, rasterio, pillow, pyproj, shapely,
    affine, tqdm
"""

import argparse
import csv
import json
import math
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from itertools import islice
from pathlib import Path

import geopandas as gpd
import numpy as np
from affine import Affine
from PIL import Image
from pyproj import CRS, Transformer
from rasterio.features import rasterize
from shapely.geometry import Polygon
from tqdm import tqdm


def parse_angles(value):
    """Parse finite angles, normalize to [0, 360), and remove duplicates."""
    try:
        angles = [float(part.strip()) for part in value.split(",")]
    except ValueError:
        raise argparse.ArgumentTypeError(
            "Angles must be a comma-separated list, e.g. 0,22.5,45,67.5."
        ) from None

    if not angles or any(not math.isfinite(angle) for angle in angles):
        raise argparse.ArgumentTypeError("Angles must be finite numbers.")

    return list(dict.fromkeys(angle % 360 for angle in angles))


def read_manifest(folder):
    """Read and validate the feature names and tags used for CSV columns."""
    config = json.loads(
        (folder / "manifest.json").read_text(encoding="utf-8")
    )

    seen_names = set()

    for feature in config["features"]:
        name = feature.get("name")
        tag = feature.get("tag")

        if not isinstance(name, str) or not name.strip():
            raise ValueError(
                f"Each feature must have a nonempty string 'name': {folder}"
            )

        if name in seen_names:
            raise ValueError(
                f"Duplicate feature name {name!r} in cache: {folder}"
            )

        if not isinstance(tag, str) or not tag.strip():
            raise ValueError(
                f"Feature {name!r} must have a nonempty string 'tag': "
                f"{folder}"
            )

        seen_names.add(name)

    return config


def load_cache(folder, config):
    """Load geometry and spatial indexes for one cache."""
    crs = CRS.from_user_input(config["metric_crs"])

    if not crs.is_projected or any(
        not math.isclose(axis.unit_conversion_factor, 1)
        for axis in crs.axis_info[:2]
    ):
        raise ValueError(
            f"Cache CRS must be projected and use metres: {folder}"
        )

    def read(filename):
        gp = gpd.read_parquet(folder / filename)

        if gp.crs is None or CRS.from_user_input(gp.crs) != crs:
            raise ValueError(f"CRS mismatch: {folder / filename}")

        return gp.loc[
            gp.geometry.notna() & ~gp.geometry.is_empty
        ].copy()

    layers = []

    for feature in config["features"]:
        gp = read(feature["file"])
        gp = gp[[gp.geometry.name]]
        layers.append((feature, gp, gp.sindex))

    boundaries = read(config["boundaries"]["selected_file"])

    if boundaries.empty:
        raise ValueError(f"No selected boundaries found: {folder}")

    return crs, layers, boundaries


def tile_layout(center_x, center_y, size, pixels, angle):
    """Return the exact world footprint and pixel-to-world transform."""
    resolution = size / pixels

    # Rotating the sampling footprint clockwise in world coordinates
    # makes the image contents appear rotated counterclockwise.
    # At angle=0, the image is north-up.
    transform = (
        Affine.translation(center_x, center_y)
        * Affine.rotation(-angle)
        * Affine.scale(resolution, -resolution)
        * Affine.translation(-pixels / 2, -pixels / 2)
    )

    footprint = Polygon([
        transform * (0, 0),
        transform * (pixels, 0),
        transform * (pixels, pixels),
        transform * (0, pixels),
    ])

    return footprint, transform


def tile_stream(boundaries, size, move, pixels, angles):
    """Yield eligible tile/angle combinations for one cache."""
    minx, miny, maxx, maxy = map(float, boundaries.total_bounds)
    index = boundaries.sindex
    tile_id = 0

    row_count = math.floor((maxy - miny) / move) + 1
    column_count = math.floor((maxx - minx) / move) + 1

    for tile_y in range(row_count):
        center_y = miny + tile_y * move + size / 2

        for tile_x in range(column_count):
            center_x = minx + tile_x * move + size / 2

            for angle in angles:
                footprint, transform = tile_layout(
                    center_x,
                    center_y,
                    size,
                    pixels,
                    angle,
                )

                # Query evaluates:
                # footprint.within(boundary)
                #
                # The entire rotated square must fit within ONE boundary.
                if len(index.query(footprint, predicate="within")):
                    yield (
                        tile_id,
                        tile_x,
                        tile_y,
                        center_x,
                        center_y,
                        angle,
                        footprint,
                        transform,
                    )
                    tile_id += 1


def render_tile(
    task,
    config,
    layers,
    output,
    pixels,
    filename_prefix="",
):
    """Render one tile and calculate visible class/tag coverage."""
    (
        tile_id,
        tile_x,
        tile_y,
        center_x,
        center_y,
        angle,
        footprint,
        transform,
    ) = task

    bounds = footprint.bounds
    pixel_count = pixels * pixels

    image = np.empty((pixels, pixels, 3), dtype=np.uint8)
    image[:] = config["backgroundColor"]

    # Zero denotes uncovered background.
    # Layer IDs start at 1 and follow the rendering order.
    ownership = np.zeros((pixels, pixels), dtype=np.int32)

    def mask(geometries, all_touched=False):
        return rasterize(
            [(geometry, 1) for geometry in geometries],
            out_shape=(pixels, pixels),
            transform=transform,
            fill=0,
            dtype="uint8",
            all_touched=all_touched,
        ) == 1

    # Layers and spatial indexes are shared read-only between threads.
    # Rasterize directly into the rotated grid to avoid image resampling.
    for layer_id, (feature, gp, index) in enumerate(layers, start=1):
        positions = index.query(footprint, predicate="intersects")

        if not len(positions):
            continue

        geometries = gp.geometry.iloc[np.sort(positions)]

        fill_mask = mask(geometries)
        image[fill_mask] = feature["color"]
        ownership[fill_mask] = layer_id

        if feature["boundaryColor"] is not None:
            outlines = []

            for geometry in geometries:
                outline = geometry.boundary

                if outline is not None and not outline.is_empty:
                    outlines.append(outline)

            if outlines:
                outline_mask = mask(outlines, all_touched=True)
                image[outline_mask] = feature["boundaryColor"]
                ownership[outline_mask] = layer_id

    # Preserve the original rejection rule: reject images with >10% black.
    if np.all(image == 0, axis=-1).mean() > 0.10:
        return None

    # Count ownership IDs once. This distinguishes classes even if they
    # share identical RGB colors.
    counts = np.bincount(
        ownership.ravel(),
        minlength=len(layers) + 1,
    )

    tag_counts = {}
    class_coverage = {}

    for layer_id, (feature, _, _) in enumerate(layers, start=1):
        count = int(counts[layer_id])
        name = feature["name"]
        tag = feature["tag"]

        class_coverage[f"class_{name}_perc"] = count / pixel_count
        tag_counts[tag] = tag_counts.get(tag, 0) + count

    filename = f"{filename_prefix}{tile_id:05d}_a{angle:g}.png"
    destination = output / filename
    temporary = destination.with_suffix(".png.tmp")

    try:
        Image.fromarray(image).save(temporary, format="PNG")
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)

    return {
        "filename": filename,
        "tile_x": tile_x,
        "tile_y": tile_y,
        "angle": angle,
        "centerX": center_x,
        "centerY": center_y,
        # Axis-aligned bounds enclosing the rotated footprint.
        "minX": bounds[0],
        "minY": bounds[1],
        "maxX": bounds[2],
        "maxY": bounds[3],
        **{
            f"{tag}_perc": count / pixel_count
            for tag, count in tag_counts.items()
        },
        **class_coverage,
    }


def results(executor, worker, tasks, limit):
    """Keep a bounded queue full without waiting for whole batches."""
    tasks = iter(tasks)
    pending = set()

    def refill():
        while len(pending) < limit:
            task = next(tasks, None)

            if task is None:
                break

            pending.add(executor.submit(worker, task))

    try:
        refill()

        while pending:
            finished, pending = wait(
                pending,
                return_when=FIRST_COMPLETED,
            )

            records = [future.result() for future in finished]
            refill()
            yield from records

    finally:
        for future in pending:
            future.cancel()


def main():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--cache",
        nargs="+",
        default=["../data/02_osm_cache/noord-brabant"],
        help="One or more cache folders, processed sequentially",
    )
    parser.add_argument(
        "--output",
        default="../data/netherlands_512_2m",
        help="A new or empty output folder",
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--max-tiles",
        type=int,
        help="Maximum attempted tiles across all caches and angles",
    )
    parser.add_argument(
        "--tile-size",
        type=float,
        default=1024,
        help="Tile side length in metres",
    )
    parser.add_argument(
        "--tile-move",
        type=float,
        default=512,
        help="Distance between adjacent grid positions in metres",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=512,
        help="Image side length in pixels",
    )
    parser.add_argument(
        "--angles",
        type=parse_angles,
        default="0",
        help=(
            "Comma-separated image rotations in counterclockwise degrees, "
            "e.g. 0,22.5,45,67.5 (default: 0)"
        ),
    )

    args = parser.parse_args()

    if args.workers < 1 or args.image_size < 1:
        parser.error("--workers and --image-size must be positive.")

    if args.max_tiles is not None and args.max_tiles < 1:
        parser.error("--max-tiles must be positive.")

    if any(
        not math.isfinite(value) or value <= 0
        for value in (args.tile_size, args.tile_move)
    ):
        parser.error(
            "--tile-size and --tile-move must be positive and finite."
        )

    caches = [Path(folder).resolve() for folder in args.cache]
    output = Path(args.output).resolve()

    if len(set(caches)) != len(caches):
        parser.error("The same cache folder was specified more than once.")

    # Cache names are used in filenames and CSV identifiers, so they must
    # also be unique even when the full paths differ.
    cache_ids = [cache.name for cache in caches]

    if len(set(cache_ids)) != len(cache_ids):
        parser.error(
            "Cache folder names must be unique because they are used "
            "as filename prefixes and CSV cache identifiers."
        )

    if output.exists() and (
        not output.is_dir() or any(output.iterdir())
    ):
        parser.error("Output must be a new or empty folder.")

    # Read manifests first, without loading geometry, so the combined CSV
    # includes every cache's tags and feature classes.
    manifests = [read_manifest(cache) for cache in caches]

    tags = list(dict.fromkeys(
        feature["tag"]
        for config in manifests
        for feature in config["features"]
    ))

    class_names = list(dict.fromkeys(
        feature["name"]
        for config in manifests
        for feature in config["features"]
    ))

    tag_fields = [f"{tag}_perc" for tag in tags]
    class_fields = [f"class_{name}_perc" for name in class_names]

    fields = [
        "filename",
        "cache",
        "tile_x",
        "tile_y",
        "angle",
        "centerX",
        "centerY",
        "minX",
        "minY",
        "maxX",
        "maxY",
        *tag_fields,
        *class_fields,
        "gMapsLink",
    ]

    if len(set(fields)) != len(fields):
        parser.error(
            "Feature names or tags produce duplicate CSV column names. "
            "Rename the conflicting features or tags."
        )

    run_info = {
        "status": "running",
        "requested_caches": [str(cache) for cache in caches],
        "caches": [],
        "tile_size_metres": args.tile_size,
        "tile_move_metres": args.tile_move,
        "image_size_pixels": args.image_size,
        "metres_per_pixel": args.tile_size / args.image_size,
        "angles_degrees": args.angles,
        "image_orientation": "counterclockwise_rotation_from_north_up",
        "metadata_bounds": "axis_aligned_bounds_of_rotated_footprint",
        "max_tiles": args.max_tiles,
        "max_tiles_scope": "attempted_tiles_across_all_caches_and_angles",
        "tags": tags,
        "classes": class_names,
        "coverage": {
            "range": [0, 1],
            "denominator": "all_image_pixels",
            "method": "final_visible_pixel_ownership",
            "outlines": "assigned_to_owning_feature",
            "background": "not_assigned_to_any_class_or_tag",
        },
        "grid_coordinates": {
            "scope": "per_cache",
            "x_column": "tile_x",
            "y_column": "tile_y",
            "x_direction": "east",
            "y_direction": "north",
            "spacing_metres": args.tile_move,
            "origin": "selected_boundaries_bounding_box_minimum_xy",
            "center_x_formula": (
                "origin_x + tile_x * tile_move_metres "
                "+ tile_size_metres / 2"
            ),
            "center_y_formula": (
                "origin_y + tile_y * tile_move_metres "
                "+ tile_size_metres / 2"
            ),
            "shared_across_angles": True,
        },
    }

    output.mkdir(parents=True, exist_ok=True)

    def save_run_info():
        destination = output / "_run.json"
        temporary = output / "_run.json.tmp"

        temporary.write_text(
            json.dumps(run_info, indent=2),
            encoding="utf-8",
        )
        temporary.replace(destination)

    attempted = 0
    completed = 0
    temporary_csv = output / "_metadata.csv.tmp"
    save_run_info()

    try:
        with temporary_csv.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as file:
            writer = csv.DictWriter(file, fieldnames=fields)
            writer.writeheader()

            with tqdm(
                total=args.max_tiles,
                desc="Rendering tiles",
                unit="tile",
            ) as progress:
                for cache, config in zip(caches, manifests):
                    if (
                        args.max_tiles is not None
                        and attempted >= args.max_tiles
                    ):
                        break

                    cache_id = cache.name
                    progress.set_description(f"Rendering {cache_id}")

                    crs, layers, boundaries = load_cache(cache, config)

                    transformer = Transformer.from_crs(
                        crs,
                        "EPSG:4326",
                        always_xy=True,
                    )

                    cache_info = {
                        "id": cache_id,
                        "folder": str(cache),
                        "status": "running",
                        "metric_crs": crs.to_wkt(),
                        "boundaries_file": (
                            config["boundaries"]["selected_file"]
                        ),
                        "boundary_rule": (
                            "rotated_footprint_within_one_boundary"
                        ),
                        "boundary_admin_level": (
                            config["boundaries"]["admin_level"]
                        ),
                        "grid_origin": (
                            boundaries.total_bounds[:2].tolist()
                        ),
                        "features": [
                            {
                                "name": feature["name"],
                                "tag": feature["tag"],
                            }
                            for feature in config["features"]
                        ],
                        "attempted_tiles": 0,
                        "completed_tiles": 0,
                    }

                    run_info["caches"].append(cache_info)
                    save_run_info()

                    tasks = tile_stream(
                        boundaries,
                        args.tile_size,
                        args.tile_move,
                        args.image_size,
                        args.angles,
                    )

                    if args.max_tiles is not None:
                        tasks = islice(
                            tasks,
                            args.max_tiles - attempted,
                        )

                    def worker(task):
                        return render_tile(
                            task,
                            config,
                            layers,
                            output,
                            args.image_size,
                            filename_prefix=f"{cache_id}_",
                        )

                    # All workers finish before the next cache is loaded.
                    with ThreadPoolExecutor(
                        max_workers=args.workers
                    ) as executor:
                        for record in results(
                            executor,
                            worker,
                            tasks,
                            args.workers * 2,
                        ):
                            attempted += 1
                            cache_info["attempted_tiles"] += 1
                            progress.update(1)

                            if record is None:
                                continue

                            record["cache"] = cache_id

                            # Classes/tags absent from this cache have
                            # zero coverage in the combined CSV.
                            for field in tag_fields:
                                record.setdefault(field, 0.0)

                            for field in class_fields:
                                record.setdefault(field, 0.0)

                            lon, lat = transformer.transform(
                                record["centerX"],
                                record["centerY"],
                            )

                            record["gMapsLink"] = (
                                "https://www.google.com/maps"
                                f"?q={lat},{lon}"
                            )

                            writer.writerow(record)
                            file.flush()

                            completed += 1
                            cache_info["completed_tiles"] += 1

                    cache_info["status"] = "complete"
                    save_run_info()

                    # Release geometry before loading the next cache.
                    del tasks, worker
                    del crs, layers, boundaries, transformer

        temporary_csv.replace(output / "_metadata.csv")
        run_info["status"] = "complete"

    except BaseException:
        run_info["status"] = "incomplete"

        for cache_info in run_info["caches"]:
            if cache_info["status"] == "running":
                cache_info["status"] = "incomplete"

        raise

    finally:
        run_info["attempted_tiles"] = attempted
        run_info["completed_tiles"] = completed
        save_run_info()

    print(
        f"Saved {completed:,} tiles from "
        f"{len(run_info['caches'])} cache(s) to {output}"
    )


if __name__ == "__main__":
    main()
