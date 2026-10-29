import argparse, json, os, random, shutil, time, uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image, ImageFile
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from accelerate import Accelerator
from accelerate.utils import DataLoaderConfiguration, GradientAccumulationPlugin, InitProcessGroupKwargs, broadcast_object_list, set_seed
from diffusers import DDIMScheduler, DDPMScheduler, UNet2DModel, get_cosine_schedule_with_warmup
from diffusers.training_utils import EMAModel


ImageFile.LOAD_TRUNCATED_IMAGES = True

RESOLUTION = 512
TRAIN_TIMESTEPS = 1000

COLUMNS = ["rural_perc", "urban_perc", "water_perc"]
TILE_MODES = ("middle", "left", "right", "top", "bottom", "top_left", "top_right", "bottom_left", "bottom_right")

# 8: diagonal neighbors count as connected.
# 4: only horizontal/vertical neighbors count.
ROAD_CONNECTIVITY = 4

# Per-channel tolerance around RGB (0, 255, 0), measured after
# the same clipping/rounding used to save PNGs.
# 0 = strictly pure green. Try 32 for approximate generated greens.
ROAD_GREEN_TOLERANCE = 5

# ---------------------------------------------------------------------------
# Command-line arguments
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()

    # Data and output locations. No pretrained VAE is required.
    p.add_argument("--data-dir", type=Path, default=Path("data/netherlands_512_2m"))
    p.add_argument("--metadata-csv", type=Path)
    p.add_argument("--output-root", type=Path, default=Path("runs"))
    p.add_argument("--run-name", default="map-pixel-diffusion")
    p.add_argument("--resume", type=Path)

    # Precision, memory usage, and tracking.
    p.add_argument("--mixed-precision", choices=["bf16", "fp16", "no"], default="bf16")
    p.add_argument("--gradient-checkpointing", action="store_true")
    p.add_argument("--report-to", choices=["wandb", "none"], default="wandb")
    p.add_argument("--tracker-project", default="map-pixel-diffusion")
    p.add_argument("--tracker-init", type=Path, help="JSON with arbitrary wandb.init options.")
    p.add_argument("--tracker-config", type=Path, help="JSON with additional experiment metadata.")

    p.add_argument(
        "--road-mse-multiplier",
        type=float,
        default=2.0,
        help=(
            "MSE weight for ground-truth road pixels. "
            "1.0 disables extra weighting; 2.0 doubles it."
        ),
    )

    # Start with a small microbatch because activations are now full resolution.
    for name, default in dict(batch_size=1, gradient_accumulation=4, num_workers=2, max_steps=100000, warmup_steps=4000, log_every=20, sample_every=1000, sample_steps=100, save_every=1000, seed=42).items():
        p.add_argument("--" + name.replace("_", "-"), type=int, default=default)

    for name, default in dict(learning_rate=1e-4, weight_decay=0.01, max_grad_norm=1.0, condition_dropout=0.1, condition_scale=1.0).items():
        p.add_argument("--" + name.replace("_", "-"), type=float, default=default)

    args = p.parse_args()

    if min(args.batch_size, args.gradient_accumulation, args.max_steps, args.log_every, args.sample_steps) < 1:
        p.error("Batch size, accumulation, max steps, log interval, and sampling steps must be positive.")

    if min(args.num_workers, args.warmup_steps, args.sample_every, args.save_every, args.weight_decay) < 0:
        p.error("Counts, intervals, and weight decay must be nonnegative.")

    if not 0 <= args.condition_dropout <= 1:
        p.error("Condition dropout must be in [0, 1].")

    if args.sample_steps > TRAIN_TIMESTEPS:
        p.error(f"Sampling steps cannot exceed {TRAIN_TIMESTEPS}.")

    if min(args.condition_scale, args.learning_rate, args.max_grad_norm) <= 0:
        p.error("Condition scale, learning rate, and max gradient norm must be positive.")

    if (
        not np.isfinite(args.road_mse_multiplier)
        or args.road_mse_multiplier < 1.0
    ):
        p.error("--road-mse-multiplier must be finite and >= 1.0.")

    return args


# ---------------------------------------------------------------------------
# Configuration and tracking helpers
# ---------------------------------------------------------------------------

def read_json(path):
    value = json.loads(path.expanduser().read_text()) if path else {}

    if not isinstance(value, dict):
        raise ValueError("Tracking JSON must contain an object.")

    return value


def save_json(value, path):
    path.write_text(json.dumps(value, indent=2, default=str))


def redact(value):
    if isinstance(value, dict):
        return {k: "[REDACTED]" if any(s in str(k).lower() for s in ("api_key", "apikey", "password", "secret", "token")) else redact(v) for k, v in value.items()}

    if isinstance(value, (list, tuple)):
        return [redact(v) for v in value]

    return value


def init_tracking(acc, args, run_dir, **metadata):
    if args.report_to == "none":
        return

    options = {"name": run_dir.name, "dir": str(run_dir), **read_json(args.tracker_init)}

    if "config" in options:
        raise ValueError("Use --tracker-config for configuration, not tracker-init.config.")

    if acc.is_main_process:
        import wandb

        mode = options.get("mode", os.environ.get("WANDB_MODE", "online"))

        if mode not in ("offline", "disabled"):
            key = os.environ.get("WANDB_API_KEY")

            if not key:
                raise RuntimeError("Set WANDB_API_KEY, or select offline tracking / --report-to none.")

            wandb.login(key=key, relogin=True)

    config = json.loads(json.dumps(redact({**vars(args), **metadata, "extra": read_json(args.tracker_config)}), default=str))
    acc.init_trackers(args.tracker_project, config=config, init_kwargs={"wandb": options})

    if acc.is_main_process:
        run = acc.get_tracker("wandb", unwrap=True)
        run.define_metric("global_step")
        run.define_metric("*", step_metric="global_step")


def flatten_metrics(values, prefix=""):
    result = {}

    for key, value in values.items():
        name = f"{prefix}/{key}" if prefix else str(key)

        if isinstance(value, dict):
            result.update(flatten_metrics(value, name))
        elif isinstance(value, torch.Tensor) and value.numel() == 1:
            result[name] = value.detach().item()
        elif isinstance(value, (int, float, bool, np.number)):
            result[name] = float(value)

    return result


def log_metrics(acc, values, step, prefix=""):
    if not acc.is_main_process or not acc.trackers:
        return

    payload = flatten_metrics(values, prefix)

    if payload:
        acc.log({**payload, "global_step": int(step)})


def log_images(acc, paths, step, key="samples"):
    if not acc.is_main_process or not acc.trackers:
        return

    import wandb

    images = [wandb.Image(str(path), caption=path.stem) for path in paths if path.is_file()]

    if images:
        acc.log({key: images, "global_step": int(step)})


# ---------------------------------------------------------------------------
# Full-resolution mask generation
# A value of 1 marks known context; 0 marks a region to generate.
# All dimensions below are image pixels, not latent coordinates.
# ---------------------------------------------------------------------------

def tile_mask(mode=None, overlap=None):
    mode = mode or random.choice(TILE_MODES)
    overlap = overlap if overlap is not None else random.randint(64, 192)

    m = np.zeros((1, RESOLUTION, RESOLUTION), dtype=np.uint8)

    if mode == "middle" or "top" in mode:
        m[:, :overlap] = 1

    if mode == "middle" or "bottom" in mode:
        m[:, -overlap:] = 1

    if mode == "middle" or "left" in mode:
        m[:, :, :overlap] = 1

    if mode == "middle" or "right" in mode:
        m[:, :, -overlap:] = 1

    return m


def holes(m, count_min, count_max, size_min, size_max):
    for _ in range(random.randint(count_min, count_max)):
        h, w = random.randint(size_min, size_max), random.randint(size_min, size_max)
        y, x = random.randint(0, RESOLUTION - h), random.randint(0, RESOLUTION - w)

        m[:, y:y+h, x:x+w] = 0

    return m


def create_mask():
    p = random.random()

    # Fully generated images and border-conditioned images.
    if p < .25:
        return np.zeros((1, RESOLUTION, RESOLUTION), dtype=np.uint8)

    if p < .65:
        return tile_mask()

    m = np.ones((1, RESOLUTION, RESOLUTION), dtype=np.uint8)

    # A large, roughly central hole, optionally with smaller holes.
    if p < .83:
        h, w = random.randint(128, 320), random.randint(128, 320)
        cy, cx = RESOLUTION // 2 + random.randint(-64, 64), RESOLUTION // 2 + random.randint(-64, 64)
        y, x = max(0, min(RESOLUTION - h, cy - h // 2)), max(0, min(RESOLUTION - w, cx - w // 2))

        m[:, y:y+h, x:x+w] = 0

        return holes(m, 1, 4, 24, 96) if random.random() < .6 else m

    return holes(holes(m, 1, 1, 96, 256), 2, 10, 24, 128)


# ---------------------------------------------------------------------------
# Dataset and preprocessing
# ---------------------------------------------------------------------------

class MapDataset(Dataset):
    def __init__(self, root, metadata_csv, scale):
        self.paths = sorted(root.rglob("*.png"))

        if not self.paths:
            raise ValueError(f"No PNGs under {root}")

        df = pd.read_csv(metadata_csv)

        if not {"file_name", *COLUMNS} <= set(df.columns):
            raise ValueError(f"Metadata requires filename and {COLUMNS}")

        df["file_name"] = df["file_name"].astype(str).str.replace("\\", "/", regex=False)

        if df["file_name"].duplicated().any():
            raise ValueError("Duplicate metadata filenames.")

        # Match relative paths first, then unique basenames.
        metadata = df.set_index("file_name")
        counts = df["file_name"].map(lambda x: x.rsplit("/", 1)[-1]).value_counts()
        lookup = {k.rsplit("/", 1)[-1]: k for k in metadata.index if counts[k.rsplit("/", 1)[-1]] == 1}

        conditions = []

        for path in self.paths:
            relative = path.relative_to(root).as_posix()
            key = relative if relative in metadata.index else lookup.get(path.name)

            if key is None:
                raise KeyError(f"No unique metadata row for {relative}")

            conditions.append(metadata.loc[key, COLUMNS].to_numpy(dtype=np.float32))

        c = np.stack(conditions) / scale

        if not np.isfinite(c).all() or c.min() < -1e-6 or c.max() > 1.000001:
            raise ValueError("Conditions must be in [0,1]. Use --condition-scale 100 for CSV percentages.")

        # Fourth conditioning channel indicates that metadata is present.
        self.conditions = np.concatenate([np.clip(c, 0, 1), np.ones((len(c), 1), dtype=np.float32)], axis=1)
        self.preprocess = transforms.Compose([transforms.Resize(RESOLUTION, interpolation=transforms.InterpolationMode.NEAREST), transforms.CenterCrop(RESOLUTION), transforms.ToTensor()])

    def __len__(self):
        return len(self.paths)

    def get_pixels(self, index, augment=True):
        with Image.open(self.paths[index]) as source:
            if source.mode in ("RGBA", "LA") or (source.mode == "P" and "transparency" in source.info):
                rgba = source.convert("RGBA")
                image = Image.alpha_composite(Image.new("RGBA", rgba.size, (0, 0, 0, 255)), rgba).convert("RGB")
            else:
                image = source.convert("RGB")

            pixels = self.preprocess(image)

        if augment:
            if torch.rand(()).item() < .5:
                pixels = pixels.flip(-1)

            if torch.rand(()).item() < .5:
                pixels = pixels.flip(-2)

            pixels = torch.rot90(pixels, int(torch.randint(4, ()).item()), (-2, -1))

        # Diffusion operates directly on RGB values normalized to [-1, 1].
        return pixels * 2 - 1

    def __getitem__(self, index):
        return {"pixels": self.get_pixels(index), "mask": torch.from_numpy(create_mask()), "condition": torch.from_numpy(self.conditions[index].copy())}


def seed_worker(_):
    seed = torch.initial_seed() % 2**32

    random.seed(seed)
    np.random.seed(seed)


# ---------------------------------------------------------------------------
# Pixel-space diffusion objective
# ---------------------------------------------------------------------------

def compute_loss(unet, scheduler, batch, args):
    pixels = batch["pixels"].float()
    masks = batch["mask"].float()
    conditions = batch["condition"].float().clone()

    # Unknown context is zero in normalized image space.
    context = pixels * masks

    # Drop only metadata; retain the spatial context and mask.
    conditions[
        torch.rand(len(conditions), device=pixels.device)
        < args.condition_dropout
    ] = 0

    condition_maps = conditions[:, :, None, None].expand(
        -1, -1, RESOLUTION, RESOLUTION
    )

    noise = torch.randn_like(pixels)
    timesteps = torch.randint(
        scheduler.config.num_train_timesteps,
        (len(pixels),),
        device=pixels.device,
    )

    noisy = scheduler.add_noise(pixels, noise, timesteps)
    target = scheduler.get_velocity(pixels, noise, timesteps)

    # Noisy RGB (3) + mask (1) + context RGB (3) + conditions (4).
    model_input = torch.cat(
        [noisy, masks, context, condition_maps],
        dim=1,
    )
    prediction = unet(model_input, timesteps).sample

    # Average velocity-MSE across RGB channels, leaving [B, H, W].
    pixel_mse = (
        prediction.float() - target.float()
    ).square().mean(dim=1)

    if args.road_mse_multiplier == 1.0:
        return pixel_mse.mean()

    # Identify roads from ground-truth RGB using the same rounding and
    # per-channel tolerance as the evaluation metric.
    # All operations stay on the GPU.
    with torch.no_grad():
        rgb_255 = (
            (pixels + 1.0)
            .mul(127.5)
            .clamp(0, 255)
            .round()
        )

        road_color = pixels.new_tensor(
            [40.0, 210.0, 40.0]
        ).view(1, 3, 1, 1)

        road_pixels = (
            (rgb_255 - road_color).abs() <= ROAD_GREEN_TOLERANCE
        ).all(dim=1)

        # Road pixels receive multiplier M; all other pixels receive 1.
        weights = (
            1.0
            + (args.road_mse_multiplier - 1.0)
            * road_pixels.float()
        )

    return (pixel_mse * weights).mean()


# ---------------------------------------------------------------------------
# Evaluation inputs and image utilities
# ---------------------------------------------------------------------------

def save_rgb(rgb, path):
    Image.fromarray(rgb.detach().float().clamp(0, 1).mul(255).round().byte().permute(1, 2, 0).cpu().numpy()).save(path)


def stats(x):
    x = x.detach().float()

    return dict(mean=x.mean().item(), std=x.std().item(), min=x.min().item(), max=x.max().item(), finite=bool(torch.isfinite(x).all().item()))


@torch.no_grad()
def make_evaluation(dataset, device, run_dir):
    folder = run_dir / "evaluation"
    folder.mkdir(parents=True)

    # Fourteen source images: 4000, 4135, 4270, ...
    source_indices = [
        (4000 + 135 * i) % len(dataset)
        for i in range(14)
    ]

    sources = [
        dataset.get_pixels(index, False)
        for index in source_indices
    ]
    a = sources[0]

    rural = int(np.argmax(dataset.conditions[:, 0] - dataset.conditions[:, 1]))
    urban = int(np.argmax(dataset.conditions[:, 1] - dataset.conditions[:, 0]))

    zero = np.zeros((1, RESOLUTION, RESOLUTION), dtype=np.uint8)

    # Preserve the original four cases:
    # 0: rural metadata, no known pixels
    # 1: urban metadata, no known pixels
    # 2: top-left border context
    # 3: context on all four borders
    mask_arrays = [
        zero.copy(),
        zero.copy(),
        tile_mask("top_left", 128),
        tile_mask("middle", 128),
    ]

    # Seven new border-conditioned cases, each with another source image.
    border_cases = [
        ("left", 96),
        ("right", 128),
        ("top", 160),
        ("bottom", 96),
        ("top_right", 128),
        ("bottom_left", 160),
        ("bottom_right", 96),
    ]

    for mode, overlap in border_cases:
        mask_arrays.append(tile_mask(mode, overlap))

    # Five new inpainting cases. Rectangles are (y, x, height, width).
    # These fixed masks keep evaluation comparable across training steps.
    hole_cases = [
        # Small central hole.
        [(192, 192, 128, 128)],

        # Large central hole.
        [(96, 96, 320, 320)],

        # Off-center rectangular hole.
        [(64, 224, 192, 224)],

        # Four separated holes.
        [
            (64, 64, 96, 96),
            (64, 352, 96, 96),
            (352, 64, 96, 96),
            (352, 352, 96, 96),
        ],

        # Cross-shaped missing region.
        [
            (64, 208, 384, 96),
            (208, 64, 96, 384),
        ],
    ]

    for rectangles in hole_cases:
        mask = np.ones((1, RESOLUTION, RESOLUTION), dtype=np.uint8)

        for y, x, h, w in rectangles:
            mask[:, y:y+h, x:x+w] = 0

        mask_arrays.append(mask)

    # Two fully generated cases plus fourteen source-conditioned cases.
    pixels = torch.stack([
        torch.zeros_like(a),
        torch.zeros_like(a),
        *sources,
    ]).to(device)

    masks = torch.from_numpy(np.stack(mask_arrays)).to(device).float()
    conditions = torch.from_numpy(
        dataset.conditions[[rural, urban, *source_indices]].copy()
    ).to(device)
    context = pixels * masks

    for i, image in enumerate(sources):
        save_rgb((image + 1) / 2, folder / f"context_{i}.png")

    for i in range(len(pixels)):
        save_rgb((context[i] + 1) / 2, folder / f"masked_context_{i}.png")
        save_rgb(masks[i].expand(3, -1, -1), folder / f"mask_{i}.png")

    evaluation = dict(
        pixels=pixels,
        masks=masks,
        conditions=conditions,
        context=context,
    )

    torch.save(
        {k: v.cpu() for k, v in evaluation.items()},
        folder / "inputs.pt",
    )
    save_json(
        {
            "columns": COLUMNS + ["condition_present"],
            "conditions": conditions.cpu().tolist(),
        },
        folder / "conditions.json",
    )

    return evaluation

# --------
# Road connectivity
# --------

def count_interior_components(mask, connectivity=8):
    """
    Count True-valued connected components that do NOT touch an image edge.

    Uses horizontal runs + union-find, rather than per-pixel flood-fill.
    No minimum component size: even a single isolated pixel counts.

    mask: boolean-compatible array of shape [H, W].
    """
    mask = np.asarray(mask, dtype=bool)

    if mask.ndim != 2:
        raise ValueError("Expected a 2D mask.")
    if connectivity not in (4, 8):
        raise ValueError("connectivity must be 4 or 8.")

    height, width = mask.shape
    if height == 0 or width == 0:
        return 0

    # Vectorized detection of horizontal runs.
    # Each run occupies [start, end), with end exclusive.
    padded = np.pad(
        mask.astype(np.int8), ((0, 0), (1, 1)), constant_values=0
    )
    changes = np.diff(padded, axis=1)
    start_rows, starts = np.nonzero(changes == 1)
    _, ends = np.nonzero(changes == -1)

    if starts.size == 0:
        return 0

    # Runs returned by nonzero are already in row-major order.
    runs_per_row = np.bincount(start_rows, minlength=height)
    offsets = np.concatenate(([0], np.cumsum(runs_per_row)))

    # Python lists keep the small union-find operations inexpensive.
    starts = starts.tolist()
    ends = ends.tolist()
    n_runs = len(starts)

    parent = list(range(n_runs))
    sizes = [1] * n_runs
    touches_edge = (
        (start_rows == 0)
        | (start_rows == height - 1)
        | (np.asarray(starts) == 0)
        | (np.asarray(ends) == width)
    ).tolist()

    def find(node):
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(a, b):
        a, b = find(a), find(b)
        if a == b:
            return

        if sizes[a] < sizes[b]:
            a, b = b, a

        parent[b] = a
        sizes[a] += sizes[b]
        touches_edge[a] = touches_edge[a] or touches_edge[b]

    # For 8-connectivity, allow a one-column diagonal offset.
    padding = 1 if connectivity == 8 else 0

    for y in range(1, height):
        previous = int(offsets[y - 1])
        previous_end = int(offsets[y])

        for current in range(int(offsets[y]), int(offsets[y + 1])):
            left, right = starts[current], ends[current]

            # Discard previous-row runs too far left to connect.
            while (
                previous < previous_end
                and ends[previous] + padding <= left
            ):
                previous += 1

            candidate = previous

            # Merge every overlapping/diagonally adjacent previous-row run.
            while (
                candidate < previous_end
                and starts[candidate] < right + padding
            ):
                union(current, candidate)
                candidate += 1

    return sum(
        parent[i] == i and not touches_edge[i]
        for i in range(n_runs)
    )


def road_connectivity_metrics(
    rgb,
    tolerance=ROAD_GREEN_TOLERANCE,
    connectivity=ROAD_CONNECTIVITY,
):
    """
    rgb: torch tensor [3, H, W], nominally in [0, 1].

    Detect roads matching RGB (40, 210, 40), within a per-channel
    tolerance. tolerance=0 requires an exact match after PNG rounding.
    """
    if not 0 <= tolerance <= 255:
        raise ValueError("Green tolerance must be in [0, 255].")

    pixels = (
        rgb.detach()
        .float()
        .clamp(0, 1)
        .mul(255)
        .round()
        .byte()
        .permute(1, 2, 0)
        .cpu()
        .numpy()
    )

    # Signed arithmetic prevents uint8 subtraction from wrapping around.
    road_color = np.array([40, 210, 40], dtype=np.int16)
    difference = np.abs(pixels.astype(np.int16) - road_color)
    roads = np.all(difference <= tolerance, axis=-1)

    return {
        "disconnected_segments": count_interior_components(
            roads, connectivity=connectivity
        ),
        "road_pixel_fraction": float(roads.mean()),
    }

# ---------------------------------------------------------------------------
# Pixel-space DDIM sampling
# ---------------------------------------------------------------------------

@torch.no_grad()
def sample(model, scheduler, evaluation, args, run_dir, step, device):
    # Pixel-space clean samples have a known [-1, 1] range.
    sampler = DDIMScheduler.from_config(scheduler.config, prediction_type="v_prediction", timestep_spacing="trailing", clip_sample=True, clip_sample_range=1.0)
    sampler.set_timesteps(args.sample_steps, device=device)

    folder = run_dir / "samples" / f"step_{step:07d}"
    folder.mkdir(parents=True)

    diagnostics = {"images": {}}

    # Parameters remain float32; autocast optionally reduces inference memory.
    amp_enabled = args.mixed_precision != "no"
    amp_dtype = torch.bfloat16 if args.mixed_precision == "bf16" else torch.float16

    # Generate examples individually rather than holding 16 large activations.
    for i in range(len(evaluation["pixels"])):
        known = evaluation["pixels"][i:i+1]
        masks = evaluation["masks"][i:i+1]
        context = evaluation["context"][i:i+1]
        conditions = evaluation["conditions"][i:i+1]

        condition_maps = conditions[:, :, None, None].expand(-1, -1, RESOLUTION, RESOLUTION)

        # Fixed, independent noise makes each example comparable across steps.
        generator = torch.Generator(device=device).manual_seed(args.seed + 12345 + i)

        image = torch.randn(known.shape, generator=generator, device=device, dtype=torch.float32)
        known_noise = torch.randn(known.shape, generator=generator, device=device, dtype=torch.float32)

        for t in sampler.timesteps:
            timesteps = t.expand(len(image))

            # Pin known regions to the source image at the current noise level.
            noisy_known = sampler.add_noise(known, known_noise, timesteps)
            image = masks * noisy_known + (1 - masks) * image

            model_input = torch.cat([image, masks, context, condition_maps], dim=1)

            with torch.autocast("cuda", dtype=amp_dtype, enabled=amp_enabled):
                velocity = model(model_input, timesteps).sample

            # Keep scheduler calculations in float32.
            image = sampler.step(velocity.float(), t, image, eta=0.0, use_clipped_model_output=True).prev_sample

        if not torch.isfinite(image).all():
            raise RuntimeError(f"Nonfinite sampling output for evaluation image {i}.")

        # Save the model output and an exact-context composite separately.
        rgb = (image[0] + 1) / 2
        source_rgb = (known[0] + 1) / 2
        composited = masks[0] * source_rgb + (1 - masks[0]) * rgb

        road_metrics = road_connectivity_metrics(composited)

        diagnostics["images"][f"image_{i}"] = {
            "pixels": stats(image),
            "rgb": stats(rgb),
            "outside_0_1": (
                ((rgb < 0) | (rgb > 1)).float().mean().item()
            ),
            "roads": road_metrics,
        }

        save_rgb(rgb, folder / f"sample_{i:02d}.png")
        save_rgb(composited, folder / f"sample_{i:02d}_composited.png")

    # Aggregate across all evaluation images, including images with no roads.
    per_image_roads = [
        values["roads"]
        for values in diagnostics["images"].values()
    ]

    diagnostics["roads"] = {
        "mean_disconnected_segments": float(np.mean([
            values["disconnected_segments"]
            for values in per_image_roads
        ])),
        "mean_road_pixel_fraction": float(np.mean([
            values["road_pixel_fraction"]
            for values in per_image_roads
        ])),
        "connectivity": ROAD_CONNECTIVITY,
        "green_tolerance": ROAD_GREEN_TOLERANCE,
    }

    print(
        "Road connectivity: "
        f"{diagnostics['roads']['mean_disconnected_segments']:.3f} "
        "disconnected segments/image",
        flush=True,
    )

    save_json(diagnostics, folder / "diagnostics.json")

    return folder, diagnostics


# ---------------------------------------------------------------------------
# Model configuration
# ---------------------------------------------------------------------------

class Progress:
    def __init__(self):
        self.step, self.epoch, self.next_batch = 0, 0, 0

    def state_dict(self):
        return vars(self).copy()

    def load_state_dict(self, state):
        self.__dict__.update(state)


def build_unet():
    # Down resolutions: 512, 256, 128, 64, 32, 16.
    # Attention is used only at 32 and 16 pixels, including the middle block.
    # Up resolutions: 16, 32, 64, 128, 256, 512.
    return UNet2DModel(sample_size=RESOLUTION, in_channels=11, out_channels=3, layers_per_block=2, block_out_channels=(64, 96, 128, 256, 384, 512), down_block_types=("DownBlock2D", "DownBlock2D", "DownBlock2D", "DownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D"), up_block_types=("AttnUpBlock2D", "AttnUpBlock2D", "UpBlock2D", "UpBlock2D", "UpBlock2D", "UpBlock2D"), norm_num_groups=32, attention_head_dim=8)


def training_config(args):
    # Include optimizer and schedule settings so incompatible resumes fail early.
    return dict(representation="rgb_pixels", architecture="pixel_unet_v1", prediction_type="v_prediction", objective="road_weighted_velocity_mse", road_mse_multiplier=args.road_mse_multiplier, resolution=RESOLUTION, in_channels=11, out_channels=3, condition_scale=args.condition_scale, condition_dropout=args.condition_dropout, batch_size=args.batch_size, gradient_accumulation=args.gradient_accumulation, learning_rate=args.learning_rate, weight_decay=args.weight_decay, max_grad_norm=args.max_grad_norm, warmup_steps=args.warmup_steps, max_steps=args.max_steps, seed=args.seed, mixed_precision=args.mixed_precision)


# ---------------------------------------------------------------------------
# Checkpoint saving
# ---------------------------------------------------------------------------

def save_checkpoint(acc, unet, ema, scheduler, progress, args, run_dir):
    started = time.perf_counter()
    folder = run_dir / "checkpoints" / f"step_{progress.step:07d}"

    acc.wait_for_everyone()
    acc.save_state(str(folder / "accelerate"))

    if acc.is_main_process:
        raw = acc.unwrap_model(unet)

        raw.save_pretrained(folder / "unet")
        scheduler.save_pretrained(folder / "scheduler")
        torch.save(ema.state_dict(), folder / "ema.pt")

        ema.store(raw.parameters())
        ema.copy_to(raw.parameters())

        try:
            raw.save_pretrained(folder / "unet_ema")
        finally:
            ema.restore(raw.parameters())

        save_json(training_config(args), folder / "training_config.json")
        save_json({"world_size": acc.num_processes}, folder / "distributed_config.json")

        print(f"Checkpoint: {folder}", flush=True)

    acc.wait_for_everyone()
    log_metrics(acc, {"seconds": time.perf_counter() - started}, progress.step, "checkpoint")


# ---------------------------------------------------------------------------
# Training setup
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    acc = Accelerator(log_with="wandb" if args.report_to == "wandb" else None, mixed_precision=args.mixed_precision, gradient_accumulation_plugin=GradientAccumulationPlugin(num_steps=args.gradient_accumulation, sync_with_dataloader=False), dataloader_config=DataLoaderConfiguration(use_seedable_sampler=True, data_seed=args.seed), kwargs_handlers=[InitProcessGroupKwargs(timeout=timedelta(hours=1))])

    device = acc.device

    if device.type != "cuda":
        raise RuntimeError("CUDA is required.")

    set_seed(args.seed)
    torch.backends.cudnn.benchmark = True

    args.data_dir = args.data_dir.expanduser().resolve()

    if args.metadata_csv:
        metadata = args.metadata_csv.expanduser().resolve()
    else:
        metadata = next((args.data_dir / name for name in ("metadata.csv", "_metadata.csv") if (args.data_dir / name).is_file()), None)

    if metadata is None or not metadata.is_file():
        raise FileNotFoundError("Cannot find metadata CSV.")

    dataset = MapDataset(args.data_dir, metadata, args.condition_scale)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True, persistent_workers=args.num_workers > 0, worker_init_fn=seed_worker)

    # Pixel-space and latent-space checkpoints are intentionally incompatible.
    if args.resume:
        args.resume = args.resume.expanduser().resolve()
        previous = read_json(args.resume / "training_config.json")

        if previous != training_config(args):
            raise ValueError("Checkpoint configuration mismatch. Resume a pixel-space checkpoint with the same training settings.")

        distributed = read_json(args.resume / "distributed_config.json")

        if distributed["world_size"] != acc.num_processes:
            raise ValueError("Resume with the same number of GPU processes.")

        scheduler = DDPMScheduler.from_pretrained(args.resume / "scheduler")
        unet = UNet2DModel.from_pretrained(args.resume / "unet", torch_dtype=torch.float32)

        if scheduler.config.prediction_type != "v_prediction":
            raise ValueError("This script requires velocity-prediction checkpoints.")

    else:
        scheduler = DDPMScheduler(num_train_timesteps=TRAIN_TIMESTEPS, beta_schedule="squaredcos_cap_v2", prediction_type="v_prediction", clip_sample=True, clip_sample_range=1.0)
        unet = build_unet()

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    optimizer = torch.optim.AdamW(unet.parameters(), lr=args.learning_rate, betas=(0.9, 0.999), weight_decay=args.weight_decay, eps=1e-8, fused=True)
    lr = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=args.max_steps)

    unet, optimizer, loader = acc.prepare(unet, optimizer, loader)

    if not len(loader):
        raise RuntimeError("No batches; reduce batch size or check the dataset.")

    set_seed(args.seed, device_specific=True)

    progress, ema = Progress(), None

    if acc.is_main_process:
        ema = EMAModel(acc.unwrap_model(unet).parameters(), decay=0.9999, use_ema_warmup=True)
        ema.to(device=device)

    acc.register_for_checkpointing(lr)
    acc.register_for_checkpointing(progress)

    if args.resume:
        acc.load_state(str(args.resume / "accelerate"))

        if acc.is_main_process:
            ema.load_state_dict(torch.load(args.resume / "ema.pt", map_location="cpu", weights_only=False))
            ema.to(device=device)

    if progress.step >= args.max_steps:
        raise ValueError("--max-steps must exceed the checkpoint step.")

    # All processes share one output directory.
    run_id = [f"{args.run_name}-{datetime.now(timezone.utc):%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:6]}" if acc.is_main_process else None]
    broadcast_object_list(run_id, from_process=0)

    run_dir = args.output_root.expanduser().resolve() / run_id[0]
    global_batch = args.batch_size * args.gradient_accumulation * acc.num_processes

    config = {**training_config(args), "dataset_images": len(dataset), "world_size": acc.num_processes, "global_batch_size": global_batch, "model_parameters": sum(p.numel() for p in unet.parameters()), "torch_version": torch.__version__, "gpu": torch.cuda.get_device_name(device), "starting_step": progress.step, "output_directory": str(run_dir)}

    if acc.is_main_process:
        run_dir.mkdir(parents=True)
        save_json(redact({**vars(args), **config, "extra": read_json(args.tracker_config)}), run_dir / "run_config.json")

    acc.wait_for_everyone()
    init_tracking(acc, args, run_dir, **config)

    evaluation, inference = None, None

    if acc.is_main_process and args.sample_every:
        evaluation = make_evaluation(dataset, device, run_dir)
        log_images(acc, sorted((run_dir / "evaluation").glob("*.png")), progress.step, "evaluation/baselines")

        # Keep an independent inference model so sampling cannot alter training state.
        with torch.random.fork_rng(devices=[]):
            inference = UNet2DModel.from_config(acc.unwrap_model(unet).config)

        inference.requires_grad_(False).eval().to(device=device, dtype=torch.float32)

    acc.wait_for_everyone()
    acc.print(f"Run: {run_dir}\nImages: {len(dataset):,}; global batch: {global_batch}; representation: RGB pixels; objective: velocity MSE; starting step: {progress.step}")

    unet.train()
    optimizer.zero_grad(set_to_none=True)

    totals = torch.zeros(2, device=device, dtype=torch.float64)
    skips, consecutive_skips, last_log = 0, 0, time.perf_counter()

    # -----------------------------------------------------------------------
    # Training loop
    # -----------------------------------------------------------------------

    while progress.step < args.max_steps:
        loader.set_epoch(progress.epoch)

        if progress.next_batch >= len(loader):
            progress.epoch += 1
            progress.next_batch = 0
            continue

        start = progress.next_batch
        batches = acc.skip_first_batches(loader, start) if start else loader

        for index, batch in enumerate(batches, start=start):
            with acc.accumulate(unet):
                loss = compute_loss(unet, scheduler, batch, args)
                acc.backward(loss)

                if acc.sync_gradients:
                    grad_norm = acc.clip_grad_norm_(unet.parameters(), args.max_grad_norm)

                    # FP16 uses the gradient scaler to skip overflowing updates.
                    # For BF16/FP32, stop rather than applying nonfinite gradients.
                    if args.mixed_precision != "fp16" and not torch.isfinite(grad_norm).all().item():
                        raise RuntimeError("Nonfinite gradient norm. Check the loss and inputs, or reduce the learning rate.")

                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            progress.next_batch = index + 1

            totals[0] += loss.detach().double() * len(batch["pixels"])
            totals[1] += len(batch["pixels"])

            if not acc.sync_gradients:
                continue

            skipped = acc.optimizer_step_was_skipped

            if skipped:
                skips += 1
                consecutive_skips += 1

                if consecutive_skips >= 100:
                    raise RuntimeError("100 consecutive skipped optimizer updates.")

            else:
                consecutive_skips = 0
                lr.step()
                progress.step += 1

                if acc.is_main_process:
                    ema.step(acc.unwrap_model(unet).parameters())

            finished = progress.step >= args.max_steps

            # Aggregate metrics across all GPU processes.
            if (not skipped and progress.step % args.log_every == 0) or finished or (skipped and skips % 10 == 0):
                values = acc.reduce(totals, reduction="sum").cpu().tolist()
                now = time.perf_counter()

                event = dict(step=progress.step, epoch=progress.epoch, weighted_velocity_mse=values[0] / max(values[1], 1), lr=lr.get_last_lr()[0], wall_img_s=values[1] / max(now - last_log, 1e-9), skips=skips, peak_gpu_gib=torch.cuda.max_memory_allocated(device) / 2**30)

                log_metrics(acc, {k: v for k, v in event.items() if k != "step"}, progress.step, "train")

                if acc.is_main_process:
                    with (run_dir / "metrics.jsonl").open("a") as f:
                        f.write(json.dumps(event) + "\n")

                acc.print(
                    f"step={progress.step:7d} "
                    f"weighted_v_mse={event['weighted_velocity_mse']:.5f} "
                    f"lr={event['lr']:.2e} skips={skips}",
                    flush=True,
                )

                totals.zero_()
                skips, last_log = 0, time.perf_counter()

            # Only the main process generates images, using EMA parameters.
            if not skipped and args.sample_every and (progress.step % args.sample_every == 0 or finished):
                acc.wait_for_everyone()

                if acc.is_main_process:
                    started = time.perf_counter()
                    ema.copy_to(inference.parameters())

                    folder, diagnostics = sample(inference, scheduler, evaluation, args, run_dir, progress.step, device)

                    log_metrics(acc, {**diagnostics, "seconds": time.perf_counter() - started}, progress.step, "sampling")
                    log_images(acc, sorted(folder.glob("*_composited.png")), progress.step, "samples/composited")
                    log_images(acc, sorted(p for p in folder.glob("sample_*.png") if "_composited" not in p.stem), progress.step, "samples/raw")

                    print(f"Samples: {folder}", flush=True)

                acc.wait_for_everyone()

            if not skipped and not finished and args.save_every and progress.step % args.save_every == 0:
                save_checkpoint(acc, unet, ema, scheduler, progress, args, run_dir)

            if finished:
                break

        if progress.step < args.max_steps:
            progress.epoch += 1
            progress.next_batch = 0

    save_checkpoint(acc, unet, ema, scheduler, progress, args, run_dir)

    acc.print(f"Complete: {run_dir}")
    acc.end_training()


if __name__ == "__main__":
    main()