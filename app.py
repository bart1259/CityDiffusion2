from __future__ import annotations

import argparse
import base64
import binascii
import io
import json
import logging
import os
import secrets
import threading
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import torch
from diffusers import DDIMScheduler, DDPMScheduler, UNet2DModel
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from huggingface_hub import snapshot_download
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, Field, StrictBool, field_validator, model_validator

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

MODEL_ID = os.getenv("CITYGEN_MODEL_ID", "bart1259/CityDiffusion2")
MODEL_REVISION = os.getenv("CITYGEN_MODEL_REVISION", "main")
CHECKPOINT_DIR = Path(os.getenv("CITYGEN_CHECKPOINT", "")) if os.getenv("CITYGEN_CHECKPOINT") else None

INDEX_FILE = Path(__file__).resolve().parent / "index.html"

RESOLUTION = 512
TILE_GRID_SIZE = 4
TILE_SIZE = RESOLUTION // TILE_GRID_SIZE
MAX_IMAGE_BYTES = 8 * 1024 * 1024
CONDITION_NAMES = ("rural_perc", "urban_perc", "water_perc")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
amp_dtype = (
    torch.bfloat16
    if device.type == "cuda" and torch.cuda.is_bf16_supported()
    else torch.float16
)

unet = None
train_scheduler = None
loaded_checkpoint = None
loaded_revision = None

generation_lock = threading.Lock()
mask_tracking_lock = threading.Lock()

tile_true_counts = np.zeros((TILE_GRID_SIZE, TILE_GRID_SIZE), dtype=np.int64)
tile_false_counts = np.zeros((TILE_GRID_SIZE, TILE_GRID_SIZE), dtype=np.int64)
total_generation_requests = 0


# ---------------------------------------------------------------------
# API models
# ---------------------------------------------------------------------

class GenerateRequest(BaseModel):
    image_b64: str = Field(max_length=12_000_000)
    # true preserves the corresponding 128x128 tile; false regenerates it.
    tile_mask: list[list[StrictBool]] | None = None
    # White preserves context; black generates. Supports pixel-accurate holes.
    mask_b64: str | None = Field(default=None, max_length=12_000_000)
    # Training order: rural, urban, water. Each value is a fraction in [0, 1].
    conditions: list[float] = Field(min_length=3, max_length=3)
    seed: int | None = Field(default=None, ge=0, le=2_147_483_647)
    num_inference_steps: int = Field(default=100, ge=10, le=200)

    @field_validator("tile_mask")
    @classmethod
    def validate_tile_mask(cls, value: list[list[StrictBool]] | None) -> list[list[StrictBool]] | None:
        if value is None:
            return value
        if len(value) != TILE_GRID_SIZE or any(len(row) != TILE_GRID_SIZE for row in value):
            raise ValueError("tile_mask must be a 4x4 boolean array.")
        return value

    @model_validator(mode="after")
    def validate_mask_source(self):
        if (self.tile_mask is None) == (self.mask_b64 is None):
            raise ValueError("Provide exactly one of tile_mask or mask_b64.")
        return self

    @field_validator("conditions")
    @classmethod
    def validate_conditions(cls, value: list[float]) -> list[float]:
        if not all(np.isfinite(item) and 0 <= item <= 1 for item in value):
            raise ValueError("conditions must contain three fractions in [0, 1].")
        return value


class GenerateResponse(BaseModel):
    image_b64: str
    seed: int
    resolution: int
    inference_steps: int


# ---------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------

def decode_image(value: str) -> Image.Image:
    try:
        if value.startswith("data:"):
            value = value.split(",", 1)[1]

        raw = base64.b64decode(value, validate=True)
        if len(raw) > MAX_IMAGE_BYTES:
            raise ValueError("Image exceeds the 8 MiB limit.")

        with Image.open(io.BytesIO(raw)) as source:
            if source.width > 4096 or source.height > 4096:
                raise ValueError("Maximum input dimensions are 4096x4096.")
            # Retain alpha until mask construction: transparent pixels are
            # unknown, never valid black context.
            return source.convert("RGBA")

    except (
        ValueError,
        IndexError,
        binascii.Error,
        OSError,
        UnidentifiedImageError,
        Image.DecompressionBombError,
    ) as exc:
        raise HTTPException(status_code=400, detail=f"Invalid context image: {exc}") from exc


def encode_png(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


# ---------------------------------------------------------------------
# Tile-mask tracking
# ---------------------------------------------------------------------

def record_tile_mask_request(tile_mask: np.ndarray) -> None:
    global total_generation_requests
    with mask_tracking_lock:
        total_generation_requests += 1
        tile_true_counts[:] += tile_mask.astype(np.int64)
        tile_false_counts[:] += (~tile_mask).astype(np.int64)


def get_tile_mask_stats() -> dict:
    with mask_tracking_lock:
        total_counts = tile_true_counts + tile_false_counts
        return {
            "total_generation_requests": total_generation_requests,
            "grid_size": TILE_GRID_SIZE,
            "tile_size": TILE_SIZE,
            "true_means": "preserve this context tile",
            "false_means": "generate this tile",
            "tiles": [
                [
                    {
                        "row": row,
                        "column": column,
                        "requests": int(total_counts[row, column]),
                        "preserve_true": int(tile_true_counts[row, column]),
                        "generate_false": int(tile_false_counts[row, column]),
                    }
                    for column in range(TILE_GRID_SIZE)
                ]
                for row in range(TILE_GRID_SIZE)
            ],
        }


# ---------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------

def find_checkpoint(root: Path) -> Path:
    def has_model(path: Path) -> bool:
        has_subfolder_model = any(
            (path / name / "config.json").is_file()
            for name in ("unet_ema", "unet")
        )
        has_root_model = (path / "config.json").is_file() and any(
            (path / filename).is_file()
            for filename in (
                "diffusion_pytorch_model.safetensors",
                "diffusion_pytorch_model.bin",
            )
        )
        return has_subfolder_model or has_root_model

    if has_model(root):
        return root

    candidates = set()
    for config in root.rglob("config.json"):
        if config.parent.name in ("unet_ema", "unet") and has_model(config.parent.parent):
            candidates.add(config.parent.parent)

    if not candidates:
        raise FileNotFoundError(
            f"No train_maps_pixel.py U-Net checkpoint found in {root}"
        )

    # Choose the most recent step folder when the repository contains several.
    return sorted(candidates, key=lambda path: path.as_posix())[-1]


def load_models(checkpoint_dir: Path | None) -> None:
    global unet, train_scheduler, loaded_checkpoint, loaded_revision

    if device.type != "cuda":
        raise RuntimeError("CUDA is required for this application.")

    if checkpoint_dir is None:
        print(f"Downloading Hugging Face model: {MODEL_ID}", flush=True)
        model_root = Path(snapshot_download(repo_id=MODEL_ID, revision=MODEL_REVISION))
        loaded_revision = model_root.name
    else:
        model_root = checkpoint_dir.expanduser().resolve()
        loaded_revision = None

    checkpoint = find_checkpoint(model_root)
    ema_path = checkpoint / "unet_ema"
    if (ema_path / "config.json").is_file():
        model_path = ema_path
        weight_source = "EMA"
    elif (checkpoint / "unet" / "config.json").is_file():
        model_path = checkpoint / "unet"
        weight_source = "raw"
    else:
        # Hub repos can publish the EMA U-Net directly at their root.
        model_path = checkpoint
        weight_source = "repository"

    config_path = checkpoint / "training_config.json"
    training_config = json.loads(config_path.read_text()) if config_path.is_file() else {}
    if not training_config:
        logging.warning("Checkpoint has no training_config.json; training provenance cannot be verified.")
    expected_config = {
        "representation": "rgb_pixels",
        "architecture": "pixel_unet_v1",
        "prediction_type": "v_prediction",
        "resolution": RESOLUTION,
        "in_channels": 11,
        "out_channels": 3,
    }
    for name, expected in expected_config.items():
        actual = training_config.get(name)
        if actual is not None and actual != expected:
            raise RuntimeError(f"Incompatible model config: {name}={actual!r}; expected {expected!r}.")

    print(f"Loading {model_path} ({weight_source} weights)", flush=True)
    model = UNet2DModel.from_pretrained(model_path, torch_dtype=torch.float32)
    print(f"Checkpoint architecture: layers_per_block={model.config.layers_per_block}; revision={loaded_revision}", flush=True)
    if model.config.in_channels != 11 or model.config.out_channels != 3:
        raise RuntimeError(
            "Expected the train_maps_pixel.py 11-input / 3-output RGB U-Net; "
            f"got {model.config.in_channels}-input / {model.config.out_channels}-output."
        )
    if model.config.sample_size != RESOLUTION:
        raise RuntimeError(f"Expected a {RESOLUTION}px model; got sample_size={model.config.sample_size}.")

    scheduler_path = checkpoint / "scheduler"
    if not (scheduler_path / "scheduler_config.json").is_file():
        scheduler_path = model_root / "scheduler"

    if (scheduler_path / "scheduler_config.json").is_file():
        scheduler = DDPMScheduler.from_pretrained(scheduler_path)
    else:
        # train_maps_pixel.py's scheduler defaults; the Hub export contains
        # only the U-Net weights/config, not the training scheduler folder.
        scheduler = DDPMScheduler(
            num_train_timesteps=1000,
            beta_schedule="squaredcos_cap_v2",
            prediction_type="v_prediction",
            clip_sample=True,
            clip_sample_range=1.0,
        )
    if scheduler.config.prediction_type != "v_prediction":
        raise RuntimeError("Expected the model's v_prediction training scheduler.")

    model.requires_grad_(False).eval().to(device=device, dtype=torch.float32)
    unet = model
    train_scheduler = scheduler
    loaded_checkpoint = f"{MODEL_ID} ({checkpoint})" if checkpoint_dir is None else str(checkpoint)
    print(f"Ready: {loaded_checkpoint}; resolution={RESOLUTION}; AMP={amp_dtype}", flush=True)


# ---------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------

def prepare_context(init_image: Image.Image, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # Match the training dataset's nearest-neighbor preprocessing. The UI sends
    # native 512px images, so its generation path performs no resizing at all.
    rgba_image = init_image.convert("RGBA")
    if rgba_image.size != (RESOLUTION, RESOLUTION):
        rgba_image = rgba_image.resize((RESOLUTION, RESOLUTION), Image.Resampling.NEAREST)
    rgba = np.asarray(rgba_image, dtype=np.uint8)
    background = Image.new("RGBA", rgba_image.size, (0, 0, 0, 255))
    rgb = np.asarray(Image.alpha_composite(background, rgba_image).convert("RGB"), dtype=np.uint8)
    if mask.shape == (TILE_GRID_SIZE, TILE_GRID_SIZE):
        preserve = np.repeat(np.repeat(mask, TILE_SIZE, axis=0), TILE_SIZE, axis=1)
    elif mask.shape == (RESOLUTION, RESOLUTION):
        preserve = mask
    else:
        raise ValueError("Mask must be 4x4 tiles or 512x512 pixels.")
    return rgb, np.asarray(preserve, dtype=np.bool_) & (rgba[:, :, 3] == 255)


@torch.inference_mode()
def generate_masked_city(
    init_image: Image.Image,
    tile_mask: np.ndarray,
    conditions: list[float],
    seed: int,
    num_inference_steps: int,
) -> Image.Image:
    assert unet is not None and train_scheduler is not None

    rgb, preserve_pixels = prepare_context(init_image, tile_mask)

    context = (
        torch.from_numpy(rgb.copy()).permute(2, 0, 1).unsqueeze(0)
        .to(device=device, dtype=torch.float32).div(127.5).sub(1.0)
    )
    preserve_mask = torch.from_numpy(preserve_pixels.copy()).unsqueeze(0).unsqueeze(0)
    preserve_mask = preserve_mask.to(device=device, dtype=torch.float32)
    known_context = context * preserve_mask

    condition_maps = torch.tensor(conditions, device=device, dtype=torch.float32)
    condition_maps = torch.cat([condition_maps, condition_maps.new_ones(1)])
    condition_maps = condition_maps.view(1, 4, 1, 1).expand(-1, -1, RESOLUTION, RESOLUTION)

    sampler = DDIMScheduler.from_config(
        train_scheduler.config,
        prediction_type="v_prediction",
        timestep_spacing="trailing",
        clip_sample=True,
        clip_sample_range=1.0,
    )
    sampler.set_timesteps(num_inference_steps, device=device)

    generator = torch.Generator(device=device).manual_seed(seed)
    latents = torch.randn(context.shape, generator=generator, device=device, dtype=torch.float32)
    known_noise = torch.randn(context.shape, generator=generator, device=device, dtype=torch.float32)

    for timestep in sampler.timesteps:
        timestep_batch = timestep.reshape(1).to(dtype=torch.long)
        noisy_known = sampler.add_noise(context, known_noise, timestep_batch)
        latents = preserve_mask * noisy_known + (1.0 - preserve_mask) * latents
        model_input = torch.cat([latents, preserve_mask, known_context, condition_maps], dim=1)

        with torch.autocast("cuda", dtype=amp_dtype):
            velocity = unet(model_input, timestep_batch).sample

        latents = sampler.step(
            velocity.float(), timestep, latents, eta=0.0, use_clipped_model_output=True
        ).prev_sample

    if not torch.isfinite(latents).all():
        raise RuntimeError("Nonfinite sampling output.")

    output = (
        latents[0].float().add(1).div(2).clamp(0, 1)
        .permute(1, 2, 0).mul(255).round().byte().cpu().numpy()
    )
    output[preserve_pixels] = rgb[preserve_pixels]
    return Image.fromarray(output)


# ---------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    load_models(CHECKPOINT_DIR)
    try:
        yield
    finally:
        global unet, train_scheduler
        unet = None
        train_scheduler = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


app = FastAPI(title="City Diffusion Generator", version="4.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    if not INDEX_FILE.is_file():
        raise HTTPException(status_code=404, detail="index.html not found.")
    return FileResponse(INDEX_FILE)


@app.get("/health")
def health():
    loaded = unet is not None and train_scheduler is not None
    return {
        "status": "ok" if loaded else "loading",
        "device": str(device),
        "model_id": MODEL_ID,
        "checkpoint": loaded_checkpoint,
        "revision": loaded_revision,
        "requested_revision": MODEL_REVISION,
        "layers_per_block": unet.config.layers_per_block if unet is not None else None,
        "resolution": RESOLUTION,
        "tile_grid_size": TILE_GRID_SIZE,
        "tile_size": TILE_SIZE,
        "model_loaded": loaded,
    }


@app.get("/mask-stats")
def mask_stats():
    return get_tile_mask_stats()


@app.post("/generate", response_model=GenerateResponse)
def generate(request: GenerateRequest):
    if unet is None or train_scheduler is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")

    image = decode_image(request.image_b64)
    if request.mask_b64 is not None:
        mask_image = decode_image(request.mask_b64)
        if mask_image.size != (RESOLUTION, RESOLUTION):
            raise HTTPException(status_code=400, detail="mask_b64 must be a 512x512 image.")
        tile_mask = (np.asarray(mask_image.convert("L")) >= 128) & (np.asarray(mask_image)[:, :, 3] == 255)
    else:
        tile_mask = np.asarray(request.tile_mask, dtype=np.bool_)
    _, effective_mask = prepare_context(image, tile_mask)
    # Track fully preserved tiles using the effective, alpha-aware mask.
    record_tile_mask_request(effective_mask.reshape(TILE_GRID_SIZE, TILE_SIZE, TILE_GRID_SIZE, TILE_SIZE).all(axis=(1, 3)))
    seed = request.seed if request.seed is not None else secrets.randbelow(2_147_483_648)

    with generation_lock:
        try:
            output = generate_masked_city(
                image, tile_mask, request.conditions, seed, request.num_inference_steps
            )
        except torch.cuda.OutOfMemoryError as exc:
            torch.cuda.empty_cache()
            raise HTTPException(status_code=503, detail="CUDA out of memory. Free GPU memory and retry.") from exc
        except Exception as exc:
            logging.exception("Generation failed")
            raise HTTPException(status_code=500, detail="Generation failed; see server logs.") from exc

    return GenerateResponse(
        image_b64=encode_png(output),
        seed=seed,
        resolution=RESOLUTION,
        inference_steps=request.num_inference_steps,
    )


if __name__ == "__main__":
    import uvicorn

    parser = argparse.ArgumentParser(description="City Diffusion server")
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT_DIR)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    if args.checkpoint is not None:
        CHECKPOINT_DIR = args.checkpoint
    uvicorn.run(app, host=args.host, port=args.port)
