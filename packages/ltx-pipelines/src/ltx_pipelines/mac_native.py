import argparse
import logging
import os
from datetime import datetime
from pathlib import Path

import torch

from ltx_core.model.video_vae import TilingConfig, get_video_chunks_number
from ltx_pipelines.distilled import DistilledPipeline
from ltx_pipelines.utils.constants import AUDIO_SAMPLE_RATE, DEFAULT_SEED
from ltx_pipelines.utils.helpers import get_mac_profile_values, is_apple_silicon
from ltx_pipelines.utils.media_io import encode_video


def _resolve_non_fp8_checkpoint(models_root: Path) -> Path:
    candidates = (
        models_root / "ltx-2-19b-distilled.safetensors",
        models_root / "ltx-2-19b-dev.safetensors",
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    msg = (
        "No non-FP8 checkpoint found. Expected one of:\n"
        f"  - {candidates[0]}\n"
        f"  - {candidates[1]}"
    )
    raise FileNotFoundError(msg)


def _default_output_path(base_output_dir: Path, profile: str) -> Path:
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = base_output_dir / f"{ts}-{profile}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir / "output.mp4"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Mac-native launcher for LTX-2 distilled pipeline.")
    parser.add_argument("--prompt", type=str, required=True, help="Prompt used for generation.")
    parser.add_argument(
        "--profile",
        type=str,
        choices=("fast", "quality"),
        default="fast",
        help="Preset tuned for Apple Silicon.",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed.")
    parser.add_argument(
        "--models-root",
        type=Path,
        default=Path("models/LTX-2"),
        help="Path containing checkpoints, upsampler, and Gemma assets.",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default=None,
        help="Optional explicit non-FP8 checkpoint path. If omitted, picks a local non-FP8 checkpoint automatically.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/mac-native"),
        help="Base directory where timestamped run folders are created.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help="Optional explicit output video path. Overrides --output-dir timestamped output.",
    )
    return parser


@torch.inference_mode()
def main() -> None:
    logging.getLogger().setLevel(logging.INFO)
    parser = _build_parser()
    args = parser.parse_args()

    if not is_apple_silicon():
        logging.warning("mac_native launcher is intended for Apple Silicon; proceeding anyway.")

    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

    models_root = args.models_root.expanduser().resolve()
    checkpoint_path = (args.checkpoint_path.expanduser().resolve() if args.checkpoint_path else None) or (
        _resolve_non_fp8_checkpoint(models_root).resolve()
    )
    if "fp8" in checkpoint_path.name.lower():
        raise ValueError("mac_native does not support FP8 checkpoints on Apple MPS. Provide a non-FP8 checkpoint.")

    upsampler_path = (models_root / "ltx-2-spatial-upscaler-x2-1.0.safetensors").resolve()
    gemma_root = (models_root / "gemma-3-12b-it-qat-q4_0-unquantized").resolve()

    if not upsampler_path.exists():
        raise FileNotFoundError(f"Missing spatial upsampler: {upsampler_path}")
    if not gemma_root.exists():
        raise FileNotFoundError(f"Missing Gemma root: {gemma_root}")

    output_path = (
        args.output_path.expanduser().resolve() if args.output_path else _default_output_path(args.output_dir, args.profile).resolve()
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    profile_values = get_mac_profile_values(profile=args.profile, is_two_stage=True)
    logging.info(f"Using mac-native profile '{args.profile}': {profile_values}")
    logging.info(f"Checkpoint: {checkpoint_path}")
    logging.info(f"Output: {output_path}")

    pipeline = DistilledPipeline(
        checkpoint_path=str(checkpoint_path),
        spatial_upsampler_path=str(upsampler_path),
        gemma_root=str(gemma_root),
        loras=[],
        fp8transformer=False,
    )

    tiling_config = TilingConfig.default()
    video_chunks_number = get_video_chunks_number(profile_values["num_frames"], tiling_config)

    video, audio = pipeline(
        prompt=args.prompt,
        seed=args.seed,
        height=profile_values["height"],
        width=profile_values["width"],
        num_frames=profile_values["num_frames"],
        frame_rate=profile_values["frame_rate"],
        images=[],
        tiling_config=tiling_config,
    )

    encode_video(
        video=video,
        fps=profile_values["frame_rate"],
        audio=audio,
        audio_sample_rate=AUDIO_SAMPLE_RATE,
        output_path=str(output_path),
        video_chunks_number=video_chunks_number,
    )
    print(output_path)


if __name__ == "__main__":
    main()
