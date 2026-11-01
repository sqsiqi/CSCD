#!/usr/bin/env python3
"""Run CSCD on a single sample.

The core stages are withheld before paper acceptance. Use ``--prepare-only`` to
exercise the released data, mask, and pixel-composition code without model
weights.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cscd.image_io import save_mask, save_rgb  # noqa: E402
from cscd.pipeline import CSCDConfig, CSCDPipeline  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CSCD demo")
    parser.add_argument("--input", required=True, help="Sample folder")
    parser.add_argument("--output", default="outputs", help="Output folder")
    parser.add_argument("--prompt", default=None, help="Optional text prompt")
    parser.add_argument("--sdxl", default="", help="Path to SDXL base")
    parser.add_argument("--ip-adapter", default="", help="Path to IP-Adapter SDXL")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Only run data loading, mask transfer, and pixel composition",
    )
    return parser.parse_args()


def load_config(path: str) -> CSCDConfig:
    if not Path(path).is_file():
        return CSCDConfig()
    with open(path, "r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    sampling = raw.get("sampling", {})
    iawvg = raw.get("iawvg", {})
    afsa = raw.get("afsa", {})
    sslo = raw.get("sslo", {})
    return CSCDConfig(
        num_steps=sampling.get("num_steps", 20),
        inversion_stop=sampling.get("inversion_stop", 10),
        guidance_scale=sampling.get("guidance_scale", 7.5),
        seed=sampling.get("seed", 3407),
        alpha=iawvg.get("alpha", 0.9),
        beta=iawvg.get("beta", 0.0),
        eps=afsa.get("eps", 1.0e-6),
        alpha_t_mid=afsa.get("alpha_t_mid", 0.10),
        alpha_t_high=afsa.get("alpha_t_high", 0.05),
        t_start=sslo.get("t_start", 14),
        t_end=sslo.get("t_end", 18),
        eta=sslo.get("eta", 0.05),
        K=sslo.get("steps", 3),
        lambda_sty=sslo.get("lambda_sty", 1.0),
        lambda_sem=sslo.get("lambda_sem", 0.5),
    )


def main() -> None:
    args = parse_args()
    sample = Path(args.input)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    pipe = CSCDPipeline(
        sdxl_path=args.sdxl,
        ip_adapter_path=args.ip_adapter or None,
        config=load_config(args.config),
    )

    state = pipe.prepare(
        sample / "background_image.png",
        sample / "ref_image.png",
        sample / "background_mask.png",
        sample / "ref_mask.png",
        prompt=args.prompt,
    )

    save_rgb(state["blend"], output / "blend.png")
    save_mask(state["insertion_mask"], output / "insertion_mask.png")
    print(f"Saved the released intermediate results to {output}")

    if args.prepare_only:
        return

    try:
        pipe(
            sample / "background_image.png",
            sample / "ref_image.png",
            sample / "background_mask.png",
            sample / "ref_mask.png",
            prompt=args.prompt,
        )
    except NotImplementedError as exc:
        print("\n[core withheld]")
        print(exc)


if __name__ == "__main__":
    main()
