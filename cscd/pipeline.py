"""Pipeline skeleton for CSCD.

The released code covers model loading, mask preparation, pixel composition,
and the stage interface. The core stages are imported from :mod:`cscd.core`
and are withheld until paper acceptance.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from .image_io import load_rgb
from .mask import compose_blend, load_mask, transfer_mask


@dataclass
class CSCDConfig:
    """Configuration used in the paper."""

    num_steps: int = 20
    inversion_stop: int = 10
    guidance_scale: float = 7.5
    seed: int = 3407

    alpha: float = 0.9
    beta: float = 0.0

    eps: float = 1.0e-6
    alpha_t_mid: float = 0.10
    alpha_t_high: float = 0.05

    t_start: int = 14
    t_end: int = 18
    eta: float = 0.05
    K: int = 3
    lambda_sty: float = 1.0
    lambda_sem: float = 0.5


class CSCDPipeline:
    """CSCD inference pipeline.

    Parameters
    ----------
    sdxl_path:
        Path to the Stable Diffusion XL base model.
    ip_adapter_path:
        Path to the IP-Adapter SDXL checkpoint.
    device:
        Torch device string.
    dtype:
        ``float16`` or ``float32``.
    """

    def __init__(
        self,
        sdxl_path: str | Path,
        ip_adapter_path: Optional[str | Path] = None,
        device: str = "cuda",
        dtype: str = "float16",
        config: Optional[CSCDConfig] = None,
    ) -> None:
        self.sdxl_path = str(sdxl_path)
        self.ip_adapter_path = None if ip_adapter_path is None else str(ip_adapter_path)
        self.device = device
        self.dtype = dtype
        self.config = config or CSCDConfig()

    def prepare(
        self,
        background_path: str | Path,
        reference_path: str | Path,
        background_mask_path: str | Path,
        reference_mask_path: str | Path,
        prompt: Optional[str] = None,
    ) -> dict:
        """Load a sample and build the pixel-level blend.

        This part is fully released. The returned dictionary is the input to
        the core stages.
        """
        background = load_rgb(background_path)
        reference = load_rgb(reference_path)
        background_mask = load_mask(
            background_mask_path, (background.shape[1], background.shape[0])
        )
        reference_mask = load_mask(reference_mask_path)

        insertion_mask = transfer_mask(reference_mask, background_mask)
        blend = compose_blend(background, reference, reference_mask, background_mask)

        return {
            "background": background,
            "reference": reference,
            "background_mask": background_mask,
            "reference_mask": reference_mask,
            "insertion_mask": insertion_mask,
            "blend": blend,
            "prompt": prompt,
        }

    def __call__(
        self,
        background_path: str | Path,
        reference_path: str | Path,
        background_mask_path: str | Path,
        reference_mask_path: str | Path,
        prompt: Optional[str] = None,
    ) -> np.ndarray:
        """Run the pipeline.

        The five stages are called in the order used in the paper:

        1. IAWVG anchors identity in foreground-restricted cross-attention.
        2. AFSA transfers background statistics to foreground tokens.
        3. SSLO refines the latent in a narrow denoising window.
        4. BDTH corrects residual tone mismatch after decoding.
        5. PSBF restores the original background outside the insertion mask.

        The core implementations are temporarily withheld; calling this method
        raises ``NotImplementedError`` with a clear message.
        """
        state = self.prepare(
            background_path,
            reference_path,
            background_mask_path,
            reference_mask_path,
            prompt,
        )

        # The full release applies the five stages to ``state`` in this order:
        #   1. IAWVG anchors identity in foreground-restricted cross-attention.
        #   2. AFSA transfers background statistics to foreground tokens.
        #   3. SSLO refines the latent in a narrow denoising window.
        #   4. BDTH corrects residual tone mismatch after decoding.
        #   5. PSBF restores the original background outside the insertion mask.
        raise NotImplementedError(
            "The core CSCD stages (IAWVG, AFSA, SSLO, BDTH, PSBF) are "
            "temporarily withheld and will be released upon paper acceptance. "
            "The released code already provides data loading, mask "
            "preparation, pixel composition, inversion, and evaluation."
        )
