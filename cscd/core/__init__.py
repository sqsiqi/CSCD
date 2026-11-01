"""Core modules of CSCD.

These functions are intentionally withheld before paper acceptance. The
signatures below document the interface used by :class:`cscd.pipeline.CSCDPipeline`
so that the released data, inversion, and evaluation code can be integrated
with the full implementation once it is available.
"""

_MESSAGE = (
    "The core CSCD modules are temporarily withheld and will be released "
    "upon paper acceptance. See the README for the released components."
)


def iawvg_attention(
    query,
    key_text,
    value_text,
    key_image,
    value_image,
    foreground_mask,
    alpha=0.9,
    beta=0.0,
):
    """Identity anchoring with weighted visual guidance (IAWVG)."""
    raise NotImplementedError(_MESSAGE)


def afsa(hidden_states, foreground_mask, alpha_t, eps=1.0e-6):
    """Adaptive foreground stylization AdaIN (AFSA)."""
    raise NotImplementedError(_MESSAGE)


def sslo(latent, prompt_embeds, insertion_mask, config):
    """Semantic-style latent optimization (SSLO)."""
    raise NotImplementedError(_MESSAGE)


def bdth(image, background, insertion_mask, config):
    """Background-driven tone harmonization (BDTH)."""
    raise NotImplementedError(_MESSAGE)


def psbf(image, background, insertion_mask):
    """Pixel-space background fusion (PSBF)."""
    raise NotImplementedError(_MESSAGE)
