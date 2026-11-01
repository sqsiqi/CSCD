# Method Overview

CSCD treats cross-domain composition as a conflict between change and
preservation. The inserted object must adopt the background style, remain
recognizable, and leave the background untouched. Existing methods balance
these goals in one latent representation, where a stronger style loss erodes
identity, a stronger identity constraint suppresses stylization, and the VAE
decoder changes the background.

CSCD separates the constraints and enforces each one in the representation
where it is best defined.

## 1. Identity Anchoring with Weighted Visual Guidance (IAWVG)

Identity is a correspondence between the reference and the generated tokens,
so it is enforced in cross-attention. A content-dominant visual prompt is
injected only into the cross-attention queries inside the insertion mask. The
text branch remains global. This keeps the reference from overwriting the
background.

## 2. Adaptive Foreground Stylization AdaIN (AFSA)

Style is a set of feature statistics, so it is enforced in self-attention.
Mask-guided AdaIN transfers background statistics to foreground tokens in the
conditional branch. A time gate controls when the transfer is applied, and a
per-token renormalization keeps the feature magnitude stable. The unconditional
branch keeps a clean structural prior.

## 3. Semantic-Style Latent Optimization (SSLO)

Statistics matching can transfer color and texture, but it does not know what
the object is. In a narrow denoising window, SSLO minimizes a task-specific
energy that combines a CLIP semantic term with a VGG Gram style term. The
gradient is restricted to the insertion region, so the background stays intact.

## 4. Background-Driven Tone Harmonization (BDTH)

After denoising, a residual tone mismatch can remain on sketch-like or
monochromatic backgrounds. A simple, interpretable prior detects these
backgrounds, and a first-order luminance transfer shifts the foreground mean
toward the background while preserving local contrast. The correction is
inactive on colorful backgrounds.

## 5. Pixel-Space Background Fusion (PSBF)

The background must be preserved exactly. Instead of fusing in latent space,
PSBF composites the decoded foreground with the original background outside
the insertion mask. Background preservation is exact by construction, not by a
loss term.

## Design Rationale

The five stages act in different spaces and have limited support: IAWVG and
AFSA act on foreground tokens, SSLO acts on the masked latent, and PSBF acts
outside the mask. Because the interventions do not share a common variable,
they do not undo one another. The target background is the only style source,
so the method needs no style reference, domain classifier, or paired training
data. The text prompt is optional: the reference supplies content and the
background supplies style.

The full technical details, equations, and experimental results are in the
paper.
