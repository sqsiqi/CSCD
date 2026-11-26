# Method Overview

CSCD treats cross-domain composition as a conflict between change and
preservation. The inserted object must adopt the background style, remain
recognizable, and preserve the original background outside the fusion mask.
Existing methods balance these goals in one latent representation, where a
stronger style loss erodes identity, a stronger identity constraint suppresses
stylization, and the VAE decoder reconstructs the whole image and can change
background details.

CSCD separates the constraints and enforces each one in the representation
where it is best defined.

## 1. Identity Anchoring with Weighted Visual Guidance (IAWVG)

Identity is a correspondence between the reference and the generated tokens, so
it is enforced in cross-attention. IAWVG uses the pretrained IP-Adapter to
supply visual information. The image-attention output is applied only to
foreground queries, while text attention remains global. Reference and
background image features are weighted by α and β; the paper uses reference-only
guidance with α = 0.9 and β = 0. This keeps the reference from overwriting the
background.

## 2. Adaptive Foreground Stylization AdaIN (AFSA)

Style is a set of feature statistics, so it is enforced in self-attention.
Mask-guided AdaIN transfers background statistics to foreground tokens in the
conditional branch. A denoising schedule controls the transfer strength, and a
per-token renormalization keeps the feature magnitude stable. The unconditional
branch keeps a clean structural prior.

## 3. Semantic-Style Latent Optimization (SSLO)

Statistics matching can transfer color and texture, but it does not specify
whether the generated foreground represents the intended object. During selected
denoising steps, SSLO minimizes a task-specific energy that combines a CLIP
semantic term with a VGG Gram style term. The gradient is restricted to the
insertion region, and the background inversion trajectory is restored outside
the latent mask.

## 4. Background-Driven Tone Harmonization (BDTH)

After decoding, a residual tone mismatch can remain on sketch-like or nearly
monochromatic backgrounds. BDTH uses fixed image cues to detect these
backgrounds and, when the rule is active, blends the decoded foreground toward
its luminance with a soft foreground mask. The correction is inactive on
colorful backgrounds.

## 5. Pixel-Space Background Fusion (PSBF)

PSBF composites the decoded image with the original background under the
image-resolution fusion mask `M_d`. Where `M_d = 0`, the final pixels equal the
original background at the working resolution. Inside `M_d`, the generated
composition is retained. The fusion mask can differ from the evaluation
placement mask, so this pixel guarantee applies outside `M_d`.

## Design Rationale

The five stages act in cross-attention, self-attention, latent, and pixel
spaces. IAWVG and AFSA act on foreground tokens, SSLO acts on the masked latent,
BDTH adjusts the decoded foreground, and PSBF acts outside the fusion mask.
These complementary controls avoid balancing all goals in a single latent
variable. The target background is the only style source, so the method needs no
separate style reference, domain classifier, or paired training data. The text
prompt is optional: the reference supplies content and the background supplies
style.

The full technical details, equations, and experimental results are in the
paper.
