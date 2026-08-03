"""Principled colour grading for photographic point colours.

Aerial RGB often arrives over-saturated and flat. Rather than nudging channels
ad hoc, this applies the standard grading chain, each step in the space where it
is actually meaningful:

    white balance -> [decode to linear] -> exposure -> contrast -> [encode to sRGB]
                  -> saturation -> gamma

Exposure and contrast belong in **linear light**, where they correspond to
physical gain and to a power law about mid-grey. Saturation belongs in the
display-encoded space, where it matches what the eye calls "colourfulness" and
where a luma-preserving blend keeps brightness intact.

Grading is meant for photographic layers only. Applying it to a categorical
palette (semantic, forest, ...) would break the correspondence between a figure
and its legend, so callers pass an explicit list of layers to touch.
"""
import numpy as np

# Rec.709 luma weights — the standard perceptual brightness of R, G, B
LUMA = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)

# Mid-grey in linear light: the pivot contrast rotates about
MID_GREY = 0.18


def srgb_to_linear(x):
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(x):
    x = np.clip(x, 0.0, None)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * x ** (1 / 2.4) - 0.055)


def grade(rgb, saturation=1.0, contrast=1.0, exposure=0.0, gamma=1.0,
          white_balance=None):
    """Grade float RGB in [0, 1] (sRGB-encoded). Returns the same shape.

    Args:
        saturation: 1 keeps as-is, <1 mutes, 0 is greyscale, >1 boosts.
        contrast: power law about mid-grey in linear light. 1 is neutral,
            >1 deepens shadows and brightens highlights.
        exposure: stops of linear gain. +1 doubles the light.
        gamma: final tweak on the encoded signal; <1 lifts midtones.
        white_balance: per-channel multiplier, e.g. [1.0, 1.0, 0.95] to warm.
    """
    out = np.asarray(rgb, dtype=np.float32).copy()

    if white_balance is not None and tuple(white_balance) != (1.0, 1.0, 1.0):
        out *= np.asarray(white_balance, dtype=np.float32)
        out = np.clip(out, 0.0, 1.0)

    if exposure != 0.0 or contrast != 1.0:
        linear = srgb_to_linear(out)
        if exposure != 0.0:
            linear *= 2.0 ** exposure
        if contrast != 1.0:
            # power law about mid-grey: preserves 0.18 exactly
            linear = MID_GREY * np.power(
                np.clip(linear, 1e-8, None) / MID_GREY, contrast)
        out = linear_to_srgb(linear).astype(np.float32)

    if saturation != 1.0:
        luma = (out * LUMA).sum(axis=-1, keepdims=True)
        out = luma + saturation * (out - luma)

    out = np.clip(out, 0.0, 1.0)

    if gamma != 1.0:
        out = np.power(out, gamma, dtype=np.float32)

    return np.clip(out, 0.0, 1.0)


def is_neutral(settings):
    """True when a grading config would leave the image untouched."""
    return (settings.get("saturation", 1.0) == 1.0
            and settings.get("contrast", 1.0) == 1.0
            and settings.get("exposure", 0.0) == 0.0
            and settings.get("gamma", 1.0) == 1.0
            and tuple(settings.get("white_balance") or (1.0, 1.0, 1.0)) == (1.0, 1.0, 1.0))
