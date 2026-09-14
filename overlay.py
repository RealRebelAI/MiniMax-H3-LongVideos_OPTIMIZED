# H3-LongVideos -- https://github.com/Smite79/MiniMax-H3-LongVideos
# Copyright (c) 2026 Smite79. All rights reserved.
# Redistribution, in whole or in part, requires written permission.
# This notice may not be removed or altered. See LICENSE.
"""
PIL text overlays for H3-LongVideos -- watermark and intro title.

Text is COMPOSITED onto the decoded frames, never asked of the model. H3 (like
every video diffusion model) renders text as plausible-looking letterforms that
drift, warp and re-spell themselves frame to frame; a watermark that changes
shape every frame is worse than none. Compositing gives pixel-identical text on
every frame at zero sampling cost, and keeps the words out of the prompt where
they would otherwise steal conditioning from the actual shot.

Both overlays are WHITE text drawn on a fully transparent RGBA layer, then
alpha-blended over the video -- so only the glyphs themselves land on the frame
and the picture shows through everywhere else.

Everything here is best-effort: any failure returns the frames untouched with a
note, because a cosmetic overlay must never lose a finished render.
"""

import torch

BLEND_CHUNK = 64          # frames blended per slice -- bounds peak RAM on long chains

# Auto-fit: text is wrapped, then shrunk in FIT_SHRINK steps until the block fits
# inside the margins. MIN_FONT_PX is the point below which the text would be
# unreadable anyway, so the loop stops there and lets PIL clip rather than spin.
MIN_FONT_PX = 8
FIT_SHRINK = 0.92
FIT_STEPS = 48

# Fonts to try when the requested one cannot be loaded. PIL resolves bare names
# against the system font directory, so "arial.ttf" works on Windows as-is.
FONT_FALLBACKS = ("arial.ttf", "segoeui.ttf", "DejaVuSans.ttf", "LiberationSans-Regular.ttf")

# Anchor -> (x, y) as a fraction of the free space: 0 = hard against the left/top
# margin, 1 = hard against the right/bottom, 0.5 = centered.
POSITIONS = {
    "bottom-right":  (1.0, 1.0),
    "bottom-left":   (0.0, 1.0),
    "bottom-center": (0.5, 1.0),
    "top-right":     (1.0, 0.0),
    "top-left":      (0.0, 0.0),
    "top-center":    (0.5, 0.0),
    "center":        (0.5, 0.5),
    "lower-third":   (0.5, 0.72),
}


def _load_font(name, px):
    """A truetype font at px, falling back through the known-present faces and
    finally to PIL's bitmap default (which ignores size -- ugly, but never fatal)."""
    from PIL import ImageFont
    px = max(8, int(px))
    for cand in ([name] if name else []) + list(FONT_FALLBACKS):
        try:
            return ImageFont.truetype(cand, px)
        except Exception:
            continue
    return ImageFont.load_default()


def _measure(draw, text, font, stroke_px, spacing):
    """(x0, y0, x1, y1) of a multi-line block, tolerant of older Pillow builds."""
    try:
        return draw.multiline_textbbox((0, 0), text, font=font, align="center",
                                       stroke_width=stroke_px, spacing=spacing)
    except TypeError:                      # older Pillow: no stroke/spacing kwargs
        return draw.multiline_textbbox((0, 0), text, font=font, align="center")


def _wrap(draw, text, font, max_w, stroke_px, spacing):
    """Greedy word-wrap every hard line to max_w. A single word wider than the
    frame cannot be broken -- the shrink loop in render_text_layer handles that."""
    out = []
    for hard in text.split("\n"):
        words = hard.split()
        if not words:
            out.append("")
            continue
        cur = words[0]
        for wd in words[1:]:
            trial = cur + " " + wd
            b = _measure(draw, trial, font, stroke_px, spacing)
            if b[2] - b[0] <= max_w:
                cur = trial
            else:
                out.append(cur)
                cur = wd
        out.append(cur)
    return "\n".join(out)


def _fit(draw, text, font_name, px, max_w, max_h, stroke_px, line_spacing, wrap=True):
    """Largest size at or below px whose wrapped block fits (max_w, max_h).

    Without this, a title is drawn at the requested size and whatever runs past the
    frame is simply CLIPPED by PIL -- silently, with no error and no note. That is
    the whole "overlays don't work at other resolutions" failure: the size is a
    percentage, so the same text that fits 1344x768 overflows a 512-wide portrait
    canvas and loses its outer characters."""
    px = max(MIN_FONT_PX, int(px))
    for _ in range(FIT_STEPS):
        font = _load_font(font_name, px)
        spacing = int(max(0.0, px * (line_spacing - 1.0)))
        fitted = _wrap(draw, text, font, max_w, stroke_px, spacing) if wrap else text
        box = _measure(draw, fitted, font, stroke_px, spacing)
        if (box[2] - box[0] <= max_w and box[3] - box[1] <= max_h) or px <= MIN_FONT_PX:
            return font, fitted, box, spacing, px
        px = max(MIN_FONT_PX, int(px * FIT_SHRINK))
    return font, fitted, box, spacing, px


def render_text_layer(width, height, text, font_px, position="bottom-right",
                      margin_pct=3.0, font_name="", stroke_px=0, line_spacing=1.15,
                      wrap=True):
    """White text on a transparent RGBA canvas the size of one frame.

    The block is WRAPPED and SHRUNK until it fits inside the margins, so the same
    settings render legibly on every supported preset -- portrait canvases and the
    512 tier included -- instead of being clipped at the frame edge.

    Returns (rgb, alpha, bbox): rgb [H,W,3] float 0..1, alpha [H,W,1] float 0..1
    (zero everywhere except the glyphs and their optional stroke), and the tight
    (x0, y0, x1, y1) box of non-transparent pixels so the blend only has to touch
    the region the text actually occupies. None when there is nothing to draw."""
    from PIL import Image, ImageDraw
    import numpy as np

    text = (text or "").strip()
    if not text:
        return None

    img = Image.new("RGBA", (int(width), int(height)), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    stroke_px = max(0, int(stroke_px))

    margin = int(min(width, height) * max(0.0, margin_pct) / 100.0)
    ax, ay = POSITIONS.get(position, POSITIONS["bottom-right"])

    # Measure first, so the block is placed by its real size rather than a guess --
    # and fit it to the space the margins actually leave.
    max_w = max(1, int(width) - 2 * margin)
    max_h = max(1, int(height) - 2 * margin)
    font, text, box, spacing, font_px = _fit(draw, text, font_name, font_px, max_w, max_h,
                                             stroke_px, line_spacing, wrap)
    tw, th = box[2] - box[0], box[3] - box[1]

    free_w = max(0, int(width) - 2 * margin - tw)
    free_h = max(0, int(height) - 2 * margin - th)
    x = margin + free_w * ax - box[0]
    y = margin + free_h * ay - box[1]

    kwargs = dict(font=font, fill=(255, 255, 255, 255), align="center")
    if stroke_px:
        kwargs.update(stroke_width=stroke_px, stroke_fill=(0, 0, 0, 255))
    try:
        draw.multiline_text((x, y), text, spacing=spacing, **kwargs)
    except TypeError:
        draw.multiline_text((x, y), text, **kwargs)

    arr = np.asarray(img, dtype=np.float32) / 255.0        # [H, W, 4]
    alpha = arr[..., 3:4]
    if not alpha.any():
        return None
    # Tight bbox of drawn pixels: blending a whole 1344x768 frame for a corner
    # watermark would cost ~50x more work on a 3000-frame chain.
    ys, xs = np.nonzero(alpha[..., 0] > 0.0)
    bbox = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
    return (torch.from_numpy(arr[..., :3].copy()),
            torch.from_numpy(alpha.copy()),
            bbox)


def blend_layer(frames, layer, frame_alpha=None, opacity=1.0):
    """Alpha-composite a rendered layer over frames [N,H,W,3] in 0..1, in place.

    frame_alpha is an optional per-frame multiplier (length N) -- that is what
    makes an intro title hold and then fade instead of sitting on the whole
    video. Frames whose multiplier is 0 are skipped entirely."""
    if layer is None:
        return frames
    rgb, alpha, (x0, y0, x1, y1) = layer
    n = frames.shape[0]
    if frame_alpha is None:
        frame_alpha = torch.ones(n, dtype=torch.float32)
    frame_alpha = frame_alpha.to(torch.float32).clamp(0.0, 1.0) * float(opacity)

    a_crop = alpha[y0:y1, x0:x1, :].to(frames.dtype)
    c_crop = rgb[y0:y1, x0:x1, :].to(frames.dtype)
    live = (frame_alpha > 0).nonzero().flatten().tolist()
    for s in range(0, len(live), BLEND_CHUNK):
        idx = live[s:s + BLEND_CHUNK]
        fa = frame_alpha[idx].to(frames.dtype).view(-1, 1, 1, 1)
        sub = frames[idx, y0:y1, x0:x1, :]
        a = a_crop * fa
        frames[idx, y0:y1, x0:x1, :] = sub * (1.0 - a) + c_crop * a
    return frames


def hold_fade_alpha(total_frames, hold_frames, fade_frames):
    """Per-frame opacity for an intro: full through hold_frames, then a linear
    ramp to zero over fade_frames, then nothing. Returns a length-N tensor."""
    a = torch.zeros(int(total_frames), dtype=torch.float32)
    hold = max(0, min(int(hold_frames), int(total_frames)))
    a[:hold] = 1.0
    fade = max(0, min(int(fade_frames), int(total_frames) - hold))
    if fade:
        a[hold:hold + fade] = torch.linspace(1.0, 0.0, fade + 2)[1:-1]
    return a


def apply_overlays(frames, fps, watermark="", wm_position="bottom-right", wm_size_pct=4.0,
                   wm_opacity=0.75, wm_margin_pct=3.0, intro="", intro_seconds=3.0,
                   intro_fade=0.6, intro_size_pct=9.0, intro_position="center",
                   font_name="", stroke_px=0):
    """Composite the watermark (every frame) and the intro title (first seconds
    only, then faded out). Returns (frames, note). Never raises -- a cosmetic
    overlay must not be able to destroy a finished render."""
    notes = []
    if frames is None or frames.ndim != 4 or frames.shape[0] == 0:
        return frames, ""
    n, h, w = frames.shape[0], frames.shape[1], frames.shape[2]
    frames = frames.contiguous()
    # Size from the SHORT edge, not the height. Height is the long edge on every
    # portrait preset, so a height-based percentage drew 9:16 text ~1.75x larger
    # than the same setting at 16:9 -- on the canvas with the LEAST room for it.
    # The short edge makes one setting mean the same apparent size at every ratio.
    short = min(int(w), int(h))

    if (watermark or "").strip():
        try:
            layer = render_text_layer(w, h, watermark, short * max(0.5, wm_size_pct) / 100.0,
                                      wm_position, wm_margin_pct, font_name, stroke_px)
            if layer is not None:
                blend_layer(frames, layer, None, wm_opacity)
                notes.append(f"watermark composited ({wm_position}, {wm_opacity:.0%})")
        except Exception as e:
            notes.append(f"watermark skipped ({type(e).__name__}: {e})")

    if (intro or "").strip():
        try:
            layer = render_text_layer(w, h, intro, short * max(0.5, intro_size_pct) / 100.0,
                                      intro_position, 6.0, font_name, stroke_px)
            if layer is not None:
                hold = round(max(0.0, float(intro_seconds)) * fps)
                fade = round(max(0.0, float(intro_fade)) * fps)
                fa = hold_fade_alpha(n, hold, fade)
                if fa.max() > 0:
                    blend_layer(frames, layer, fa, 1.0)
                    notes.append(f"intro title composited ({hold}f hold + {fade}f fade)")
                else:
                    notes.append("intro title skipped (no hold or fade frames)")
        except Exception as e:
            notes.append(f"intro title skipped ({type(e).__name__}: {e})")

    return frames, "; ".join(notes)


# --- the node --------------------------------------------------------------
# Overlays used to be eleven widgets on the sampler. They are compositing, not
# generation: they run after the render, touch no conditioning, and most renders
# never use them. Eleven widgets is a fifth of that node's surface spent on a
# feature you reach for occasionally, so they live out here instead -- wire this
# node when you want a watermark or a title card, and the sampler stays about
# making video.
class H3Overlay:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "fps": ("INT", {"default": 24, "min": 1, "max": 120,
                    "tooltip": "Only used to convert the intro's SECONDS into frames. "
                               "Match your save node; H3 always renders 24."}),
            },
            "optional": {
                "watermark_text": ("STRING", {"default": "",
                    "tooltip": "Composited onto every frame with PIL -- never asked of the "
                               "model, which would render drifting, re-spelling letterforms. "
                               "Empty disables it."}),
                "watermark_position": (["bottom-right", "bottom-left", "bottom-center",
                                        "top-right", "top-left", "top-center", "center"],
                    {"default": "bottom-right"}),
                "watermark_size": ("FLOAT", {"default": 4.0, "min": 0.5, "max": 40.0, "step": 0.5,
                    "tooltip": "Cap height as a percentage of the frame's SHORT edge, so it "
                               "looks the same at every aspect ratio."}),
                "watermark_opacity": ("FLOAT", {"default": 0.75, "min": 0.0, "max": 1.0, "step": 0.05}),
                "watermark_margin": ("FLOAT", {"default": 3.0, "min": 0.0, "max": 25.0, "step": 0.5,
                    "tooltip": "Inset from the frame edge, as a percentage of the short edge."}),
                "intro_text": ("STRING", {"multiline": True, "default": "",
                    "tooltip": "Title card held over the opening frames, then faded out. "
                               "Empty disables it."}),
                "intro_position": (["center", "lower-third", "top-center", "bottom-center"],
                    {"default": "center"}),
                "intro_seconds": ("FLOAT", {"default": 3.0, "min": 0.0, "max": 30.0, "step": 0.5,
                    "tooltip": "How long the title holds at full opacity before the fade."}),
                "intro_fade": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 10.0, "step": 0.1,
                    "tooltip": "Fade-out length in seconds, after the hold."}),
                "intro_size": ("FLOAT", {"default": 9.0, "min": 1.0, "max": 40.0, "step": 0.5,
                    "tooltip": "Cap height as a percentage of the short edge."}),
                "overlay_font": ("STRING", {"default": "arial.ttf",
                    "tooltip": "TrueType font for BOTH overlays: a bare name resolved against "
                               "the system font folder (arial.ttf, arialbd.ttf, segoeui.ttf) or "
                               "a full path. Falls back to the first font that loads."}),
                "overlay_stroke": ("INT", {"default": 0, "min": 0, "max": 20,
                    "tooltip": "Black outline thickness around the white text. 0 keeps it pure "
                               "white; 2-3 makes it survive a bright sky or a white wall."}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("images", "info")
    FUNCTION = "run"
    CATEGORY = "MiniMax-H3/utils"
    DESCRIPTION = ("Composite a watermark and/or an intro title onto finished frames. "
                   "Put it AFTER any upscale, so glyphs are rasterized at the final "
                   "pixel size instead of being interpolated up with the picture.")

    def run(self, images, fps=24, watermark_text="", watermark_position="bottom-right",
            watermark_size=4.0, watermark_opacity=0.75, watermark_margin=3.0,
            intro_text="", intro_position="center", intro_seconds=3.0, intro_fade=0.6,
            intro_size=9.0, overlay_font="arial.ttf", overlay_stroke=0):
        # Compositing is in-place, so copy only when there is something to draw.
        wanted = bool((watermark_text or "").strip() or (intro_text or "").strip())
        if not wanted:
            return (images.detach().cpu(),
                    "no overlay text given -- frames unchanged")
        frames = images.detach().cpu().clone()
        frames, note = apply_overlays(
            frames, max(1, int(fps)), watermark_text, watermark_position, watermark_size,
            watermark_opacity, watermark_margin, intro_text, intro_seconds,
            intro_fade, intro_size, intro_position, overlay_font, overlay_stroke)
        if not note:
            note = "overlays applied"
        return (frames, note)


NODE_CLASS_MAPPINGS = {"H3Overlay": H3Overlay}
NODE_DISPLAY_NAME_MAPPINGS = {"H3Overlay": "H3 Overlay (watermark + title)"}
__all__ = ["apply_overlays", "NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
