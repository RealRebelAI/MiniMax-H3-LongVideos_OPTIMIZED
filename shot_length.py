# H3-LongVideos -- https://github.com/Smite79/MiniMax-H3-LongVideos
# Copyright (c) 2026 Smite79. All rights reserved.
# Redistribution, in whole or in part, requires written permission.
# This notice may not be removed or altered. See LICENSE.
"""
H3 Shot Length  (single model-free source for shot length)
==========================================================
Holds ONE shot-length value and emits it as both seconds and a grid-aligned
H3 frame count. Because it never reads the model, it can sit upstream of
Kijai's Model Preview Override without creating a cycle -- unlike the preview
node or the sampler, which read the model to compute their frame counts and so
cannot feed anything that produces the model.

Wire:
  H3 Shot Length (seconds) -> H3-LongVideos FL2VA (shot_seconds)
  H3 Shot Length (frames)  -> Model Preview Override (preview_frames)

One value, entered once here, drives both -- no manual re-entry, no cycle.

Note: this is a FIXED shot length you choose. The sampler's *auto* (VRAM-
picked) length can't be used for preview_frames, because computing it requires
reading the model, which is the very dependency that creates the loop. Set the
sampler's shot_seconds from this node's `seconds` output so the two agree.
"""

H3_MAX_FRAMES = 362


def align_up_grid(n):
    n = max(5, int(n))
    while n % 17 != 5:
        n += 1
    return n


class H3ShotLength:
    CATEGORY = "MiniMax-H3/utils"
    FUNCTION = "emit"
    RETURN_TYPES = ("FLOAT", "INT", "STRING")
    RETURN_NAMES = ("seconds", "frames", "info")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "shot_seconds": ("FLOAT", {"default": 5.0, "min": 0.2, "max": 15.1, "step": 0.5,
                    "tooltip": "Length of each shot. Feeds the sampler's shot_seconds AND (as frames) "
                               "the preview override. Max ~15s (362 frames)."}),
                "fps": ("INT", {"default": 24, "min": 1, "max": 60}),
            },
            "optional": {
                "cap_to_h3_max": ("BOOLEAN", {"default": True,
                    "tooltip": "Clamp frames to 362 (~15s), H3's single-clip maximum."}),
            },
        }

    def emit(self, shot_seconds, fps, cap_to_h3_max=True):
        fps = max(1, int(fps))
        frames = align_up_grid(round(float(shot_seconds) * fps))
        capped = cap_to_h3_max and frames > H3_MAX_FRAMES
        if capped:
            frames = H3_MAX_FRAMES
        info = (f"{shot_seconds:g}s/shot @ {fps}fps -> {frames} frames"
                f"{' (capped 362)' if capped else ''}")
        return (round(float(shot_seconds), 3), frames, info)


NODE_CLASS_MAPPINGS = {"H3ShotLength": H3ShotLength}
NODE_DISPLAY_NAME_MAPPINGS = {"H3ShotLength": "H3 Shot Length"}
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
