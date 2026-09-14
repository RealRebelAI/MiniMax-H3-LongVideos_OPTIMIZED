# H3-LongVideos -- https://github.com/Smite79/MiniMax-H3-LongVideos
# Copyright (c) 2026 Smite79. All rights reserved.
# Redistribution, in whole or in part, requires written permission.
# This notice may not be removed or altered. See LICENSE.
"""
H3-LongVideos
=============
Make long (up to ~120s) MiniMax-H3 videos from a single prompt + a single
length, in ComfyUI.

Nodes:
  * H3-LongVideos      (sampler.py)      - one prompt -> a chain of shots with
                                           synchronised audio. Your text is passed
                                           through verbatim; the node does the
                                           chaining, not the writing.
  * H3 Shot Length     (shot_length.py)  - one shot length as seconds AND a valid
                                           H3 frame count (17k+5 grid, 362 cap);
                                           wire `seconds` -> the sampler and
                                           `frames` -> a preview override
  * H3 Model Inspector (inspector.py)    - report base precision (BF16/FP8/NVFP4/MXFP8)
                                           and whether this card runs it natively

Sizing: the sampler's `resolution` preset supplies the ASPECT RATIO and its
`megapixels` widget supplies the SIZE (1 MP = 1024x1024, ComfyUI's own
convention). At 1.00MP every native preset reproduces its own dimensions.
Scaling from the preset rather than a nominal ratio is what makes that exact --
1344x768 is 1.750, i.e. 7:4, NOT 16:9 (1.778), and 1536x672 is 16:7, not 21:9.
Set megapixels to 0 to use the preset's dimensions verbatim.

The sampler registers under four keys -- H3LongVideos, H3LongVideosFL2VA,
H3LongVideosV1 and H3LongVideosREF2VA -- all aliases onto the same class, so every
workflow saved under any previous name keeps loading. REF2VA was briefly a second,
duplicated sampler; it was 94% the same file, every shared fix had to be made
twice, and one such mirror silently emitted a clause twice. It is now folded in.

Install: put this whole folder in ComfyUI/custom_nodes/ and restart ComfyUI.
"""

from .sampler import NODE_CLASS_MAPPINGS as _s_c, NODE_DISPLAY_NAME_MAPPINGS as _s_d
from .shot_length import NODE_CLASS_MAPPINGS as _sl_c, NODE_DISPLAY_NAME_MAPPINGS as _sl_d
from .inspector import NODE_CLASS_MAPPINGS as _i_c, NODE_DISPLAY_NAME_MAPPINGS as _i_d
from .overlay import NODE_CLASS_MAPPINGS as _o_c, NODE_DISPLAY_NAME_MAPPINGS as _o_d

NODE_CLASS_MAPPINGS = {**_s_c, **_sl_c, **_i_c, **_o_c}
NODE_DISPLAY_NAME_MAPPINGS = {**_s_d, **_sl_d, **_i_d, **_o_d}

# No WEB_DIRECTORY: the only frontend script was autoshift.js, which wrote
# auto-derived flow shifts back into the widgets. auto_shift is gone -- its premise
# was wrong for distilled LoRAs -- and nothing else here needs a browser-side hook.

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
