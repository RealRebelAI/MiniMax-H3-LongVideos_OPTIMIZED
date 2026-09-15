# H3-LongVideos -- https://github.com/Smite79/MiniMax-H3-LongVideos
# Copyright (c) 2026 Smite79. All rights reserved.
# Redistribution, in whole or in part, requires written permission.
# This notice may not be removed or altered. See LICENSE.
"""Decisions about which pictures may condition a shot."""

import torch
import node_helpers
from h3_runtime import (H3_FPS, AUDIO_LATENT_FPS, _empty_av_latent, _resize, ref_image_canvas,
                        frame_levels)
from h3_audio import _SILENCE_STATUS, _silent_audio_latent, _pin_audio_silence


def may_carry_room(previous_cast, current_cast, tagged_names):
    """A previous frame is safe as a reference only when it adds no subject."""
    previous = [name for name in (previous_cast or ()) if name]
    current = set(current_cast or ())
    tagged = set(tagged_names or ())
    return bool(previous) and all(name in current for name in previous) \
        and not any(name in tagged for name in previous)


def recoverable_subject(cast, tagged_names, returning_names, captured):
    """Return the sole safe recovered subject, or an empty string."""
    people = [name for name in (cast or ()) if name]
    if len(people) != 1:
        return ""
    name = people[0]
    return name if (name not in set(tagged_names or ())
                    and name in set(returning_names or ())
                    and captured.get(name) is not None) else ""


KEYFRAME_SAFE_AUG = 0.99       # below this, a ref aug would soften the keyframe too

# What ONE boundary is allowed to claim it measured. Wider than any real per-pass drift,
# narrow enough that a bad frame -- a flash, a cut to black, a frame the model lost --
# cannot swing the estimate. The median across boundaries does the real rejecting.
LEVEL_GAIN_CAP = 0.12          # in log-gain, so +-12.7% of contrast
LEVEL_OFFSET_CAP = 0.05
# The within-shot term is believed only when boundaries AGREE on its sign, and even then
# only this far: within-shot change is often the author's (a light switched off), so it is
# the half of the signal that cannot be trusted on its own.
LEVEL_SHOT_GAIN_CAP = 0.015
LEVEL_SHOT_OFFSET_CAP = 0.010
LEVEL_AGREE = 2.0 / 3.0
LEVEL_MIN_OBS = 3
# What the correction may do to one handoff, whatever it measured. A cut should not carry
# a visible grade step: shot N's last frame reaches the video uncorrected while N+1 is
# sampled from a corrected keyframe, so an uncapped correction trades burn-in for a pop at
# every join -- the same class of complaint, differently shaped.
LEVEL_GAIN_LO, LEVEL_GAIN_HI = 0.80, 1.25
LEVEL_OFFSET_BOUND = 0.02
# Below this a frame is too flat for a contrast RATIO to mean anything.
LEVEL_MIN_SIGMA = 0.01


class HandoffLevels:
    """Takes the grade the chain adds to itself back out of the handoff.

    THE MEASUREMENT, which is the whole reason this needs no scene list. At every
    boundary the render holds two pictures that are SUPPOSED to be the same frame: K,
    the handoff it gave the shot, and R, frame one of what came back -- the model's own
    reproduction of K, from a keyframe labelled sigma 0.001. Nothing was asked to change
    between them, so everything separating them is the chain's own doing and none of it
    is the author's intent. That is the one difference in the loop that can be corrected
    without guessing at anybody's lighting, and R costs nothing to look at: it is the
    frame trim_seam throws away.

    A beat that walks into a darker room moves K, and R follows it there. So the level is
    never anchored, never compared to shot 1, and never compared to a target -- only K
    against its own reproduction, boundary by boundary.

    WHAT IT WILL NOT FIX. Clipping already baked into earlier shots, because the VAE
    clamps every decode and headroom spent is gone. Softening, which is a different
    measurement and a different cause. Anything spatial -- ghosting, local burn, identity
    drift. A tone curve with a knee in it, since this is affine per channel; the residual
    in the report is how that would show itself. The first boundary, which has nothing to
    measure yet. And a deliberate monotone move -- a film that dims every single beat --
    loses a bounded, reported fraction of itself."""

    def __init__(self):
        self._bg, self._bo = [], []      # per boundary: K -> R, the chain's own drift
        self._sg, self._so = [], []      # per shot: R -> last frame, believed only on agreement
        self.applied = []                # (gain, offset) actually used, for the report

    def observe(self, given, repro, last=None, pre_up_last=None):
        """Record one boundary. given is the keyframe this shot got, repro is frame one
        of what it produced, last is its final frame, pre_up_last the handoff it hands on.

        last/pre_up_last are how the pre-upscale handoff and the post-upscale output are
        put in the same frame of reference: their difference IS the pipeline's own offset,
        measured on one frame that went through both, so it can be subtracted from the
        K->R reading instead of being mistaken for drift. With latent_upscale off they are
        the same frame and the term is zero."""
        gm, gs = frame_levels(given)
        rm, rs = frame_levels(repro)
        if gm is None or rm is None:
            return False
        if float(gs.min()) < LEVEL_MIN_SIGMA or float(rs.min()) < LEVEL_MIN_SIGMA:
            return False
        ug = torch.zeros(3)
        uo = torch.zeros(3)
        lm, ls = frame_levels(last) if last is not None else (None, None)
        if pre_up_last is not None and lm is not None:
            pm, ps = frame_levels(pre_up_last)
            if pm is not None and float(ps.min()) >= LEVEL_MIN_SIGMA:
                ug = torch.log(ls / ps)
                uo = lm - pm
        self._bg.append((torch.log(rs / gs) - ug).clamp(-LEVEL_GAIN_CAP, LEVEL_GAIN_CAP))
        self._bo.append((rm - gm - uo).clamp(-LEVEL_OFFSET_CAP, LEVEL_OFFSET_CAP))
        if lm is not None and float(ls.min()) >= LEVEL_MIN_SIGMA:
            self._sg.append(torch.log(ls / rs))
            self._so.append(lm - rm)
        return True

    def _agreed(self, rows, cap):
        """The median of rows, but only per channel where at least LEVEL_AGREE of them
        share its sign. A within-shot change the boundaries disagree about is content, not
        drift, and content must not be corrected."""
        out = torch.zeros(3)
        if len(rows) < LEVEL_MIN_OBS:
            return out
        st = torch.stack(rows)
        med = st.median(dim=0).values
        agree = ((st * med.sign().unsqueeze(0)) > 0).float().mean(dim=0)
        keep = agree >= LEVEL_AGREE
        return torch.where(keep, med.clamp(-cap, cap), out)

    def estimate(self):
        """(gain_log, offset) the chain is drifting by per boundary, per channel."""
        if not self._bg:
            return None, None
        g = torch.stack(self._bg).median(dim=0).values + self._agreed(self._sg, LEVEL_SHOT_GAIN_CAP)
        o = torch.stack(self._bo).median(dim=0).values + self._agreed(self._so, LEVEL_SHOT_OFFSET_CAP)
        return g, o

    def gains(self, strength):
        """(gain, offset) as 3-vectors, or (None, None) when there is nothing worth doing.

        Separate from note() because more than one frame leaves a shot -- the handoff,
        and any face captured for a return several shots later -- and they have to carry
        the SAME grade. A recovered face arriving at a different exposure from the shot
        around it would be a new bug of exactly the kind this is fixing."""
        g, o = self.estimate()
        if g is None or strength <= 0:
            return None, None
        gain = torch.exp(-float(strength) * g).clamp(LEVEL_GAIN_LO, LEVEL_GAIN_HI)
        off = (-float(strength) * o).clamp(-LEVEL_OFFSET_BOUND, LEVEL_OFFSET_BOUND)
        # The next thing this frame meets is an 8-bit quantisation, so a correction under
        # 1/255 would be erased on the way there. Claiming it would be worse than silence.
        if float((gain - 1.0).abs().max()) < 1e-3 and float(off.abs().max()) < 1.0 / 255.0:
            return None, None
        return gain, off

    def note(self, gain, off):
        """Record what was applied, and say it in one clause."""
        self.applied.append((gain.clone(), off.clone()))
        return (f"gain {'/'.join(f'{float(v):.3f}' for v in gain)} "
                f"level {'/'.join(f'{float(v):+.4f}' for v in off)}")

def _keyframe_latent(vae, hand_img):
    """The keyframe latent for this shot: an ENCODE of the previous shot's last frame.

    This was briefly an optimisation -- pass the previous shot's own latent straight
    through and skip a VAE round trip per boundary. It was wrong, and it degraded
    every shot after the first.

    A keyframe is ONE pixel frame, and H3's grid puts that at 5f -> TWO latent
    frames. Slicing [:, :, -1:] off a finished shot hands over one. Worse, the video
    VAE is causal: the last latent of a 72-frame sequence encodes its temporal
    context, not a standalone opening frame, so even at the right count it does not
    mean what a keyframe means. The spatial-size guard could not see either problem.

    The round trip is real but it is one lossy step on a correctly formed anchor,
    which beats a cheap malformed one."""
    return vae.encode(hand_img)


def _build_ref_images(vae, images, gen_w, gen_h, mode="match"):
    """(tokenizer items, DiT blocks) for a list of reference IMAGE tensors.

    The tokenizer labels each one `<Picture N>:` itself, in the order given here --
    so the roster the prompt refers to is decided by input order, not by anything
    written in the prompt."""
    items, blocks = [], []
    for img in images:
        if img is None:
            continue
        h, w = int(img.shape[1]), int(img.shape[2])
        tw, th = ref_image_canvas(w, h, gen_w, gen_h, mode)
        resized = _resize(img[:1], tw, th, "disabled")
        items.append({"type": "image", "data": resized})
        blocks.append({"kind": "image", "latent_h": th // 16, "latent_w": tw // 16,
                       "latent": vae.encode(resized)})
    return items, blocks


def build_conditioning(clip, vae, audio_vae, prompt, width, height, length,
                       handoff=None, refs=None,
                       ref_noise_aug=0.999, silent=False, ref_image_size="match",
                       handoff_as_ref=False, speech_lead_seconds=0.0, speech_tail_frames=0):
    """Encode prompt, identity references, keyframe, and audio constraints for a shot."""
    latent, fc = _empty_av_latent(width, height, length, H3_FPS)
    refs = [r for r in (refs or []) if r is not None]

    hand_img = None
    if handoff is not None:
        hand_img = _resize(handoff[:1], width, height, "disabled")

    # REFERENCES AND THE KEYFRAME RIDE TOGETHER. This is the arrangement the node
    # had before I broke it, and the reason is in ComfyUI's own layout:
    #
    #   model_base.py:2183-2191  cond_video_latents = keyframe latents THEN ref latents
    #   model.py PackedLayout    emits keyframe "cond" segments THEN ref "ref_img" ones
    #
    # The two orders agree, so both channels coexist. A shot takes its references AND
    # a real keyframe: the keyframe ANCHORS the first frame, which is what continuity
    # needs, while a reference only supplies identity. They are not alternatives.
    #
    # I had read "<Picture 1>" as MEANING the first frame on fl2va, and rearranged the
    # roster around that. It does not. Which image is the first frame is decided by
    # resolved_frame_index in minimax_keyframes, not by a label's number -- the labels
    # are only how the images are shown to the VLM, and what they have to line up with
    # is the <Picture N> tags in the prompt.
    #
    # So references come FIRST and keep slots 1..N, which is what a sheet line's
    # `Name: <Picture 1>, ...` points at, and the handoff is appended AFTER them where
    # it disturbs no numbering. It has to be in the list at all because
    # tokenize_with_weights is either/or: passing minimax_ref_items makes it ignore
    # `images` outright, so leaving the handoff out means the VLM is never shown where
    # the shot left off and re-imagines the scenery -- same place, new room.
    keyframe_ok = ref_noise_aug is None or float(ref_noise_aug) >= KEYFRAME_SAFE_AUG
    # One aug covers every visual condition row, references AND the keyframe. Below
    # KEYFRAME_SAFE_AUG the keyframe latent would be noised and labelled at the wrong
    # timestep, so the handoff stops being an anchor and rides as an extra reference
    # instead: weaker continuity, but nothing pretending to anchor while carrying noise.
    # ...or because the caller asked for it. A shot that introduces somebody already
    # in position wants the room this picture carries and NOT the first frame it
    # would force, and that is a demotion the aug knows nothing about.
    carry_as_ref = bool(hand_img is not None
                        and (handoff_as_ref or (refs and not keyframe_ok)))

    enc_refs = refs + ([hand_img] if carry_as_ref else [])
    items, blocks = ([], [])
    if enc_refs:
        items, blocks = _build_ref_images(vae, enc_refs, width, height, ref_image_size)
    if hand_img is not None and not carry_as_ref:
        items = items + [{"type": "image", "data": hand_img}]

    if items:
        tokens = clip.tokenize(prompt, minimax_ref_items=items)
    else:
        tokens = clip.tokenize(prompt)
    cond = clip.encode_from_tokens_scheduled(tokens)

    vals = {}
    if blocks:
        vals["minimax_refs"] = blocks
        # How CLEAN the references are shown. One aug covers every conditioning
        # latent, keyframe included -- which is why softening references below
        # KEYFRAME_SAFE_AUG would soften the anchor too.
        if ref_noise_aug is not None:
            vals["minimax_visual_cond_noise_aug"] = float(ref_noise_aug)

    kfs = []
    if hand_img is not None and not carry_as_ref:
        kfs.append({"resolved_frame_index": 0,
                    "latent": _keyframe_latent(vae, hand_img)})
    # Audio keyframes are extra conditioning rows in H3's PackedLayout. Pin the
    # generated target stream instead, so the joint model also sees a quiet mouth.
    # A dialogue shot pins its opening (the lead) and, past the line's estimated end,
    # its close (the tail); the span between is the model's.
    if silent or float(speech_lead_seconds or 0.0) > 0.0 or int(speech_tail_frames or 0) > 0:
        _SILENCE_STATUS["asked"] += 1
        if audio_vae is None:
            _SILENCE_STATUS["why"] = "no audio VAE is wired to the node"
        else:
            sil = _silent_audio_latent(audio_vae, fc, H3_FPS)
            if sil is None:
                _SILENCE_STATUS["why"] = ("the audio VAE would not encode a silent "
                                          "second -- the wrong VAE is on the "
                                          "audio_vae input")
            else:
                lead = None if silent else round(float(speech_lead_seconds) *
                                                 AUDIO_LATENT_FPS)
                tail = 0 if silent else int(speech_tail_frames or 0)
                if _pin_audio_silence(latent, sil, lead, tail):
                    _SILENCE_STATUS["applied"] += 1
                else:
                    _SILENCE_STATUS["why"] = "the silent latent did not match the shot"
    if kfs:
        vals["minimax_keyframes"] = kfs
    if vals:
        cond = node_helpers.conditioning_set_values(cond, vals)
    return cond, latent, fc, carry_as_ref
