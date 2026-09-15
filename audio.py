# H3-LongVideos -- https://github.com/Smite79/MiniMax-H3-LongVideos
# Copyright (c) 2026 Smite79. All rights reserved.
# Redistribution, in whole or in part, requires written permission.
# This notice may not be removed or altered. See LICENSE.
"""Audio policy shared by conditioning and soundtrack assembly."""

from dataclasses import dataclass

import torch
import comfy.nested_tensor
from h3_runtime import temporal_shape


@dataclass(frozen=True)
class ShotAudio:
    speech: bool
    sounded: bool
    voiced_only: bool
    silence_enabled: bool
    lead_seconds: float
    latent_fps: int
    # The tail. Everything after the line's expected end is pinned the way the lead
    # pins everything before its start. All three default off, so a ShotAudio built
    # the old way -- six positional arguments -- behaves exactly the old way.
    line_seconds: float = 0.0      # planner's estimate of the spoken line
    tail_seconds: float = 0.0      # free audio kept after that estimate; 0 = no tail pin
    frame_count: int = 0           # the shot in pixel frames; the audio T comes from it

    @property
    def pinned(self):
        return self.silence_enabled and not self.speech and not self.sounded

    @property
    def lead_frames(self):
        if not self.speech or self.lead_seconds <= 0:
            return 0
        return round(self.lead_seconds * self.latent_fps)

    @property
    def tail_frames(self):
        """Audio latent frames pinned at the END of a dialogue shot.

        The lead pins the opening so the line cannot start early; nothing pinned the
        close, and a 2s line in a 9s shot left 7s of open branch in a shot the model
        knows has a voice in it -- which is where speech carries on past the line, or
        doubles it. The free span is lead + the line's estimate + tail_seconds; the
        rest is held at encoded silence. The model chooses WHEN to speak, so the
        margin is the author's dial: a clipped word costs more than a second of babble.
        Off unless the shot speaks, the margin is set, and at least half a second would
        be pinned -- a sliver is not worth the risk of clipping."""
        if (not self.speech or self.tail_seconds <= 0 or self.line_seconds <= 0
                or self.frame_count <= 0):
            return 0
        total = temporal_shape(self.frame_count)[2]
        free = self.lead_frames + round((self.line_seconds + self.tail_seconds) * self.latent_fps)
        tail = total - free
        return tail if tail >= round(0.5 * self.latent_fps) else 0


_SILENT_UNIT = {"lat": None, "key": None}

# ---------------------------------------------------------------------------
# NOTHING HERE IS SYNTHESISED ANY MORE. Removed on the report: "Just get rid of
# the ambient sounds all together. They sound horrid. Go back to the model's
# natural audio."
#
# What was here built the soundtrack's non-vocal half out of shaped noise: a room
# tone from the scene's own wording (synth_ambient, over a table of recipes, with
# plain_bed under it as a floor) and 21 foley recipes laid into the shots whose
# audio branch is pinned to silence (foley_for, over _hits/_band/_room and later
# _contact/_flow/_creak, timed off the picture's own movement for footsteps).
#
# It went in because a shot pinned to silence cannot get audio from the model at
# all -- prompt text never opens a branch -- so auto_sound was writing sounds into
# prompts that could not make them. That reasoning was sound and the thing it built
# still did not pass: reported first as footsteps sounding like heartbeats and a
# bathroom that tapped, and then, once both of those measured clean, as horrid
# anyway. Synthesis that measures right and sounds wrong is the end of that road.
#
# So the audio is the model's, whole. H3 is a joint model and the audio branch is
# where its sound comes from; the prompt still describes what a shot sounds like,
# which is the half that was always doing the real work.
#
# The consequence, which is real and is reported in info rather than left to be
# discovered: a shot with no line and no sound you wrote is pinned to silence and
# is now SILENT. The pin is not a bug and is deliberately untouched -- it is what
# stops a free branch filling itself with babble and a face lip-syncing to it.
# Write the sound into the beat to open the branch on purpose, or wire a recording
# to ambient_audio, which is played under the finished track and conditions
# nothing. mix_ambient below is that path, and it is all that is left here.
# ---------------------------------------------------------------------------


def _seamless_loop(x, n, sr):
    """[C, M] -> [C, n], looped with a crossfade so the join does not click.

    Plain tiling puts a discontinuity at every repeat, once per loop length. In a
    bed that is meant to sit under everything unnoticed, a regular click is the one
    thing that gets noticed -- the same objection that made the silence latent
    ping-pong its interior rather than tile it. Here the material is real audio
    being PLAYED rather than a latent being conditioned on, so it cannot be
    reversed: a room tone read backwards is fine, but footsteps are not. Crossfade
    instead, which works on both."""
    m = int(x.shape[-1])
    if m <= 0:
        return None
    if m >= n:
        return x[..., :n]
    fade = min(int(0.25 * sr), m // 4)
    if fade < 1:
        reps = -(-n // m)
        return x.repeat(1, reps)[..., :n]
    # OVERLAP-ADD the tail onto the head, and shorten the unit by the overlap. The
    # unit then runs x[m-fade] .. x[m-fade-1], so tiling it steps between samples
    # that were adjacent in the source and there is no discontinuity anywhere.
    #
    # Measured, because the obvious construction is wrong: appending the crossfade
    # to the END of a full-length unit leaves it finishing on x[fade-1] while the
    # next repeat starts on x[0], which are not adjacent -- a 2s tone that does not
    # divide evenly gave a 64x jump at the join, worse than plain tiling's 41x.
    t = torch.linspace(0.0, 1.0, fade, dtype=x.dtype, device=x.device)
    head = x[..., :fade] * t + x[..., m - fade:] * (1.0 - t)
    unit = torch.cat([head, x[..., fade:m - fade]], dim=-1)
    if int(unit.shape[-1]) < 1:
        reps = -(-n // m)
        return x.repeat(1, reps)[..., :n]
    reps = -(-n // int(unit.shape[-1]))
    return unit.repeat(1, reps)[..., :n]


def mix_ambient(audio, sr, bed, level):
    """Lay an ambient bed UNDER a finished soundtrack. -> (waveform, note).

    The bed is PLAYED, not conditioned on: it is the file, at the level asked for,
    under whatever the model generated. That is the whole reason to do it here
    rather than in the sampler -- ambience needs no cooperation from a joint model,
    has nothing to lip-sync to, and so cannot put a voice in a wordless shot. The
    conditioning path can only steer the branch toward something bed-LIKE, and on a
    shot with a line it competes with the line.

    Defensive throughout, like the silence latent: any failure returns the audio
    untouched with a note saying so, because a bed is a nicety and a render is not.
    """
    try:
        if audio is None or bed is None or float(level or 0.0) <= 0.0:
            return audio, ""
        w = bed.get("waveform") if isinstance(bed, dict) else None
        if w is None or not int(getattr(w, "ndim", 0)):
            return audio, ("ambient_audio is wired but carries no waveform, so nothing "
                           "was laid under the soundtrack")
        w = w[0] if w.dim() == 3 else w              # [B, C, M] -> [C, M]
        if w.dim() != 2 or w.shape[-1] < 2:
            return audio, ("ambient_audio is too short to loop, so nothing was laid "
                           "under the soundtrack")
        w = w.detach().to(dtype=audio.dtype, device=audio.device)
        b_sr = int((bed.get("sample_rate") if isinstance(bed, dict) else 0) or 0)
        # RESAMPLE, or the bed plays at the wrong speed and pitch. Linear is coarse
        # for music and inaudible on a room tone, which is what this input is for.
        resampled = ""
        if b_sr > 0 and b_sr != int(sr):
            want = max(2, int(round(w.shape[-1] * float(sr) / float(b_sr))))
            w = torch.nn.functional.interpolate(
                w.unsqueeze(0), size=want, mode="linear", align_corners=False)[0]
            resampled = f", resampled from {b_sr} Hz"
        ch = int(audio.shape[1])
        if int(w.shape[0]) != ch:
            w = (w.mean(dim=0, keepdim=True).repeat(ch, 1) if int(w.shape[0]) > ch
                 else w[:1].repeat(ch, 1))
        n = int(audio.shape[-1])
        loop = _seamless_loop(w, n, int(sr))
        if loop is None:
            return audio, ""
        out = audio + loop.unsqueeze(0) * float(level)
        # NORMALISE rather than clip. Clipping a bed that pushed a loud line over
        # the top distorts the LINE, which is the thing worth keeping.
        peak = float(out.abs().max())
        gain = ""
        if peak > 1.0:
            out = out / peak
            gain = f", and the mix was scaled by {1.0 / peak:.2f} to stop it clipping"
        secs = w.shape[-1] / float(sr)
        return out, (f"an ambient bed was laid under the whole soundtrack at level "
                     f"{float(level):.2f} -- {secs:.1f}s of audio{resampled}, looped "
                     f"with a crossfade so the join does not click{gain}. It is your "
                     f"file, played under what the model generated: it conditions "
                     f"nothing, so it cannot put a voice in a wordless shot the way "
                     f"an inferred bed did. Shots pinned to silence keep their silent "
                     f"conditioning and get the bed on top, which is what makes a "
                     f"wordless shot sound like a room instead of a mute")
    except Exception as exc:
        return audio, (f"the ambient bed could not be mixed ({type(exc).__name__}), so "
                       f"the soundtrack is unchanged")


_SILENCE_STATUS = {"asked": 0, "applied": 0, "why": ""}


_SILENT_SECONDS = 2


_SILENT_EDGE = 4


def _silent_audio_latent(audio_vae, frame_count, fps):
    """A keyframe audio latent of actual SILENCE, or None if it cannot be made.

    H3 is a JOINT model: the mouth follows the audio branch. On a shot with no
    scripted line the branch is otherwise unconditioned, and an unconditioned audio
    branch invents a voice -- which the picture then lip-syncs to. The lips-closed
    sentence is arguing with a stream that has already decided someone is talking.

    REBUILT 2026-09-05, from measurements against the real VAE rather than from
    reasoning. The previous version encoded one second, kept a SINGLE interior
    frame and repeated it, on the argument that silence is homogeneous. It is not,
    in latent space: encoded silence has genuine frame-to-frame variation (delta
    mean 0.002-0.004, max 0.021), and a repeated frame has a delta of exactly
    0.000000. That is a flat signal no encoder produces, and a model handed
    conditioning outside its own distribution has every reason to disregard it --
    which is an audio branch back to inventing a voice, with the report saying
    silence went on.

    The fix that version was avoiding is real too: tiling the whole encoded second
    end to end leaves a 25x spike at each join (0.554 against 0.022), once per
    second, which is a metronome in the conditioning of a joint model.

    So: encode two seconds, drop the padded ends, and PING-PONG the interior --
    forward, reversed, forward. Every join repeats a frame, so there is no seam,
    and the interior statistics are the encoder's own. Measured over a 9s shot:

        one frame repeated   peak 0.000686   delta mean 0.000000   max 0.000000
        whole 2s tiled       peak 0.000314   delta mean 0.017451   max 0.554715
        interior ping-pong   peak 0.000566   delta mean 0.002039   max 0.021159

    where the encoder's own interior is mean 0.0021, max 0.0212. Decoded peak
    0.000566 on a +/-1.0 scale is about -65 dBFS: silence.

    Everything here stays defensive. Shapes are CHECKED against what the layout
    expects rather than assumed, and any failure returns None so the shot falls
    back to an unconditioned branch instead of breaking the render -- the caller
    reports when that happens, so it is no longer a silent failure.
    """
    try:
        sr = int(getattr(audio_vae, "audio_sample_rate", 0) or 0)
        if sr <= 0:
            return None
        _, _, want_t = temporal_shape(frame_count, fps)
        if want_t <= 0:
            return None
        key = (id(audio_vae), sr)
        block = _SILENT_UNIT.get("lat") if _SILENT_UNIT.get("key") == key else None
        if block is None:
            # CHANNELS LAST. comfy.sd.VAE.encode() does `pixel_samples.movedim(-1, 1)`
            # before handing off, so the audio VAE -- which wants [B, 2, L] -- must be
            # given [B, L, 2]. Passing [B, 2, L] raises inside the encoder, and an
            # early version did exactly that: swallowed by the guard below, so the
            # whole layer silently did nothing.
            #
            # Two seconds, encoded ONCE and cached. Encoding a full 15s shot instead
            # cost a VAE pass big enough to OOM mid-render on a 16GB card, where the
            # failure again degraded silently to no conditioning at all.
            enc = audio_vae.encode(torch.zeros((1, sr * _SILENT_SECONDS, 2)))
            if enc is None or enc.dim() != 4 or enc.shape[1] != 32:
                return None
            if enc.shape[-1] <= 2 * _SILENT_EDGE + 1:
                return None
            block = enc[..., _SILENT_EDGE:-_SILENT_EDGE].detach().to("cpu").clone()
            _SILENT_UNIT["lat"] = block
            _SILENT_UNIT["key"] = key
        n = block.shape[-1]
        if n < 1:
            return None
        # Forward, reversed, forward... Each join repeats a frame, so the seam that
        # plain tiling leaves is gone while the interior variation is the encoder's.
        pieces, have, i = [], 0, 0
        while have < want_t:
            piece = block if i % 2 == 0 else torch.flip(block, dims=[-1])
            pieces.append(piece)
            have += n
            i += 1
        out = torch.cat(pieces, dim=-1)[..., :want_t].clone()
        if out.shape[-1] != want_t:
            return None
        return out
    except Exception:
        return None                         # never fail a render for a nicety


def _pin_audio_silence(latent, silence, lead_frames=None, tail_frames=0):
    """Start target audio at encoded silence and preserve the requested span(s).

    lead_frames None pins the whole shot. Otherwise the first lead_frames and the
    last tail_frames are held at silence and the span between is left to the model
    -- that is where the line goes. The tail is clipped to what the lead leaves, so
    the two can never overlap. Nothing pinned at all is a no-op, reported as False
    so the caller does not count it as applied."""
    try:
        video, audio = latent["samples"].unbind()
        silence = silence.to(device=audio.device, dtype=audio.dtype)
        if silence.shape != audio.shape:
            return False
        audio_mask = torch.ones_like(audio[:, :1])
        if lead_frames is None:
            audio_mask.zero_()
        else:
            t = audio.shape[-1]
            n = min(t, max(0, int(lead_frames)))
            m = min(t - n, max(0, int(tail_frames or 0)))
            if n <= 0 and m <= 0:
                return False
            if n > 0:
                audio_mask[..., :n] = 0
            if m > 0:
                audio_mask[..., t - m:] = 0
        latent["samples"] = comfy.nested_tensor.NestedTensor((video, silence))
        latent["noise_mask"] = comfy.nested_tensor.NestedTensor(
            (torch.ones_like(video[:, :1]), audio_mask))
        return True
    except Exception:
        return False
