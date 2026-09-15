# H3-LongVideos -- https://github.com/Smite79/MiniMax-H3-LongVideos
# Copyright (c) 2026 Smite79. All rights reserved.
# Redistribution, in whole or in part, requires written permission.
# This notice may not be removed or altered. See LICENSE.
"""Plan MiniMax-H3 shots, render their audio/video, and preserve continuity.

The node interface and prompt planning live here. Audio policy and synthesis,
conditioning assembly, and tensor/runtime operations have separate owner modules.
"""

import math
import os
import re
import sys
import time

import torch

import nodes
import comfy.utils
import comfy.sample
import comfy.samplers
import comfy.nested_tensor
import comfy.model_management as mm

# The prompt engine: scene state, read beat by beat, rendered once per shot.
# Imported by file path rather than by name so it resolves the same whether
# ComfyUI loads this package as `custom_nodes.H3-LongVideos-V1` or bare.
import importlib.util as _ilu


def _load_local(name, filename):
    spec = _ilu.spec_from_file_location(
        name, os.path.join(os.path.dirname(os.path.abspath(__file__)), filename))
    module = _ilu.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


engine = _load_local("h3_engine", "engine.py")
_plan_module = _load_local("h3_shot_plan", "shot_plan.py")
_runtime_module = _load_local("h3_runtime", "runtime.py")
_audio_module = _load_local("h3_audio", "audio.py")
_cond_module = _load_local("h3_conditioning", "conditioning.py")
ShotPlan = _plan_module.ShotPlan
PreparedVideo = _plan_module.PreparedVideo

# Internal helper exports retained for existing callers.
ShotAudio = _audio_module.ShotAudio
FrameAccumulator = _runtime_module.FrameAccumulator
frame_levels = _runtime_module.frame_levels
apply_levels = _runtime_module.apply_levels
H3_FPS = _runtime_module.H3_FPS
AUDIO_LATENT_FPS = _runtime_module.AUDIO_LATENT_FPS
KEYFRAME_SAFE_AUG = _cond_module.KEYFRAME_SAFE_AUG
AUTO_TILE_T = _runtime_module.AUTO_TILE_T
MAX_FRAMES = _runtime_module.MAX_FRAMES
CANVAS_MULTIPLE = _runtime_module.CANVAS_MULTIPLE
REF_IMAGE_SHORT_EDGE = _runtime_module.REF_IMAGE_SHORT_EDGE
_SILENT_UNIT = _audio_module._SILENT_UNIT
align_frame_count = _runtime_module.align_frame_count
video_latent_t = _runtime_module.video_latent_t
temporal_shape = _runtime_module.temporal_shape
ref_image_canvas = _runtime_module.ref_image_canvas
_resize = _runtime_module._resize
_empty_av_latent = _runtime_module._empty_av_latent
_auto_tile_t = _runtime_module._auto_tile_t
_decode_video = _runtime_module._decode_video
_decode_audio = _runtime_module._decode_audio
_seamless_loop = _audio_module._seamless_loop
mix_ambient = _audio_module.mix_ambient
_is_oom = _runtime_module._is_oom
_deep_cleanup = _runtime_module._deep_cleanup
DECODE_HEADROOM = _runtime_module.DECODE_HEADROOM
SAMPLE_HEADROOM = _runtime_module.SAMPLE_HEADROOM
_decode_headroom = _runtime_module._decode_headroom
_resident = _runtime_module._resident
_image_out_dtype = _runtime_module._image_out_dtype
_evict_all_but = _runtime_module._evict_all_but
_SILENCE_STATUS = _audio_module._SILENCE_STATUS
_SILENT_SECONDS = _audio_module._SILENT_SECONDS
_SILENT_EDGE = _audio_module._SILENT_EDGE
_silent_audio_latent = _audio_module._silent_audio_latent
_pin_audio_silence = _audio_module._pin_audio_silence
HandoffLevels = _cond_module.HandoffLevels
_keyframe_latent = _cond_module._keyframe_latent
_build_ref_images = _cond_module._build_ref_images
_sample_on_sigmas = _runtime_module._sample_on_sigmas
RESIZE_CHUNK = _runtime_module.RESIZE_CHUNK
_stream_chunks = _runtime_module._stream_chunks
_resize_short_edge = _runtime_module._resize_short_edge
_upscale_frames = _runtime_module._upscale_frames
_find_node = _runtime_module._find_node
_invoke_node = _runtime_module._invoke_node
build_conditioning = _cond_module.build_conditioning

RES_MULTIPLE = 32
# Latent frames decoded from the PRE-upscale latent to source the handoff. Enough
# for the VAE's temporal context to produce a clean last frame, and cheap.
HANDOFF_LATENT_TAIL = 8
GB = 1024 ** 3

# H3-Base is trained at 768 on the short edge; below that the whole frame softens.
NATIVE_RES = {
    "16:9": (1344, 768),
    "9:16": (768, 1344),
    "4:3":  (1024, 768),
    "3:4":  (768, 1024),
    "1:1":  (768, 768),
    "21:9": (1536, 672),
    "9:21": (672, 1536),
}


_LAST_MODEL_FP = {"fp": None}


def _call_node(cls, model, shift_video, shift_audio):
    """Call the H3 sampling node whether it uses the V1 (INPUT_TYPES/FUNCTION) or
    V3 (define_schema/execute) API, mapping the shift args by name."""
    inst = cls()
    # V1 API
    if hasattr(cls, "INPUT_TYPES") and getattr(cls, "FUNCTION", None):
        req = cls.INPUT_TYPES().get("required", {})
        kwargs = {}
        for name in req:
            low = name.lower()
            if low == "model":
                kwargs[name] = model
            elif "video" in low:
                kwargs[name] = float(shift_video)
            elif "audio" in low:
                kwargs[name] = float(shift_audio)
        out = getattr(inst, cls.FUNCTION)(**kwargs)
        # A V3 node exposes INPUT_TYPES and a truthy FUNCTION ('EXECUTE_NORMALIZED')
        # for compatibility, so this branch runs on 0.31+ too -- and there it returns
        # a NodeOutput, not a tuple. Without the unwrap the caller got the wrapper
        # object where a MODEL belonged. Unreachable today because the direct patch
        # succeeds first, which is exactly why it went unnoticed.
        out = getattr(out, "result", out)
        return out[0] if isinstance(out, (tuple, list)) else out
    # V3 API: an execute()/patch() classmethod taking model + shift kwargs
    fn = None
    for cand in ("execute", "patch", "apply"):
        if hasattr(inst, cand):
            fn = getattr(inst, cand); break
    if fn is None:
        raise RuntimeError("unknown node API")
    out = fn(model=model, shift_video=float(shift_video), shift_audio=float(shift_audio))
    out = getattr(out, "result", out)                 # V3 NodeOutput
    return out[0] if isinstance(out, (tuple, list)) else out


def _is_audio_vae(v):
    """True when v looks like the H3 audio VAE (DAC/BigVGAN), False when it looks
    like a video/image VAE, None when it can't be told. The video VAEs carry a
    3-tuple upscale_ratio (t, y, x); the audio VAE carries a scalar and reports
    latent_dim 2 with an audio_sample_rate."""
    ur = getattr(v, "upscale_ratio", None)
    if isinstance(ur, (tuple, list)):
        return False
    if getattr(v, "audio_sample_rate", None) or getattr(v, "audio_sample_rate_output", None):
        return True
    if isinstance(ur, (int, float)) and getattr(v, "latent_dim", None) == 2:
        return True
    return None


def align_frame_count_nearest(n):
    """The NEAREST 17k+5 grid point, not the next one up.

    align_frame_count always rounds up, which is right for a length you asked for
    -- never give back less than requested. It is wrong for an ESTIMATE: the grid
    steps 17 frames (~0.7s), and rounding an estimate up lengthens the shot in the
    one direction that causes trouble."""
    n = max(5, int(n))
    lo = n - ((n - 5) % 17)
    hi = lo + 17
    return min(MAX_FRAMES, lo if (n - lo) <= (hi - n) else hi)


def parse_resolution(choice):
    text = (choice or "").strip()
    if text in NATIVE_RES:
        return NATIVE_RES[text]
    m = re.search(r"(\d+)\s*x\s*(\d+)", text)
    if m:
        return int(m.group(1)), int(m.group(2))
    return NATIVE_RES["16:9"]


def scale_to_megapixels(w, h, mp, multiple=RES_MULTIPLE):
    """Scale (w, h) to `mp` megapixels keeping the ratio, snapped to the grid.
    mp <= 0 keeps the preset's own size."""
    if not mp or mp <= 0:
        return w, h
    scale = math.sqrt((mp * 1024 * 1024) / float(w * h))
    sw = max(multiple, int(round(w * scale / multiple)) * multiple)
    sh = max(multiple, int(round(h * scale / multiple)) * multiple)
    return sw, sh


# --- prompt -> beats --------------------------------------------------------

def split_beats(prompt):
    """(scene, beats). Paragraphs are separated by a BLANK line.

    The first paragraph is the SCENE: it is prepended to every shot verbatim, and
    nothing is stripped from it. Every paragraph after it is one beat, one shot.
    A single-paragraph prompt is one shot with no separate scene text.

    Deliberately the whole of the text handling. The previous version rewrote beats
    -- binding descriptions, collapsing repeated names, scrubbing the scene, adding
    continuity clauses -- and the result was a shot whose own action was a few
    percent of what the model was told. What you type is what the shot gets."""
    paras = paragraphs(prompt)
    if not paras:
        return "", []
    # A CHARACTER SHEET WRITTEN FIRST IS NOT THE SCENE. Opening a script with who is
    # in it is the natural order, and taking that paragraph as the scene stamped the
    # WHOLE sheet onto every shot as prose: every person described in every shot,
    # "Owen feeds the ducks." sent with Maya's full description beside it, and none
    # of the per-shot scoping, the count or the mouth guard run, because the node
    # believed there was no sheet. Leading sheet paragraphs stay in the beat list,
    # where pull_character_sheets takes them, and the scene is the first paragraph
    # that is not one.
    #
    # Stricter than is_character_sheet on purpose: every line has to declare a
    # pronoun or an age, which a person's entry does and "Interior: a kitchen at
    # night." does not -- that one is a scene heading and stays the scene.
    lead = []
    while paras and is_character_sheet(paras[0]) and all(
            sheet_pronoun(ln) or age_in(ln)
            for ln in paras[0].splitlines() if ln.strip()):
        lead.append(paras.pop(0))
    if not paras:
        return "", lead
    if len(paras) == 1:
        return "", lead + paras
    return paras[0], lead + paras[1:]


def paragraphs(text):
    """Non-empty paragraphs, separated by a BLANK line."""
    return [p.strip() for p in re.split(r"\n\s*\n", (text or "").strip()) if p.strip()]


# A line of a character sheet: `Name: attributes`. The directive lines are excluded
# by name -- they are instructions to this node, not people.
#
# UP TO THREE WORDS, AS sheet_lines READS THEM. This allowed one, so "Mistress Vale:
# she, 45, black dress." was not a sheet line: the whole sheet paragraph rendered as a
# shot of its own and every later shot described nobody -- the people the model
# invented there were not the ones in the first shot. The first word keeps its old
# case-insensitivity; each extra word has to be capitalised, which keeps
# "Both women: tired" out, as sheet_lines does.
_SHEET_LINE = re.compile(r"^\s*(?!(?i:remove|off|add|wear|wardrobe)\s*:)"
                         r"[A-Za-z][\w'’-]{0,24}(?:\s+[A-Z][\w'’-]{0,24}){0,2}\s*:\s*\S")


def is_character_sheet(par):
    """A paragraph that DESCRIBES people rather than staging an action.

    Every line reads `Name: attributes` -- "McKenna: 22, blonde, grey coat." Handed
    to the model as a beat, a sheet spends a whole shot rendering a static
    description. Worse, the wardrobe then lives in ONE shot instead of being
    re-stamped into all of them: later shots describe no clothing at all, so the
    model invents it, and a removal has nothing to scrub because what it would
    scrub was never in the scene.

    A sheet lists ATTRIBUTES. A line that stages an action is a beat, however it is
    labelled -- "McKenna: thrashes in her restraints" and "Camera: pushes in slowly"
    are shots, not descriptions. Getting that wrong is expensive in one direction
    only: a sheet mistaken for a beat costs one visible shot, while a beat mistaken
    for a sheet never renders AND has its words stamped onto every other shot. So
    anything that opens with a verb is treated as a beat.

    A line with speech in it is a beat too -- 'Dan: "Hello."' stages something."""
    lines = [ln for ln in (par or "").splitlines() if ln.strip()]
    if not lines or _QUOTED.search(par) or _DIALOGUE_TAG.search(par):
        return False
    return all(_SHEET_LINE.match(ln) and not _ACTION_AFTER_LABEL.search(ln)
               for ln in lines)


# What follows `Name:` in a sheet is an attribute -- a pronoun, an age, a colour, a
# <Picture N> tag. An inflected verb there means the line stages something instead.
# The participles excepted below introduce attributes rather than actions.
_ACTION_AFTER_LABEL = re.compile(
    r":\s*(?!(?:wearing|dressed|carrying|holding|sporting|wrapped|covered)\b)"
    r"(?:is|are|was|were|has|have|had|does|do|[\w-]+(?:s|es|ed|ing))\b", re.I)


def pull_character_sheets(beats):
    """(the beats that stage something, the sheet paragraphs joined)."""
    beats = beats or []
    sheets = [b for b in beats if is_character_sheet(b)]
    return [b for b in beats if not is_character_sheet(b)], "\n".join(sheets)


def sheet_lines(sheet):
    """[(name or None, line)] for a character sheet, in order. A line with no
    `Name:` label belongs to everyone and is never dropped."""
    out = []
    for ln in (sheet or "").splitlines():
        if not ln.strip():
            continue
        # UP TO THREE CAPITALISED WORDS. One word only, and "Mistress Vale:",
        # "Miss Kane:", "Aunt May:" all failed to parse -- so the line kept its
        # description and lost its name, and an unlabelled line belongs to
        # everyone and is never dropped. A full physical description of a woman
        # then rode into EVERY shot with no name on it, beside the character it
        # was meant to be. Reported as a duplicate Mistress in the first beat.
        #
        # Each extra word has to be capitalised too, so "Both women: tired" and
        # "The room: dim" stay unlabelled and global, as they were.
        m = re.match(r"\s*([A-Z][\w'’-]{0,24}(?:\s+[A-Z][\w'’-]{0,24}){0,2})"
                     r"\s*:\s*\S", ln)
        out.append((m.group(1) if m else None, ln.strip()))
    return out


# A beat about the GROUP. "They sit down", "both of them wait", "the two of them
# walk out" -- none of these names anybody, and "they" sits in _PRONOUN_SET as a
# SINGULAR group (the pronoun a nonbinary character declares), so a plural "they"
# resolved to whoever the last beat happened to keep. One of the two people in the
# shot then had no sheet line, and a person the text does not describe is a person
# the model invents -- including their clothes. Reported as clothing invented for
# somebody who had been out of shot.
#
# "each other" and "one another" are plural by definition: they need two people.
_PLURAL_CUE = re.compile(
    r"\b(?:both|each\s+other|one\s+another|the\s+two\s+of\s+(?:them|us|you)|"
    r"the\s+pair\s+of\s+(?:them|us|you)|all\s+of\s+(?:them|us|you))\b", re.I)
# ...and a bare THEY -- nominative only, and only when nobody's sheet claims it.
#
# NOT "them" or "their". Those are the object and possessive forms, and a garment
# claims them as often as a person does: "takes off her shorts and steps out of
# THEM" is the shorts, "puts THEIR keys down" is the keys. Reading either as the
# group put the other character into a shot he was not in -- which is the very
# failure the pronoun resolver below exists to avoid, reintroduced by the group
# fix. A bare "they" cannot be an object, so it is always a subject and always
# more than one person.
_THEY = re.compile(r"\bthey\b", re.I)


def group_beat(beat, rows):
    """Does this beat talk about the people as a GROUP rather than an individual?

    `rows` is sheet_lines(sheet). A they/them that some entry DECLARES as its own
    pronoun is that person, not the group -- so it is only a group cue when nobody
    on the sheet uses it."""
    b = beat or ""
    if _PLURAL_CUE.search(b):
        return True
    if not _THEY.search(b):
        return False
    return not any(sheet_pronoun(ln) == "they" for n, ln in (rows or []) if n)


def entry_heads(line):
    """Every head noun in one sheet entry's wardrobe, whatever kind of thing it is.

    garments_in knows garments and restraint_words knows hardware, and a chastity
    belt is neither: it is in no garment list and "belt" is not a restraint word,
    so both readers return nothing for it. This is the list used to decide WHOSE
    thing a beat is handling, and for that the category does not matter -- only
    that the sheet gave this person that item.

    Age, pronoun and bare adjectives are not things: an entry has to end in a word
    that could be a noun, and the numeric and pronoun entries are dropped."""
    out = []
    for item in re.split(r"[,;.]", str(line or "").split(":", 1)[-1]):
        # A <Picture N> tag ANYWHERE in the entry, not just at its head. "chastity
        # belt <Picture 2>" ends in "2>", so the head noun was the tag and the
        # wearer was never matched -- the same trap scene_name_for hit.
        item = re.sub(r"<\s*picture\s+\d+\s*>", " ", item, flags=re.I)
        item = _LEADING_TAG.sub("", re.sub(r"\s+", " ", item)).strip()
        if not item:
            continue
        head = item.split()[-1].lower().strip("-")
        if (len(head) < 3 or head.isdigit() or head in _NOT_A_GARMENT
                or head in {"she", "he", "they", "her", "his", "them", "old"}):
            continue
        if head not in out:
            out.append(head)
    return out


# Speech-stripping lives in the engine: this file and that one had identical
# copies, written the same day, which is the duplication this port exists to
# end. A name inside a line of dialogue is being SAID, not staged.
_SPOKEN_SPAN = engine._SPOKEN_SPAN
_outside_speech = engine._outside_speech


# AN OBJECT PRONOUN AFTER A PREPOSITION OF PROXIMITY IS SOMEBODY ELSE.
#
# "Tess kneels beside her" cannot mean Tess kneels beside herself, but with two women
# on the sheet the pronoun resolver credited "her" to Tess -- she declares "she", so it
# read as already accounted for -- and McKenna, who is in the shot and is what "her"
# refers to, lost her sheet line. A person in frame with no description is a person the
# model dresses out of nothing: reported as hair changing between shots, restraints not
# matching, and a body that is not the same size twice.
#
# DELIBERATELY NARROW, because the rule it refines is right in the ordinary case and
# there is a recorded regression on either side of it.
#   * `behind` is NOT in this list. "Jon walks out and shuts the door behind him" is
#     behind HIMSELF, and reading it as another person is the exact bug the resolver's
#     own comment records fixing.
#   * A pronoun followed by a noun is possessive, not an object: "look at her hands" is
#     Nora's hands, and the two-name case is excluded anyway.
# The pronoun must end its phrase -- punctuation, a conjunction, or the end of the beat.
_PRONOUN_AT = re.compile(
    r"\b(?:beside|alongside|next\s+to|opposite|toward|towards|at|to|over|onto|into|"
    r"against|with|near|by)\s+(her|him|them)\b"
    r"(?=\s*[.,;:!?]|\s+(?:and|but|then|while|as|so|who|before|after)\b|\s*$)", re.I)


def pronoun_points_away(beat):
    """Does this beat aim a pronoun at somebody OTHER than the person it names?"""
    return bool(_PRONOUN_AT.search(str(beat or "")))


def sheet_for_beat(sheet, beat, previous=None):
    """(the sheet lines for the people this beat involves, the names kept).

    The sheet is re-stamped into every shot so clothing holds -- but describing
    EVERYONE in every shot puts everyone in every shot. A beat about one person
    renders two, because the text standing beside it says the other one is there,
    and a described person is a person the model draws.

    A PRONOUN counts as naming someone: "Jon takes her jacket off" is about both of
    them, and dropping Maya there would leave the garment being removed undescribed
    in the very shot that removes it. Who "her" refers to is not resolvable from the
    sentence, so it keeps whoever the last beat kept.

    A beat that names nobody at all keeps the last beat's people too, so "She lies
    still." does not empty the frame."""
    rows = sheet_lines(sheet)
    # CASE-SENSITIVE. Prose capitalises a name, and matching without case made the
    # word "will" find a character called Will, and "grace" find Grace.
    #
    # A NAME INSIDE SPEECH IS BEING SAID, NOT STAGED. Reported: a beat where one
    # character calls for another --
    #
    #     Dana opens the door and calls out: "McKenna where are you?"
    #
    # -- put McKenna's whole sheet line into the shot, so the model was handed
    # "McKenna: she, 27, green dress" and drew her standing there. She is the one
    # person the beat says is NOT in the room. Calling for somebody is the
    # commonest way to write their absence and it was reading as their presence.
    #
    # So presence is decided on the beat with its spoken spans removed. A name
    # said aloud AND staged outside the quote still counts -- "Dana turns to
    # McKenna and says: 'McKenna, wait'" keeps her, because the staging half
    # names her. Only a name that appears nowhere but inside the speech is
    # dropped.
    # One reader, in the engine: it strips speech and matches case-sensitively,
    # and this file's rows keep their own order because nothing here needs the
    # sentence order the engine's wearer logic does.
    _here = set(engine.names_in(beat, [n for n, _ in rows if n]))
    named = [n for n, _ in rows if n in _here]
    # THE WEARER of anything the beat handles. "Dan unlocks the chastity belt"
    # names only Dan, so the shot described only Dan -- and her sheet line went,
    # taking BOTH her <Picture N> tags with it. The shot then unlocked her belt
    # while carrying no reference at all: the belt had nothing to look like, and
    # she was in the frame undescribed and unpinned, which renders as somebody
    # else. A garment cannot be acted on without the person wearing it.
    #
    # Head nouns only, and only from that person's own entry: "jeans" in Dan's
    # entry must not pull McKenna in because her shorts are jean shorts.
    for n, ln in rows:
        if not n or n in named:
            continue
        if any(re.search(r"\b" + re.escape(g) + r"\b", beat or "", re.I)
               for g in entry_heads(ln)):
            named.append(n)
    # THE GROUP. A plural cue means more than one person is in the shot, so it can
    # never resolve to a single name. Whoever the beat names plus whoever the last
    # beat kept; if that still does not reach two, everyone on the sheet.
    #
    # Erring towards MORE people here on purpose: one too many is a person
    # described who is not in frame, which the beat's own words contradict. One too
    # few is a person in frame with no description at all, and that is the one the
    # model dresses out of nothing.
    if group_beat(beat, rows):
        everyone = [n for n, _ in rows if n]
        for n in (previous or []):
            if n in everyone and n not in named:
                named.append(n)
        named = ([n for n in everyone if n in named] if len(named) >= 2
                 else everyone)
        return "\n".join(ln for n, ln in rows if n in named), named
    used = {m.group(0).lower() for m in _PRONOUN.finditer(beat or "")}
    if used:
        # Resolve a pronoun to the person whose sheet DECLARES it. Adding the whole
        # previous cast on any pronoun put someone in a shot they were not in --
        # "Jon walks out and shuts the door behind him" kept the other character,
        # because "him" was read as evidence that somebody else was present.
        # ONE PRONOUN IS ONE PERSON. Resolved per pronoun GROUP, not per sheet entry:
        # walking the entries and taking everyone who declares "she" is fine with one
        # woman on the sheet and a guess with two, and it used to take BOTH -- a third
        # character pulled into a shot that named two.
        matched = False
        for group, words in _PRONOUN_SET.items():
            if not used & words:
                continue
            # Already accounted for by somebody the beat names outright: "Nora and Dan
            # look at her hands" needs nobody else for "her".
            #
            # UNLESS THE PRONOUN POINTS AWAY FROM THEM. "Tess kneels beside her" names
            # one person and aims the pronoun at another, and crediting it to Tess left
            # McKenna in the shot with no description at all. Only with exactly one name
            # in the beat -- with two there is somebody for the pronoun to belong to --
            # and only where exactly one other person on the sheet declares that
            # pronoun, so nothing is guessed between two candidates. See _PRONOUN_AT.
            # ...AND ONLY AT SOMEBODY WHO IS ACTUALLY IN THE SCENE. Read from the
            # previous shot's cast, not from the sheet: off the sheet, "Tess looks at
            # her" dragged whichever other woman was WRITTEN DOWN into the shot, even
            # one who had left two beats earlier or never appeared at all. That is a
            # random in the scene, which is the thing character_guard exists to
            # prevent, reintroduced by the fix for the opposite problem.
            _present = {n for n in (previous or []) if n}
            _away = (len(named) == 1 and pronoun_points_away(beat))
            if _away and _present:
                _others = [n for n, ln in rows
                           if n and n not in named and n in _present
                           and sheet_pronoun(ln) == group]
                if len(_others) == 1:
                    named.append(_others[0])
                    matched = True
                    continue
            if any(sheet_pronoun(ln) == group for n, ln in rows if n and n in named):
                matched = True
                continue
            cands = [n for n, ln in rows
                     if n and n not in named and sheet_pronoun(ln) == group]
            if len(cands) == 1:
                named.append(cands[0])
                matched = True
            elif len(cands) > 1:
                # Two people declare it. The scene continuing is the only evidence
                # available, so take the one who was in the last beat -- and if that
                # does not single anybody out, add NOBODY. Naming a person the beat
                # did not is the failure being fixed; leaving them to the keyframe is
                # recoverable.
                narrowed = [n for n in cands if n in (previous or [])]
                if len(narrowed) == 1:
                    named.append(narrowed[0])
                    matched = True
        # A sheet that declares no pronouns tells us nothing, so fall back to the
        # last beat's people rather than guessing.
        if not matched:
            named += [n for n in (previous or []) if n not in named]
    # Somebody is in it, but the beat does not say who -- "Someone knocks at the
    # door." Keep the last beat's people, since a scene usually continues with them.
    # With nobody before it, describing the WHOLE sheet is the same failure in
    # miniature: it puts everyone in a shot on the strength of not knowing. One
    # person on the sheet is unambiguous and still resolves; two or more is a guess,
    # and the guard exists precisely not to make it.
    if not named:
        named = list(previous or [])
    if not named:
        _all = [n for n, _ in rows if n]
        named = _all if len(_all) == 1 else []
    keep = [ln for n, ln in rows if n is None or n in named]
    return "\n".join(keep), named


# A beat that stages somebody ARRIVING. The chain is right for these: the previous
# shot's last frame is where they walk in from. A beat that stages no entrance is
# describing where somebody already IS, and there is no frame to inherit that has them
# in it.
_ENTRANCE = re.compile(
    r"\b(?:walk|step|come|run|stride|hurry|move|wander|burst|barge|slip|climb)"
    r"(?:s|ed|ing)?\s+(?:in|into|through|up|over|back|out\s+of)\b"
    r"|\benter(?:s|ed|ing)?\b|\barriv(?:es?|ed|ing)\b"
    r"|\bjoin(?:s|ed|ing)?\b|\breturn(?:s|ed|ing)?\b|\bfollow(?:s|ed|ing)?\b"
    r"|\blets?\s+\w+\s+in\b", re.I)
# APPEARING IS NOT ARRIVING, and the difference is the whole reason this list
# exists. A staged arrival keeps the previous frame as the keyframe, because
# somebody walking in through a door has a path into a frame that does not have
# them in it -- they cross the edge of it. "Appears", "shows up", "turns up"
# describe the RESULT, not the movement: there is no path, so the only way for
# the model to put them into that frame is to fade them up inside it. Reported as
# ghosting on a character introduction, which is exactly what that looks like.
#
# So they are introductions in position instead, and the shot cuts to her already
# there -- which is what the words mean.
#
# Taking them out of the list above is the whole fix. A guard that ALSO looked
# for them and cancelled an arrival was written here and removed: with the words
# gone from the list it never changed an answer, and the one case it did reach --
# a beat with a real entrance and an "appears" in it, "walks in and appears calm"
# -- it got wrong, cancelling an arrival that plainly happens. The disable-check
# is what showed it was dead: reverting it left every case green.


def arrives_in(text):
    """Does this beat stage somebody arriving -- moving into the frame?

    A word that only says they are suddenly THERE does not count, however much it
    reads like an entrance -- see the note on _ENTRANCE."""
    return bool(_ENTRANCE.search(text or ""))


def unresolved_pronouns(sheet, beat, previous=None):
    """[(pronoun group, the people who could answer to it)] this beat cannot settle.

    Two people declaring "she" and a beat saying "her" is a guess, and the guard makes
    none: it adds nobody rather than both. Nobody being described is recoverable --
    the keyframe still carries them -- but it is worth saying, because the fix is to
    write the name instead of the pronoun."""
    rows = sheet_lines(sheet)
    named = [n for n, _ in rows
             if n and re.search(r"\b" + re.escape(n) + r"\b", beat or "")]
    used = {m.group(0).lower() for m in _PRONOUN.finditer(beat or "")}
    out = []
    for group, words in _PRONOUN_SET.items():
        if not used & words:
            continue
        if any(sheet_pronoun(ln) == group for n, ln in rows if n and n in named):
            continue
        cands = [n for n, ln in rows
                 if n and n not in named and sheet_pronoun(ln) == group]
        if len(cands) > 1 and len([n for n in cands if n in (previous or [])]) != 1:
            out.append((group, cands))
    return out


_PRONOUN_SET = {"she": {"she", "her", "hers"},
                "he": {"he", "him", "his"},
                "they": {"they", "them", "their", "theirs"}}


def sheet_pronoun(line):
    """Which pronoun this sheet entry declares for its person, or None.

    Writing the pronoun into the sheet -- "Maya: 27, she, grey coat" -- is what lets
    "her coat" in a beat be resolved to Maya rather than to whoever was in the last
    shot."""
    body = (line or "").split(":", 1)[-1]
    for group, words in _PRONOUN_SET.items():
        if any(re.search(r"\b" + w + r"\b", body, re.I) for w in words):
            return group
    return None


ADULT_AGE = 18                 # below this the node describes no body at all

# HOW OLD THE SHEET SAYS SOMEBODY IS. Inert text until now: the age went to the model
# inside the author's own words and nothing here read it, so every clause this file
# writes about a body said only "a woman's body" -- and an attribute a prompt does not
# state is not left to the model, it is left to the model's PRIOR. The prior for an
# adult woman is a woman in her twenties whatever the sheet says, which is how a
# character written as 45 renders as 22.
#
# DELIBERATELY NARROW. A number is an age only where it stands as its own attribute in
# the list, or where an age word is attached to it. A sheet says "size 10 boots" and
# "5'7" and "a 9mm" and a <Picture 2> tag, and reading any of those as an age would
# describe a body nobody asked for -- worse than describing none.
_AGE_WORD = r"(?:y\.?o\.?|yrs?|years?(?:\s+old)?|year-old)"
_AGE_AT = re.compile(
    # "aged 24", "age 24", "24yo", "24 years old", "24-year-old"
    r"\bage[d]?\s+(\d{1,3})\b"
    r"|\b(\d{1,3})\s*-?\s*" + _AGE_WORD + r"\b"
    # ...or a bare number alone between the commas of the attribute list. The list can
    # END on it -- "Kate: she, 28." is how most entries are written -- so a full stop or
    # a semicolon closes the attribute as well as a comma does. Requiring a comma read
    # that entry as having no age at all, which is silent twice over: no body named, and
    # the refusal below never fired either.
    r"|(?:^|,)\s*(\d{1,3})\s*(?=[,;.]|$)", re.I)
# "in her forties", "early thirties", "mid-50s", "late 20s". The decade's MIDDLE,
# except where the qualifier says otherwise -- and it is read as an age only for the
# decades an adult has, because a bare "teens" names no single year and is not a
# licence to guess one.
_DECADE = {"twenties": 20, "thirties": 30, "forties": 40, "fifties": 50,
           "sixties": 60, "seventies": 70, "eighties": 80}
_DECADE_AT = re.compile(
    r"\b(early|mid|middle|late)?\s*-?\s*"
    r"(?:(twenties|thirties|forties|fifties|sixties|seventies|eighties)"
    r"|(\d0)\s*s)\b", re.I)


def age_in(line):
    """The age this sheet entry declares, or 0 when it declares none.

    Read off the attribute list, never off a beat: a beat saying "twenty years later"
    is not somebody's age, and the sheet is where the author states what is true of a
    person for the whole film."""
    body = str(line or "").split(":", 1)[-1]
    # The tag carries digits of its own, and they are a slot number.
    body = re.sub(r"<\s*picture[\s_]*\d+\s*>", " ", body, flags=re.I)
    m = _AGE_AT.search(body)
    if m:
        got = int(next(g for g in m.groups() if g))
        return got if 1 <= got <= 120 else 0
    m = _DECADE_AT.search(body)
    if m:
        base = _DECADE.get((m.group(2) or "").lower())
        if base is None and m.group(3):
            base = int(m.group(3))
        if base in _DECADE.values():
            q = (m.group(1) or "").lower()
            return base + (2 if q == "early" else 8 if q == "late" else 5)
    return 0


_PRONOUN = re.compile(r"\b(?:she|he|her|hers|his|him|they|them|their|theirs)\b", re.I)


# A determiner in front means the capitalised word DESCRIBES something rather than
# doing something: "her Nike leggings" names a garment, not somebody in the room.
_DETERMINER = frozenset("a an the her his its their our my your this that".split())
_CAPITALISED = re.compile(r"\b([A-Z][a-z\u2019'-]{1,24})\b")
# A WORD THAT IS NEVER SOMEBODY'S NAME, however it is capitalised.
#
# The mid-sentence test was supposed to make this list unnecessary -- an ordinary word
# only opens a sentence, a name appears inside one -- and it is defeated by the
# commonest punctuation in a script:
#
#     "Nearly there," The guard says.
#
# "The" follows a comma, so it IS mid-sentence, so it was reported as a character with
# no sheet entry: "shot(s) 4, 7, 8, 9 name The, who has no entry in the character
# sheet". Every pronoun reaches the same way out of a speech tag -- '"Wait," She says'
# -- and the warning then sends the author looking for a person who does not exist
# while saying nothing about the one who does.
#
# Only words that are never a name go in here. Grace, Will, Hope, Faith and May are
# names and are deliberately absent: this file has already been bitten by matching
# "will" and "grace" case-insensitively.
_NEVER_A_NAME = _DETERMINER | frozenset("""
i we you he she it they me him us them myself yourself himself herself itself
themselves mine yours hers ours theirs
and but or nor so yet then than as at in on of off to into onto from with without
if when while because though although after before until once since
there here what which who whom whose why how where whether
no not now never always again also just only even still both each either neither
one two three four five six seven eight nine ten first second next last another
yes ok okay oh ah well right left up down out over under across back forward
""".split())


def unknown_people(beats, sheet):
    """{name: [1-based shot numbers]} -- names the beats use as PEOPLE that the
    character sheet never describes.

    A person the sheet does not describe is a person no shot describes. The guard
    keeps the entries for the people a beat names, and there is no entry to keep, so
    the beat stages somebody the model has been told nothing about -- no age, no
    clothes, no face -- and it invents them, differently in each shot. Worse, a beat
    whose ONLY person is undescribed falls back to the previous beat's cast, so the
    shot describes someone who is not in it and stays silent about the one who is.

    It is also how one person written under two names becomes two people, one of
    them a stranger.

    A capitalised word only counts once it has appeared MID-sentence somewhere in
    the script. That is what separates a name from an ordinary word that happens to
    open a sentence, and it needs no list of ordinary words to do it.

    Reported, never acted on: whether a name is somebody already on the sheet under
    another name or a third person in the room is not answerable from the text, and
    guessing would be the node rewriting the script."""
    known = {n.lower() for n, _ in sheet_lines(sheet) if n}
    seen, mid_sentence = {}, set()
    for i, beat in enumerate(beats or [], 1):
        for m in _CAPITALISED.finditer(beat or ""):
            # "Jon's kitchen" is Jon. The apostrophe is in the class for O'Neill.
            word = re.sub(r"['’]s$", "", m.group(1))
            # Never a name, however the punctuation capitalised it.
            #
            # NAMED _NEVER_A_NAME, not _NOT_A_NAME: that one already exists further
            # down as a regex STRING, and shadowing it turned this membership test into
            # a silent substring match against a regex -- "one" passed because it
            # appears inside the pattern and "the" failed because the pattern spells it
            # "The". The test caught it; `in` on a string never raises.
            if word.lower() in _NEVER_A_NAME:
                continue
            before = (beat[:m.start()]).rstrip()
            prev = re.search(r"([\w’'-]+)\W*$", before)
            if prev and prev.group(1).lower() in _DETERMINER:
                continue
            # Opening a sentence -- or a quoted line -- capitalises anything, so
            # only a mid-sentence appearance is evidence of a name.
            if before and before[-1] not in ".!?:\"”":
                mid_sentence.add(word)
            if i not in seen.setdefault(word, []):
                seen[word].append(i)
    return {w: s for w, s in seen.items()
            if w in mid_sentence and w.lower() not in known}


# Where a beat says something becomes VISIBLE. The other half of a removal: "cuts
# off her coat to expose the jumper" names the coat as coming off AND the jumper as
# what was under it.
_EXPOSE_CUE = re.compile(r"\b(?:to\s+expose|to\s+reveal|to\s+show|exposing|revealing|"
                         r"showing|uncovering|baring)\b", re.I)


# LAYERING LIVES IN THE ENGINE, beside the garment vocabulary it reads --
# keeping them apart is what let a chastity belt be underwear to one file
# and a bare "belt" to the other. The CLAUSES stay here, because saying a
# thing in a sentence belongs where a shot is assembled.
_UNDER_BY_REGION = engine._UNDER_BY_REGION
_OUTER_BY_REGION = engine._OUTER_BY_REGION
implied_layers = engine.implied_layers
hidden_layers = engine.hidden_layers
is_undergarment = engine.is_undergarment


def exposed_by(beat, scene):
    """Garments this beat says become visible. [] when none."""
    out = []
    for m in _EXPOSE_CUE.finditer(beat or ""):
        tail = beat[m.end():]
        cut = re.search(r"[,;.]|\band\s+(?:then|he|she|they)\b", tail, re.I)
        span = tail[:cut.start()] if cut else tail
        for word in re.findall(r"\b[\w-]{3,}\b", span):
            low = word.lower().strip("-")
            if not low or low in out or low in _NOT_A_GARMENT:
                continue
            if _RESTRAINT_WORD.match(low) or not _is_entry_head(word, scene):
                continue
            # "jeans shorts" is one garment; "jeans" there is a modifier, and
            # matching it against another character's entry took their trousers off.
            if _modifier_of_a_named_entry(word, span, scene):
                continue
            out.append(low)
    return out


def infer_layers(bodies, scene):
    """{under: over} -- which garment covers which, read from the script's own words.

    A sheet lists every layer at once, which tells the model all of them are on show
    simultaneously. Nothing says which is hidden, so the under layer bleeds through
    the top one -- and by the last frame, where only the text governs, it is simply
    drawn on top.

    The script already says what covers what: a beat that takes A off "to expose B"
    has stated that B was under A. Read it from there rather than asking for it."""
    covers = {}
    for body in bodies or []:
        off = infer_removals(body, scene)
        for under in exposed_by(body, scene):
            for over in off:
                if under != over:
                    covers.setdefault(under, over)
    return covers


# Layering that needs no telling: underwear goes under. infer_layers only learns what
# the SCRIPT states -- "takes A off to expose B" -- so a sheet listing panties beside
# shorts, with no beat ever saying one is under the other, left both described in every
# shot. A layer the model is told about is a layer it draws, and it draws it through
# whatever is over it. Reported as underwear and a chastity belt showing through the
# clothes.
#
# By REGION, because that is what covering means: a bra is not hidden by trousers.
def revealed_by(covers, gone):
    """Under-layers brought into view because the thing over them has just come off."""
    return [u for u, o in (covers or {}).items() if o in (gone or [])]


# Which region of the body a garment leaves uncovered when it comes off. Only what
# the node can place with certainty; a garment it cannot place gets no clause, since
# a wrong region is worse than none.
_REGION_OF = engine._REGION_RX


def body_of(pronoun, age=0):
    """The body the sheet's declared pronoun and age mean. "" where nothing is declared.

    AN UNSPECIFIED BODY IS FILLED FROM THE PRIOR, which is the lesson this file
    already recorded for the chest -- "this said 'The arms and shoulders are bare' and
    stopped there, so the one region a bra occupies was unspecified, and an unspecified
    region is filled by the model's own prior". The hip-down clause had the same gap
    and it was never closed: "The legs are bare from the hip down" names the region and
    says nothing about whose body it is, so the anatomy at the hip came from the prior
    too. Reported as the wrong anatomy on a female character.

    Read from the pronoun the author DECLARED, which the README already requires for
    every entry, so this asserts nothing the sheet does not already say. `they` returns
    nothing: an undeclared body is not a licence to guess one.

    ...AND THE AGE THEY DECLARED, for the same reason one step further on. "A woman's
    body" is true of a woman of 22 and a woman of 62, so it settles nothing between
    them, and what fills the gap is the prior -- which is a woman in her twenties
    whatever the sheet says. Reported as a character written at one age rendering at
    another. The age is the author's own word, already in the sheet and already going
    to the model inside it; this only stops it being the one attribute nothing here
    reads.

    NO BODY IS DESCRIBED FOR A DECLARED AGE UNDER 18. Not a softer description -- none,
    and this returns "" so every clause built on it stays silent. An age the author
    states is the one fact here that is not a guess, and a generator has no business
    composing anatomy for a child. See also the refusal in _prepare: a script that
    declares a minor and stages nudity or sex does not render at all."""
    who = {"she": "woman", "he": "man"}.get(str(pronoun or "").strip().lower(), "")
    if not who:
        return ""
    age = int(age or 0)
    if age and age < ADULT_AGE:
        return ""
    return f"a {who}'s body" if not age else f"the body of a {who} of {age}"


# HOW AN ADULT CHEST DIFFERS WITH AGE. Plain physical description -- fullness, where it
# sits, how firm, what the skin does -- because those are the facts that separate one
# adult decade from another, and the prior collapses all of them onto the twenties.
#
# Asked for directly: "Breast development should also be correct, given the age of a
# person." The clause is scoped hard. It is said only where the chest is ALREADY being
# described as bare, so it adds nothing to a clothed shot; only for a declared age of
# 18 or over, with no entry below that; and only for a sheet that declares "she",
# because the request was about breasts and a pronoun this file was not given is not a
# licence to guess an anatomy.
_FIGURE = (
    (18, 24, "grown and firm, sitting high on the chest"),
    (25, 34, "fully grown and full, sitting a little lower than in her early twenties"),
    (35, 44, "full and softer, settled lower with the weight of middle age"),
    # No "skin" in these two: the sentence they join already ends on "the skin itself
    # the outermost surface there", and the word arriving twice in one clause reads as
    # two different things being described.
    (45, 54, "mature and heavier, softened and lower again, with less tension in them"),
    (55, 120, "older and slacker, hanging low and soft, loose and lined"),
)


# A SHEET THAT DECLARES A CHILD AND A SCRIPT THAT STAGES SEX DO NOT RENDER TOGETHER.
#
# This file reads an age now, and the age drives anatomy -- see body_of and figure_of,
# which describe no body at all below ADULT_AGE. That floor is necessary and it is not
# sufficient: withholding the node's own clauses does nothing about a script whose own
# words stage nudity or sex, and those words reach the model verbatim. So the two
# together are refused outright, before anything is sampled.
#
# Read off the SHEET for the age, because that is where an author states a person's
# age, and off the whole script for the staging. Deliberately blunt: no attempt to work
# out who the nudity is about. A film that declares a minor anywhere and stages this
# anywhere is refused whole, and a legitimate scene with a child in it -- which this
# node will render, with no body described for them -- does not contain either.
_SEXUAL_STAGING = re.compile(
    r"\b(?:sex|sexual|fucks?|fucking|fucked|intercourse|penetrat\w*|blow\s?job|"
    r"handjob|masturbat\w*|orgasms?|orgasmic|climax(?:es|ed|ing)?|cums?|cumming|"
    r"aroused|arousal|horny|erotic\w*|nipples?|genitals?|vagina\w*|penis\w*|"
    r"cocks?|dicks?|pussy|clit\w*|erections?|foreplay|straddl\w*|"
    r"topless|bottomless|naked|nude|nudity|undress\w*|strips?\s+(?:off|naked|bare)|"
    r"moans?|moaning|moaned)\b", re.I)


def minor_with_sexual_staging(sheet, script):
    """A refusal message when a sheet declares a minor and the script stages sex. "" otherwise.

    Both halves required. An age under 18 on its own renders -- children exist in
    films -- and gets no body described for them by anything here. Sexual staging on
    its own renders, which is what this node is for."""
    named = [(n, age_in(ln)) for n, ln in sheet_lines(sheet or "") if n]
    minors = sorted({n for n, a in named if 0 < a < ADULT_AGE})
    if not minors:
        return ""
    m = _SEXUAL_STAGING.search(str(script or ""))
    if not m:
        return ""
    return (f"REFUSED, and nothing was rendered. The character sheet declares "
            f"{_join_names(minors)} as under {ADULT_AGE}, and the script stages sexual "
            f"or nude content -- it contains {m.group(0)!r}. This node will not "
            f"generate that combination, whichever character the wording is about and "
            f"whatever was intended by it. Nothing here tried to work out who: a film "
            f"holding both is refused whole.\n\n"
            f"If an age is a typo, fix the sheet and run again -- an adult age renders "
            f"normally. If the character is an adult, state an adult age. A scene with "
            f"a child in it and no sexual or nude content renders as it always did, "
            f"and no body is described for them by this node.")


def _pron_age(sheet, name):
    """(pronoun, age) off one person's own sheet entry. ("" , 0) when it has neither."""
    line = dict(sheet_lines(sheet)).get(name, "")
    return sheet_pronoun(line), age_in(line)


def figure_of(pronoun, age=0):
    """Age-consistent adult chest description, or "". See _FIGURE and body_of.

    Returns nothing at all without BOTH a declared "she" and a declared adult age:
    with no age there is nothing to be consistent with, and the old silence is better
    than a guess."""
    if str(pronoun or "").strip().lower() != "she":
        return ""
    age = int(age or 0)
    if age < ADULT_AGE:
        return ""
    for lo, hi, said in _FIGURE:
        if lo <= age <= hi:
            # THE AGE IS NOT REPEATED HERE. body_of already states it in the same
            # sentence, and "the breasts those of a woman of 45 ... on the body of a
            # woman of 45" says one fact twice -- which is the vice this file spends
            # most of its comments on. figure_of is only ever reached through a
            # declared "she", so the body phrase is always there to carry it.
            return f"the breasts {said}"
    return ""


def bare_clause(gone, covers=None, worn="", body="", figure=""):
    """Say the uncovered region is BARE, when the sheet names nothing under it.

    A removal clause is emphatic -- off the body, dropped out of frame -- and then
    says nothing about what occupies the space it left. An unspecified region is
    where the model's own prior fills in, and for legs that prior is legwear: the
    shot invents leggings, tights or stockings that appear nowhere in the prompt,
    and the keyframe then carries the invention into every later shot.

    Positively phrased, and it names a BODY PART, never a garment. At cfg 1 there
    is no negative prompt, so "no leggings" would be read as leggings; "the legs
    are bare" fills the same region with something that is actually wanted.

    Silent when the sheet already answers the question -- reveal_clause covers the
    case where something IS underneath, and the two must never both speak -- and
    silent when another garment the character still wears covers the same region."""
    if not gone:
        return ""
    regions = []
    for item in gone:
        r = engine.region_of(item)
        if r and r not in regions:
            regions.append(r)
    return bare_hold(regions, covers, worn, gone, body=body, figure=figure)


def bare_hold(regions, covers=None, worn="", gone=(), whose="", body="", figure=""):
    """Say those regions are bare -- from STATE, so it outlives its beat.

    The same suppression as the removal beat, because it is the same sentence:
    silent when the sheet names a layer underneath (reveal_clause has that one),
    and silent when a garment still worn covers the region.

    The reason it exists apart from bare_clause is the report: a bra coming back
    on somebody topless, on a character with no bra anywhere on the sheet. The
    clause only ever fired on the beat that uncovered the region, so every shot
    after it said nothing about that region -- and an unspecified region is
    filled by the model's own prior. Nothing was restoring the bra. The prior was
    inventing one, and the keyframe then carried the invention forward."""
    if not regions:
        return ""
    spoke = []                     # the regions this clause actually speaks about
    # ...AND ONLY WHILE IT IS STILL ON. `covers` is read off the SHEET, and the
    # sheet is never edited, so a thong listed under a skirt went on suppressing
    # this clause long after the thong had come off as well -- and a full strip is
    # the one case this clause matters most in. The hips then had no sentence at
    # all, an unspecified region is filled by the model's own prior, the prior for
    # a hip is underwear, and the keyframe carried what it invented into every
    # later shot. Reported as a thong restored a beat after she undressed.
    #
    # It cost the bra half too, which is the report this function was written for:
    # a sheet that layered the bra under a shirt suppressed the chest clause by
    # this same line, so the fix only ever worked for a sheet that did not.
    #
    # reveal_clause already filters itself the same way -- it is silent for an
    # under-layer coming off in the same breath -- so the two still never both
    # speak, which is the only thing this suppression was for.
    under = {str(u).lower() for u in (covers or {}) if not names_any(u, gone)}
    said, out = set(), []
    for _region in regions:
        for rx, region, sentence in _REGION_OF:
            if region != _region or region in said:
                continue
            # Something else still on the body covers this region: not bare.
            if any(rx.search(w) for w in (worn or "").split(",")
                   if not names_any(w, gone)):
                said.add(region)
                break
            # The sheet named a layer underneath: reveal_clause has this one, and
            # the two must never both speak. Matched against the region's UNDER
            # vocabulary as well as its own -- panties sit in the leg region but
            # are not legwear, and testing only the outer list let this clause
            # call the legs bare while reveal_clause said the panties show.
            if any(rx.search(u) or re.search(_UNDER_BY_REGION.get(
                       "lower" if region == "legs" else
                       "upper" if region == "torso" else "", "(?!)"), u, re.I)
                   for u in under):
                said.add(region)
                break
            said.add(region)
            out.append(sentence)
            spoke.append(region)
            break
    if not out:
        return ""
    # One region is the normal case. Two is a full strip, and past that the clause
    # would outweigh the beat it is protecting. Only the first stays capitalised:
    # joined as written it read "and The feet and ankles are bare".
    out = out[:2]
    joined = out[0] + "".join(", and " + s[0].lower() + s[1:] for s in out[1:])
    # WHOSE, when the shot describes somebody else as well. An unattributed "the
    # chest is bare" in a shot about two people is a region belonging to nobody,
    # and the model picks. The hardware hold has said "on <name>" for the same
    # reason since it was written.
    if whose:
        joined = f"{whose}'s " + joined[4:] if joined.startswith("The ") else \
            f"{whose}: " + joined
    # ...AND WHOSE BODY IT IS. A bare region with no body named is anatomy left to the
    # prior, and at cfg 1 nothing later takes back what the prior draws. See body_of.
    # POSITIVELY PHRASED, to the last clause. This ended ", with nothing else worn
    # there" -- a negation, in the one sentence whose whole purpose is to stop the
    # model filling a region from its own prior, and at cfg 1 there is no negative
    # prompt to carry it: "nothing else worn" offers the word worn and no picture.
    # It went unnoticed because the clause could only reach a shot whose sheet put
    # no layer under the garment, and the suite that checks every guard sentence for
    # a negation had no such scene until the suppression was fixed.
    #
    # What replaces it says the same thing as a surface, which is what a model
    # renders: the skin is the outermost thing on that part of the body. Same move
    # under_clause made for the cover it describes.
    # ...AND WHAT THAT PART OF THE BODY IS LIKE AT THE AGE THE SHEET STATES, but only
    # where the chest is one of the regions this sentence actually reached. A clause
    # about bare legs that describes a chest is describing a region it was not asked
    # about, and `out` is capped at two, so "torso was in `regions`" is not the same
    # question as "torso got said". See figure_of.
    # AFTER the body, not before it: the body phrase is what the figure is a fact
    # about, and "the breasts ..., on the body of a woman of 45" puts the attribute
    # ahead of the thing it belongs to.
    said_fig = f", {figure}" if (figure and "torso" in spoke[:2]) else ""
    return " " + joined + (f", on {body}" if body else "") + said_fig + \
        ", the skin itself the outermost surface there."


def defer_tag_for(text, items):
    """Take the <Picture N> off an item that is covered THIS SHOT, keeping its
    words. The tag comes back the moment the cover comes off.

    THIS IS A DEFERRAL, NOT A REMOVAL, and the distinction is the whole point.
    The item stays in the character memory in every shot, exactly as written. What
    waits is its reference, and only on the shots where the thing is under
    something else.

    It waits because a reference is an instruction to REPRODUCE AN IMAGE. At the
    near-clean ref_noise_aug this node runs at, the node's own report says so:
    "that asks the model to REPRODUCE them, framing and background included". A
    picture of a chastity belt, handed to the model for a shot in which the belt
    is under a skirt, is an instruction to draw the belt, and it outweighs any
    sentence about what is on top of what. Measured twice, from two different
    directions: every configuration that sent the picture while the garment was
    covered rendered it through the cover, including one where the cover was
    described as whole, opaque and unbroken.

    There is no third option available. Reference strength is ref_noise_aug and it
    is one number for every image, so the belt's picture cannot be weakened
    without weakening the face. The tag is what routes the image, so the tag is
    what waits -- leaving it in while withholding the image would name a picture
    the shot does not carry, which is its own bug."""
    out = str(text or "")
    for item in items or []:
        if not str(item).strip():
            continue
        w = re.escape(str(item).strip())
        # Either side of the item, which is where a sheet puts it: "<Picture 2> a
        # chastity belt" and "a chastity belt <Picture 2>" are both written.
        out = re.sub(r"<\s*Picture\s*\d+\s*>\s*((?:a|an|the)\s+)?" + w,
                     lambda m: (m.group(1) or "") + str(item).strip(), out,
                     flags=re.I)
        out = re.sub(w + r"\s*<\s*Picture\s*\d+\s*>", str(item).strip(), out,
                     flags=re.I)
    return out


def under_clause(pairs):
    """Say that an under-layer is UNDER, rather than deleting it from the sheet.

    Layering used to work by scrubbing: a garment read as covered came out of the
    shot text entirely, and its <Picture N> with it. The reasoning was sound as
    far as it went -- a described thing is a drawn thing, and an under-layer
    described flatly beside its cover gets drawn on top of it -- but the cost was
    the author's own words disappearing, which was reported three times, the last
    of them a chastity belt with a reference image attached to it.

    Deleting a thing is not the only way to stop it being drawn on top. Saying
    where it is works better and keeps the text: the model is told the belt is
    under the jeans, which is a spatial fact it can render, rather than being
    told nothing and left to guess. Positively phrased, because at cfg 1 there is
    no negative prompt -- this says where the thing IS, never where it is not.

    Panties, knickers, thongs, briefs, boxers, underwear, bras, corsets and
    chastity belts, devices and cages are all in _UNDER_BY_REGION, so they are
    always the under-layer whatever order the sheet lists them in."""
    pairs = [(p[0], p[1], p[2] if len(p) > 2 else "")
             for p in (pairs or []) if p[0] and p[1]]
    if not pairs:
        return ""

    def _plural(w):
        return w.endswith("s") and not w.endswith("ss")

    def _one(u, o, who=""):
        # WHOSE, when more than one person is in the shot. "The chastity belt is
        # worn under the skirt" beside two women says nothing about which of them
        # wears it, and an unattributed garment lands on whoever the model finds
        # convenient -- the same failure as hardware on nobody's wrists. Named
        # once, at the front, and never run through .capitalize(), which lowers
        # the rest of a name and turned McKenna into Mckenna.
        #
        # Each garment takes its own number: panties ARE worn, a bra IS; jeans
        # cover THEM, a skirt covers IT.
        #
        # THE COVER IS THE PART TO DESCRIBE. "the belt is under the jeans" asks
        # the model to work out an occlusion from a spatial word, which it does
        # badly, and the belt came through the denim. What it renders well is a
        # surface: say the jeans are whole and unbroken over that part of the
        # body and there is nothing for the belt to show through. Positively
        # phrased, as everything here has to be at cfg 1 -- this describes the
        # cloth that IS there, never the thing that must not show.
        cover = "cover" if _plural(o) else "covers"
        whose = f"{who}'s " if who else "The "
        # THE COVER ONLY. This used to open with "{u} is worn under the {o}",
        # which names the hidden garment in the one shot that must not show it --
        # and at cfg 1 there is no negative prompt, so naming a thing draws it.
        # Measured: with the picture already withheld, the belt was still named
        # twice in a covered shot, once by the author's sheet entry and once
        # here. This clause was the half that could be removed.
        #
        # What survives is the half that works: a SURFACE, which the model
        # renders well, described as unbroken over the part of the body in
        # question. `u` is deliberately unused -- it is the thing not to mention.
        return (f"{whose}{o} {cover} the hips and waist completely: whole, "
                f"opaque and unbroken, the outermost layer there and the only "
                f"one in view.")

    return " " + " ".join(_one(*p) for p in pairs[:2])


def reveal_clause(items):
    """Say what is underneath is what shows now, on the shot that uncovers it.

    The removal clause is emphatic and specific -- off the body, dropped out of frame
    -- while the layer beneath is one item in an attribute list. Against a model whose
    prior for trousers coming off is bare skin, a list entry does not compete. It has
    to be told what fills the space the garment left."""
    if not items:
        return ""
    said = " and ".join(f"the {i}" for i in items[:2])
    plural = len(items) > 1 or items[0].endswith("s")
    return (f" {said[0].upper()}{said[1:]} underneath {'are' if plural else 'is'} what "
            f"shows there now, on and unchanged.")


def merge_sheets(*sources):
    """(one sheet, the names that were described more than once).

    character_memory and a `Name:` paragraph in the prompt are the same channel by
    two routes, and using both -- the natural thing to do once the widget exists --
    put the person in every shot TWICE:

        A basement. Maya: 27, silver hair, grey coat. Maya: 27, silver hair, grey
        coat. Maya lies still on the floor.

    A model told about one person twice renders two of them. One entry per name, and
    no line repeated. The earlier source wins, so character_memory overrides a sheet
    left in the prompt."""
    seen_names, seen_lines, out, dupes = set(), set(), [], []
    # ...AND ONE PERSON UNDER TWO FORMS OF THE NAME. "Maya Brooks" in
    # character_memory and "Maya:" in the prompt matched as two keys, so both entries
    # went into every shot -- one woman in a green sweater and one in a red coat,
    # under "There is one person in the shot". A name made of words the other name
    # already has is the same person, unless the two entries say otherwise: a
    # different pronoun or a different age is somebody else ("May: she, 24" and
    # "Aunt May: she, 60").
    seen_rows = []
    def _same_person(name, line):
        words = set(name.lower().split())
        for other, other_line in seen_rows:
            theirs = set(other.lower().split())
            if not (words <= theirs or theirs <= words):
                continue
            p1, p2 = sheet_pronoun(line), sheet_pronoun(other_line)
            a1, a2 = age_in(line), age_in(other_line)
            if (p1 and p2 and p1 != p2) or (a1 and a2 and a1 != a2):
                continue
            return True
        return False
    for src in sources:
        for name, line in sheet_lines(src):
            key = name.lower() if name else None
            if key and (key in seen_names or _same_person(name, line)):
                if name not in dupes:
                    dupes.append(name)
                continue
            if key:
                seen_rows.append((name, line))
            if line in seen_lines:
                continue
            if key:
                seen_names.add(key)
            seen_lines.add(line)
            out.append(line)
    return "\n".join(out), dupes


def terminate_lines(text):
    """Give every line a full stop, so what follows does not run into it.

    The sheet is assembled ahead of the beat, and a line ending "grey coat" welds
    onto the beat as "grey coat Maya lies still". A name fused to the end of an
    attribute list reads as one more item in the list -- another person in shot."""
    out = []
    for ln in (text or "").splitlines():
        s = ln.rstrip().rstrip(",;:")
        if s and s[-1] not in ".!?":
            s += "."
        if s:
            out.append(s)
    return "\n".join(out)


def build_scene(anchor, first_para, character_memory, sheet):
    """The text every shot carries, in reading order: the anchor frames the film,
    the opening paragraph sets the scene, and the character sheet says who is in it
    and what they are wearing.

    One string on purpose -- a removal scrubs all of it. The previous node kept the
    anchor immutable, and clothing written there could never be taken off: the
    anchor put it back on every shot, under a beat that had just removed it."""
    parts = [(anchor or "").strip(), (first_para or "").strip(),
             (character_memory or "").strip(), (sheet or "").strip()]
    return "\n".join(terminate_lines(p) for p in parts if p)


# THE OPENING PARAGRAPH NAMES PEOPLE TOO, AND IT IS IN EVERY SHOT.
#
# sheet_for_beat scopes the sheet to the people a beat involves, because "describing
# EVERYONE in every shot puts everyone in every shot". The anchor and the opening
# paragraph ride into every shot beside it and were never scoped the same way, so an
# opening written the ordinary way -- "Maya and Owen wait in a train station." --
# put both names in a shot the node had cut down to one:
#
#     Maya and Owen wait in a train station. Owen checks the departure board.
#     Owen: he, 42, blue shirt. There is one person in the shot: one body, one face.
#
# Two names, one description, one body: the model is told a second person stands
# there and given nobody to draw but Owen, which is how a character is rendered
# twice. So a sentence there that names someone NOT in the shot gives up its
# setting and loses the person; with no setting to give up, it goes.
# A place that ENCLOSES is preferred over a spot beside a thing: "On the couch in a
# dark living room." reads as somebody on the couch, where "In a dark living room."
# is only the room.
_SETTING_PHRASE = re.compile(
    r"\b(?:in|inside|outside|at)\s+(?:a|an|the|this|that)\b", re.I)
_SPOT_PHRASE = re.compile(
    r"\b(?:on|by|near|beside|under|behind)\s+(?:a|an|the|this|that)\b", re.I)
_PERSON_WORD = re.compile(r"\b(?:her|his|their|him|them|herself|himself)\b", re.I)
_LEADING_PRONOUN = re.compile(r"^\s*(?:she|he|they|her|his|their)\b", re.I)


def _name_forms(name):
    """A name as a script writes it: whole, and a two-part name by either part."""
    forms = {name}
    parts = [p for p in name.split() if len(p) >= 3 and p[:1].isupper()]
    if len(parts) > 1:
        forms.update(parts)
    return forms


def _setting_of(sentence):
    """Where a sentence happens, without who is there. "" when it names no place.

    From the first "in a / at the / on the ..." to the end of the sentence, so
    "Maya sits at her desk in an office" gives "In an office" -- "at her desk" is
    hers, not the room's, and is passed over because it is not "at a" or "at the"."""
    m = _SETTING_PHRASE.search(sentence or "") or _SPOT_PHRASE.search(sentence or "")
    if not m:
        return ""
    phrase = sentence[m.start():].strip().rstrip(".!?;, ")
    if not phrase or _PERSON_WORD.search(phrase):
        return ""
    return phrase[0].upper() + phrase[1:] + "."


def static_for_shot(static, sheet, shot_sheet):
    """The anchor and opening paragraph for one shot: nobody named who is not in it.

    Names are the sheet's, matched case-sensitively as sheet_for_beat matches them,
    so "will" is never Will. A sentence that opens on a pronoun straight after one
    that was cut goes with it -- "Maya sits at her desk. She types." in a shot
    without Maya leaves no "She" behind to be drawn."""
    if not (static or "").strip() or not (sheet or "").strip():
        return static
    here = {n for n, _ in sheet_lines(shot_sheet or "") if n}
    absent = set()
    for name, _ in sheet_lines(sheet):
        if name and name not in here:
            absent |= _name_forms(name)
    for name in here:
        absent -= _name_forms(name)
    if not absent:
        return static
    named = re.compile(r"(?<![\w'\u2019-])(?:" + "|".join(
        re.escape(f) for f in sorted(absent, key=len, reverse=True)) + r")(?![\w-])")

    out = []
    for line in static.split("\n"):
        kept, cut_last = [], False
        for sentence in re.split(r"(?<=[.!?])\s+", line.strip()):
            if not sentence:
                continue
            if named.search(sentence) or (cut_last and _LEADING_PRONOUN.match(sentence)):
                setting = _setting_of(sentence)
                if setting and not named.search(setting):
                    kept.append(setting)
                cut_last = True
                continue
            cut_last = False
            kept.append(sentence)
        if kept:
            out.append(" ".join(kept))
    return "\n".join(out)


_QUOTED = re.compile(r'["“][^"”]+["”]')
# H3's OWN dialogue delimiter. comfy/text_encoders/minimax.py registers <d> and </d>
# as special tokens, alongside a caption channel (<|caption_start|>...) and a lyrics
# one -- so the model distinguishes speech, captions and lyrics explicitly. Text in
# plain quotes is not marked as any of them, and a model with a caption channel is
# entitled to read it as a caption, which renders as text ON the picture.
_DIALOGUE_TAG = re.compile(r"<\s*d\s*>(.+?)<\s*/\s*d\s*>", re.I | re.S)
# Tokens that ASK for text on the frame. If one of these is in the prompt, the
# subtitles are being requested, not invented.
_CAPTION_TOKEN = re.compile(r"<\|(?:caption|lyrics)_(?:start|end)\|>", re.I)


# A shot LONGER than its action does not get filled with more action -- it gets
# filled by performing the same action more slowly, which reads as the whole film
# being in slow motion. Measured: "Maya walks to the window" is a few steps, under
# two seconds of real movement, and the old constants gave it a 4.5s shot.
#
# The base was the larger error. It was meant as setup and settle, but a chained shot
# continues from the previous frame -- it opens mid-scene, with nothing to set up.
BEAT_BASE_SEC = 0.8            # a little room to settle, not a whole beat of it
SECONDS_PER_ACTION = 2.2       # screen time one staged action clause needs
WORDS_PER_SEC = 2.5            # spoken delivery
# A new coordinated verb phrase starts a new action.
#
# A PLAIN COMMA between verb phrases is one too, and it is the commonest way
# anybody writes a sequence: "walks in, drops her bag, takes off her coat, hangs
# it up". Only " and " used to split that, so ten actions counted as TWO and the
# beat was sized for two -- the shot then performed all ten inside it, which is a
# walk down a hallway arriving as a cut to the far end. Reported as scenes being
# cut short and missing their detail.
#
# The comma has to be followed by an INFLECTED verb, so a list of adjectives or of
# garments does not split: "a red, tattered coat" is one thing, and a character
# sheet is not a sequence of actions.
_CLAUSE_SPLIT = re.compile(
    r"(?:[.!?;]+|,?\s+(?:and then|then|and|before|after|while|as|until)\s+"
    r"|,\s+(?=\w+(?:ing|es|s|ed)\b))")


def travel_spaces(beat):
    """How many distinct spaces this beat shows on screen. 0 when it goes nowhere.

    THE WALK IS THE EXPENSIVE PART OF A TRANSIT, AND IT WAS INVISIBLE TO THE SIZING.
    beat_seconds counts ACTION CLAUSES, so the grammar of the sentence set the time
    and the ground covered did not: "McKenna walks down the hallway to the living
    room" is one verb phrase, so it was sized for one action -- 3.0s, the floor, the
    SHORTEST shot in its script -- and then told to show three rooms inside it, while
    "gets up and comes out of her bedroom" got 5.2s to stand up in one room. The beats
    doing the most spatial work were getting the least time to do it.

    A model handed 73 frames, a bedroom keyframe and instructions to reach a living
    room cannot TRAVEL, so it blends the two into one hybrid space -- which is a
    living room with a bed in it, the third route to a bug already fixed twice in the
    text. _CLAUSE_SPLIT's own comment names this failure exactly, "a walk down a
    hallway arriving as a cut to the far end", and fixed it only for comma lists.

    AN INTRA-ROOM WALK IS NOT THIS and must stay short: test_pace measured "Maya walks
    to the window" as under two seconds of real movement and the constants were tuned
    down for it. A window is not a place, so it crosses nothing here. Only a beat that
    actually ARRIVES somewhere counts, which is the same test travel_anchor applies
    before it will say a journey happened at all.

    The origin counts even when the beat does not name it: the shot opens in the room
    it was already in, that room is on screen at frame one, and it has to be left."""
    text = _DIALOGUE_TAG.sub(" ", _QUOTED.sub(" ", str(beat or "")))
    frm, via, to = travel_legs(text)
    if not to:
        # A place the list cannot name still has to be walked to, and getting there
        # still costs screen time. Origin plus destination. See moved_to.
        return 2 if moved_to(text) else 0
    named = [p for p in (frm, via, to) if p]
    return len(named) + (0 if frm else 1)


def beat_seconds(beat):
    """Roughly how much screen time this beat's content asks for.

    Action and dialogue OVERLAP -- people talk while they move -- so it is the
    larger of the two, not the sum. Deliberately rough: the point is not to size
    the shot (the node does not), it is to notice when a shot is much longer than
    anything the beat gives it to do."""
    text = _DIALOGUE_TAG.sub(" ", _QUOTED.sub(" ", beat or ""))
    text = _REMOVE_LINE.sub("", _ADD_LINE.sub("", text))
    clauses = [p for p in _CLAUSE_SPLIT.split(text) if p and len(p.split()) >= 2]
    # A ROOM BOUNDARY CROSSED ON SCREEN COSTS WHAT A STAGED ACTION COSTS. Rooms have
    # to be established to be left, and crossing into one is work the grammar of the
    # sentence does not show: one verb phrase can move somebody through three rooms.
    # See travel_spaces. Reused constant rather than a new one, because this IS the
    # same quantity -- screen time that something has to happen in.
    crossings = max(0, travel_spaces(text) - 1)
    action = (BEAT_BASE_SEC + SECONDS_PER_ACTION * (len(clauses) + crossings)) \
        if (clauses or crossings) else 0.0
    spoken = sum(len(q.split()) for q in _QUOTED.findall(beat or "")) \
        + sum(len(q.split()) for q in _DIALOGUE_TAG.findall(beat or ""))
    return max(action, (spoken / WORDS_PER_SEC + 1.0) if spoken else 0.0)


MIN_AUTO_FRAMES = 73           # ~3.0s: the shortest shot that can hold one action


def plan_lengths(beats, ceiling_frames, from_beat, pace=1.0):
    """Frames for each shot. Returns (lengths, note).

    'fixed' gives every shot the ceiling. 'from the beat' sizes each shot from what
    its own line stages, capped by that same ceiling and floored at one action's
    worth -- so a beat with one action stops getting a shot with room for two, which
    is what makes an action carry on past its end.

    The estimate leans SHORT deliberately. A shot that ends before its action does
    hands a mid-motion frame to the next shot, and the chain is built to continue
    from exactly that. A shot that outlasts its action does not invent more action --
    it performs the same action more slowly, which is what slow-looking footage is.

    `pace` scales the whole estimate: below 1.0 the shots get shorter and the motion
    in them brisker, above 1.0 they get longer and slower."""
    if not from_beat:
        return [ceiling_frames] * len(beats), ""
    pace = max(0.05, float(pace if pace else 1.0))
    lens, capped = [], []
    for b in beats:
        need = beat_seconds(b) * pace
        want = align_frame_count_nearest(int(round(need * H3_FPS))) if need else MIN_AUTO_FRAMES
        # A beat that wants MORE than shot_seconds allows is compressed into it,
        # silently. The shot then performs the whole beat faster -- a walk down a
        # hallway becomes a cut to the far end -- and nothing in the report said the
        # length was the reason. Reported as scenes being cut short.
        if want > ceiling_frames:
            capped.append((len(lens) + 1, want))
        lens.append(max(MIN_AUTO_FRAMES, min(want, ceiling_frames)))
    note = ""
    if capped:
        note = ("shot(s) "
                + ", ".join(f"{n} (wants {w / H3_FPS:.1f}s)" for n, w in capped[:6])
                + f" stage more than shot_seconds allows, so they are cut to "
                  f"{ceiling_frames / H3_FPS:.1f}s and perform the whole beat faster "
                  f"-- which is a walk down a hallway arriving as a cut to the far "
                  f"end. Raise shot_seconds (H3's own ceiling is "
                  f"{MAX_FRAMES / H3_FPS:.1f}s), or give the beat fewer actions and "
                  f"let the next one carry the rest. ")
    if len(set(lens)) > 1:
        note += (
                "shot lengths are sized from each beat ("
                + ", ".join(f"{n}f/{n / H3_FPS:.1f}s" for n in lens)
                + "). They differ, so one seed does not give them one noise field -- "
                  "noise is drawn to the latent's shape -- and surface detail resets at "
                  "each cut. Set shot_length to 'fixed' if that matters more than pacing")
    return lens, note


def pace_clause(need, have):
    """Spread a short action across a long shot. "" when the shot is not long.

    thin_beats has always been able to SEE this -- one action sitting in a ten
    second shot -- and only ever reported it. The shot was still told what happens
    and nothing about when, so the action was performed at once and the spare
    seconds filled by carrying on: the same movement repeated on whatever was
    nearest. Reported as actions running way ahead of schedule.

    A timing anchor, the same shape the node already uses for a door ("open at the
    first frame and shut by the last") and for a removal ("away by the last
    frame"). It names WHEN, not how fast: "slowly" is a style instruction and this
    is not one -- it says the action occupies the shot it was given.

    Only where the gap is real. thin_beats' own thresholds: at least 2.5 spare
    seconds and a quarter again longer than the content, so a shot that only
    slightly outlasts a long beat stays quiet."""
    try:
        need, have = float(need), float(have)
    except (TypeError, ValueError):
        return ""
    if need <= 0 or (have - need) < 2.5 or have <= need * 1.25:
        return ""
    return (" What the beat stages runs at an even pace across the whole shot, "
            "beginning at the first frame and finishing on the last.")


def thin_beats(beats, seconds):
    """Beats with far less content than the shot they are given.

    A shot that outlasts its action leaves the model seconds it was told nothing
    about, and the cheapest way to fill them is to CARRY ON: an action that has
    finished its object repeats it on whatever is nearest. Pure arithmetic -- it
    cannot know whether "walks
    across the room" is two seconds or ten, but it can see one action sitting in a
    ten second shot and say so before the render."""
    out = []
    for i, b in enumerate(beats or [], 1):
        need = beat_seconds(b)
        # The GAP matters more than the ratio: "takes off her coat and hangs it up"
        # asks for about 7s, and in a 10s shot the three spare seconds are enough for
        # the action to run on past the thing it was given. A small ratio guard
        # keeps it quiet when the shot only slightly outlasts a long beat.
        if need and (seconds - need) >= 2.5 and seconds > need * 1.25:
            out.append(f"shot {i}: ~{need:.0f}s of content in a {seconds:.0f}s shot")
    return out


# Effort and reaction: the beats where a face has something to do.
_EXERTION = re.compile(
    r"\b(?:thrash(?:es|ing|ed)?|struggl(?:e|es|ing|ed)|writh(?:e|es|ing|ed)|"
    r"strain(?:s|ing|ed)?|fight(?:s|ing)?|kick(?:s|ing|ed)?|jerk(?:s|ing|ed)?|"
    r"gasp(?:s|ing|ed)?|pant(?:s|ing|ed)?|cr(?:y|ies|ying)|sob(?:s|bing|bed)?|"
    r"scream(?:s|ing|ed)?|shout(?:s|ing|ed)?|yell(?:s|ing|ed)?|moan(?:s|ing|ed)?|"
    r"whimper(?:s|ing|ed)?|laugh(?:s|ing|ed)?|flinch(?:es|ing|ed)?|"
    r"trembl(?:e|es|ing|ed)|shak(?:e|es|ing)|shiver(?:s|ing|ed)?|"
    r"freak(?:s|ing)?\s+out|wakes?\s+up|woke\s+up|panic(?:s|king|ked)?)\b", re.I)

# What a hand closes on under effort. A railing, a wheel, a bag or a door handle is
# somebody steadying themselves and is deliberately not here.
_EFFORT_OBJ = (r"(?:her|his|their|the)\s+(?:backs?|hips?|thighs?|shoulders?|arms?|"
               r"wrists?|waist|hair|neck|sheets?|bedding|blankets?|pillows?|"
               r"mattress|headboard|bars?|restraints?)")

# The generic MOTION verbs, which mean effort only in context.
#
# These were added bare, and bare they are wrong: arch, buck, clench, clutch, grind,
# grip, rock and thrust are ordinary English. "He grinds the coffee", "she grips the
# railing", "the truck rocks over the kerb" and "she arches an eyebrow" all read as
# vocal effort, which put "unsteady breathing, with gasps and moans of effort" into
# the prompt of a scenery beat, opened its audio branch and took the mouth guard off
# it. On a joint model that is a close-up of a panting face where a wide shot of a
# hallway was asked for, and an invented speaker to go with the invented voice.
#
# THE TRADE RUNS THE OTHER WAY FROM WHAT I ASSUMED. A wrong OPEN branch costs
# moaning text, a free mouth and a babbling stream that drags the framing with it; a
# wrong CLOSED one costs a silent shot -- and that is the whole cost again now that
# nothing is built to cover it. So these must corroborate, never fire alone.
#
# `clench` is gone except standing alone: a clenched jaw or fist is silent tension,
# which is the opposite of a sound cue.
_EXERTION_NARROW_SRC = (
    # arching a back, not an eyebrow
    r"arch(?:es|ed|ing)?\s+(?:her|his|their)\s+backs?\b|"
    r"arch(?:es|ed|ing)?\s+(?:up|upwards?|off)\b|"
    # sustained movement, always against or with something
    r"(?:rock|grind|thrust|buck|push|move)(?:s|ed|ing)?\s+"
    r"(?:against|into|together|beneath|underneath|under|onto)\b|"
    # ...or the same verbs with no object at all, which is the intransitive sense
    r"(?:rocks?|rocked|rocking|grinds?|ground|grinding|thrusts?|thrusting|"
    r"bucks?|bucked|bucking|clench(?:es|ed|ing)?)\s*(?=[.,;!?]|$)|"
    # a hand closing on a body or the bedding, not on a railing
    r"(?:clutch(?:es|ed|ing)?|grip(?:s|ped|ping)?|claw(?:s|ed|ing)?)\s+"
    r"(?:at\s+)?" + _EFFORT_OBJ + r"|"
    r"(?:clutch(?:es|ed|ing)?|grip(?:s|ped|ping)?|claw(?:s|ed|ing)?)\s+at\b|"
    # involuntary, and rarely said of a prop
    r"shudder(?:s|ed|ing)?\b")
_EXERTION_NARROW = re.compile(r"\b(?:" + _EXERTION_NARROW_SRC + r")", re.I)


def exertion_in(beat):
    """Does this beat stage effort or reaction -- something a face performs?

    Two lists: verbs that are inherently about distress or exertion and mean it
    wherever they appear, and generic motion verbs that mean it only with the right
    complement. See _EXERTION_NARROW for why the second group may not fire alone."""
    b = beat or ""
    return bool(_EXERTION.search(b) or _EXERTION_NARROW.search(b))


# Sound the text asks for. H3 is joint, so the same prose conditions the audio
# branch -- a scene is scored by describing it, not by a setting.
_SOUND_CUE = re.compile(
    r"\b(?:sounds?|noises?|echo(?:e?s|ing)?|rattl(?:e|es|ing)|clank(?:s|ing)?|"
    r"clink(?:s|ing)?|creak(?:s|ing)?|scrap(?:e|es|ing)|thud(?:s|ding)?|bang(?:s|ing)?|"
    r"slam(?:s|ming)?|clatter(?:s|ing)?|jingl(?:e|es|ing)|squeak(?:s|ing)?|"
    r"footsteps?|breath(?:s|es|ing)?|pant(?:s|ing)?|gasp(?:s|ing)?|sigh(?:s|ing)?|"
    r"whimper(?:s|ing)?|moan(?:s|ing)?|groan(?:s|ing)?|sob(?:s|bing)?|"
    r"scream(?:s|ing)?|shout(?:s|ing)?|whisper(?:s|ing)?|laugh(?:s|ing|ter)?|"
    r"hum(?:s|ming)?|buzz(?:es|ing)?|hiss(?:es|ing)?|drip(?:s|ping)?|"
    r"rustl(?:e|es|ing)|click(?:s|ing)?|snap(?:s|ping)?|zip(?:s|ping)?|"
    r"rings?|ringing|wind|rain|thunder|traffic|music|hollow|muffled|reverb|"
    # How a sound is usually written when the noun is not itself a sound word.
    # "her boots loud on the concrete" describes a sound and named none of the above,
    # so it was read as staging nothing audible and the shot was silenced -- which is
    # the one thing the docs tell you to do to score a silent shot.
    # Adverbs only where the bare adjective describes something other than a sound --
    # "quietly closes the door" is a sound being made, while "the workshop is quiet",
    # "she is quiet" and "a faint smile" are the absence of one or nothing to do with
    # one. Opening the branch on those is a free branch with no line in the shot,
    # which is where an invented voice comes from.
    # `silence` sat at the head of this list as a noun and did exactly that: "she
    # sits in silence" read as a sound being asked for, opened the branch, and the
    # one word that most plainly asks for a pinned shot was the one that unpinned
    # it. A beat that names silence names nothing to make; it gets the default.
    r"loud(?:ly)?|quietly|faintly|audible|noisy|deafening|"
    r"scuff(?:s|ing|ed)?|crunch(?:es|ing|ed)?|thump(?:s|ing|ed)?|"
    r"patter(?:s|ing)?|whirr?(?:s|ing)?|whine(?:s|d)?|whining|rumbl(?:e|es|ing)|"
    r"growl(?:s|ing)?|roar(?:s|ing)?|chime(?:s|d)?|ticking|"
    r"knock(?:s|ing)?|tap(?:s|ping)?|whoosh(?:es|ing)?|sizzl(?:e|es|ing))\b", re.I)


# A BREATH IS NOT ENOUGH SOUND TO HOLD A BRANCH OPEN.
#
# "Dana takes a breath." is the beat people write immediately before a line, and
# it read as the author asking for a sound -- so the audio branch stayed open for
# the whole shot with nothing in it but half a second of breathing. An open
# branch on a joint model fills itself, and at 4-8 steps the last audio step
# clears 30-50% of the denoising in one jump, so what it fills with is a voice.
# Reported as micro-babble at the start of a scene, just as somebody goes to talk.
#
# Only the PREPARATORY breath, and only when it is all there is.
#
# Sustained breathing is a different thing and does fill a shot: "she breathes
# hard through the gag" is the sound of that shot, and silencing it would be
# taking away a sound somebody asked for by name. So would sighs, gasps, moans.
# What this catches is the single indrawn breath before a line -- one gesture,
# half a second, against a whole shot of open branch.
#
# And only when nothing else is making a noise: "takes a breath as the chain
# rattles" still opens it, because the chain has something to say for the rest.
_BREATH_WORD = re.compile(r"\bbreath(?:s|es|ing)?\b|\bbreathe[sd]?\b", re.I)
_BREATH_PREP = re.compile(
    r"\b(?:takes?|took|taking|draws?|drew|drawing|catch(?:es)?|caught|"
    r"suck(?:s|ed)?|pull(?:s|ed)?|lets?\s+out|releases?)\s+"
    r"(?:in\s+)?(?:a|an|her|his|their|one|another|deep|long|slow|sharp|\s)*"
    r"breath\b|\bwith\s+a\s+breath\b|\ba\s+(?:deep\s+|long\s+|slow\s+|sharp\s+)?"
    r"breath\b", re.I)


def sound_described(text):
    """Does this beat ask for a sound the audio branch should make?

    A breath on its own does not: see _BREATH_ONLY."""
    t = text or ""
    hits = [h for h in (m.group(0).strip() for m in _SOUND_CUE.finditer(t)) if h]
    if not hits:
        return False
    if all(_BREATH_WORD.fullmatch(h) for h in hits) and _BREATH_PREP.search(t):
        return False
    return True


# What a staged action sounds like. The beat already says what happens; the sound it
# makes follows from that, so it does not have to be written twice.
#
# Matched against the BEAT only, never the scene. Sourcing it from the scene as well
# would put a chain rattling into a shot where nobody moves, because the scene says
# there is a chain -- the beat is what decides whether anything makes a noise.
# The six, as their own table, because two readers need them and a second copy
# would drift. sounds_for suppresses a LONE vocal -- the beat carries it verbatim
# and the node has nothing to add over the top -- and named_vocals_in below does
# not, because a shot whose clause is spoken as a CLOSED list has to name it.
_VOCAL_FROM = (
    (r"\bwhimper(?:s|ing|ed)?\b",                   "whimpering"),
    (r"\bsob(?:s|bing|bed)?\b",                     "sobbing"),
    (r"\bmoan(?:s|ing|ed)?\b",                      "moaning"),
    (r"\bgroan(?:s|ing|ed)?\b",                     "groaning"),
    (r"\bscream(?:s|ing|ed)?\b",                    "screaming"),
    (r"\bwhin(?:e|es|ing|ed)\b",                    "whining"),
)

_SOUND_FROM = (
    # A VOCAL THE BEAT NAMES IS THE ONE THE SHOT MAKES, and it goes FIRST.
    #
    # These were missing entirely: sound_described() reads them off _SOUND_CUE and
    # opens the audio branch, but nothing put them into the sound clause, so the
    # word the author wrote reached neither branch. What the shot was told instead
    # was inferred from the MOTION verb beside it -- "she whimpers and thrashes"
    # produced "unsteady breathing, with gasps and moans of effort" and no whimper.
    #
    # Two failures came back from that, and they are the same substitution:
    #
    #   - The clause is emitted with only=True, a CLOSED list. "The only sounds
    #     are ... moans of effort" does not merely omit the whimpering, it asserts
    #     the whimpering is not there, against a beat that says it is.
    #   - "moans" is the one vocal that reads as readily as pleasure. The face
    #     follows the audio branch on a joint model (see sound_clause), so a shot
    #     of distress conditioned on moans of effort renders a woman smiling.
    #     Reported exactly that way.
    #
    # FIRST in the tuple because sounds_for stops at MAX_SOUNDS, and on the beat
    # this was reported from the budget was already full of engine, restraints and
    # the inferred effort phrase before any vocal could be reached. What the author
    # wrote outranks what the node inferred; that is the whole of the ordering rule.
    #
    # Speech verbs are NOT here. shout and whisper are lines being delivered and
    # belong to the dialogue path, which suppresses the mouth guard and opens the
    # branch on purpose. These six are non-speech vocalisations only.
    *_VOCAL_FROM,
    (r"\b(?:walk(?:s|ed|ing)?|step(?:s|ped|ping)?|pace[sd]?|enters?|runs?|"
     r"approach(?:es|ed)?|creep(?:s|ing)?|crept|sneak(?:s|ing)?|shuffl(?:e|es|ing)|"
     r"stumbl(?:e|es|ing)|stagger(?:s|ing)?|feet)\b",  "footsteps"),
    (r"\bchains?\b",                                "chain links dragging"),
    # BEFORE the generic cuffs entry, because both match and the first wins. Cuffs
    # being APPLIED are a ratchet, which is the sound anyone picturing the moment
    # expects; "cuffs knocking" is what they do afterwards, hanging on a wrist.
    # BOTH conditions as lookaheads anchored at \A, so each scans the WHOLE beat.
    # A lookahead placed mid-pattern only looks FORWARD from wherever the engine is
    # standing, so "the cuffs ratchet closed" failed -- the hardware is named before
    # the verb, and by the time the verb matched the cuffs were behind it. Written
    # this way the order in the sentence stops mattering. See the restraint entry
    # below, which had the same defect and lost its sound on exactly that wording.
    (r"\A(?=[\s\S]*\b(?:handcuff|cuff|shackle|manacle)\w*\b)"
     r"(?=[\s\S]*\b(?:ratchet(?:s|ed|ing)?|clos(?:e|es|ing|ed)|snap(?:s|ped|ping)?|"
     r"lock(?:s|ed|ing)?|tighten(?:s|ed|ing)?|click(?:s|ed|ing)?)\b)",
                                                    "cuffs ratcheting closed"),
    (r"\b(?:handcuff(?:s|ed)?|cuffs?|cuffed|shackle[sd]?|manacle[sd]?)\b",
                                                    "cuffs knocking"),
    # A bolt is not something dragging on the floor, which is what the drag entry
    # below was giving it. Ahead of that entry, because "slides the bolt" matches
    # both and the first match is the one that is kept.
    (r"\b(?:bolt|latch|catch)(?:es|ed|ing)?\b",     "a metal bolt sliding"),
    # NOT "locks eyes with her" -- that is a look, and it was giving the shot the
    # sound of a padlock closing.
    (r"\b(?:padlock(?:s|ed)?|locks?|locked|locking)\b(?!\s+(?:eyes|gaze|horns|onto))",
                                                    "a lock snapping shut"),
    (r"\b(?:drag(?:s|ged|ging)?|haul(?:s|ed|ing)?|shov(?:e|es|ing)|slid(?:e|es|ing))\b",
                                                    "something dragging on the floor"),
    (r"\b(?:buckle(?:s|d)?|unbuckle(?:s|d)?|clasp(?:s|ed)?|strap(?:s|ped)?|harness)\b",
                                                    "a buckle and leather creaking"),
    (r"\b(?:pour(?:s|ed|ing)?|water|splash(?:es|ed)?|wet|puddle)\b",
                                                    "water"),
    (r"\b(?:van|car|engine|truck|motor)\b",         "an engine outside"),
    (r"\b(?:fabric|cloth|coat|jacket|shirt|dress|skirt)\b", "fabric rustling"),
    (r"\b(?:scissors|shears|cut(?:s|ting)?)\b",     "blades through fabric"),
    (r"\bdoors?\b",                                 "a door on its hinges"),
    (r"\b(?:drops?|dropped|throw(?:s|n)?|threw|toss(?:es|ed)?)\b",
                                                    "something landing"),
    (r"\b(?:smack(?:s|ed)?|slap(?:s|ped)?|hits?|strikes?|struck)\b", "a sharp impact"),
    # Only where there is something to pull against. "McKenna thrashes on the bed"
    # was getting restraints she is not wearing, because the verb alone armed it.
    # Anchored at \A with BOTH conditions as lookaheads, so the hardware and the
    # verb may appear in either order. Before this the lookahead sat mid-pattern and
    # only looked forward: "she strains against the cuffs" worked and "the cuffs
    # hold her wrists as she strains" silently did not, which is the same sentence.
    (r"\A(?=[\s\S]*\b(?:cuffs?|handcuffs?|shackles?|manacles?|chains?|ropes?|cords?|"
     r"straps?|restraints?|bindings?|ties?|tape|harness|collar)\b)"
     r"(?=[\s\S]*\b(?:thrash(?:es|ing|ed)?|struggl(?:e|es|ing|ed)|writh(?:e|es|ing|ed)|"
     r"strain(?:s|ing|ed)?|pull(?:s|ing|ed)?\s+against)\b)",
                                                    "restraints pulling taut"),
    # A body under effort makes a VOICE, not only movement. H3 is joint, so this is
    # also what stops the face going flat: conditioning the audio on silence tells the
    # model the person makes no sound, and a person making no sound is rendered still.
    # A beat that already names the sound is left alone -- "she moans" is in
    # _SOUND_CUE, so what you wrote wins and none of this is added.
    # THE TWO LISTS HAVE TO AGREE, and they are now built from the same source so
    # they cannot drift again. The generic motion verbs were bare here too, so "he
    # grinds the coffee" was given moans of effort as prompt text -- and prompt text
    # on a joint model steers the picture, which is how a wide shot became a
    # close-up of a panting face. See _EXERTION_NARROW.
    (r"\b(?:thrash(?:es|ing|ed)?|struggl(?:e|es|ing|ed)|writh(?:e|es|ing|ed)|"
     r"strain(?:s|ing|ed)?|trembl(?:e|es|ing|ed)|shiver(?:s|ed|ing)?)\b"
     r"|\b(?:" + _EXERTION_NARROW_SRC + r")",
                                                    "unsteady breathing, with gasps and "
                                                    "moans of effort"),
    (r"\b(?:zip(?:s|ped|ping)?|unzip(?:s|ped|ping)?|zipper)\b", "a zip running"),
    (r"\btap(?:e|es|ed|ing)\b",                     "tape pulling off"),
    # Gaps found by listing the beats this is actually asked for and reading what
    # came back. Each of these returned NOTHING, on a shot whose whole point is the
    # sound: velcro, a rope going tight, and the lower-body garments -- the fabric
    # entry listed coat, jacket, shirt, dress, skirt and stopped there, so taking
    # off a pair of shorts was silent while taking off a coat was not.
    # FURNITURE UNDER SUSTAINED MOVEMENT. Both conditions, either order, because a
    # bed standing in the scene must not creak in a shot where nobody moves -- the
    # same rule the room tone follows. This is the NON-VOCAL half: a frame and a
    # mattress working. The vocal half is not built anywhere and cannot be, since
    # this synthesiser shapes noise and a voice is not noise; it comes from the
    # model, on a branch the effort verbs open. See _EXERTION.
    (r"\A(?=[\s\S]*\b(?:bed|mattress|springs?|bunk|couch|sofa|headboard|"
     r"frame|table|desk|floorboards?)\b)"
     r"(?=[\s\S]*\b(?:rock(?:s|ed|ing)?|thrust(?:s|ing)?|grind(?:s|ing)?|"
     r"buck(?:s|ed|ing)?|writh(?:e|es|ing|ed)|arch(?:es|ed|ing)?|"
     r"thrash(?:es|ing|ed)?|struggl(?:e|es|ing|ed)|move(?:s|d)?\s+together|"
     r"shift(?:s|ed|ing)?\s+under)\b)",              "a bed frame working"),
    (r"\bvelcro\b",                                 "velcro tearing open"),
    (r"\b(?:rope|cord|twine|zip\s?tie)s?\b",        "rope creaking as it goes tight"),
    (r"\b(?:shorts|trousers|pants|jeans|leggings|tights|socks|boots|shoes|"
     r"gloves|top|vest|jumper|sweater|hoodie|trousers)\b", "fabric rustling"),
    (r"\bkeys?\b",                                  "keys on a ring"),
    (r"\b(?:wakes?\s+up|woke|gasp(?:s|ing)?|pant(?:s|ing)?|breath(?:es|ing)?)\b",
                                                    "breathing"),
)
MAX_SOUNDS = 3      # a shot's audio needs a cue, not an inventory
# {specific: (generals it retires)} -- see sounds_for.
# The inferred effort phrase and the bare breath are what a NAMED vocal replaces:
# one mouth is making one sound, and saying it twice spends two of three slots on
# the same thing -- the crowding this table exists to stop. The effort phrase also
# retires the bare "breathing" on its own, with no vocal named at all: "wakes up"
# and "thrashes" both fired and a shot came back listing "unsteady breathing, with
# gasps and moans of effort AND breathing".
_VOCAL_RETIRES = ("unsteady breathing, with gasps and moans of effort", "breathing")
# WHAT IS HAPPENING BETWEEN THE MOANS.
#
# A named vocal opens the audio branch on purpose -- it is meant to be heard -- and
# then sounds_for said nothing at all, because the vocal was the whole list and the
# beat already carries it. Reported as babble between the moans, and that is exactly
# where it came from: a moan is INTERMITTENT, the branch is open for the whole shot,
# and nothing described the gaps. An open branch on a joint model fills itself, and
# what it fills itself with, next to a face, is speech.
#
# There is no way to ask for the absence of speech -- cfg is 1, there is no negative
# prompt, and naming it would ask for it. The only move is to say what IS there, and
# between moans what is there is breath. It is continuous where the vocal is not,
# which is the whole point: it gives the gaps something to be.
#
# This is why a lone vocal no longer returns nothing. The old reasoning was that the
# node would be restating the author to the author -- true of the vocal, and the
# vocal is still not restated for its own sake; what is added is the half the author
# did not write and the branch cannot do without.
_VOCAL_BETWEEN = "breathing"
# The six above, as a set: see the tail of sounds_for for why they are special-cased.
_NAMED_VOCALS = frozenset(("whimpering", "sobbing", "moaning", "groaning",
                           "screaming", "whining"))
_SOUND_SUPERSEDES = {
    "cuffs ratcheting closed": ("cuffs knocking",),
    "unsteady breathing, with gasps and moans of effort": ("breathing",),
    "whimpering": _VOCAL_RETIRES,
    "sobbing": _VOCAL_RETIRES,
    "moaning": _VOCAL_RETIRES,
    "groaning": _VOCAL_RETIRES,
    "screaming": _VOCAL_RETIRES,
    "whining": _VOCAL_RETIRES,
}

# The SPACE, as opposed to the things in it. Read from the scene, and this is the one
# thing that safely can be: a chain standing in the scene must not rattle in a shot
# where nobody moves, but a concrete room is hard in every shot whatever happens in
# it. That is the difference between a recording and a sound effect -- real footage
# has a bed under the events, and digital silence between them is what makes a scene
# sound staged.
_ROOM_TONE = (
    (r"\b(?:bathroom|shower|tiled?|tiles)\b",       "tiled walls ringing"),
    (r"\b(?:basement|cellar|warehouse|garage|hangar|tunnel|stairwell|"
     r"corridor|concrete|stone|brick|bare walls?)\b", "hard walls giving the sound back"),
    # "shallow depth of field" and "field of view" are the LENS, not a location.
    # Every anchor written for this node says one of them, so every interior scene
    # was being told it sounds like open air.
    (r"\b(?:outside|outdoors|street|road|yard|garden|forest|beach|park)\b"
     r"|(?<!depth of )\bfield\b(?! of view)",       "open air with no walls close by"),
    (r"\b(?:carpet(?:ed)?|curtains?|bedroom|sofa|cushions?)\b",
                                                    "a soft room with little echo"),
    (r"\b(?:barn|attic|loft|shed|workshop|hall|church)\b", "a large room with a long tail"),
)


# AMBIENT read from the scene, not from the beat. room_tone says how a space
# SOUNDS -- an acoustic, "hard walls giving the sound back" -- which shapes a sound
# that is already there and makes none of its own. A shot with no line and no
# written sound was therefore pinned to real silence: not "no speech" but no
# footsteps, no room tone, nothing, which is what makes a scene sound staged.
#
# Asked for: read the anchor and the scene and score the film from them, so the
# ambience does not have to be typed into every beat.
#
# Ordered most specific first -- weather and named places before the generic
# interior, so "a rainy street" is rain and traffic rather than "a room".
# Each phrase has to read naturally after "The only sound is ...", so none of them
# is a plural noun phrase, and NONE of them names a voice: "low talk" was in this
# table for one revision and is exactly the thing that puts a second speaker in a
# silent shot. Ambience is weather, machines, rooms and distance -- never people.
_AMBIENT = (
    (r"\brain(?:ing|y)?\b|\bdownpour\b|\bdrizzl", "rain against the glass"),
    (r"\bstorm|\bthunder", "a storm somewhere outside"),
    (r"\bwind(?:y)?\b|\bgale\b", "wind against the building"),
    (r"\bbeach\b|\bsea\b|\bocean\b|\bshore\b", "the sea a long way off"),
    (r"\bforest\b|\bwoods?\b", "wind in the trees"),
    (r"\bgarden\b|\byard\b|\bpark\b", "birdsong"),
    (r"\bstreet\b|\broad\b|\btraffic\b|\bcity\b|\bpavement\b",
     "traffic somewhere off the street"),
    (r"\bcar\b|\bvan\b|\btruck\b|\bdriving\b", "an engine idling"),
    (r"\bkitchen\b", "a fridge humming"),
    (r"\bbathroom\b|\bshower\b", "water moving in the pipes"),
    (r"\bnursery\b|\bbaby\b|\bcot\b|\bcrib\b", "a clock ticking"),
    (r"\bbedroom\b", "the quiet of a bedroom"),
    (r"\boffice\b|\bstudy\b", "a computer fan"),
    (r"\bworkshop\b|\bgarage\b|\bfactory\b", "a strip light humming"),
    (r"\bbasement\b|\bcellar\b|\bboiler\b", "a low hum off the strip light"),
    (r"\bhospital\b|\bward\b|\bclinic\b", "a monitor somewhere down the corridor"),
    (r"\bcafe\b|\bbar\b|\brestaurant\b|\bpub\b", "cutlery and moving chairs"),
    (r"\bschool\b|\bclassroom\b", "a corridor beyond the door"),
    (r"\bchurch\b|\bhall\b", "the air of a large empty room"),
    (r"\bstairs?\b|\bstairwell\b|\bhallway\b|\bcorridor\b",
     "the hollow quiet of a hallway"),
    (r"\bnight\b|\blate evening\b", "the quiet of a night"),
    # The generic interior LAST, so a named room wins.
    (r"\bhome\b|\bhouse\b|\bflat\b|\bapartment\b|\bliving room\b|\blounge\b"
     r"|\bindoors?\b|\broom\b", "the quiet of a house"),
)

def scene_ambient(*texts):
    """One ambient bed for the film, read from the anchor and the scene. "" if none.

    First match wins, and the table is ordered most specific first: weather and
    named places before the generic interior. ONE bed, not a list -- a shot told
    four things to sound like is a shot inventing which."""
    joined = " ".join(str(t or "") for t in texts)
    if not joined.strip():
        return ""
    for pat, phrase in _AMBIENT:
        if re.search(pat, joined, re.I):
            return phrase
    return ""


def room_tone(scene, opening=""):
    """How the space itself sounds. One room, one acoustic -- the first match wins.

    `opening` is the first beat, and it is read only when the scene names no space at
    all. With `anchor` set there is no scene PARAGRAPH -- the anchor is the whole of
    it -- and an anchor describes the CAMERA, not the room. The location is then
    written in the first beat, so reading nothing but the lens line left the acoustic
    to be guessed from words like "depth of field"."""
    for text in (scene, opening):
        for pat, phrase in _ROOM_TONE:
            if re.search(pat, text or "", re.I):
                return phrase
    return ""


# Sounds that are a THING IN MOTION, and which thing. H3 is joint: the prose
# conditions the audio branch and the picture follows the audio, so "a door on its
# hinges" is not a decoration on a shot with a door in it -- it is a request for a
# door to swing. Asked for beside a sentence holding that same door shut, the sound
# wins, because it describes something happening and the hold describes something
# not happening.
#
# Reported exactly that way: the doors started closed, as the hold asked, and were
# then opened. Two guards, one contradicting the other.
_SOUND_OF_MOVING = {"a door on its hinges": ("door",)}


def sounds_for(beat, held=()):
    """The sounds this beat's own action implies. [] when it stages nothing audible.

    `held` is the scenery this shot is holding still. A sound of one of those moving
    is dropped: the shot cannot be asked to keep the doors shut and to sound like a
    door swinging."""
    held = set(held or ())
    out = []
    for pat, phrase in _SOUND_FROM:
        if len(out) >= MAX_SOUNDS:
            break
        if held.intersection(_SOUND_OF_MOVING.get(phrase, ())):
            continue
        if phrase not in out and re.search(pat, beat or "", re.I):
            out.append(phrase)
    # A specific sound retires the general one for the same object. Cuffs being
    # applied are a ratchet; "cuffs knocking" is what they do afterwards, hanging on
    # a wrist. Both in one shot is one object described making two noises, and the
    # budget is three sounds -- spending two of them on the same pair of cuffs
    # crowds out whatever else the beat stages.
    for specific, general in _SOUND_SUPERSEDES.items():
        if specific in out:
            out = [p for p in out if p == specific or p not in general]
    # WHAT YOU WROTE WINS -- and when it is ALL you wrote, winning means the node
    # says nothing. "She moans." is already the sound of its shot, in the beat, going
    # to the model verbatim; a sentence adding "the only sound is moaning" over the
    # top of it is the node restating the author to the author's own reader.
    #
    # But that only holds while the vocal is the WHOLE list. The clause is emitted
    # closed -- "The only sounds are ..." -- so as soon as anything else is in it,
    # leaving the vocal out stops being silence and becomes a denial: a beat reading
    # "she starts whimpering and thrashes in her restraints" was conditioned on "the
    # only sounds are an engine outside, restraints pulling taut and unsteady
    # breathing, with gasps and moans of effort", which asserts the whimpering is not
    # happening and substitutes a vocal that is not a distress word. The face follows
    # the audio branch, so that shot came back smiling. Both halves were reported.
    if out and all(p in _NAMED_VOCALS for p in out):
        return []
    return out


def named_vocals_in(beat):
    """The non-speech vocals THIS BEAT NAMES, in the order the table lists them.

    sounds_for deliberately returns [] when a vocal is all the beat says: the beat
    goes to the model verbatim and the node has nothing to add over the top of it.
    That is right where the node then says nothing -- and wrong the moment it says
    something EXCLUSIVE.

    "She screams." is sound_described, so _own is true and the inferred list is
    zeroed; exertion_in is also true, so _voiced keeps the branch open rather than
    letting the shot be muted; then the ambient bed is appended and only=not _speaks
    closes the list. The shot was conditioned on "The only sound is an engine
    idling" -- an exclusive claim, against a beat that says she screams, on the one
    kind of shot whose branch is open and therefore has to fill itself with
    something. Reproduced on "She screams.", "She sobs quietly." and "She starts
    whimpering and thrashes in her restraints."

    So the closed list gets the author's own vocal put back into it. This adds
    nothing the node inferred -- these are the author's words, matched literally --
    and it is what keeps the exclusive sentence true."""
    b = str(beat or "")
    return [phrase for pat, phrase in _VOCAL_FROM if re.search(pat, b, re.I)]


def sound_clause(phrases, only=False):
    """One sentence naming what the shot is heard as.

    `only` closes the list. H3 is joint, so the audio branch drives the face: a shot
    whose audio is left free but only loosely described will fill the rest with a
    VOICE, and the mouth moves to it in a shot that has no line. Saying these are the
    only sounds leaves nothing for a voice to fill.

    Positively phrased, because that is the only phrasing this model gets: at cfg 1
    H3 is CFG-free and no negative prompt is evaluated, so "nobody speaks" is not a
    prohibition, it is the word "speaks" in the prompt. "The only sound is X" excludes
    speech by saying what IS there.

    Plain prose, and deliberately not a labelled line: `sound:` at the start of a
    line is read as text to DRAW and turns up on screen, which is the whole reason
    the old node's field labels had to be stripped out."""
    if not phrases:
        return ""
    if len(phrases) == 1:
        heard = phrases[0]
    else:
        heard = ", ".join(phrases[:-1]) + " and " + phrases[-1]
    if only:
        verb = "is" if len(phrases) == 1 else "are"
        return f" The only sound{'' if len(phrases) == 1 else 's'} {verb} {heard}."
    return f" It sounds like {heard}."


# A LINE THAT IS NOT COMING OUT OF ANYBODY IN THE ROOM.
#
# Reported: she appeared to be mouthing what was on the television. H3 is joint, so
# the face follows the audio branch -- and the branch has no idea a voice belongs to
# a device. A shot with 'The TV says: "..."' in it reads as a speaking shot, which
# opens the branch AND suppresses the mouth guard, so the only face in frame gets
# handed the line.
#
# The branch must stay open: the television is supposed to be heard. What has to
# change is who the voice is attributed to.
_TALKER_DEVICE = (r"(?:televisions?|tvs?|telly|screens?|radios?|speakers?|stereos?|"
                  r"tannoys?|intercoms?|phones?|telephones?|laptops?|monitors?|"
                  r"record\s+players?|pa\s+systems?|answerphones?|announcements?)")
_DEVICE_SAYS = re.compile(
    r"\b" + _TALKER_DEVICE + r"\b(?:\s+[\w,']+){0,3}?\s+"
    r"(?:says?|said|announces?|announced|blares?|blared|plays?|played|calls?|called|"
    r"reads?|talks?|talking|goes|went|crackles?|drones?|repeats?|asks?)\b", re.I)
# Somebody in the room speaking. Kept deliberately generous: if there is any chance a
# person has the line, the person keeps it. Muting a real line is far worse than a
# mouth moving, and this decides whether the mouth guard applies.
# The capitalised-word branch is a stand-in for a name, so it has to refuse the words
# that are capitalised for being at the start of a sentence -- "The TV says" was
# reading as a person called The -- and the machines themselves, which are capitalised
# as often as not ("TV", "PA").
_NOT_A_NAME = (r"(?!(?:The|A|An|It|This|That|These|Those|There|Then|Here|His|Her|Their|"
               r"Its|Our|My|Your|When|While|As|But|And|One|Now|So|No|Yes|Somebody|"
               r"Someone|Nobody|Everyone|"
               r"TV|TVs|PA|Television|Televisions|Telly|Radio|Radios|Screen|Screens|"
               r"Speaker|Speakers|Stereo|Intercom|Phone|Telephone|Laptop|Monitor)\b)")
# The verbs that give somebody a line. ONE list: this was written out three times
# -- in _PERSON_SAYS, in the sheet-name check inside speech_is_a_devices, and in
# speakers_in -- and the three had already drifted apart. The middle copy was
# missing a dozen of them, so "Mara murmured: ..." read as a person speaking in
# two places and not in the third, which decides whether a line belongs to a
# person or to a television.
_SAYS = (r"says?|said|asks?|asked|whispers?|whispered|shouts?|shouted|calls?|"
         r"called|repl(?:y|ies|ied)|answers?|answered|adds?|added|murmurs?|"
         r"murmured|mutters?|muttered|tells?|told|begs?|begged|snaps?|snapped|"
         r"breathes?|breathed|hisses|hissed")


# HOW FAR A SUBJECT REACHES TO ITS VERB -- and the asymmetry that gave a woman's
# line to the phone in her hand.
#
# _DEVICE_SAYS reaches THREE words to find its speech verb. This reached two. So
# "Mara picks up the phone and says" -- four words between the person and the verb,
# one between the phone and it -- read as the phone talking, and every beat of the
# shape "somebody handles a machine, then speaks" hit it. On a joint model that is
# the worst reading of the beat available: the branch opens because there IS a line,
# the mouth guard shuts the only face in frame because the line is judged not hers,
# and her own words play out of the object she just picked up.
#
# This file already states the rule -- if there is any chance a person has the line,
# the person keeps it -- so the person's reach is now the wider one. Bounded by the
# SENTENCE, never past it: 'Mara sits on the sofa. The TV says: "..."' is a real
# device line, and a gap that crossed the full stop would take it straight back off
# the television.
_TO_VERB = r"[^.!?\n]{0,80}?"

# A possessive is not a speaker. "Dana's phone says" is the phone talking.
_NOT_POSSESSIVE = r"(?!['\u2019]s\b)"

_PERSON_SAYS = re.compile(
    r"\b(?:he|she|they|i|we|you|" + _NOT_A_NAME + r"[A-Z][\w-]+)\b" + _NOT_POSSESSIVE
    + _TO_VERB + r"\s(?:" + _SAYS + r")\b")


def speech_is_a_devices(beat, sheet=""):
    """Is the only spoken line in this beat coming out of a machine?

    False whenever a person might have it, including when nothing attributes the
    line at all -- an unattributed quote in a beat about people is a person talking."""
    b = beat or ""
    if not has_speech(b) or not _DEVICE_SAYS.search(b):
        return False
    # WHO IS TALKING IS SETTLED OUTSIDE THE QUOTE. What a machine SAYS is not
    # evidence about who said it -- an answerphone playing "Mara, Dan called you
    # back." names two people and a speech verb, and reading the line's own
    # contents as an attribution handed the message back to whoever it mentioned.
    # Strip the spoken spans and attribute what is left. This is also what lets the
    # reach above be widened safely: the only text it can now cross is narration.
    outside = _DIALOGUE_TAG.sub(" ", _QUOTED.sub(" ", b))
    if _PERSON_SAYS.search(outside):
        return False
    # A name from the sheet with a speech verb after it, which the pattern above
    # only catches when the name happens to be capitalised in the beat.
    for n, _ in sheet_lines(sheet):
        if n and re.search(r"\b" + re.escape(n) + r"\b" + _NOT_POSSESSIVE + _TO_VERB
                           + r"\s(?:" + _SAYS + r")\b", outside, re.I):
            return False
    return True


def device_voice_clause(beat):
    """Say which machine the voice is coming out of, so no face is given it."""
    m = re.search(r"\b" + _TALKER_DEVICE + r"\b", beat or "", re.I)
    if not m:
        return ""
    # As the author spelled it. Lowercasing turned "TV" into "tv", and a set is not
    # improved by the node correcting its capitalisation.
    thing = re.sub(r"\s+", " ", m.group(0))
    # "hold still" was a freeze on everybody in the room, and a television scene is
    # mostly people watching one. What this clause has to buy is that no face in the
    # room is given the machine's line -- that is a closed mouth, not a still body.
    return (f" The voice in this shot is the {thing}'s, coming out of it across the "
            f"room, and the people listening let it play, their own mouths closed.")


# SAYING NOTHING IS NOT SAYING SOMETHING.
#
# "Mara says nothing" matched Name-then-speech-verb and credited her with a line.
# That is bad on its own and worse in context: with both people counted as
# speakers, nobody was left silent, so the lock clause -- which is only emitted
# when there IS somebody to hold -- was cancelled outright. A negation switching
# the guard off is the worst available reading of it.
_SAYS_NOTHING = re.compile(
    r"\b(?:says?|said|speaks?|spoke)\s+(?:absolutely\s+|almost\s+)?"
    r"(?:nothing|not\s+a\s+word|no\s+more|none)\b"
    r"|\b(?:does|do|did|would|will|could)\s*n[o']?t\s+(?:say|speak|answer|reply)\b"
    r"|\bnever\s+(?:says?|said|speaks?|spoke)\b"
    r"|\b(?:stays?|stayed|remains?|remained|keeps?|kept)\s+(?:quiet|silent)\b"
    r"|\bin\s+silence\b|\bwithout\s+(?:a\s+word|speaking|answering)\b", re.I)


def _in_beat_order(names, beat):
    """The names sorted by where the BEAT first mentions them.

    speakers_in walks the sheet, so it returned sheet order -- and the lock clause
    now says "Dan speaks first, then Mara", which is a claim about the beat."""
    b = beat or ""
    def at(n):
        m = re.search(r"\b" + re.escape(n) + r"\b", b, re.I)
        return m.start() if m else len(b)
    return sorted([n for n in names if n], key=at)


def speakers_in(beat, sheet=""):
    """Who this beat gives a line to. [] when it names nobody.

    A shot where one of two people speaks is a SPEAKING shot, so the mouth guard
    stood down for both -- and the listener's mouth was left as free as the
    speaker's. That is the commonest scene there is, and the lip-sync problem the
    guard exists for lands squarely on the person saying nothing."""
    b, out = beat or "", []
    # A DENIAL OF SPEECH CANCELS THE CLAUSE IT SITS IN, not the whole beat: "Dan
    # says: 'Wait.' Mara says nothing." has one speaker and one person who
    # explicitly does not speak, and both halves have to survive. So the beat is
    # split on sentence boundaries and only the denying halves are dropped.
    # The terminator is usually INSIDE the quote -- `says: "Wait here."` ends on a
    # quote mark, not a full stop -- so the closing quote counts as a boundary too.
    b = " ".join(part for part in re.split(r"(?<=[.!?\"\u201d>])\s+", b)
                 if not _SAYS_NOTHING.search(part))
    for n, _ in sheet_lines(sheet):
        if not n:
            continue
        # The gap may not contain a CONJUNCTION. "Kate approaches Sam and asks"
        # gave the line to Sam: he is nearer the verb, but "and" starts a new
        # predicate whose subject is still Kate, so the shot was told the wrong
        # person speaks -- and the mouth guard then held the actual speaker's mouth
        # shut. Filler like "then"/"quietly" is still allowed through.
        if re.search(r"\b" + re.escape(n) + r"\b"
                     r"(?:\s+(?!and\b|but\b|then\b|who\b|,\s*who\b)[\w,']+){0,2}?\s+"
                     r"(?:" + _SAYS + r")\b", b, re.I):
            out.append(n)
    # INVERTED attribution: the verb comes first. '"Sure thing," says Dan.' is the
    # commonest form in prose after the plain one, and the pattern above only ever
    # looked for name-then-verb, so it resolved nobody -- and a line nobody is
    # credited with leaves both mouths free, which is where the second voice comes
    # from.
    #
    # ONLY AFTER A CLOSING QUOTE. Bare verb-then-name is far more often the
    # ADDRESSEE than the speaker -- "she tells Dan", "she asks Dan", "she begs Dan"
    # -- and crediting the addressee is worse than crediting nobody: the shot then
    # says "Only Dan speaks; every other mouth closed", which holds the actual
    # speaker's mouth shut and moves the listener's. The voice comes out of the
    # wrong face. The quote is what marks the real inversion.
    if not out:
        for n, _ in sheet_lines(sheet):
            if not n:
                continue
            if re.search(r"[\"'”’]|</d>", b) and re.search(
                    r"(?:[\"'”’]|</d>)\s*[,.;]?\s*(?:" + _SAYS + r")\s+"
                    + re.escape(n) + r"\b", b, re.I):
                out.append(n)
    # Still nobody, and somebody is speaking. The name nearest the START of the beat
    # is the subject: "In the living room, Dan looks up. '...'" and "The door opens
    # and Dan walks in. '...'" are both Dan, and neither begins with his name -- the
    # old fallback read only the beat's FIRST WORD, so any beat that opened with
    # scenery credited nobody.
    if not out and has_speech(b):
        # Names AND declared pronouns, whichever comes FIRST. A name alone is not
        # enough: "She tells Dan to wait" holds one name and he is the ADDRESSEE,
        # so taking the only name credited the listener -- and the shot then said
        # "Only Dan speaks", holding the actual speaker's mouth shut and moving
        # his. A pronoun in subject position beats a name that comes after it.
        at = {}
        for n, ln in sheet_lines(sheet):
            if not n:
                continue
            m = re.search(r"\b" + re.escape(n) + r"\b", b)
            if m:
                at[n] = m.start()
            group = sheet_pronoun(ln)
            if not group:
                continue
            # Only where this pronoun picks out ONE person: with two women on the
            # sheet "she" resolves nobody, and guessing is how a line lands on the
            # wrong face.
            if sum(1 for _n, _l in sheet_lines(sheet)
                   if _n and sheet_pronoun(_l) == group) != 1:
                continue
            pm = re.search(r"\b(?:" + "|".join(sorted(_PRONOUN_SET[group]))
                           + r")\b", b, re.I)
            if pm and (n not in at or pm.start() < at[n]):
                at[n] = pm.start()
        if at:
            out.append(min(at, key=at.get))
    return _in_beat_order(out, beat)


# The mouth half AND the voice half. This said only that the other mouths stay
# closed, which is the PICTURE -- and on a joint model the face follows the audio:
# a second voice in the stream puts a second mouth in motion whatever the prose
# says about jaws. So the shot has to be told how many voices there are, not just
# how many mouths, and the prose is what conditions the audio branch.
#
# Positively phrased: "one voice" names what IS there. "Nobody else speaks" asks
# the model to render an absence, and at cfg 1 there is no negative prompt to carry
# it. {who} is named ONCE -- naming a person twice in one shot is what put a second
# copy of them in frame.
# SAID ONCE, and what fills the rest. A line is a second or two; the shot is five
# to ten, and the audio branch is open for all of it. Told only that there is one
# voice, the model still has seconds of open branch to fill on either side of the
# line -- and the only thing it knows is happening in this shot is somebody
# talking, so it invents more talking to occupy the lead-in. Reported exactly that
# way: babble before the dialogue starts.
#
# Two statements fix the gap, and both name something that IS there rather than an
# absence, because at cfg 1 there is no negative prompt: the line is said ONCE, and
# what occupies the time around it is ROOM TONE. A branch with a bed to lay down
# does not need to invent a voice to fill the space.
# The specific acoustic belongs to the sound clause, which already says it where
# the scene names a space. Here it is the generic bed, so the sentence reads the
# same whatever room this is.
# SHORT. Every word here is speech vocabulary -- speaks, voice, line, said -- and
# on a joint model the prose conditions the AUDIO branch as much as the picture.
# A longer version of this clause ("one voice in the shot, the line said once, with
# room tone either side of it") was added to stop a listener babbling and was
# reported as causing it: more speech words on a shot is more reason for the branch
# to make speech. Say who has the line and hold the other mouths; nothing else.
# WHICH LANGUAGE the line is in. H3 is joint and multilingual: the prose conditions
# the audio branch, and a branch told a line is spoken but never told in what will
# pick one. Babble that is not babble at all -- a real language, fluently delivered,
# and not the one the script is written in -- sounds exactly like gibberish to
# somebody expecting English.
#
# Positively phrased, and stated once: at cfg 1 there is no negative prompt, so
# "not in another language" would name the other language. Naming the wanted one is
# the whole mechanism.
# The FALLBACK, not the rule. This was the rule -- the clause named English and
# only English -- so a script written in any other language was told its own line
# is spoken in English, and the delivery fought the words. Users asked for that
# restriction to come out.
#
# Naming NOTHING is not the way out: unnamed is where the branch picks a language
# on its own, which is the "sounds like gibberish" report this clause answers. So
# the language is read off the line, and this is only what stands in when the line
# is too short to tell.
SPOKEN_LANGUAGE = "English"
# THE LANGUAGE, AND NOT THE FACT THAT IT IS SPOKEN. ea58d3c took the speech
# vocabulary back out of the speech guard on the evidence of a render, and wrote
# the finding down: "Every word I added is speech vocabulary -- speaks, voice,
# line, said -- and on a joint model the prose conditions the AUDIO branch as much
# as the picture. A clause meant to suppress a second voice was itself priming
# speech." What it kept is "who has the line and holds the other mouths, which is
# what it said before this session and what was not babbling".
#
# 6943916 put `line` and `spoken` back, on EVERY speaking shot, twelve and a half
# hours later (2026-09-05 23:16 -> 2026-09-06 11:53). Nothing was wrong with its
# purpose -- a branch told a line is spoken but never told in WHAT picks a language,
# and that was a real report -- but it carried two of the four words the render had
# just convicted, into the one clause that lands on exactly the shots with a voice
# in them.
#
# The dropped half was redundant anyway, which is why this costs nothing. <d> and
# </d> (151669/151670) are real tokens the model was trained with, and they are
# what marks a span as spoken; the language is the one thing they cannot carry,
# and it is all this sentence needs to say. Verified across English, Spanish,
# French, German, Russian and Japanese.
LANGUAGE_HOLD = " The language is {lang}."

# Characters that are not plain Latin text. A stray CJK, Cyrillic or Arabic glyph in
# a prompt is a strong signal to a multilingual model about what language to speak,
# and one pasted quotation mark is easy to miss by eye. Reported rather than
# stripped: the node passes the author's words through, and silently editing them is
# the thing it does not do.
#
# Latin-1 and Latin Extended cover the accented letters, and U+0300-U+036F the
# COMBINING marks -- "cafe" plus a combining acute is the decomposed spelling of
# the same word, and flagging it would report every accented character typed on a
# Mac. Curly quotes and dashes are ordinary punctuation, not a language signal.
_NON_LATIN = re.compile(
    r"[^\x00-\x7F\u00C0-\u024F\u0300-\u036F"
    r"\u2018\u2019\u201C\u201D\u2013\u2014\u2026]")


# Things inside a line that have no single spoken form: a number, a time, a date,
# an abbreviation, an acronym, a symbol. The model reads the line as text and
# picks one -- "7:30" as "seven thirty" or "seven three zero", "Dr." as "doctor"
# or "dee arr" -- and the picking is what mispronounced dialogue is.
#
# The abbreviations are a LIST, not a shape. "[A-Z][a-z]{0,3}\." also matches the
# end of any short sentence, so "No." would have been reported as an abbreviation
# in every script that has somebody saying no.
_HARD_TO_SAY = re.compile(
    r"\b\d[\d:.,/\-]*\d\b|\b\d\b"
    r"|\b(?:Mr|Mrs|Ms|Dr|Prof|Sgt|Lt|Capt|Rev|Hon|St|Ave|Rd|Blvd|Jr|Sr|"
    r"vs|etc|approx|dept|Inc|Ltd|Co)\."
    r"|[&%$#@+=]", re.I)


def non_latin_in(text):
    """The distinct non-Latin characters in this text, in order. [] when clean."""
    out = []
    for ch in str(text or ""):
        if _NON_LATIN.match(ch) and ch not in out:
            out.append(ch)
    return out


# A LINE THAT ORDERS AN ACTION. "Dana says to McKenna: \"Take off your shorts and
# lie down on the change table.\"" -- the node no longer STAGES that (the readers
# refuse quoted speech), but the words are still in the shot, because beats are
# passed through verbatim and that is the oldest promise this file makes. A video
# model does not distinguish a quoted instruction from a stage direction: it
# renders what the words describe, and the action arrives a beat early.
#
# The words cannot be removed. What can be added is something for the LISTENER to
# be doing, so the shot has an answer for them other than the instruction --
# positively phrased, because at cfg 1 "does not do it yet" names the thing.
_ORDERED = re.compile(
    r"\b(?:take|takes|taking|pull|pulls|remove|removes|undo|undoes|unfasten|"
    r"unbuckle|unzip|slip|slips|step|steps|get|gets|lie|lies|lay|lays|sit|sits|"
    r"kneel|kneels|stand|stands|turn|turns|come|comes|go|goes|put|puts|hold|"
    r"holds|open|opens|close|closes)\b", re.I)


def told_to_act(beat, speakers, described):
    """Who is being TOLD to do something in this beat's dialogue. [] when nobody.

    Only where the quoted line contains an action verb, and only for people the
    shot describes who are not the one speaking -- the listener is the one whose
    body the instruction is about, and the one the model will move early."""
    b = str(beat or "")
    if not b:
        return []
    said = " ".join(m.group(0) for m in _QUOTED.finditer(b))
    if not said or not _ORDERED.search(said):
        return []
    talking = {n for n in (speakers or []) if n}
    return [n for n in (described or []) if n and n not in talking]


def told_hold(listeners):
    """Give the listener something to be doing while the line is said."""
    who = [n for n in (listeners or []) if n]
    if not who:
        return ""
    # ONE naming each. A described person is a person the model draws, and naming
    # somebody twice in one shot is what put a second copy of them in frame.
    if len(who) == 1:
    # NOT "listens, still". This clause exists to give the listener something to BE
    # DOING -- its own docstring says so -- and what it gave them was an instruction
    # to be motionless, set off in commas so it could only be read as the adjective.
    # It lands on the reaction shot, which is where acting happens. `listens` is the
    # activity; the comma was doing the opposite of the clause's whole purpose.
        return f" {who[0]} listens, wearing what the sheet already lists."
    said = ", ".join(who[:-1]) + " and " + who[-1]
    return f" {said} listen, wearing what the sheet already lists."


# The tail both voice guards end on, defined once so they cannot drift apart.
MOUTH_HOLD_REST = "every other mouth in the shot stays closed, those expressions moving"
MOUTH_HOLD_OTHERS = " Only {who} speaks; " + MOUTH_HOLD_REST + "."


# A VOCAL BELONGS TO SOMEBODY.
#
# Reported: "her whimpering is opening up his ability to babble. Dialogue is not
# being localized to the characters."
#
# _voiced is exertion_in(body) -- a SHOT-LEVEL flag with no owner -- and both mouth
# guards stand down on it, for everybody in the shot. The comment says exactly why
# they stand down: "straining is vocal and that mouth should be open." THAT mouth.
# Not every mouth. So "McKenna sobs while Dan watches" left Dan's mouth as free as
# hers, on a shot whose audio branch her sob had just opened -- which is precisely
# the machinery the speech guard exists to stop, switched off by the one kind of
# beat that opens the branch without giving anybody words.
#
# Measured on a six-shot scene of a woman gagged in a van: not one shot carried any
# mouth guard at all.
#
# Attribution table of its own, NOT _VOCAL_FROM. That one feeds the sound clause and
# is the six vocals the node will name as a sound; this is about whose face moves,
# which is a wider list and must not change what the shot is heard as.
_VOCAL_SOURCE = (
    (r"whimper(?:s|ing|ed)?", "whimpering"), (r"sob(?:s|bing|bed)?", "sobbing"),
    (r"moan(?:s|ing|ed)?", "moaning"),       (r"groan(?:s|ing|ed)?", "groaning"),
    (r"scream(?:s|ing|ed)?", "screaming"),   (r"whin(?:e|es|ing|ed)", "whining"),
    (r"gasp(?:s|ing|ed)?", "gasping"),       (r"pant(?:s|ing|ed)?", "panting"),
    (r"cr(?:y|ies|ying|ied)", "crying"),     (r"sigh(?:s|ing|ed)?", "sighing"),
    (r"shriek(?:s|ing|ed)?", "shrieking"),   (r"yelp(?:s|ing|ed)?", "yelping"),
    (r"grunt(?:s|ing|ed)?", "grunting"),     (r"weep(?:s|ing)?", "weeping"),
    (r"wail(?:s|ing|ed)?", "wailing"),       (r"laugh(?:s|ing|ed)?", "laughing"),
)


def vocal_sources_in(beat, sheet=""):
    """Who this beat says is making a non-speech vocal, and what it is.

    [(name, phrase)], in sheet order. Same shape as speakers_in, including its
    conjunction guard: "Dan holds the door and McKenna sobs" must not credit Dan,
    because `and` starts a new predicate with its own subject -- and crediting the
    wrong person here is worse than crediting nobody, since the shot would then hold
    the mouth of whoever is actually making the noise."""
    b, out = beat or "", []
    for n, _ in sheet_lines(sheet):
        if not n:
            continue
        for pat, phrase in _VOCAL_SOURCE:
            if re.search(r"\b" + re.escape(n) + r"\b"
                         r"(?:\s+(?!and\b|but\b|then\b|who\b|,\s*who\b)[\w,']+){0,2}?\s+"
                         r"(?:" + pat + r")\b", b, re.I):
                out.append((n, phrase))
                break
            # A COMPOUND SUBJECT IS TWO SOURCES, NOT ONE.
            #
            # "Mia and Tess laugh over breakfast" credited only Tess, so the shot said
            # "the laughing is Tess's; every other mouth in the shot stays closed" --
            # holding Mia's mouth shut in a beat that says she laughs. Reported as the
            # acting not matching the scene.
            #
            # The conjunction guard above is RIGHT about "Dan holds the door and
            # McKenna sobs", where `and` starts a new predicate with its own subject.
            # What it cannot tell apart is two names sharing ONE verb, and the
            # difference is whether a verb intervenes: here nothing stands between the
            # names and the verb they share. Same reading posture_in already uses for
            # "Kate and Sam sit down", which seats both.
            if re.search(r"\b" + re.escape(n) + r"\b(?:\s*,\s*[\w'\u2019-]+)*"
                         r"\s+and\s+[\w'\u2019-]+\s+"
                         r"(?:" + pat + r")\b", b, re.I):
                out.append((n, phrase))
                break
    return out


def _joined(names):
    """'Dan', 'Dan and Sam', 'Dan, Sam and Mara'."""
    ns = [n for n in (names or []) if n]
    if len(ns) < 2:
        return ns[0] if ns else ""
    return ", ".join(ns[:-1]) + " and " + ns[-1]


def voice_sources(talkers, vocal, vocalisers, silent):
    """Say whose voice is whose, and close the mouths that are neither.

    Two jobs, and the second is the reported one. Closing the rest stops the
    listener babbling on a branch somebody else's sob opened. NAMING THE SOURCES
    stops the model swapping them -- two voices in one shot with nothing saying
    which is which is a shot where he can be given her whimper and she his line.

    So the sentence is emitted for two DIFFERENT sources even when nobody is left to
    hold: with one source and nobody silent there is nothing to disambiguate and
    nothing to close, and the shot is left alone."""
    parts = []
    if len(talkers or []) == 1:
        parts.append(f"only {talkers[0]} speaks")
    elif talkers:
        # TWO LINES, TWO MOUTHS, AND NOTHING SAYING WHICH IS WHICH. A beat with two
        # speakers left the shot free to put either line on either face. Said in the
        # order the BEAT gives them, which is the only ordering there is.
        parts.append(f"{talkers[0]} speaks first, then "
                     + ", then ".join(talkers[1:]))
    if vocalisers and vocal:
        parts.append(f"the {vocal} is {_joined(vocalisers)}'s")
    # Emitted for two DIFFERENT sources even when nobody is left to hold, and for
    # two speakers for the same reason: the ordering is the whole point of it.
    if not parts or (len(parts) == 1 and not silent and len(talkers or []) < 2):
        return ""
    if silent:
        parts.append(MOUTH_HOLD_REST)
    said = "; ".join(parts)
    return f" {said[0].upper()}{said[1:]}."

# ...and when the line has no name on it. Two people, one line, nobody named: the
# speaker cannot be identified, so neither mouth could be held and BOTH were free
# to move -- which on a joint model is two voices in the stream and the second one
# is the babble. Saying how many voices there are does not require knowing whose.
# A JAW THAT "STAYS STILL" IS A FROZEN FACE. The guarantee here is one voice, and a
# closed mouth delivers it -- lip-sync needs lips to part. "Stays still" asked for
# something stronger than the guarantee needs and put it on the listener, which is
# the face the audience is watching. Same wording as MOUTH_HOLD_OTHERS now, because
# they are the same situation with and without a name to put on it.
ONE_VOICE = (" Only the person speaking has their mouth moving; every other mouth "
             "in the shot stays closed, those expressions moving.")


# H3'S OWN DIALOGUE MARKER. <d> and </d> are special tokens the model was trained
# with -- comfy/text_encoders/minimax.py registers them as 151669 and 151670 -- and
# they mark a span as SPEECH rather than as scene description.
#
# This file warned about them for a long time and never used them, which left the
# real problem unfixed: a quoted line is just words in the prompt, and a video
# model renders what the words describe. "Take off your shorts and lie down on the
# change table" is an imperative sentence, and it was performed a beat before
# anybody said it. Refusing to STAGE it -- which every reader here now does -- does
# nothing about the model reading it.
#
# So the quotes become the marker the model actually understands. Every word the
# author wrote is kept, in order; only the quotation marks are exchanged for the
# tokens that say "this is spoken". That is less of an edit than the sentences this
# node already appends, and it is the difference between a line being heard and a
# line being acted out.
_PLAIN_QUOTED = re.compile(r"[\"“]([^\"“”]{1,400}?)[\"”]")


def mark_dialogue(beat):
    """Wrap plainly-quoted speech in H3's <d>...</d>. Unchanged when there is none.

    Left alone where the author has already marked it, and where a quote is not
    speech at all. A LINE ends in terminal punctuation and a scare quote does not:
    "Wait." is one word and is speech, a "vintage" coat is emphasis. Counting words
    got both of those backwards."""
    b = str(beat or "")
    if not b or "<d>" in b:
        return b

    def _wrap(m):
        said = m.group(1).strip()
        if not said:
            return m.group(0)
        # A LINE ends in terminal punctuation; a scare quote does not. "Wait." is
        # one word and is speech; a "vintage" coat is two characters of emphasis.
        # Word count alone got that backwards both ways.
        if said[-1] in ".!?":
            return "<d>" + said + "</d>"
        # No terminal punctuation: it needs BOTH a speech cue and more than one
        # word. A cue alone is not enough -- _SAYS contains "called", so 'he
        # called it a "problem"' read as an introduction to a line. A determiner
        # and one word is a noun, whatever verb came before it.
        if len(said.split()) < 2:
            return m.group(0)
        before = b[max(0, m.start() - 40):m.start()]
        if re.search(r"(?:" + _SAYS + r")\b[^.]{0,12}$|[:,]\s*$", before, re.I):
            return "<d>" + said + "</d>"
        # ...or the cue comes AFTER it. '"Come here," Dana says.' is how half of
        # written dialogue is punctuated, and only the text BEFORE the quote was
        # ever consulted -- so that form was never marked at all, and an unmarked
        # line is a line the audio branch was never told is spoken. Quotation
        # marks say nothing to the model on their own.
        after = b[m.end():m.end() + 40]
        if re.match(r"[\s,]*(?:[A-Za-z][\w'’-]*\s+){0,2}?(?:" + _SAYS + r")\b",
                    after, re.I):
            return "<d>" + said + "</d>"
        return m.group(0)

    return _PLAIN_QUOTED.sub(_wrap, b)


def has_speech(beat):
    """Does this beat contain a scripted line?

    Either H3's own <d>...</d> marker or plain double quotes. Only checking quotes
    meant a beat written the way the model expects was treated as silent, and its
    audio muted."""
    text = beat or ""
    return bool(_DIALOGUE_TAG.search(text) or _QUOTED.search(text))


_PICTURE_TAG = re.compile(r"<\s*picture[\s_\-]*(\d+)\s*>", re.I)


def picture_tags(text):
    return sorted({int(m.group(1)) for m in _PICTURE_TAG.finditer(text or "")})


def resolve_tags(text, ref_list):
    """(text with its tags renumbered, the images that shot carries, dropped slots).

    A <Picture N> tag is the BINDING between an image and the subject the prompt
    describes, and it belongs IN the prompt. comfy_extras/nodes_minimax_h3.py says so
    outright: "Ordinals are 1-based per type, so the prompt refers to them as
    <Picture i>", and the node's own description is "Use the same tags when
    prompting."

    The rule that follows governs every reference decision in this file:

        a picture the prompt REFERS TO is that subject;
        a picture the prompt does NOT refer to is ANOTHER subject.

    So taking a tag out of the text does not remove a spare person, it CREATES one --
    the image arrives labelled and unclaimed, and the model renders it as somebody
    else. It is also why the handoff frame must not enter this channel at all: no
    wording refers to it, so it would arrive as a stranger.

    comfy/text_encoders/minimax.py writes the "<Picture N>: " label itself, numbering
    by the order it receives the images -- so a shot that uses only <Picture 2>
    receives that image labelled <Picture 1>, and text still saying <Picture 2> points
    at nothing. The tags are renumbered per shot to match what the shot actually
    carries: slot 2 alone becomes <Picture 1>; slots 2 and 4 become <Picture 1> and
    <Picture 2>.

    A tag naming a slot with no image connected refers to nothing at all, so it is
    removed from the text rather than left for the encoder to puzzle over."""
    wanted = picture_tags(text)
    live = [n for n in wanted if 1 <= n <= len(ref_list or [])]
    dropped = [n for n in wanted if n not in live]
    renum = {old: new for new, old in enumerate(live, 1)}

    def sub(m):
        n = int(m.group(1))
        return f"<Picture {renum[n]}>" if n in renum else ""

    out = _PICTURE_TAG.sub(sub, text or "")
    out = re.sub(r"\s+([,.;:])", r"\1", out)      # " ," left by a removed tag
    out = re.sub(r"([:,;])\s*,", r"\1", out)      # ",," where the tag was the only item
    out = re.sub(r"\s{2,}", " ", out)
    return out.strip(), [ref_list[n - 1] for n in live], dropped

def check_audio_vae_loaded(audio_vae):
    """Catch an UNCONVERTED audio VAE checkpoint.

    comfy/ldm/minimax/audio_vae.py loads a checkpoint whose weight-norm has been
    folded into plain "*.weight" tensors. Feed it the raw upstream file (172
    weight_g/weight_v pairs, no latents_mean/latents_std) and load_state_dict
    reports the misses as a WARNING, not an error: every weight-normed conv keeps
    its random init and the two normalization buffers stay torch.empty(), i.e.
    uninitialized memory. Decoding then multiplies the latents by garbage and the
    audio comes out as noise -- with nothing in the log at render time to say why.

    latents_std is the cheapest tell: it is a real per-channel scale, so a
    non-finite or absurd value means the buffer was never filled."""
    m = getattr(audio_vae, "first_stage_model", None)
    mean, std = getattr(m, "latents_mean", None), getattr(m, "latents_std", None)
    if mean is None or std is None:
        return
    try:
        bad = (not torch.isfinite(mean).all() or not torch.isfinite(std).all()
               or float(std.min()) <= 0.0 or float(std.max()) > 1e3
               or float(mean.abs().max()) > 1e3)
    except Exception:
        return                       # never block a render on a failed introspection
    if bad:
        raise RuntimeError(
            "the audio VAE loaded but its weights are NOT initialized -- this is the raw "
            "upstream MiniMax-H3 audio checkpoint (weight_g/weight_v weight-norm pairs, no "
            "latents_mean/latents_std). ComfyUI's loader needs the CONVERTED file, with "
            "weight-norm folded into plain '*.weight' tensors. Look for the 'Missing VAE keys' "
            "warning in the log when the VAE loaded. Download the repackaged H3 audio VAE from "
            "the Comfy-Org release; rendering with this one produces noise, not speech.")


def shot_latent_cells(w, h, frames, fps):
    """Latent cells in one shot: what sampling VRAM actually scales with.

    Not a byte figure -- the constant depends on the quantisation path -- but it is
    exactly linear in both shot length and area, so ratios between settings are
    right even though the absolute number is not a prediction."""
    _, lt, _ = temporal_shape(frames, fps)
    return max(1, int(lt)) * max(1, w // 16) * max(1, h // 16)


def model_fingerprint(model):
    """A cheap, stable identity for the loaded DiT: (quant format, layer count,
    weight bytes, class name). Changes whenever the checkpoint changes -- a
    different quant, a pruned-vs-full build, or a different model entirely -- while
    staying identical across shots of the same run. Deliberately avoids hashing
    weights, which would cost more than the flush it guards."""
    try:
        dm = getattr(getattr(model, "model", None), "diffusion_model", None)
        fmts, n = {}, 0
        if dm is not None and hasattr(dm, "modules"):
            for mod in dm.modules():
                n += 1
                f = getattr(mod, "quant_format", None)
                if f:
                    fmts[f] = fmts.get(f, 0) + 1
        top = max(fmts.items(), key=lambda kv: kv[1])[0] if fmts else "none"
        size = 0
        try:
            size = int(model.model_size())
        except Exception:
            pass
        cls = type(dm).__name__ if dm is not None else "unknown"
        return (top, n, size, cls)
    except Exception:
        return None


def check_vae_wiring(vae, audio_vae):
    """Catch the commonest miswire -- the video VAE dropped into BOTH VAE inputs.
    Without this the run samples a whole shot, decodes the video fine, then dies
    deep inside comfy/sd.py with 'IndexError: tuple index out of range' when the
    video memory estimator indexes shape[4] of the 4-D audio latent."""
    if _is_audio_vae(audio_vae) is False:
        raise RuntimeError(
            "audio_vae is a video/image VAE, not the H3 audio VAE. Load the audio "
            "autoencoder (the DAC/BigVGAN one shipped with MiniMax-H3, e.g. "
            "minimax_h3_audio_vae.safetensors) in its own VAELoader and wire that "
            "into 'audio_vae'; the video VAE belongs on 'vae' only.")
    check_audio_vae_loaded(audio_vae)
    if _is_audio_vae(vae) is True:
        raise RuntimeError(
            "vae is the H3 audio VAE -- the video and audio VAE inputs are swapped. "
            "Wire the video VAE into 'vae' and the audio VAE into 'audio_vae'.")


def flush_for_model_change(model):
    """Detect a checkpoint swap since the last run and, if one happened, hard-flush
    GPU state before doing anything else.

    Why this matters: ComfyUI keeps previously-loaded models in current_loaded_models
    and only evicts reactively. Swapping checkpoints mid-session (e.g. NVFP4 -> FP8 ->
    MXFP8 while comparing quality) leaves the OLD DiT resident alongside the new one,
    plus any hooks/injections a previous LoRA installed and stale cached allocator
    blocks sized for the old model's layers. The result is a card that is already
    half full before the first shot samples -- which looks exactly like the node
    over-spilling, when in fact the budget was computed against memory the previous
    checkpoint never released.

    Returns a note for `info` when a change was detected (empty string otherwise)."""
    fp = model_fingerprint(model)
    prev = _LAST_MODEL_FP.get("fp")
    _LAST_MODEL_FP["fp"] = fp
    if prev is None or fp is None or prev == fp:
        return ""
    try:
        mm.unload_all_models()          # drop every resident model, not just the cache
    except Exception:
        pass
    # Never let a cleanup failure abort the run: the flush is best-effort hygiene,
    # and a partially-flushed card is still better than raising here.
    try:
        _deep_cleanup()
    except Exception:
        pass
    old_fmt, _n, old_sz, _c = prev
    new_fmt = fp[0]
    return (f"model changed since last run ({old_fmt} ~{old_sz / GB:.1f}GB -> {new_fmt} "
            f"~{fp[2] / GB:.1f}GB): flushed all resident models and VRAM caches")


_POSTURE = re.compile(
    r"\b(?:lying|laying|lies|lays|kneel(?:s|ing)?|knelt|sit(?:s|ting)?|sat|"
    r"crouch(?:es|ing|ed)?|curled|sprawled|slumped|face[- ]?down|face[- ]?up|"
    r"on (?:her|his|their) (?:side|back|front|knees|stomach|belly))\b", re.I)


def posture_note(scene, has_first_frame):
    """Warn when shot 1's opening pose is left to the text alone.

    Shot 1 is the only shot with no keyframe -- there is no previous frame to
    continue from -- so its opening pose comes from the text and from nothing else.
    A posture sentence sitting at the end of a long sheet is the least-weighted
    thing the model reads, and text cannot outrank a picture anyway. This does not
    reorder anything: the node sends what you wrote, in the order you wrote it."""
    if has_first_frame or not (scene or "").strip():
        return ""
    sents = [s for s in re.split(r"(?<=[.!?])\s+", scene.strip()) if s.strip()]
    where = [i for i, s in enumerate(sents) if _POSTURE.search(s)]
    if not where:
        return ""
    return (f"shot 1 has no keyframe, so its opening pose comes from the text alone -- "
            f"and the sentence describing the pose is {where[0] + 1} of {len(sents)}. "
            f"first_frame pins it, but it pins the WHOLE opening frame, so it has to be "
            f"a composed frame of the shot you want: a head-and-shoulders picture wired "
            f"there makes the first frame a head-and-shoulders picture. An identity "
            f"portrait belongs on ref_image_1 instead")


def reference_note(n_refs, aug, has_first_frame):
    """What a near-clean reference actually asks the model to do.

    ONE aug covers every visual conditioning row. At H3's default of 0.999 a
    reference is handed over essentially noise-free, and a noise-free image is an
    invitation to REPRODUCE it -- its framing and background along with its subject.
    That is a matter of DEGREE, not a format error, and this is the dial for it: the
    symptom is a shot that opens on the reference and moves off it, and the answer is
    to lower the aug until it informs the face without being copied.

    Shot 1 is where it shows most, because it has no keyframe pinning its opening
    frame -- the reference is the only picture it has, so there is nothing competing
    with the invitation to reproduce."""
    if not n_refs or aug is None:
        return ""
    if float(aug) >= KEYFRAME_SAFE_AUG:
        note = (f"{n_refs} reference image(s) at ref_noise_aug {float(aug):.3f}, which is "
                f"near-clean -- that asks the model to REPRODUCE them, framing and "
                f"background included, in the opening frames. Lower it to say "
                f"approximate: try 0.95, then 0.90. Below 0.99 the handoff stops being "
                f"a keyframe and rides as an extra reference, so continuity weakens as "
                f"identity strengthens")
    else:
        note = (f"{n_refs} reference image(s) at ref_noise_aug {float(aug):.3f} -- "
                f"softened, so they inform the face rather than being copied. Below "
                f"0.99 one aug would also soften the keyframe, so the handoff rides as "
                f"an extra reference instead of anchoring: weaker continuity, nothing "
                f"pretending to anchor while carrying noise")
    if not has_first_frame:
        note += (". Shot 1 has no keyframe, so the reference is its only picture and "
                 "nothing competes with reproducing it -- that shot is where a "
                 "near-clean reference shows up as the opening frame, AND IT DOES NOT "
                 "STAY THERE: every later shot opens on the previous shot's last "
                 "frame, so whatever composition shot 1 settles on is handed down the "
                 "whole chain. A portrait reproduced at shot 1 is therefore a portrait "
                 "framing for the film, which is what 'the camera is fixated on her' "
                 "is. Wire a wide establishing frame into first_frame and shot 1 is "
                 "pinned to that composition instead -- it is the one input that "
                 "outranks a reference, because it IS frame one")
    return note


def frame_detail(img):
    """(detail, contrast) for one frame in 0..1, HWC.

    Detail is mean absolute neighbour difference -- a cheap stand-in for how much
    fine structure survives. Contrast is the luminance spread. Neither is an
    absolute measure of anything; what matters is the TREND across shots.

    Every shot boundary decodes a latent to pixels, takes the last frame and
    re-encodes it as the next shot's keyframe. That round trip is lossy, and the
    frame it runs on is the model's own output, so shot 11 is sampled from a
    picture that has been through ten decode/encode cycles. Softening that
    compounds is invisible shot to shot and obvious end to end -- so measure it."""
    # This is diagnostic only. Sampling at most roughly 256 points per axis avoids
    # allocating a full-resolution float32 copy of every shot's final frame.
    step = max(1, max(int(img.shape[0]), int(img.shape[1])) // 256)
    x = img[::step, ::step].float()
    if x.dim() == 3 and x.shape[-1] >= 3:
        x = x[..., :3].mean(dim=-1)
    elif x.dim() == 3:
        x = x[..., 0]
    if x.dim() != 2 or x.shape[0] < 2 or x.shape[1] < 2:
        return 0.0, 0.0
    gx = (x[:, 1:] - x[:, :-1]).abs().mean()
    gy = (x[1:, :] - x[:-1, :]).abs().mean()
    return float((gx + gy) * 0.5), float(x.std())


def levels_report(levels, shots):
    """What hold_levels measured, and what it did about it.

    Worth printing even when it corrected nothing: the measurement is the evidence that
    the chain is or is not cooking, and a run that measured a drift too small to act on
    is a different thing from a run that never looked."""
    if levels is None:
        return ""
    g, o = levels.estimate()
    if g is None:
        return ""
    pct = "/".join(f"{(float(torch.exp(v)) - 1.0) * 100.0:+.1f}%" for v in g)
    lvl = "/".join(f"{float(v):+.4f}" for v in o)
    line = (f"hold_levels: measured the chain drifting {pct} of contrast and {lvl} of level "
            f"per boundary, per R/G/B channel, from {len(levels._bg)} boundary(ies)")
    n = len(levels.applied)
    if not n:
        line += (" -- below the 8-bit floor a handoff is quantised to, so nothing was "
                 "applied rather than claiming a correction that would be erased")
    else:
        last = levels.applied[-1][0]
        line += (f", and took it back out of {n} handoff(s); the last gain applied was "
                 f"{'/'.join(f'{float(v):.3f}' for v in last)}. The contrast line above is "
                 f"measured on the corrected frames, so it is the residual, not the defect")
    return line


def detail_report(per_shot):
    """Two lines: whether the chain is softening, and whether it is COOKING.

    per_shot is [(detail, contrast), ...] measured on each shot's last frame.

    Contrast used to be measured here and thrown away, which was the worst possible
    arrangement: the surviving metric RISES with burn-in -- expanding contrast creates
    neighbour differences -- so a chain visibly cooking printed "UP n%, so the chain is
    not softening" and read as reassurance. The reported symptom was being measured on
    exactly the right frame and never shown. Both trends are reported now, and the
    detail line no longer pronounces on a rise it cannot explain by itself."""
    ds = [d for d, _ in per_shot if d > 0]
    cs = [c for _, c in per_shot if c > 0]
    if len(ds) < 2:
        return ""
    out = []
    drop = (ds[0] - ds[-1]) / ds[0] * 100.0 if ds[0] else 0.0
    line = "detail per shot (last frame): " + " ".join(f"{d:.4f}" for d, _ in per_shot)
    if drop >= 10.0:
        line += (f" -- DOWN {drop:.0f}% from shot 1 to shot {len(ds)}. Each boundary "
                 f"decodes a shot, takes its LAST frame and re-encodes it as the next "
                 f"shot's keyframe, so the loss of one round trip is carried into the "
                 f"next and compounds. Break the chain to stop it accumulating: "
                 f"restart_after_removal starts a shot from the text instead of the "
                 f"previous frame, at the cost of a visible cut there")
    elif drop <= -10.0:
        line += (f" -- UP {-drop:.0f}%. Read the contrast line before taking that as good "
                 f"news: expanding contrast raises this number too")
    else:
        line += f" -- flat within {abs(drop):.0f}%"
    out.append(line)
    if len(cs) >= 2:
        rise = (cs[-1] - cs[0]) / cs[0] * 100.0 if cs[0] else 0.0
        cl = "contrast per shot (last frame): " + " ".join(f"{c:.4f}" for _, c in per_shot)
        if rise >= 10.0:
            cl += (f" -- UP {rise:.0f}% from shot 1 to shot {len(cs)}, which is the chain "
                   f"COOKING: every shot is sampled from the previous shot's last frame, "
                   f"the model reproduces it with a little more contrast, and the VAE "
                   f"clamps the result to 0..1 -- so the headroom each pass spends is "
                   f"never given back, and it shows as crushed blacks and blown "
                   f"highlights rather than merely as more contrast. hold_levels takes "
                   f"the per-boundary part of it back out")
        elif rise <= -10.0:
            cl += f" -- DOWN {-rise:.0f}%, so the chain is flattening rather than cooking"
        else:
            cl += f" -- flat within {abs(rise):.0f}%"
        out.append(cl)
    return " | ".join(out)


def _find_h3_sampling_node():
    """Locate the H3 sigma-shift node under ANY registered name. It was renamed
    to 'ModelSamplingMiniMaxH3' in a later patch (kijai PR #15243); older 0.30.x
    builds register it under a different id, so exact-key lookup misses it. Try
    the known names, then fuzzy-scan all node mappings for the H3 model-sampling
    node. Returns (class, key) or (None, None)."""
    maps = getattr(nodes, "NODE_CLASS_MAPPINGS", {}) or {}
    for key in ("ModelSamplingMiniMaxH3", "ModelSamplingMinimaxH3", "ModelSamplingMinimax", "ModelSamplingH3"):
        if key in maps:
            return maps[key], key
    for k, v in maps.items():
        kl = k.lower()
        if "sampl" in kl and (("minimax" in kl and "h3" in kl) or ("h3" in kl and "shift" in kl)):
            return v, k
    for k, v in maps.items():
        kl = k.lower()
        if ("minimax" in kl or "h3" in kl) and ("shift" in kl or "sampling" in kl):
            return v, k
    return None, None


def _direct_model_sampling(model, shift_video, shift_audio):
    """Fallback that sets the shift on the model's own model_sampling object
    without any node -- version-tolerant and V3-proof, since it uses model-level
    APIs (get_model_object / set_parameters / add_object_patch) rather than
    calling a node. Copies the sampling object so the base model isn't mutated,
    and applies audio_shift only if the installed set_parameters accepts it."""
    import inspect, copy
    m = model.clone()
    # deepcopy, not copy: model_sampling is an nn.Module, and a SHALLOW copy shares
    # its `_buffers` dict with the original. set_parameters() re-registers `sigmas`
    # into that shared dict, so a shallow copy silently rewrites the BASE model's
    # sigma table -- the very thing this copy exists to prevent. Our own run reads
    # the patched object either way, but ComfyUI caches the model across queue
    # runs, so the damage outlives this execution and reaches anything else holding
    # that model. The buffer is ~1000 floats; the deepcopy is free.
    ms = copy.deepcopy(m.get_model_object("model_sampling"))
    sig = inspect.signature(ms.set_parameters)
    kwargs = {}
    if "shift" in sig.parameters:
        kwargs["shift"] = float(shift_video)
    if "audio_shift" in sig.parameters:
        # NOTE: on ComfyUI 0.31 the audio latent is carried on the video schedule
        # scaled by audio_scale = shift_video / shift_audio (12/3 = 4.0), applied in
        # process_latent_in and undone in process_latent_out. Forcing that ratio to
        # 1.0 (audio_shift == shift_video) as a "legacy 0.30" emulation produces
        # SILENT output -- the model needs the scaling -- so it is not offered.
        kwargs["audio_shift"] = float(shift_audio)
    if not kwargs:
        raise RuntimeError("set_parameters takes no shift")
    ms.set_parameters(**kwargs)
    m.add_object_patch("model_sampling", ms)
    return m


# WHERE THE AUDIO BRANCH LANDS FROM, AND HOW TO SHORTEN THE FALL.
#
# Reported over and over as babble at the OPENING of a beat, and none of the prose
# in this file could touch it. Every clause here changes what the branch is TOLD.
# None of them changes how much noise it still has to clear when it stops.
#
# Computed from ComfyUI's own scheduler code at shift 12/3 -- the last AUDIO sigma
# before zero, which the final step has to clear in a single jump:
#
#     scheduler     5 steps   8 steps
#     simple         0.4286    0.3000
#     beta           0.2981    0.1559
#     normal         0.0348    0.0348
#     kl_optimal     0.0030    0.0030
#     exponential    0.0030    0.0030
#
# 43% in one step against 0.3%. A branch resolving that much at once invents
# whatever is easiest to invent, and on a branch conditioned on "somebody speaks"
# that is a voice. It surfaces at the OPENING because that is where the branch has
# least conditioning to anchor it -- the line has not started. That is also why the
# prose fixes helped and did not solve it: they reduce the empty space the invention
# lands in; this reduces the capacity to invent.
#
# THE AUDIO BRANCH HAS NO SCHEDULE OF ITS OWN. comfy/ldm/minimax/model.py derives it
# per step -- sigma_a = time_shift_sigma(sigma_v, shift_v, shift_a) -- so the audio
# tail is decided by the VIDEO schedule, and choosing a scheduler for the audio
# means giving up the one chosen for the picture. Inserting ONE step does not: it
# splits the final jump and leaves every earlier sigma exactly where it was.
#
# The formula is comfy's, restated here rather than imported, for the same reason
# last_audio_sigma restates it: this has to work when comfy is not importable.
def audio_sigma_of(video_sigma, shift_video, shift_audio):
    """The audio branch's sigma at a given video sigma. comfy's time_shift_sigma."""
    v, a, s = float(shift_video), float(shift_audio), float(video_sigma)
    base = s / (v + s * (1.0 - v))
    return a * base / (1.0 + (a - 1.0) * base)


def video_sigma_for_audio(target_audio, shift_video, shift_audio):
    """The video sigma that puts the audio branch on `target_audio`. The inverse."""
    v, a, t = float(shift_video), float(shift_audio), float(target_audio)
    base = t / (a - t * (a - 1.0))
    return base * v / (1.0 - base + base * v)


def insert_audio_landing(sigmas, shift_video, shift_audio,
                         target=0.03, coarse=0.10):
    """One extra step so the audio branch does not land from a great height.

    Returns a new list, or the input unchanged when there is nothing to do. This
    runs inside the render path, so anything unexpected -- an empty schedule, no
    trailing zero, a tail that is already soft -- returns the input rather than
    raising. It never inserts twice: after one pass the tail is below `coarse`.

    `target` is 0.03 because that is roughly what `normal` achieves on its own, and
    it is comfortably above the 0.003 kl_optimal leaves -- close enough to free, far
    enough from zero that the extra step is doing work rather than nothing."""
    try:
        out = [float(x) for x in (sigmas or [])]
    except (TypeError, ValueError):
        return sigmas
    if len(out) < 3 or out[-1] != 0.0 or out[-2] <= 0.0:
        return sigmas
    if audio_sigma_of(out[-2], shift_video, shift_audio) <= coarse:
        return sigmas
    land = video_sigma_for_audio(target, shift_video, shift_audio)
    # Strictly inside the final jump, or the schedule stops being monotonic.
    if not (0.0 < land < out[-2]):
        return sigmas
    return out[:-1] + [land, 0.0]


def last_audio_sigma(steps, shift_audio, scheduler="simple", shift_video=None):
    """How much audio noise is still left going into the FINAL sampling step.

    The audio branch runs on its own shifted timeline: time_shift_sigma inverts the
    video shift and re-applies the audio one, so what reaches the last step depends
    on the STEP COUNT and shift_audio -- and not at all on shift_video, which is the
    dial everybody reaches for.

    The base grid's last position before zero is 1/steps, so

        sigma_audio(last) = shift_audio / (steps + shift_audio - 1)

    At the 8 steps this node defaults to, shift_audio 3.0 leaves 0.30. At the 4 a
    distilled LoRA wants, the same 3.0 leaves 0.50 -- half of the audio denoising
    crammed into one step, and an audio branch resolving half its noise in a single
    jump is one that invents whatever is easiest. Reported as babble starting at
    step 3 of 4, which is that step.
    """
    try:
        n = max(1, int(steps))
        a = float(shift_audio)
    except (TypeError, ValueError):
        return 0.0
    # THE SCHEDULER DECIDES THIS, and the closed form agrees with exactly one of them.
    #
    # comfy/ldm/minimax/model.py:569 derives the audio sigma from the VIDEO sigma --
    # sigma_a = time_shift_sigma(sigma_v, shift_v, shift_a) -- so what reaches the
    # last step is whatever ladder the SCHEDULER produced, re-shifted. The formula
    # below reproduces that only for `simple`. Measured, 5 steps, shift_audio 3.0:
    #
    #     simple      0.4286   formula agrees
    #     beta        0.2981   formula is 44% high
    #     kl_optimal  0.0030   formula is 143x high
    #
    # The note this feeds fires above 0.40 and told the reader "only the step count
    # and shift_audio matter". On kl_optimal that warned about babble the scheduler
    # had already removed, and sent them to lower shift_audio -- a dial that cannot
    # reach 0.003 at any legal value -- when one dropdown does it.
    v = float(shift_video) if shift_video else _WIDGET_RANGE["shift_video"][0]
    try:
        import comfy.samplers as _cs
        import comfy.model_sampling as _cms
        _calc = getattr(_cs, "calculate_sigmas", None)
        if _calc is not None:
            _ms = _cms.ModelSamplingDiscreteFlow()
            _ms.set_parameters(shift=v)
            _sig = [float(x) for x in _calc(_ms, str(scheduler), n)]
            _last = next((x for x in reversed(_sig) if x > 0.0), 0.0)
            # invert to the base grid at shift_video, re-apply shift_audio
            _base = _last / (v + _last * (1.0 - v))
            return a * _base / (1.0 + (a - 1.0) * _base)
    except Exception:
        # No real ComfyUI (tests stub it), or a scheduler this install lacks. The
        # closed form is exact for `simple`, which is the shipped default.
        pass
    return a / (n + a - 1.0) if (n + a - 1.0) > 0 else 0.0


def scheduler_that_finishes_audio(steps, shift_audio, shift_video=None,
                                  current="simple", target=0.10):
    """The shipped scheduler that leaves the LEAST audio noise on the last step.

    ONLY ONE THAT HONOURS shift_video, and that restriction is the whole of what
    this function got wrong. Reported: switching to kl_optimal put watery waves on
    the picture.

    comfy/samplers.py grades its schedulers by `use_ms`. A handler with use_ms True
    is called as handler(model_sampling, steps) and sees the shift; one with use_ms
    False is called as handler(n, sigma_min, sigma_max) and NEVER SEES IT. kl_optimal
    exponential and karras are all in the second group, so recommending them threw
    shift_video away silently. At 5 steps and shift 12 the difference is the whole
    schedule:

        simple      1.0  0.9796  0.9474  0.8889  0.7500   <- stays high, as shift 12 asks
        kl_optimal  1.0  0.6725  0.4212  0.2082  0.0119   <- shift discarded

    The video branch gets almost no time at high sigma, so structure never resolves
    and the remaining steps polish detail with nothing underneath it. That is what
    watery looks like. The audio tail WAS better; it was better because the schedule
    had stopped being the one that was asked for.

    Returns (name, sigma) when a shift-honouring scheduler would get under `target`
    and beat what is selected, else None. Named rather than silently switched: the
    schedule shape changes the picture too, and that is the reader's call."""
    try:
        import comfy.samplers as _cs
        names = [n for n in (getattr(_cs.KSampler, "SCHEDULERS", []) or [])
                 if getattr(_cs.SCHEDULER_HANDLERS.get(n, None), "use_ms", False)]
    except Exception:
        return None
    # ...AND THE VIDEO SCHEDULE HAS TO SURVIVE IT. Honouring the shift is necessary
    # and not sufficient: ddim_uniform honours it and still starts at 0.98 rather
    # than 1.0, so the first step does not begin from full noise. Checking only the
    # audio number is the mistake that recommended kl_optimal, so the candidate's
    # own video schedule is read and has to start from noise and keep its high-sigma
    # steps -- all but two of them above 0.5, which is what shift 12 is buying.
    def _video_ok(nm):
        try:
            import comfy.model_sampling as _cms
            _ms = _cms.ModelSamplingDiscreteFlow()
            _ms.set_parameters(shift=float(shift_video or 12.0))
            sig = [float(x) for x in _cs.calculate_sigmas(_ms, nm, int(steps))]
        except Exception:
            return False
        return (len(sig) >= 3 and sig[0] >= 0.999
                and sum(1 for x in sig if x > 0.5) >= max(1, int(steps) - 1))

    now = last_audio_sigma(steps, shift_audio, current, shift_video)
    best, best_s = None, now
    for nm in names:
        if nm == current:
            continue
        try:
            sg = last_audio_sigma(steps, shift_audio, nm, shift_video)
        except Exception:
            continue
        if sg > 0.0 and sg < best_s and _video_ok(nm):
            best, best_s = nm, sg
    return (best, best_s) if (best is not None and best_s <= target) else None


# What shift_audio 3.0 leaves on the last step at the 8 this node defaults to.
DEFAULT_LAST_AUDIO_SIGMA = 0.30


def shift_audio_for(steps, target=None):
    """The shift_audio that reproduces a chosen last-step sigma at THIS step count.

    Inverting sigma = a / (steps + a - 1):

        a = sigma * (steps - 1) / (1 - sigma)

    The DIRECTION matters more than the arithmetic. sigma rises monotonically with
    shift_audio -- d/da = (steps - 1) / (steps + a - 1)**2, positive for every step
    count above one -- so fewer steps need a SMALLER shift_audio, not a larger one.

    The note this feeds scaled the other way: 3.0 * 8 / steps, which at the 4 steps
    a distilled LoRA wants advised 6.0 and took the last step from 0.50 to 0.67.
    That is the babble dial turned the wrong way, printed on the one report that
    only fires when somebody is already hearing babble. Nothing caught it because
    the tests covered last_audio_sigma, which was right, and not the advice.

    Clamped to the widget's own range so the number printed is one that can be
    typed in; where the floor binds, the caller reports the sigma it really gives
    rather than the one that was asked for.
    """
    s = DEFAULT_LAST_AUDIO_SIGMA if target is None else float(target)
    try:
        n = max(1, int(steps))
    except (TypeError, ValueError):
        return 0.0
    if not 0.0 < s < 1.0:
        return 0.0
    lo, hi = _WIDGET_RANGE["shift_audio"][1], _WIDGET_RANGE["shift_audio"][2]
    return min(max(s * (n - 1) / (1.0 - s), lo), hi)


def apply_h3_model_sampling(model, shift_video, shift_audio):
    """Apply H3's dual video/audio flow schedule from INSIDE the node so a missing
    upstream patch can't silently gibberish the audio.

    On ComfyUI 0.31+ the H3 nodes are V3-schema and don't live in the legacy
    NODE_CLASS_MAPPINGS the old way -- AND the model already defaults to the correct
    FLOW_AV schedule (12/3) at load. So the reliable path here is a DIRECT model-
    level patch (works regardless of node API); the node call is only a secondary.
    Order: direct model_sampling patch -> node under any name (V1/V3) -> give up with
    an informative, non-alarming note. Shifts aren't hardcoded (12/3 base, ~8 video
    for low-step MXFP8, ~4-6 audio for turbo)."""
    try:
        return _direct_model_sampling(model, shift_video, shift_audio), \
               f"model_sampling video {shift_video:g}/audio {shift_audio:g} (direct)"
    except Exception:
        pass
    cls, key = _find_h3_sampling_node()
    if cls is not None:
        try:
            return _call_node(cls, model, shift_video, shift_audio), \
                   f"model_sampling video {shift_video:g}/audio {shift_audio:g} (via {key})"
        except Exception:
            pass
    return model, (f"model_sampling not explicitly set (video {shift_video:g}/audio {shift_audio:g}); "
                   "on ComfyUI 0.30+ the model already defaults to the correct schedule, so this is "
                   "usually harmless -- only set shift_video/audio explicitly if you're on a low-step "
                   "MXFP8/turbo profile and the audio sounds wrong")


def sparse_dit_patched(model):
    """True when something upstream installed a per-block DiT replace patch, False when it
    provably did not, None when this model cannot say.

    That patch is how ComfyUI's Model Sparse Attention node registers itself
    (set_model_patch_replace -> model_options["transformer_options"]["patches_replace"]
    ["dit"]), whatever method it was set to. None is distinct from False on purpose: a stub
    or hand-built model carries no model_options at all, and the one caller of this turns a
    True into a refusal, so "cannot tell" must never be read as "not there"."""
    opts = getattr(model, "model_options", None)
    if not isinstance(opts, dict):
        return None
    tops = opts.get("transformer_options") or {}
    if not isinstance(tops, dict):
        return None
    return bool((tops.get("patches_replace") or {}).get("dit"))


def sparse_attention_allocator_abort(model):
    """The one configuration that does not raise, it ABORTS. Returns why, or "".

    cudaMallocAsync is stream-ordered: a block allocated on one CUDA stream must be freed
    consistently with that stream. comfy_kitchen's chunked sparse-attention producer is a
    GENERATOR consumed from inside the kernel call, across the stream boundary that
    --enable-dynamic-vram's prefetch machinery sets up, and freeing its per-chunk tensor
    there returns CUDA_ERROR_INVALID_VALUE from cuMemFreeAsync. That throws out of a tensor
    DESTRUCTOR, where there is no Python frame to catch it, so the process calls
    std::terminate: a core dump, not an exception, taking the server and the rest of the
    queue with it.

    Worth refusing rather than warning for exactly that reason -- there is nothing to
    recover from an abort, and nothing downstream gets the chance to try. Losing one render
    to a readable error is the better trade.

    Nobody chooses this, either. ComfyUI force-enables the allocator on any CUDA 13 torch
    build and does not consult its own card blacklist on that path (cuda_malloc.py), so a
    current install arrives here by default."""
    conf = str(os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "") or "")
    if "cudamallocasync" not in conf.lower():
        return ""
    if sparse_dit_patched(model) is not True:
        return ""
    return ("this render would ABORT the ComfyUI process rather than fail: a sparse-attention "
            "patch is on the model AND torch is using the cudaMallocAsync allocator "
            f"(PYTORCH_CUDA_ALLOC_CONF={conf}). Freeing the attention producer's per-chunk "
            "tensor under that allocator returns CUDA_ERROR_INVALID_VALUE from cuMemFreeAsync, "
            "inside a tensor destructor where nothing can catch it -- so the process "
            "core-dumps and the queue goes with it, which is why this stops here instead.\n\n"
            "Either restart ComfyUI with --disable-cuda-malloc, which is the flag ComfyUI's "
            "own cuda_malloc.py names for this failure, or take the Model Sparse Attention "
            "node out of the graph. ComfyUI turns that allocator on by itself on every CUDA 13 "
            "torch build without checking whether the card supports it, so this is the default "
            "rather than anything you picked.")


def sampling_oom_help(w, h, frames, fps, megapixels=0.0):
    """What to change, in this shot's own numbers, after a SAMPLING OOM.

    Tiling is a decode setting and cannot help here, so the generic "try tiling"
    advice is worse than useless -- it costs another full sampling pass before
    failing the same way. Give the two levers that do change sampling cost, each
    priced from the shot that just failed."""
    now = shot_latent_cells(w, h, frames, fps)
    secs = frames / float(fps or 24)
    out = [f"This is a SAMPLING out-of-memory, not a decode one, so tiled decode "
           f"cannot help it. The shot is {w}x{h} x {frames}f (~{secs:.1f}s) = "
           f"{now:,} latent cells, and sampling cost scales linearly with that."]
    opts = []
    for cut in (10.0, 7.0):
        if cut < secs - 0.4:
            f2 = align_frame_count(int(round(cut * (fps or 24))))
            opts.append(f"shot_seconds {cut:g} ({f2}f) is "
                        f"{100 - shot_latent_cells(w, h, f2, fps) * 100 // now}% smaller")
    if megapixels:
        for mp in (0.5, 0.35):
            if mp < megapixels - 0.02:
                w2, h2 = scale_to_megapixels(w, h, mp)
                opts.append(f"megapixels {mp:g} ({w2}x{h2}) is "
                            f"{100 - shot_latent_cells(w2, h2, frames, fps) * 100 // now}% smaller")
    if opts:
        out.append("Options: " + "; ".join(opts) + ".")
    out.append("Shot length is the stronger lever on a chain, because every shot pays it. "
               "H3's own cap is 362 frames and this shot is at or near it.")
    return " ".join(out)




# --- removals ----------------------------------------------------------------
# The one place the node edits your text, and it only ever DELETES.
#
# The scene paragraph is stamped on every shot, so a garment described there is
# still being described after a beat takes it off -- and a description of a worn
# garment beats a sentence saying it came off. The old node inferred removals from
# prose, which meant guessing, and the guessing is most of what made it
# unpredictable. This does not guess. You say what came off:
#
#     Dan cuts off her jacket and throws it away.
#     remove: jacket
#
# From that shot onward, any part of the scene naming "jacket" is dropped. The
# directive line itself never reaches the model.

_REMOVE_LINE = re.compile(r"^[ \t]*(?:remove|removed|off)[ \t]*:[ \t]*(.+?)[ \t]*$",
                          re.I | re.M)

# Field labels the OLD version of this node printed at the bottom of every shot it
# built. Paste one of those old scripts back in as a prompt and the labels now go
# to the model verbatim -- and a line reading "overall_soundscape: room tone" is
# read as text to put ON THE PICTURE. They are never scene description, so they are
# dropped, and info says so.
# A whole line that is nothing but one of those labels. Only the exact field names
# the old node emitted -- a bare "music:" could be someone's own scene note.
_LEGACY_FIELD = re.compile(
    r"^[ \t]*(?:overall_soundscape|non_diegetic_music)[ \t]*:.*$", re.I | re.M)
# ...and the shot tag it put at the FRONT of a line that also carries real text, so
# only the tag comes off.
_LEGACY_PREFIX = re.compile(r"^[ \t]*\[(?:Generation|Shot)[ \t]*\d+\][ \t]*", re.I | re.M)

# Words that ask for letterforms in the frame. H3 renders text when the prompt
# names text, and at cfg 1 there is no negative prompt to take it back -- so this
# warns rather than edits: only you know whether "a neon sign" is set dressing you
# want or a watermark you do not.
_TEXT_CUE = re.compile(
    r"\b(?:subtitle[sd]?|caption(?:s|ed)?|closed[- ]caption\w*|watermark(?:ed|s)?|"
    r"logo|logos|credits|title card|end card|lower third|chyron|"
    r"timestamp|time stamp|date stamp|timecode|"
    r"text overlay|on-?screen text|banner|karaoke)\b", re.I)


def strip_legacy_fields(text):
    """(text, how many field-label lines were dropped)."""
    text = text or ""
    n = len(_LEGACY_FIELD.findall(text)) + len(_LEGACY_PREFIX.findall(text))
    if not n:
        return text, 0
    out = _LEGACY_PREFIX.sub("", _LEGACY_FIELD.sub("", text))
    # The field lines leave blank lines behind, and a blank line is a beat boundary
    # here -- collapsing them keeps the shot count the author intended.
    out = re.sub(r"[ \t]*\n[ \t]*\n[ \t]*\n+", "\n\n", out)
    return out.strip(), n


_ADD_LINE = re.compile(r"^[ \t]*(?:add|wear|wearing)[ \t]*:[ \t]*(.+?)[ \t]*$", re.I | re.M)

# Prose that reads as taking something off. NOT used to remove anything -- inferring
# removals from prose is what made the old node unpredictable. It is used only to
# notice that a beat looks like a removal while the scene still describes the
# garment, and to say so, because that combination is a garment that comes back.
# Verbs that mean REMOVAL only with a particle. On their own, "cuts the rope",
# "takes her hand", "pulls her closer" and "throws the bag on the floor" are
# ordinary actions -- and reading one as a removal deletes that garment's entry
# from the scene, after which it is still worn but UNDESCRIBED. An undescribed
# garment is one the model invents, and what it invents is plain and pale. That
# is how a black shiny latex crop top comes back white.
#
# The particle's POSITION settles the ambiguous case. Straight after the verb it
# is a removal ("pulls down her shorts"); trailing after the object, only "off"
# and "away" are -- "takes her coat off" removes it, "pulls her crop top down"
# only adjusts it, and adjusting a garment must not cost it its description.
# One definition, in the engine. See engine._STRIP_VERB.
_STRIP_VERB = engine._STRIP_VERB
# The verbs above that stay a removal when the particle TRAILS the object -- "kicks
# her boots off". The rest are removals only with the particle straight after them:
# "steps out of her leggings" is one, "steps back" while a light goes off later in
# the sentence is not, and the trailing form would read that as a removal.
_TRAILING_VERB = (r"take[sn]?|took|taking|pull(?:s|ed|ing)?|peel(?:s|ed|ing)?|"
                  r"strip(?:s|ped|ping)?|cut(?:s|ting)?|rip(?:s|ped|ping)?|tear[s]?|"
                  r"tore|slip(?:s|ped)?|shrug(?:s|ged)?|yank(?:s|ed)?|tug(?:s|ged)?|"
                  r"toss(?:es|ed)?|throw[s]?|threw|kick(?:s|ed|ing)?|"
                  r"slide[s]?|slid|wriggle[sd]?|wiggle[sd]?")
# ...and verbs that are a removal on their own, needing no particle.
# One definition, in the engine. See engine._UNDO_VERB.
_UNDO_VERB = engine._UNDO_VERB
# Verbs that only take a garment off with the preposition that says so. Kept apart
# from _STRIP_VERB on purpose: that list also builds the DISPLACEMENT reader, and a
# bare "gets" or "pushes" there reads "gets down on her knees" and "pushes the door
# open" as garments being moved. Here they are only ever matched with "out of",
# "clear of", "off" or a destination, which is where the meaning lives.
_OUT_OF_VERB = (r"get(?:s|ting)?|got|shimm(?:y|ies|ied|ying)|squirm(?:s|ed|ing)?|"
                r"climb(?:s|ed|ing)?|ease[sd]?|easing|back(?:s|ed|ing)?|"
                r"step(?:s|ped|ping)?|wriggle[sd]?|wiggle[sd]?|struggle[sd]?")
_PUSH_VERB = (r"push(?:es|ed|ing)?|shove[sd]?|skim(?:s|med|ming)?|ease[sd]?|"
              r"easing|roll(?:s|ed|ing)?|work(?:s|ed|ing)?")

_REMOVAL_PROSE = re.compile(
    r"\b(?:" + _UNDO_VERB + r")\b"
    # "down" is NOT here. Pulling a garment down leaves it ON, around the thighs or
    # the hips -- it is displaced, not removed. Counted as a removal it was scrubbed
    # out of the scene, so every later shot stopped describing something that was
    # still in the picture, and an undescribed garment is one the model re-invents.
    # Reported as the shorts changing appearance in the next beat. The shot was also
    # told they come off and are "dropped out of frame", which is not what the beat
    # asked for at all. Displacement is handled below and keeps the garment described.
    r"|\b(?:" + _STRIP_VERB + r")\s+(?:off|away|out\s+of)\b"
    r"|\b(?:" + _TRAILING_VERB + r")\b(?=[^.;!?]{0,40}?\b(?:off|away)\b)"
    # Over the head is off. The only way a garment goes over a head is coming off
    # or going on, and the strip verbs are one-directional. A LOOKAHEAD, because
    # the garment sits between the verb and the particle -- "lifts her top over her
    # head" -- and the object span is read forward from the end of the match.
    r"|\b(?:" + _STRIP_VERB + r")\b"
    r"(?=[^.;!?]{0,40}?\bover\s+(?:her|his|their|the)\s+head\b)"
    # OUT OF IT, CLEAR OF IT, FREE OF IT. These verbs say nothing on their own --
    # "gets down", "eases back", "climbs up", "backs away" -- so they are kept out
    # of _STRIP_VERB, which also feeds the DISPLACEMENT reader, where a bare "gets"
    # would read every "gets down on her knees" as a garment being moved. With the
    # preposition in front of a garment there is no second reading: you cannot get
    # out of a thong and still have it on. The engine's own state reader has had
    # `gets out of` since it was written; this one did not, so the state knew the
    # garment was off while the text went on describing it as worn.
    r"|\b(?:" + _OUT_OF_VERB + r")\s+(?:out|clear|free)\s+of\b"
    # PUSHED OR SHOVED OFF. push and shove live in the displacement reader and not
    # in the strip verbs, so "pushes the thong off her hips" was a displacement at
    # best -- and in practice nothing at all, because the displacement pattern
    # wants the direction word where this sentence puts a body part. The garment
    # stayed described as worn in every later shot.
    r"|\b(?:" + _PUSH_VERB + r")\s+(?:off|away)\b"
    r"|\b(?:" + _PUSH_VERB + r")\b(?=[^.;!?]{0,40}?\b(?:off|away)\b)"
    # DOWN PAST THE HIPS. "down" on its own stays a displacement, for exactly the
    # reason the comment above gives: it leaves the garment ON, "around the thighs
    # or the hips". Down her LEGS, her knees, her ankles, or down to the floor is
    # the garment travelling past all of that, with nothing left holding it up.
    # The two readings are separated by the part of the body named -- and the two
    # positions this deliberately excludes are the two that comment names.
    r"|\b(?:" + _STRIP_VERB + r"|" + _PUSH_VERB + r")\b"
    r"(?=[^.;!?]{0,40}?\bdown\s+(?:(?:her|his|their|the)\s+"
    r"(?:legs?|knees?|calves|shins?|ankles?|feet)\b|(?:and\s+)?(?:off|away)\b|"
    + engine.TO_THE_FLOOR + r"))"
    # ...AND ONTO THE FLOOR. A garment dropped, let fall, kicked or thrown onto the
    # floor is off the body by the end of the sentence, whatever verb carried it
    # there. `drop` and `let` are the RESTORE vocabulary as well -- that is the
    # same ambiguity restored_garments resolves, and it resolves it the same way,
    # on where the garment lands. See engine.FLOOR.
    r"|\b(?:" + _STRIP_VERB + r"|" + _PUSH_VERB + r"|drop(?:s|ped|ping)?|"
    r"let(?:s|ting)?|lob(?:s|bed)?|fling(?:s|ing)?|flung|discard(?:s|ed|ing)?)\b"
    r"(?=[^.;!?]{0,40}?" + engine.TO_THE_FLOOR + r")",
    re.I)


_HAS_VERB = re.compile(
    r"\b(?:is|are|was|were|be|being|been|has|have|had|wears?|wearing|dressed|"
    r"walks?|walked|stands?|stood|sits?|sat|lies?|lying|holds?|holding|"
    r"cuts?|pulls?|takes?|steps?|turns?|looks?|comes?|goes)\b", re.I)


# WHOSE HANDS take a garment off. A removal clause with no agent describes the
# garment removing itself -- "the belt comes off during this shot and is away by the
# last frame" is true of a belt that drops to the floor on its own, and that is what
# it rendered. Reported after a beat where she ASKS somebody to unlock it.
#
# The beat names the person; the clause was just not carrying it. Only where the beat
# is unambiguous about who acts, which is why asking is read as the OTHER person's
# hands: "she asks Dan to take it off" is Dan's doing, not hers.
_ASKS = re.compile(r"\b(?:asks?|asked|begs?|begged|tells?|told|wants?|wanted|"
                   r"pleads?|pleaded|has|have|had|gets?|got)\b", re.I)


def _clause_about(beat, item=""):
    """The sentence/clause of `beat` that names `item`; the whole beat if it does not.

    An ask governs the garment it is ASKING about, not every garment in the beat.
    "Kate takes off her coat ... and asks him to get the scarf off" has one removal
    by her hands and one by his, and reading the ask against the whole beat gave
    both to him."""
    if not beat or not item:
        return beat or ""
    head = str(item).split()[-1]
    for part in re.split(r"(?<=[.;!?])\s+", str(beat)):
        if re.search(r"\b" + re.escape(head) + r"\b", part, re.I):
            return part
    return beat


def removal_agent(beat, cast, wearer=None, item=""):
    """Who takes the garment off in this beat. '' when the beat does not say.

    A beat with one person in it is that person undressing. With two, the one who is
    NOT the wearer is doing it when the wearer asks -- and when nobody asks, whoever
    the beat names first is acting, the same reading restrained_by_beat uses."""
    people = [n for n in (cast or []) if n]
    if not people:
        return ""
    if len(people) == 1:
        return people[0]
    b = beat or ""
    others = [n for n in people if n != wearer]
    # "She asks Dan to take it off" -- the request is hers, the hands are his. Only
    # when the ask governs THIS garment: a beat that takes a coat off and then asks
    # about a scarf had the ask applied to both, so her own coat came off by his
    # hands. Scoped to the clause the garment is named in, and when the garment is
    # not named there the beat's own first-named actor is used instead.
    if wearer and others and _ASKS.search(_clause_about(b, item)):
        return others[0]
    # First-named acts -- but in the GARMENT'S OWN clause, not the whole beat.
    # "Sam unties the scarf. Kate takes off her jumper." names Sam first overall,
    # so her jumper came off by his hands. The clause is what says who acts on what.
    scope = _clause_about(b, item)
    first, at = "", len(scope) + 1
    for n in people:
        m = re.search(r"\b" + re.escape(n) + r"\b", scope, re.I)
        if m and m.start() < at:
            first, at = n, m.start()
    if first:
        return first
    # Nobody is named in that clause: the wearer is undressing themselves.
    return wearer or people[0]


def beat_stages_removal(beat, item, agent=""):
    """Does the BEAT already say this garment comes off, by this agent's hands?

    The clause exists to guarantee the removal FINISHES inside the shot -- the last
    frame is the next shot's keyframe, and a cut mid-removal hands on a garment
    still half worn. That guarantee is needed whether or not the beat stages it.

    But when the beat already says "McKenna takes off her shorts and steps out of
    them", the full clause repeats the whole action -- who, what, and that it comes
    off -- and the shot carries the same removal twice. Two statements of one action
    is an invitation to render it twice.

    True when the beat names the garment's head noun near a removal verb, and either
    names the agent or the beat has no other actor. The caller then says only the
    part the beat does NOT cover: that it is finished by the last frame.
    """
    b = str(beat or "")
    head = str(item or "").split()[-1] if item else ""
    if not b or not head:
        return False
    if not re.search(r"\b" + re.escape(head) + r"\b", b, re.I):
        return False
    # A removal verb in the same sentence as the garment.
    for part in re.split(r"(?<=[.;!?])\s+", b):
        if not re.search(r"\b" + re.escape(head) + r"\b", part, re.I):
            continue
        if not _REMOVAL_PROSE.search(part):
            continue
        # ...and not merely ASKED for: a request is not the act. See _in_a_request.
        m = _REMOVAL_PROSE.search(part)
        if m and _in_a_request(part, m.start()):
            continue
        if not agent:
            return True
        return bool(re.search(r"\b" + re.escape(agent) + r"\b", part, re.I)
                    # "she takes off her shorts" -- a pronoun for the only actor.
                    or re.search(r"\b(?:she|he|they)\b", part, re.I))
    return False


def scene_tag_for(head, scene):
    """The <Picture N> tag on the sheet entry whose head noun is `head`. "" if none.

    The tag lives INSIDE the wardrobe entry -- "chastity belt <Picture 2>" -- so
    scrubbing the entry when the garment comes off takes the picture with it. That
    is right for the description and wrong for the reference: the shot that takes a
    thing off is the shot it is handled in and most needs to look like itself, and
    without the tag it carries no image at all. Reported as the belt not matching
    its reference on the shot that removes it."""
    head = (head or "").strip().lower()
    if not head or not scene:
        return ""
    for line in str(scene).split("\n"):
        for item in re.split(r"[,;.]", line.split(":", 1)[-1]):
            m = re.search(r"<\s*picture\s+\d+\s*>", item, re.I)
            if not m:
                continue
            bare = re.sub(r"<\s*picture\s+\d+\s*>", " ", item, flags=re.I)
            bare = re.sub(r"\s+", " ", bare).strip()
            if bare and bare.split()[-1].lower() == head:
                return m.group(0)
    return ""


def off_by_last_frame(items, agent="", scene="", beat=""):
    """State that a removal FINISHES inside this shot. Empty when nothing came off.

    Scrubbing the scene stops a garment being described. It does not tell the model
    to complete the removal, and the last frame is what the next shot inherits as
    its keyframe -- so a cut still in progress hands on a garment still half worn,
    and the next beat has moved on and never contradicts the picture. The garment
    stays. That is a garment "coming back" even though the text was right.

    Said ONCE, in the removing shot, and never again. A later shot that says "no
    longer wearing the coat" names the coat, and to a video model a mention is a
    presence cue -- that phrasing put garments back on in the previous version of
    this node. Afterwards the item is simply absent from the text."""
    items = [i.strip() for i in (items or []) if i and i.strip()]
    if not items:
        return ""
    # The SHEET's words for it, not the head noun the reader keyed it under. The
    # tokens are identity keys -- matched by head noun everywhere that scrubs and
    # compares -- but this sentence is PROSE the model reads, and "the shorts" beside
    # a sheet saying "blue jeans shorts" is two garments described, not one. The pair
    # that came back was the bare one, drawn however the model liked.
    # ...with the picture the sheet gave it. The entry is scrubbed on the removing
    # shot, so this is the only place left that can claim the image -- and a shot
    # carrying a reference whose tag it never names reads the picture as ANOTHER
    # subject, which is a duplicate rather than a belt.
    named = []
    for i in items:
        nm = scene_name_for(i, scene) or i
        tag = scene_tag_for(i, scene)
        named.append(f"{nm} {tag}" if tag else nm)
    what = " and ".join(f"the {i}" for i in named)
    plural = len(items) > 1 or bool(_PLURAL_ITEM.search(named[-1]))
    verb, are = ("come", "are") if plural else ("comes", "is")
    # Named hands where the beat gives them. Without an agent this says a garment
    # comes off by itself, and a belt with nobody touching it drops to the floor.
    # The beat already staged it: say only the part it does NOT cover -- that the
    # removal FINISHES in this shot. Restating who and what is the same action
    # written twice in one prompt, which is what rendered it twice.
    if beat and all(beat_stages_removal(beat, i, agent) for i in items):
        # The AGENT is what the beat already gave; the ACTION is not. An earlier
        # version of this cut both and returned only "the shorts are away by the
        # last frame", which asserts an end state and never says the removal
        # happens -- and the whole reason this clause exists is that scrubbing the
        # scene does not tell the model to complete one. Garments stopped coming
        # off. Say it agentlessly: the beat supplies the hands, this supplies the
        # completion.
        return (f" {what[0].upper()}{what[1:]} {verb} off during this shot and "
                f"{are} away by the last frame -- fully removed and clear of "
                f"the body.")
    if agent:
        sentence = (f"{agent} takes {what} off during this shot, with {agent}'s own "
                    f"hands, and {what} {are} away by the last frame -- fully removed "
                    f"and clear of the body.")
    else:
        sentence = (f"{what} {verb} off during this shot and {are} away by the last "
                    f"frame, fully removed and clear of the body, dropped out of "
                    f"frame.")
    # BOUND the action. Saying what comes off does not say where to STOP, and an
    # action with time left over runs on to whatever is next: a hand that finishes
    # one garment starts on the next one, or on the body under it. Said as what
    # STAYS -- at
    # cfg 1 there is no negative prompt, and a negation in the positive names the
    # thing it forbids. It also names no garment, so it summons none.
    # About what is WORN, not about the body. "Everything else on the body stays
    # exactly as it is for the whole shot" reads as an instruction to hold still.
    bound = "Everything else worn stays exactly as it is, untouched and fastened."
    return " " + sentence[0].upper() + sentence[1:] + " " + bound


# Garments that are grammatically plural, so the sentence above agrees with them.
_PLURAL_ITEM = re.compile(r"\b(?:s|shorts|trousers|pants|jeans|boots|shoes|gloves|"
                          r"tights|leggings|briefs|knickers|cuffs)$", re.I)


# PUTTING SOMETHING BACK ON. The mirror of a removal, and it had none of the same
# machinery. A removal is scrubbed from the staging shot AND given a clause saying
# it FINISHES there -- both ends, because the shot's keyframe shows the garment on
# and the text has to carry it off. An `add:` had only the scrub's opposite: the
# phrase went into the same shot's scene block as a plain worn item.
#
# So the shot inherited a last frame with the garment OFF and was told, statically,
# that it is ON. There is no change described, only a disagreement, and the model
# resolves it in the opening frames: whatever is on the body turns into the garment.
# Reported as one thing instantly becoming another, a beat before the beat that
# puts it on -- which is exactly what the opening frames of that shot are.
_PUTS_ON = re.compile(
    r"\b(?:put(?:s|ting)?|pull(?:s|ing)?|slip(?:s|ping)?|tug(?:s|ging)?|"
    r"draw(?:s|ing)?|get(?:s|ting)?|climb(?:s|ing)?|step(?:s|ping)?)\b"
    r"[^.;!?]{0,40}?\b(?:back\s+on|back\s+into|on|into)\b", re.I)
# ...and the ones that need no preposition.
_DRESSES = re.compile(r"\b(?:dress(?:es|ing)?|redress(?:es|ing)?|"
                      r"button(?:s|ing)?(?:\s+up)?|zip(?:s|ping)?\s+up|"
                      r"fasten(?:s|ing)?|laces?\s+up|puts?\s+back\s+on)\b", re.I)


def beat_stages_wearing(beat, item):
    """Does the BEAT say this garment goes ON during this shot?

    Only then is the both-ends clause right. `add:` has a second, older job -- it
    reveals a layer that was under something all along ("add: her white shirt
    underneath", after the jacket is cut off) -- and that garment was already worn.
    Telling the shot it goes on during these frames would stage a dressing that
    never happens, which is the same defect pointing the other way."""
    b = str(beat or "")
    if not b.strip():
        return False
    head = str(item or "").strip().lower()
    if not head:
        return False
    # The item has to be NAMED near the wearing verb, or a beat that puts a coat on
    # would also claim the boots an `add:` mentioned in the same breath.
    for pat in (_PUTS_ON, _DRESSES):
        for m in pat.finditer(b):
            window = b[max(0, m.start() - 60):min(len(b), m.end() + 60)]
            if re.search(r"\b" + re.escape(head.split()[-1]) + r"\b", window, re.I):
                return True
    return False


def wearing_clause(phrases):
    """Give putting something on BOTH ENDS: off as the shot opens, on by the last.

    The same shape direction_anchor uses for a door and removal_clause uses for a
    garment coming off. Phrased as where the garment IS at each end rather than as
    what it is not, because at cfg 1 there is no negative prompt and naming an
    unwanted state in the positive asks for it."""
    items = [str(p or "").strip().rstrip(".") for p in (phrases or []) if str(p or "").strip()]
    if not items:
        return ""
    what = " and ".join(items)
    plural = len(items) > 1 or bool(_PLURAL_ITEM.search(items[-1]))
    are = "are" if plural else "is"
    return (f" {what[0].upper()}{what[1:]} {are} off the body as the shot opens and "
            f"fully on by the last frame, put on during this shot.")


# --- restraints ---------------------------------------------------------------
# The one continuity fact the node asserts on its own, because it is the one that
# cannot be recovered: a cuff that renders open is not a detail that drifts, it is
# the scene stopping making sense. Once hardware is on, it stays on.
#
# ONE sentence, impersonal, positive. The previous version had a per-limb effect
# table, pose tracking and a hardware clause, and between them the beat became 4% of
# the prompt. This is the fact and nothing else.
# What a `remove:` has to name to switch the hold off again.
RESTRAINT_HOLD_KEY = ("handcuffs cuffs chains rope ropes tape gag collar restraints "
                      "shackles clamp clamps clip clips")
# Every one of these constrains the HARDWARE, never the body. An earlier wording said
# the restraint held "the same way from the first frame to the last" and the chain let
# the body reach "only as far as the metal allows before it stops" -- read plainly,
# that is an instruction to hold still, and stacked together the holds came to 64% of
# a shot whose beat was 11%. The performance died under its own continuity guards.
# Say what the metal does; leave the body to the beat.
# Staying closed is not the same as staying itself. Every hold above constrains
# the fastening; none of them says the thing is still made of what it was made
# of. A strip of tape, decoded and re-encoded once a shot, has nothing in the
# text holding it to being tape, and it drifts to the nearest commoner object.
# One short sentence, because these holds are already the longest thing a
# restrained shot carries.
# The picture side of a shot with nobody speaking. Positively phrased, because at
# cfg 1 no negative is evaluated: "nobody speaks" asks the model to render an absence
# and a closed mouth is a thing it can actually draw.
#
# This is the WEAK half and is known to be. _silent_audio_latent already records that
# a lips-closed sentence loses against an audio stream that has decided somebody is
# talking -- conditioning the branch is what settles it. So this rides along, and the
# switch also extends the silencing to the shots that were keeping the branch open.
#
# TWO THINGS ca75672 PAID FOR, both of which this has to keep:
#
# It goes AFTER the action, never in front of it. As the opening tokens it was face
# anatomy in the first thing the model reads, and a distilled LoRA settles composition
# in its first step or two -- that rendered a face at the start of shots.
#
# It is only ever said where there is a mouth to describe. On a scenery beat with
# nobody in it, a sentence about mouths describes a person who is not there, and the
# only way to satisfy it is to put a face in an empty frame. The AUDIO half has no
# such limit -- an empty room still babbles -- so the two are separate conditions and
# are gated separately below.
# AND IT DOES NOT FREEZE THE FACE. "Mouths stay closed" is the whole of the
# lip-sync guarantee -- lip-sync needs lips to part, and a closed mouth cannot do
# it. "jaws still" was a stillness instruction riding along on that guarantee,
# landing on every quiet shot in the film, and `still` is not a quiet word to a
# video model: it damps motion wherever it is pointed. Reported as bad acting.
#
# So the second half now says what the face IS doing rather than what it is not,
# which is the same rule every other clause here follows -- at cfg 1 there is no
# negative prompt, and an unreacting face is exactly what you get by asking for
# nothing. It costs one word against a guard block already measured at 47% of the
# shot, and it is the only sentence in that block with anything to say about
# performance.
MOUTH_HOLD = " Mouths in the shot stay closed, the expressions moving."


# A MOUTH THE BEAT ITSELF PUTS TO WORK.
#
# The guard above is right for a face doing nothing. It was also landing on the
# beats that ARE the performance -- and contradicting them, in one case word for
# word:
#
#     Dana grins, wide and mean.   -> Mouths in the shot stay closed, jaws still.
#     Dana's mouth falls open.     -> Mouths in the shot stay closed, jaws still.
#     Dana yawns.                  -> Mouths in the shot stay closed, jaws still.
#
# Only the VOCAL reactions stood down, because exertion_in covers laughing and
# sobbing. Every SILENT facial performance -- the ordinary currency of acting --
# was answered with an instruction to freeze it, and the beat is the only
# performance direction a shot has.
#
# ONLY THE MOUTH. A stare, a frown, a wince is a face acting with its mouth shut,
# and the guard costs it nothing; standing down for those would free a mouth for no
# gain, and a free mouth on an open branch is where invented lip-sync lands.
#
# SEPARATE FROM _voiced, which is the same stand-down for EFFORT and also unpins
# the audio branch. A smile is silent. This frees the picture and leaves the branch
# exactly where it was -- no beat that was silent before this becomes audible --
# because a silent expression is the most common beat in any script, and letting
# one open an audio branch would be the babble hole rebuilt at the widest point.
_MOUTH_WORKS = re.compile(
    r"\b(?:smil(?:e|es|ed|ing)|grin(?:s|ned|ning)?|smirk(?:s|ed|ing)?|"
    r"sneer(?:s|ed|ing)?|grimac(?:e|es|ed|ing)|pout(?:s|ed|ing)?|"
    r"yawn(?:s|ed|ing)?|gape(?:s|d|ing)?|chew(?:s|ed|ing)?|"
    r"kiss(?:es|ed|ing)?)\b"
    # Spitting needs somewhere to spit. Bare `spits` is what an engine does.
    r"|\bspits?\s+(?:it\s+)?(?:on|at|out|into|onto)\b"
    # The rest need their object, because the bare verb is ordinary English:
    # she bites her lip, not the dog bites; she licks her lips, not licks a stamp.
    r"|\b(?:bite|bites|biting|bit)\s+(?:down\s+on\s+)?(?:her|his|their|the)\s+lips?\b"
    r"|\blick(?:s|ed|ing)?\s+(?:her|his|their|the)\s+lips\b"
    r"|\bpurs(?:e|es|ed|ing)\s+(?:her|his|their|the)\s+lips\b"
    r"|\bbar(?:e|es|ed|ing)\s+(?:her|his|their|the)\s+teeth\b"
    r"|\bmouth(?:s|ed|ing)?\s+(?:the\s+)?words?\b"
    # "Dana's mouth falls open" is the same sentence as "her mouth falls open" and
    # was the one this file contradicted word for word, so the possessive NAME has
    # to be a determiner here too.
    r"|\b(?:her|his|their|the|[\w-]+['\u2019]s)\s+(?:mouth|jaw)\s+"
    r"(?:falls?|fell|drops?|dropped|hangs?|hung|opens?|opened)\b"
    r"|\b(?:her|his|their|the|[\w-]+['\u2019]s)\s+lips?\s+(?:parts?|parted)\b", re.I)


# WHAT SILENCE IN THE PROMPT ACTUALLY ASKS FOR.
#
# Reported: she smiles at the camera in a situation of duress. Dumped, a four-shot
# scene of a woman handcuffed in the back of a van -- pulling at the cuffs,
# struggling, going limp -- carried NOT ONE WORD about anybody's face. Every clause
# in it was hardware, limbs, or mouths-closed.
#
# An unstated attribute is not a neutral one. The model fills it from its prior, and
# the prior for a named, described person is a PORTRAIT: facing the lens, pleasantly,
# because that is what photographs of people are. This file already knows that half
# of it about the EYES -- it is the entire reason gaze_hold exists -- but gaze_hold
# only fires where the beat NAMES something to look at, which most beats do not, and
# nothing in this node has ever spoken for the expression at all.
#
# So the shot says the one thing the scene has already established. NOT an invented
# emotion: hardware the sheet lists, or the author's own distress verbs. A shot
# staging neither gets nothing, because a node deciding how everybody feels is a node
# writing the film -- which is why this also has a switch.
#
# The distress list is _EXERTION's, minus the ones that are not distress. `laughs` is
# in that list because it is VOCAL, which is all _EXERTION is for; stamping strain on
# a face that was written laughing would be the mouth guard's bug again in a new
# place. `wakes up` is not duress either.
_DISTRESS = re.compile(
    r"\b(?:thrash(?:es|ing|ed)?|struggl(?:e|es|ing|ed)|writh(?:e|es|ing|ed)|"
    r"strain(?:s|ing|ed)?|squirm(?:s|ing|ed)?|kick(?:s|ing|ed)?|jerk(?:s|ing|ed)?|"
    r"sob(?:s|bing|bed)?|cr(?:y|ies|ying|ied)|weep(?:s|ing)?|"
    r"scream(?:s|ing|ed)?|shriek(?:s|ing|ed)?|whimper(?:s|ing|ed)?|"
    r"beg(?:s|ging|ged)?|plead(?:s|ing|ed)?|"
    r"flinch(?:es|ing|ed)?|winc(?:e|es|ing|ed)|recoil(?:s|ing|ed)?|"
    r"trembl(?:e|es|ing|ed)|shiver(?:s|ing|ed)?|panic(?:s|king|ked)?|"
    r"freak(?:s|ing)?\s+out)\b"
    r"|\b(?:goes|went|going)\s+limp\b", re.I)

# POSITIVELY PHRASED, like everything else here. At cfg 1 there is no negative
# prompt, so "not smiling" would name the smile -- and "unsmiling" is the same word
# with a prefix on it. A mouth that is SET is a thing the model can draw, and it is
# what the shot needs drawn.
#
# It names no camera. Naming one is asking for one, and the lens is exactly what
# this sentence is trying to get her to stop looking at.
# A FILM HAS A MOOD, AND IT DOES NOT BELONG TO ONE CHARACTER'S FACE.
#
# 8418805 gave the restrained person a face and stopped there. Reported back: "the
# last run had them smiling and thinking this was a happy scene, when indeed it was
# not." THEM -- plural. Measured on a five-shot kidnapping:
#
#     shot 1  cast=          -- NOTHING --
#     shot 2  cast=McKenna   strain
#     shot 3  cast=McKenna Dan   strain     <- impersonal, so Dan gets nothing
#     shot 4  cast=Dan       -- NOTHING --
#     shot 5  cast=Dan       -- NOTHING --
#
# Every shot the captor is alone in had no tone in it at all, and an unstated tone
# is filled from the prior the same way an unstated expression is: the prior for a
# man in a work coat is a pleasant one. The face clause could never reach him --
# he is not the one under duress and should not look strained. What is wrong with
# those shots is not his expression, it is the whole frame.
#
# So the mood is read from the FILM and said in every shot, and the face clause
# rides on top of it where the person it describes is actually present. One word of
# tone conditions light, faces and framing together, which no per-face sentence can.
DURESS_MOOD = " The mood is grim."
DURESS_FACE = " The mood is grim; the face shows the strain of it, the mouth set."


# COERCION -- A KIDNAPPING IS NOT A DISTRESS VERB AND IT IS NOT IN THE SHEET.
#
# Reported: "she is still smiling in every beat, despite this being a kidnapping
# situation that was not defined in the anchor." That last clause is the bug.
# film_stages_duress read binding hardware from the CHARACTER SHEET and distress
# verbs from the beats, and an abduction is neither: nobody writes "McKenna: she,
# 26, handcuffs" for one -- the hardware goes ON during the film, in the beats --
# and the beats use verbs that were nowhere in the distress list. Ten beats of an
# explicit abduction returned False, so the film had no mood, so every face came
# from the portrait prior, which is pleasant.
#
# THE VERBS NEED A PERSON. Grabbing, dragging, forcing and shoving are all ordinary
# things to do to an OBJECT -- a coffee, a case, a window -- and a film of those is
# not a grim film. So each one has to take a person: a pronoun, or a capitalised
# name. That single requirement is what separates "drags McKenna towards the van"
# from "drags the case to the door", and it is tested both ways.
# THE OBJECT HAS TO BE THE PERSON, NOT SOMETHING THEY OWN.
#
# "grabs her keys", "snatches her coat", "seizes her chance", "forces her way to the
# bar", "pins Ellie's painting to the fridge" -- all read as coercion, because `her`
# and `him` and a name are as often possessives as objects. Measured on 93 ordinary
# domestic beats, this was most of 65 false positives, and one of them is enough to
# stamp "The mood is grim" on a whole comedy.
#
# So the pronoun or name has to be the END of the object: a clause boundary, or one
# of the words that can only follow a completed object. "grabs her from behind" is
# coercion; "grabs her keys" is a Tuesday.
_OBJ_IS_THE_PERSON = (
    r"(?=\s*(?:[.,;:!?\"\u201d]|$)|\s+(?:into|out|off|from|down|up|towards?|to|"
    r"against|across|onto|through|back|away|and|by|in|on|over|behind|while|as|"
    r"before|after|until|so|but|with|without|aside|apart|hard|roughly|violently|"
    r"bodily|clear|free|upright|sideways|forward|backwards?)\b)")

_COERCION = re.compile(
    r"\b(?:grab(?:s|bed|bing)?|drag(?:s|ged|ging)?|forc(?:e|es|ed|ing)|"
    r"shov(?:e|es|ed|ing)|haul(?:s|ed|ing)?|bundl(?:e|es|ed|ing)|"
    r"seiz(?:e|es|ed|ing)|snatch(?:es|ed|ing)?|pin(?:s|ned|ning)?|"
    r"restrain(?:s|ed|ing)?|manhandl(?:e|es|ed|ing)|overpower(?:s|ed|ing)?|"
    r"subdu(?:e|es|ed|ing)|wrestl(?:e|es|ed|ing))\s+"
    # (?-i:) MATTERS. The whole pattern is case-insensitive, which turned [A-Z]
    # into "any letter" and let "drags THE case" and "forces THE window" read as
    # coercion. The capital is the only thing separating a name from a determiner.
    r"(?:her|him|them|(?-i:[A-Z][\w-]+))" + _OBJ_IS_THE_PERSON
    # ...and the phrases that carry it without a bare transitive verb.
    + r"|\b(?:holds?|held|holding|pins?|pinned|forces?|forced)\s+"
    r"(?:her|him|them|(?-i:[A-Z][\w-]+))\s+(?:down|still|against|into|in)\b"
    r"|\bcover(?:s|ed|ing)?\s+(?:her|his|their|[\w-]+['\u2019]s)\s+mouth\b"

    r"|\bagainst\s+(?:her|his|their)\s+will\b"
    # PASSIVE VOICE. "She was grabbed from behind", "is bundled into the back",
    # "were hauled out of the church" -- the victim is the SUBJECT, so nothing
    # follows the verb and every active pattern above misses. This was the single
    # largest family of misses: an author writing an abduction reaches for the
    # passive precisely because the victim is the one the sentence is about.
    #
    # The trailing preposition is what keeps "the photo is taken at noon" out: a
    # person is taken FROM, INTO, OUT OF, AWAY. A thing is just taken.
    r"|\b(?:was|were|is|are|been|being|got)\s+(?:\w+\s+){0,2}?"
    r"(?:grabbed|dragged|forced|shoved|hauled|bundled|seized|snatched|pinned|"
    r"restrained|manhandled|overpowered|subdued|taken|carried|marched|walked|"
    r"loaded|bundled|driven|led)\s+"
    r"(?:from|into|out|off|away|down|to|in|through|aboard|across|onto|with)\b"
    # CAPTIVITY, which often has no verb of violence in it at all.
    r"|\bheld\s+(?:captive|prisoner|hostage)\b"
    r"|\b(?:captors?|hostages?|abduction|kidnapping)\b"
    r"|\b(?:held|taken|kept)\s+captive\b"
    r"|\blocked\s+(?:in|inside|up)\b"
    r"|\bkidnap(?:s|ped|ping)?\b|\babduct(?:s|ed|ing|ion)?\b|\bhostage\b"
    # Trying to get out is duress by definition.
    r"|\btr(?:y|ies|ied|ying)\s+to\s+(?:get\s+away|get\s+out|escape|run|pull\s+free)\b"
    r"|\b(?:break(?:s|ing)?|broke|pull(?:s|ed|ing)?)\s+free\b"
    r"|\bescap(?:e|es|ed|ing)\b", re.I)

# Hardware being APPLIED, in a beat. The same words as _BOUND_HARDWARE plus the
# forms an action uses -- a sheet says "tied", a beat says "ties" -- and each one
# still has to reach a person or a part of one, so taping a box shut is not an
# abduction.
# WHAT A RESTRAINT IS PUT ON: A BODY.
#
# The first version of this asked whether binding words appeared near a person, and
# a sweep of 512 beats showed what English does with those words when nobody is
# being restrained at all:
#
#     She is bound for Lisbon on the early flight.
#     At full time it is still tied at two apiece.
#     She's tied up in meetings until four.
#     He has been chained to that desk for eleven years.
#     The ledger is bound in green cloth.
#     The boat is tied up at the jetty.
#     He gagged at the smell coming off the bins.
#     He pulls the hood on his parka up against the drizzle.
#
# Every one read as STRONG evidence, and one strong beat is enough to stamp "The
# mood is grim" on a whole film. `bound`, `tied`, `chained` and `gagged` are all
# idioms before they are restraints.
#
# So the state forms are gone. Binding has to reach a BODY PART, or a person plus
# the furniture people actually get tied to. That is what a restraint is; the rest
# is a figure of speech.
_BINDABLE = (r"wrists?|ankles?|hands|feet|legs?|arms?|mouth|thumbs?|knees|elbows")
_TIE_TO = (r"chair|bed|bedframe|headboard|radiator|pipe|post|stake|banister|"
           r"bannister|frame|hook|ring|beam|column|tree")
_BINDING_ACT = re.compile(
    # ties her wrists, cuffs his ankles, gags her, tapes McKenna's mouth
    r"\b(?:ties?|tying|tied|bind(?:s|ing)?|bound|cuff(?:s|ed|ing)?|"
    r"shackl(?:e|es|ed|ing)|chain(?:s|ed|ing)?|zip-?ti(?:e|es|ed))\s+(?:up\s+)?"
    r"(?:her|his|their|(?-i:[A-Z][\w-]+)'s)\s+(?:" + _BINDABLE + r")\b"
    # taping is strapping unless it reaches a mouth, or wrists held together
    r"|\btap(?:e|es|ed|ing)\s+(?:up\s+)?(?:her|his|their|(?-i:[A-Z][\w-]+)'s)\s+"
    r"(?:mouth\b|(?:wrists?|ankles?|hands)\s+(?:together|behind|to)\b)"
    # wrists cable-tied, ankles taped -- the participle fragment
    r"|\b(?:" + _BINDABLE + r")\s+(?:\w+\s+){0,2}?"
    r"(?:tied|taped|cuffed|bound|chained|shackled|zip-?tied|strapped)\b"
    # tied TO the things people get tied to
    r"|\b(?:tied|cuffed|bound|shackled|chained|strapped|handcuffed)\s+"
    r"(?:her|him|them|(?-i:[A-Z][\w-]+)\s+)?(?:to|against)\s+"
    r"(?:the|a|an|that|this|his|her|their)\s+(?:" + _TIE_TO + r")\b"
    # hardware named as being ON a body
    r"|\b(?:handcuffs?|cuffs|rope|ropes|cord|cords|chains?|shackles|zip\s*ties?|"
    r"cable\s*ties?|duct\s*tape|tape|gag|blindfold)\s+(?:\w+\s+){0,2}?"
    r"(?:on|around|round|over|across|behind)\s+"
    r"(?:her|his|their|(?-i:[A-Z][\w-]+)'s|the)\s+(?:" + _BINDABLE + r"|head|eyes|face)\b"
    # A person gagged -- the person, not a smell he gagged at.
    r"|\bgag(?:s|ged|ging)\s+(?:her|him|them|(?-i:[A-Z][\w-]+))\b"
    # A bag or hood put over SOMEBODY ELSE'S head. Bare `hooded` and `blindfolded`
    # are out: a hooded parka, a hooded dressing gown, a hooded teenager and a
    # blindfold wine tasting all read as abduction, and one strong beat is enough
    # to call a whole film grim. The article is what carries it -- "a hood over her
    # head" is done TO her, "her hood over her head" is her own coat in the rain.
    r"|\b(?:a|the|another)\s+(?:bag|hood|sack|pillowcase)\s+over\s+"
    r"(?:her|his|their|(?-i:[A-Z][\w-]+)'s|the)\s+head\b"
    r"|\bblindfold(?:s|ed|ing)?\s+(?:her|him|them|(?-i:[A-Z][\w-]+))\b"
    # A PERSON in a bound state. `tied` and `chained` are left out of this one
    # deliberately -- "her hands are tied, politically speaking", "he has been
    # chained to that desk for eleven years" -- and `bound` needs guarding against
    # the commonest idiom of all, which is a departure board.
    r"|\b(?:she|he|they|(?-i:[A-Z][\w-]+))\s+(?:\w+\s+){0,2}?"
    r"(?:is|are|was|were|had\s+been|has\s+been|got)\s+(?:\w+\s+){0,2}?"
    r"(?:bound(?!\s+for\b)|gagged|cuffed|handcuffed|shackled)\b",
    re.I)


# STRONG EVIDENCE AND WEAK EVIDENCE, because English will not do better.
#
# Swept across 512 beats of six scenario families, the distress list alone produced
# 87 false positives, and they are not fixable by patching it:
#
#     The children scream all the way down the waterslide.
#     The baby cries in the next room.
#     She strains to hear the platform announcement.
#     She winces at the price and buys it anyway.
#     She screams with laughter as the boat slaps down off the wake.
#
# `screams`, `cries`, `strains`, `winces`, `panics` and `begs` mean distress or they
# mean a good day out, and no pattern can tell which from the words alone. What CAN
# tell is the rest of the film. So the evidence is graded:
#
#   STRONG -- says duress on its own and is almost never innocent: hardware on a
#             body, a captor, a hostage, an abduction, being locked in, something
#             done against somebody's will.
#   WEAK   -- an ambiguous verb: the distress words, and the ordinary coercion verbs
#             that are equally at home in a garden centre.
#
# A film is grim if ANY beat is strong, or if TWO are weak. One ambiguous verb is
# not enough to stamp "The mood is grim" on somebody's comedy; two is little enough
# that a real abduction -- which is nothing but coercion verbs -- always lands.
#
# And a WEAK beat gets the face clause only in a film already established as grim.
# That is the point of grading: "she screams" is terror in an abduction and delight
# on a waterslide, and the film is the only thing that knows which.
_DURESS_STRONG = re.compile(
    r"\bheld\s+(?:captive|prisoner|hostage)\b"
    r"|\b(?:captors?|hostages?|abduction|kidnapping)\b"
    r"|\b(?:held|taken|kept)\s+captive\b"
    r"|\bkidnap(?:s|ped|ping)?\b|\babduct(?:s|ed|ing|ion)?\b"
    r"|\blocked\s+(?:in|inside|up)\b"
    r"|\bagainst\s+(?:her|his|their)\s+will\b", re.I)


def beat_duress_strength(beat):
    """'' , 'weak' or 'strong'. See _DURESS_STRONG for why the grading exists."""
    b = beat or ""
    if _DURESS_STRONG.search(b) or _BINDING_ACT.search(b):
        return "strong"
    if _DISTRESS.search(b) or _COERCION.search(b):
        return "weak"
    return ""


def beat_stages_duress(beat, film_duress=True):
    """Does this BEAT stage duress?

    Strong evidence always counts. Weak evidence counts only where the FILM is
    already grim, because that is the context that says which meaning an ambiguous
    verb has. Defaults to True so a caller asking about a beat in isolation gets
    the old, generous reading."""
    strength = beat_duress_strength(beat)
    return strength == "strong" or (strength == "weak" and bool(film_duress))


# THE AUTHOR CAN JUST SAY IT, AND THAT BEATS ANY AMOUNT OF GUESSING.
#
# Swept across 512 beats of six scenario families, inference alone does not work and
# the numbers say so plainly. Simulating 8-beat films:
#
#     evidence needed   duress films read grim   ordinary films read grim
#     2 weak                     73.5%                    87.8%
#     3 weak                     57.1%                    82.5%
#     strong only                44.5%                    28.5%
#
# At every setting an ORDINARY film was as likely to be called grim as a duress one,
# because the words overlap: `screams` is a waterslide, `tied` is a boat, `bound` is
# a flight to Lisbon, `chained` is a desk job. That is not a pattern that needs more
# work, it is English, and no bag of patterns is going to separate them.
#
# So the ANCHOR is asked first. It is already the film-wide declaration -- "framing
# that belongs to the whole film" -- and a tone belongs there beside the lighting.
# Said there, it is authoritative in BOTH directions: a film declared warm is never
# given a grim mood however its beats read, and that is the escape hatch for every
# false positive above.
#
# Only where the anchor says nothing does this fall back to inference, and then only
# on STRONG evidence, because an unasked-for grim mood on somebody's comedy is a
# visible defect while a missing one is recoverable by typing six words. info says
# which of the three happened every run.
_MOOD_GRIM = re.compile(
    r"\b(?:grim|bleak|tense|menacing|sinister|harrowing|distressing|brutal|"
    r"frightening|terrifying|desperate|oppressive|claustrophobic|ominous|"
    r"threatening|violent|grave|sombre|somber|dread|hostile|cruel|"
    r"kidnap(?:ping)?|abduction|captivity|hostage|abusive|coercive)\b", re.I)
_MOOD_LIGHT = re.compile(
    r"\b(?:warm|comic|comedy|cheerful|joyful|joyous|happy|light[-\s]?hearted|"
    r"playful|romantic|tender|sunny|upbeat|gentle|affectionate|celebratory|"
    r"whimsical|carefree|domestic\s+bliss|feel[-\s]?good)\b", re.I)


def mood_declared(anchor):
    """'grim', 'light' or '' -- what the ANCHOR says the film's tone is.

    Both directions matter. A film declared warm must never be handed a grim mood
    however its beats read, because that is the one reliable way out of a wrong
    inference; and a film declared grim needs no inference at all."""
    a = anchor or ""
    if _MOOD_LIGHT.search(a):
        return "light"
    if _MOOD_GRIM.search(a):
        return "grim"
    return ""


def film_stages_duress(beats, sheet="", anchor=""):
    """Does this FILM stage duress anywhere -- binding hardware, or a distress verb?

    Read once, over the whole script, because a shot of the captor alone is grim on
    account of what is on her wrists three beats ago. The same two signals the face
    clause uses, and the same refusals: a collar alone is not duress, and a film
    that stages neither is left alone in every shot. The node does not get to decide
    that somebody's film is bleak."""
    said = mood_declared(anchor)
    if said:
        return said == "grim"
    for _, ln in sheet_lines(sheet or ""):
        if _BOUND_HARDWARE.search(ln or ""):
            return True
    return any(beat_duress_strength(b) == "strong" for b in (beats or []))


# BINDING hardware, which is narrower than restraint hardware. restraint_present is
# right for the continuity holds -- a collar is a thing that must stay fastened and
# stay the object it was -- but it is not evidence of DURESS. A collar is worn in
# scenes that are not distressing at all, and stamping strain on a face in one of
# those is the same error as stamping a closed mouth on a grin. Cuffs, rope, chain,
# tape and a gag are not ambiguous that way.
_BOUND_HARDWARE = re.compile(
    r"\b(?:handcuffs?|cuffs?|shackles?|manacles?|irons|"
    r"ropes?|cords?|twine|zip\s*ties?|cable\s*ties?|"
    r"chains?|chained|tape|taped|gag|gagged|bound|tied|bindings?)\b", re.I)


def duress_face(beat, wearers, described, film_duress=False):
    """One short sentence about the face, on a shot whose scene already stages duress.

    IMPERSONAL, the choice gaze_hold already made and for the same reason: a named
    person is a person the model draws, and naming somebody twice in one shot is what
    put a second girl in frame at the moment of cuffing. On the shot where that could
    be ambiguous -- two people, one of them restrained -- the hardware hold has
    already said "Every restraint on Nora", so the shot is not short of an
    attribution. It is short of a sentence about her face."""
    # THE AUTHOR'S OWN EMOTION WINS, and it is said back in THEIR word rather than
    # the film's generic mood. "The mood is grim; ... the mouth set" was being stamped
    # over "terrified", which is a different performance, and over a happy scene it
    # fired not at all -- so the emotional register was only ever asserted in one
    # direction and only ever generically. Said wherever the beat names a feeling,
    # duress or not, which is why this sits ahead of every duress test below.
    _emotion = emotion_in(beat)
    if _emotion and described:
        # ONE PERSON, NOBODY ELSE IT COULD BE. More than one and the feeling has to be
        # pinned, or the sentence lands on every face in the shot -- and a feeling the
        # beat pins on nobody holds nobody, exactly as a vocal does: guessing which of
        # two faces wears it is how the captor came to look terrified.
        if len(described) < 2:
            return mood_face(_emotion)
        _pairs = emotion_pairs(beat, described)
        return mood_faces(_pairs)
    # The author's own face beat wins, exactly as it does against the mouth guard.
    # Where the beat says what the face is doing, the node has nothing to add.
    if mouth_performs(beat):
        return ""
    if not described:
        return ""
    who = [n for n, ln in (wearers or []) if n and _BOUND_HARDWARE.search(ln or "")]
    if not who and beat_stages_duress(beat, film_duress):
        who = [n for n in (described or []) if n]
    if who:
        return DURESS_FACE
    # Nobody under duress IN THIS SHOT, but the film is. The frame still is not a
    # happy one, and saying nothing is what let the captor smile through it.
    return DURESS_MOOD if film_duress else ""


# AN EMOTION THE AUTHOR STATED, in their own word.
#
# Reported: under duress she does not act or respond like it, and the same in scenes
# where she is supposed to be happy. Measured, the node was contradicting the beat in
# both directions at once. "Mia hugs Tess, beaming." came out with "Mouths in the shot
# stay closed" beside it, because beaming was in no list. "McKenna is terrified and
# shaking." came out with "The mood is grim; the face shows the strain of it, the mouth
# set" -- a generic, clenched, stoic face stamped over the specific word the author
# chose, and then a mouth guard on top of that.
#
# An emotion is performed largely WITH THE MOUTH: delight is a smile, terror is an open
# mouth, fury is bared teeth. A guard that closes the mouth closes the performance, and
# at cfg 1 the flat positive instruction wins over the adjective in the beat.
#
# Read as a stand-down and as a register, never as an invention: where the author names
# no feeling, nothing here fires and the film's own mood clause is untouched.
_EMOTION = re.compile(
    r"\b(?:happy|happily|happiness|delighted|delight(?:ed)?|thrilled|overjoyed|"
    r"joyful|joyous|elated|ecstatic|beaming|beams?|gleeful|glee|cheerful|cheery|"
    r"pleased|excited|excitement|grateful|relieved|relief|proud|smug|amused|"
    r"terrified|terror|frightened|afraid|scared|fearful|panicked|panicking|panic|"
    r"furious|fury|angry|angrily|anger|enraged|livid|seething|indignant|"
    r"desperate|desperation|distraught|devastated|grief|grieving|heartbroken|"
    r"miserable|wretched|ashamed|shame|humiliated|mortified|disgusted|horrified|"
    r"anguished|anguish|agony|bereft|despair(?:ing)?)\b", re.I)


def emotion_in(beat):
    """The emotion this beat states, in the author's own word. "" when it states none."""
    m = _EMOTION.search(str(beat or ""))
    return m.group(0).lower() if m else ""


def emotion_owner(beat, names, word):
    """Whose feeling it is: the person the beat puts in front of it. "" if nobody.

    The shape subjects_for uses, conjunction guard included, so "Dan holds the door
    and McKenna is terrified" does not hand the terror to Dan. Takes a name list
    rather than a sheet because the caller already has the shot's cast."""
    b = str(beat or "")
    for n in (names or []):
        if n and re.search(r"\b" + re.escape(n) + r"\b"
                           r"(?:\s+(?!and\b|but\b|then\b|who\b|,\s*who\b)[\w,']+){0,2}?"
                           r"\s+(?:is|was|looks?|looked|seems?|feels?|felt|sounds?|"
                           r"becomes?|became|goes|went|turns?|gets?|got)?\s*"
                           + re.escape(word) + r"\b", b, re.I):
            return n
    return ""


def emotion_pairs(beat, names):
    """[(who, feeling)] for the feelings this beat pins on people. Two at most.

    TWO PEOPLE CAN FEEL DIFFERENT THINGS IN ONE SHOT. "Dan is furious and McKenna is
    terrified" gave only the first of them, so one face was performing and the other
    was left to the prior -- the same half-fix as naming one of two speakers. Two at
    most, like the layering clause: a shot carrying four feelings has stopped being
    about its beat."""
    out, seen = [], set()
    for m in _EMOTION.finditer(str(beat or "")):
        word = m.group(0).lower()
        who = emotion_owner(beat, names, word)
        if who and who not in seen:
            seen.add(who)
            out.append((who, word))
        if len(out) >= 2:
            break
    return out


def mood_faces(pairs):
    """Say whose feeling is whose, for one or two people. "" for none."""
    ps = [(w, e) for w, e in (pairs or []) if w and e]
    if not ps:
        return ""
    if len(ps) == 1:
        return mood_face(ps[0][1], ps[0][0])
    return (f" {ps[0][0]}'s face carries {ps[0][1]} and {ps[1][0]}'s carries "
            f"{ps[1][1]}, each played in the eyes and the mouth.")


def mood_face(word, who=""):
    """Say the face plays the feeling the author named. "" when they named none.

    Their word, not a synonym: "terrified" and "grim" are not the same performance,
    and the generic one was replacing the specific one on every shot.

    NAMED ONCE A SECOND PERSON IS IN THE SHOT, which is the call gaze_hold already
    makes for the same reason. Said impersonally, "the face carries it: the expression
    is terrified" is a sentence about whoever is on screen -- so in a two-hander the
    captor wore his victim's terror. Reported as actions being performed by all the
    characters at once. With one person there is nobody else it could be, and naming
    them again is a second mention of a person, which has its own cost."""
    if not word:
        return ""
    if who:
        return (f" {who}'s face carries it: the expression is {word}, played in the "
                f"eyes and the mouth together.")
    return (f" The face carries it: the expression is {word}, played in the eyes and "
            f"the mouth together.")


def mouth_performs(beat):
    """Does the beat itself put the MOUTH to work?

    The beat has already said what the mouth does, so the guard has nothing to add
    over the top -- and what it was adding contradicted it. Same shape as the LONE
    vocal that sounds_for leaves alone: where the author wrote it, the node is
    quiet."""
    return bool(_MOUTH_WORKS.search(beat or ""))

_PERSON_WORD = re.compile(
    r"\b(?:he|she|they|him|her|hers|them|his|their|theirs|himself|herself|themselves|"
    r"nobody|somebody|anyone|everyone|man|woman|men|women|boy|girl|person|people|"
    r"figure|guard|driver|doctor|nurse|officer)\b", re.I)


def beat_puts_somebody_on_screen(beat, sheet=""):
    """Does the BEAT itself put a person in the shot?

    Deliberately not "is a person described in this shot's text": the character
    guard carries the previous shot's cast forward so a wordless beat does not empty
    the frame, and falls back to the sole sheet entry when there is no previous. So
    a scenery beat has a person described beside it before anybody has walked in,
    and reading that as "somebody is here" is what put a face in an empty yard."""
    b = beat or ""
    if _PERSON_WORD.search(b):
        return True
    return any(n and re.search(r"\b" + re.escape(n) + r"\b", b, re.I)
               for n, _ in sheet_lines(sheet))

FORM_HOLD = ", the same object in the same material."

# THE SHOT WHERE THE HARDWARE GOES ON IS NOT A SHOT WHERE IT IS ALREADY ON.
#
# Reported: she was meant to be caught and then restrained, and came out restrained
# and then bolting for the door. The applying shot was being handed the standing hold
# -- "fastened exactly as it was put on, and still fastened at the last frame" -- and
# read at frame 1 that says the cuffs are already closed. So they close first and the
# struggle happens around them, in whatever order is left.
#
# Same fault as a door told it is shut without being told when, and the same fix:
# name both ends. This replaces the standing hold on that one shot; from the next
# shot the latch takes over and the hold is correct, because by then it IS on.
RESTRAINT_GOING_ON = (" The hardware goes on during this shot: it is open and off the "
                      "body at the first frame, and closed on it by the last.")
# WHERE THE LIMBS FINISH, on the shot that stages the fastening.
#
# The clause above says what the HARDWARE does across the shot and says nothing
# about the body, and the anchor was deliberately withheld here on the grounds
# that the author's own words are right beside it. They are -- but they describe
# the ACT, and the next shot does not inherit the act. It inherits the last
# frame. So a shot could close the cuffs with the arms wherever they happened to
# be, and the shot after it opened on a picture of somebody with their arms at
# their sides while the text insisted the wrists were behind the back. Text loses
# to an inherited picture, every time.
#
# Reported as the handcuffs breaking in the next beat. Nothing broke: the frame
# the next shot started from never had them behind her back.
RESTRAINT_ENDS_AT = " By the last frame the {part} are {where}, and stay there."
# The rigid half of CHAIN_HOLD, on its own. Steel is steel while it is being locked
# on, so the applying shot keeps this even though it must not be told the thing is
# already fastened -- dropping it there let the chain go soft for exactly the shot
# that introduces it, which is where a model's idea of the object gets set.
CHAIN_RIGID_TAIL = " Its links keep their size and the run between them stays taut."
# Applying it, as opposed to describing it already worn. The tense is what separates
# them: "Dan cuffs her" stages the act, "her wrists cuffed" and "is handcuffed to the
# rail" describe a state that already holds. Getting that backwards would put "free at
# the first frame" on a woman who has been in cuffs for five shots.
# Nearly every one of these is a noun as well as a verb, and the noun is what a beat
# about restraints is full of: "pulls against the cuffs", "the chains hang", "her
# straps". Read as verbs those turn an ordinary struggling shot into an applying one,
# and it is then told the hardware is off at the first frame -- the exact inversion
# this is here to prevent, on a woman who has been in cuffs for five shots.
#
# A determiner in front is what marks the noun. You do not "the cuffs" anybody.
_A_DETERMINER = (r"(?<!\bthe\s)(?<!\bher\s)(?<!\bhis\s)(?<!\ba\s)(?<!\bmy\s)"
                 r"(?<!\bits\s)(?<!\btheir\s)(?<!\byour\s)(?<!\bthose\s)"
                 r"(?<!\bthese\s)(?<!\bsome\s)(?<!\bboth\s)")
_APPLY_NOW = re.compile(
    _A_DETERMINER +
    r"\b(?:cuffs|handcuffs|chains|ties|binds|locks|straps|tapes|gags|shackles|"
    r"fastens|secures|padlocks|buckles|clamps|clips|snaps|trusses|lashes|wraps|"
    r"cinches|tightens)\b", re.I)
_APPLY_PHRASE = re.compile(
    r"\b(?:put|puts|putting|pull|pulls|pulling|force|forces|forcing|get|gets|"
    r"getting|work|works|snap|snaps)\s+(?:[\w,']+\s+){0,4}?"
    r"(?:on|onto|around|behind|together|shut|closed)\b", re.I)


# WHAT the hardware is, in the author's own words.
#
# Reported: the handcuffs disappeared while she still looked restrained. The holds say
# "every restraint stays whole and closed" and never name the thing, so a shot after
# the one that applied them is told a restraint EXISTS without being told what it is.
# The model renders the consequence -- hands held, restrained posture -- and no object,
# because no object was described.
#
# It only became visible after the sheet stopped listing the item: the sheet was what
# had been naming it in every shot. Taking it off the sheet is right, since the sheet
# put the cuffs in the shots before they went on; naming it here is what that costs.
#
# Ordered LONGEST FIRST inside each start position, so "duct tape gag" is latched
# whole. Matching the bare "gag" out of it made every later shot say "the gag is
# still on her" -- and a gag with no material named is a gag the model draws however
# it likes, which is a strip of tape turning into something else. The material IS the
# object here, the same way the form hold has to say what a thing is made of.
_TAPE = r"(?:duct|gaffer|packing|masking|electrical|parcel)"
_HARDWARE_NOUN = re.compile(
    r"\b(?:(steel|metal|leather|nylon|plastic|padded|heavy|thin|black|chrome|"
    r"ball|ring|bit|rubber|canvas|webbing)\s+)?"
    r"(" + _TAPE + r"\s+tape\s+gags?|tape\s+gags?|" + _TAPE + r"\s+tape|tapes|tape|"
    r"handcuffs|cuffs|manacles|shackles|leg\s+irons|chains|chain|ropes|rope|cords|"
    r"cord|straps|strap|collars|collar|gags|gag|blindfolds|blindfold|"
    r"spreader\s+bars|spreader\s+bar|zip\s+ties|zip\s+tie|cable\s+ties|cable\s+tie)\b",
    re.I)


def hardware_named(text):
    """The hardware this text names, as written. '' when it names none."""
    # The MOST SPECIFIC thing named anywhere in the beat, not the first one. "gags her
    # with duct tape" names the verb before the material, and taking the leftmost gave
    # "gags" -- so every later shot said "the gags are still on her" and the tape, the
    # part that decides what it looks like, was never mentioned again.
    best = ""
    for m in _HARDWARE_NOUN.finditer(text or ""):
        phrase = re.sub(r"\s+", " ", " ".join(g for g in m.groups() if g)).strip()
        if len(phrase) > len(best):
            best = phrase
    if not best:
        return ""
    item = best.lower()
    # "tapes her mouth shut" is the verb, and the thing it leaves behind is tape.
    # Only reached on a shot already read as restrained, so an ordinary "tapes the
    # box shut" never arrives here.
    return "tape" if item == "tapes" else item


def hardware_all_named(text):
    """EVERY piece of hardware this text names, longest phrase per match, in order.

    hardware_named returns one item -- the most specific -- and the caller appended
    that single string to the worn list. So a beat that puts on two things at once,
    which is the ordinary way to write it:

        The guard handcuffs Ana's wrists behind her back and locks a steel collar
        around her neck, chained to the wall.

    recorded the collar and lost the handcuffs. From the next shot on, the cuffs
    were not named in the prompt at all -- not "stays fastened", not mentioned --
    and hardware nobody mentions is hardware the model stops drawing. Reported as
    her breaking out of the handcuffs, which is the model rendering exactly what it
    was told: a woman with a collar and free hands.

    The across-shots case was already fixed -- worn_item used to be overwritten by
    the next shot's item -- and the same bug within a single beat was left.
    """
    out = []
    for m in _HARDWARE_NOUN.finditer(text or ""):
        phrase = re.sub(r"\s+", " ", " ".join(g for g in m.groups() if g)).strip().lower()
        if phrase == "tapes":
            phrase = "tape"
        if not phrase:
            continue
        # A longer phrase naming the same thing replaces the shorter one: "collar"
        # then "steel collar" is one item, described better the second time.
        dupe = next((i for i, p in enumerate(out)
                     if p in phrase or phrase in p), None)
        if dupe is None:
            out.append(phrase)
        elif len(phrase) > len(out[dupe]):
            out[dupe] = phrase
    return out


_UNDO_NOW = re.compile(
    r"\b(?:unlocks?|unlocked|unlocking|uncuffs?|uncuffed|unbinds?|unbound|"
    r"unties?|untied|untying|unbuckles?|unbuckled|unstraps?|unstrapped|"
    r"unclips?|unclipped|unfastens?|unfastened|unshackles?|unshackled|"
    r"ungags?|ungagged|releases?|released|frees?|freed|cuts?\s+(?:off|away|free)|"
    r"slips?\s+off|takes?\s+off|pulls?\s+off|lifts?\s+(?:off|away))\b"
    r"[^.;!?]{0,40}?"
    r"\b(?:cuffs?|handcuffs?|chains?|ropes?|cords?|ties|straps?|tape|gags?|"
    r"collars?|shackles?|clamps?|clips?|restraints?|belt|them|it)\b", re.I)
# ...and the object-first form: "the cuffs come off", "the rope is untied".
_UNDO_PHRASE = re.compile(
    r"\b(?:cuffs?|handcuffs?|chains?|ropes?|cords?|ties|straps?|tape|gags?|"
    r"collars?|shackles?|clamps?|clips?|restraints?)\b\s+"
    r"(?:[\w,']+\s+){0,3}?"
    r"\b(?:come|comes|came|drop|drops|dropped|fall|falls|fell)\s+"
    r"(?:off|away|to\s+the\s+floor|to\s+the\s+ground)\b"
    r"|\b(?:is|are|was|were|gets?|got)\s+"
    r"(?:unlocked|untied|unbound|removed|taken\s+off|cut\s+(?:off|away|free))\b",
    re.I)


def restraint_words(line):
    """The restraint HARDWARE named in one sheet entry, as its own head nouns.

    Used to take hardware out of the sheet when a beat unlocks it: the hold can be
    cleared, but while the entry still lists the cuffs the next shot reads them back
    out of the scene text and latches the hold again."""
    out = []
    for item in re.split(r"[,;.]", str(line or "")):
        item = _LEADING_TAG.sub("", re.sub(r"\s+", " ", item)).strip()
        if not item:
            continue
        head = item.split()[-1].lower().strip("-")
        if head and _RESTRAINT_WORD.match(head) and head not in out:
            out.append(head)
    return out


def restraint_coming_off(beat):
    """Does this beat stage hardware being TAKEN OFF, rather than merely mentioned?

    The hold latches, and it was cleared only by an explicit `remove:` naming the
    hardware -- deliberately, because a beat that does not mention cuffs is not a
    beat that removes them. But auto_remove never puts hardware in `toks` (restraint
    words are filtered out of infer_removals on purpose), so a script that unlocks
    the cuffs IN ITS PROSE and writes no remove: line never cleared the latch: the
    beat said they were unlocked and dropped to the floor, and every shot after went
    on insisting they stay closed and fastened. Reported as the hold still firing
    several shots after the hardware came off.

    Narrow, like the apply patterns it mirrors: an UNDOING verb with the hardware or
    a pronoun as its object. "She looks at the cuffs" or "the key is on the table"
    must not clear a restraint that is still on.
    """
    b = beat or ""
    return bool(_UNDO_NOW.search(b) or _UNDO_PHRASE.search(b))


def restraint_going_on(beat):
    """Does this beat stage hardware being APPLIED, rather than already worn?"""
    b = beat or ""
    return bool(_APPLY_NOW.search(b) or _APPLY_PHRASE.search(b))


# Keep this compact: it is repeated in every shot while restraints remain present.
RESTRAINT_HOLD = (" Every restraint stays closed and fastened as it was put on") + FORM_HOLD


def restraint_wearers(sheet):
    """The people whose own sheet entry describes hardware.

    Read from the entries rather than the beat, because the entry is what says who is
    WEARING it -- a beat can mention a chain without anyone being in it."""
    return [n for n, ln in sheet_lines(sheet) if n and restraint_present(ln)]


# Bound continuity text so it cannot overwhelm the authored beat. Clauses are
# ranked by the caller; the floor preserves essential guards for very short beats.
GUARD_FLOOR_WORDS = 90
GUARD_WORDS_PER_BEAT_WORD = 5


def fit_guards(clauses, beat_words):
    """(kept text, dropped names) for continuity clauses, ranked, within a budget.

    `clauses` is [(priority, name, text)] with 1 the most important. Order in the
    OUTPUT follows the list as given, not the priority -- the ranking decides what
    survives, not where it sits in the sentence."""
    budget = max(GUARD_FLOOR_WORDS, int(beat_words) * GUARD_WORDS_PER_BEAT_WORD)
    spent, keep = 0, set()
    for _, name, text in sorted(clauses, key=lambda c: c[0]):
        if not text:
            continue
        cost = len(text.split())
        if spent + cost > budget and spent > 0:
            continue
        spent += cost
        keep.add(name)
    kept = "".join(t for _, n, t in clauses if n in keep and t)
    dropped = [n for _, n, t in clauses if t and n not in keep]
    return kept, dropped


# PEOPLE THE BEAT STAGES WHO ARE ON NOBODY'S SHEET. Extras: a crowd, dancers,
# other girls, two men at the bar. Plural nouns only, and deliberately not "both",
# "they" or "the two of them" -- those are group cues about the NAMED cast and
# group_beat already owns them. A singular "someone" is not here either: one more
# person is what the cast clause is already counting.
_EXTRA_PEOPLE = re.compile(
    r"\b(?:crowds?|groups?|others|onlookers|bystanders|passers-?by|spectators|"
    r"people|dancers|guests|customers|patrons|strangers|students|staff|tourists|"
    r"girls|women|men|boys|guys|ladies|blondes|brunettes|figures|silhouettes)\b",
    re.I)


# PEOPLE MENTIONED ARE NOT PEOPLE STAGED. "They hear people outside" puts nobody in
# the frame and "the others have gone" says the opposite of staging them -- and both
# used to count, which then stood the body-count clause down and let a random into
# every later shot. A plural noun in an absence or an offscreen phrase is not a crowd.
_NOT_STAGED = re.compile(
    r"\b(?:gone|left|leaving|went|departed|vanished|absent|empty|alone|"
    r"outside|elsewhere|away|upstairs|downstairs|next\s+door|beyond|"
    r"no\s+one|no[- ]?body|none|without|hears?|heard|hearing|listens?|"
    r"remembers?|imagines?|thinks?\s+of|expects?|waits?\s+for)\b", re.I)


# ...and what says they have GONE. The latch below needs an explicit way out, the way
# every other state in this file has one: a garment comes off, a restraint is unlocked,
# a room is left. Without it, extras staged once would suppress the body count for the
# rest of the film even after the script empties the room.
_ALONE = re.compile(
    r"\b(?:alone|by\s+(?:her|him|them)self|on\s+(?:her|his|their)\s+own|"
    r"empty|deserted|to\s+(?:her|him|them)self)\b", re.I)


def extras_in(beat):
    """Does this beat stage people beyond the ones the sheet names, IN the frame?"""
    b = str(beat or "")
    if not _EXTRA_PEOPLE.search(b):
        return False
    return not _NOT_STAGED.search(b)


def extras_dismissed(beat):
    """Does this beat say the people the sheet does not name are no longer there?"""
    b = str(beat or "")
    if _ALONE.search(b):
        return True
    return bool(_EXTRA_PEOPLE.search(b) and _NOT_STAGED.search(b))


# WHO IS IN CONTACT WITH WHOM.
#
# Reported: girls kissing each other when they should be kissing boys. The beat said
# "Mia kisses Dan while Tess kisses Jon" and that is ALL the shot said about it -- one
# sentence among four appearance descriptions, and at cfg 1 the model reads the prompt
# as a bag of words and pairs by its own prior. This file names the owner of a gaze, a
# vocal, a feeling, a posture, a restraint and a body count; contact was the one
# relationship nothing restated.
#
# THE OBJECT HAS TO BE A NAME ON THE SHEET, which is what makes the verb list safe to
# be generous with: "holds the door" and "pulls the chain" name no person and yield no
# pair, so hold, pull, grab and take can all be here without reading furniture as a
# partner.
_CONTACT_SRC = (
    r"kiss(?:es|ed|ing)?|hug(?:s|ged|ging)?|embrac(?:e|es|ed|ing)|"
    r"straddl(?:e|es|ed|ing)|mount(?:s|ed|ing)?|caress(?:es|ed|ing)?|"
    r"strok(?:es|ed|ing)?|cuddl(?:e|es|ed|ing)|hold(?:s|ing)?|held|"
    r"grab(?:s|bed|bing)?|touch(?:es|ed|ing)?|caught|catch(?:es|ing)?|"
    r"pull(?:s|ed|ing)?|take[sn]?|took|taking|push(?:es|ed|ing)?|"
    r"danc(?:e|es|ed|ing)\s+with|lean(?:s|ed|ing)?\s+(?:on|against|into)|"
    r"press(?:es|ed|ing)?\s+(?:against|into)|sit(?:s|ting)?\s+on|"
    r"wraps?\s+(?:her|his|their)\s+arms?\s+around|"
    r"reach(?:es|ed|ing)?\s+for|undress(?:es|ed|ing)?")
_CONTACT_VERB = re.compile(r"(?:" + _CONTACT_SRC + r")", re.I)
# A clause boundary for contact: each pair gets its own, so "A kisses B while C kisses
# D" is read as two pairs rather than one four-way.
_CONTACT_SPLIT = re.compile(r"(?<=[.;!?])\s+|\s+\b(?:while|as|and|then)\b\s+|,\s+", re.I)


def contact_pairs(beat, names):
    """[(who, whom)] the beat puts in physical contact. Two at most.

    Two, like the layering clause: a shot restating four pairings has stopped being
    about its beat."""
    b = _DIALOGUE_TAG.sub(" ", _QUOTED.sub(" ", str(beat or "")))
    people = [n for n in (names or []) if n]
    out = []
    for part in _CONTACT_SPLIT.split(b):
        for a in people:
            m = re.search(r"\b" + re.escape(a) + r"\b\s+(?:\w+\s+){0,2}?(?:"
                          + _CONTACT_SRC + r")\b", part, re.I)
            if not m:
                continue
            tail = part[m.end():]
            for c in people:
                if c == a:
                    continue
                if re.match(r"\W{0,14}(?:the\s+)?" + re.escape(c) + r"\b", tail, re.I):
                    if not any({a, c} == set(p) for p in out):
                        out.append((a, c))
                    break
        if len(out) >= 2:
            break
    return out[:2]


def contact_hold(pairs):
    """Say which body is with which. "" when the beat pairs nobody.

    Positive, like every other clause here: it says what the pairing IS, never that
    anybody is not paired. Naming both sides is the point -- an unnamed "they kiss" in
    a shot with four people is the sentence that let the model choose."""
    ps = [(a, b) for a, b in (pairs or []) if a and b]
    if not ps:
        return ""
    if len(ps) == 1:
        return f" The contact is {ps[0][0]} with {ps[0][1]}: those two bodies together."
    return (f" The contact is {ps[0][0]} with {ps[0][1]}, and {ps[1][0]} with "
            f"{ps[1][1]}: two pairs, each body with its own partner.")


def cast_hold(names, beat="", extras=False):
    """A positive body-count constraint for a one- or two-person composition.

    THE SOLO SHOT HAD NO COUNT AT ALL, and a solo shot is where a duplicate of the
    one person in it has nothing standing against it. The pair case has been asserted
    since this function was written; the one-person case returned "" with no recorded
    reason for it, so the shot most at risk of being rendered as twins was the shot
    that said nothing about how many bodies were in it. Reported, repeatedly, as
    duplicate characters.

    STANDS DOWN WHERE THE BEAT STAGES EXTRAS. "There are two people in the shot,
    with one body for each person" is exactly right against a duplicated character
    and exactly wrong against "two women dance behind them": it forbids, as a
    positive fact, the people the author just asked for. Reported as extras refusing
    to appear. The author's words outrank anything inferred from them, which is this
    file's standing rule, so a beat that puts more bodies in the frame keeps them and
    the count goes unsaid."""
    people = list(dict.fromkeys(n for n in (names or []) if n))
    # `extras` is kept as a parameter so a caller can stand the count down explicitly.
    # It is no longer LATCHED for the film: one plural word anywhere -- "the others
    # have gone" included -- then silenced the count on every shot that followed, and
    # this is the clause that keeps a duplicate or a stranger out of the frame.
    if extras or extras_in(beat):
        return ""
    if len(people) == 1:
        return " There is one person in the shot: one body, one face."
    if len(people) == 2:
        return " There are two people in the shot, with one body for each person."
    # THREE OR MORE IS LEFT ALONE, as it always has been. The count comes from the
    # cast this file decided is in the shot, and the more people that decision holds
    # the likelier one of them is described without being in frame -- an assertion
    # that there are four bodies is then a request for a fourth. One and two are the
    # counts the duplicate reports are about.
    return ""


def restrained_by_beat(beat, cast):
    """Who this beat puts in the hardware. The agent is not the one wearing it.

    `restrained` was a film-level latch: once anything was on anybody, every later
    shot got the hold. So a shot describing only the man who applied it was told
    there were cuffs holding wrists behind a back -- with nobody in the text those
    wrists could belong to. The model has to draw the person the sentence describes,
    so it invents one. That is the duplicate.

    One person in the shot is the one wearing it. Two or more and the first named is
    the one doing it, which is how these beats are written: "Dan walks in and cuffs
    her wrists"."""
    people = [n for n in (cast or []) if n]
    if len(people) <= 1:
        return set(people)
    # SPOKEN NAMES ARE NOT STAGED ONES, here for the same reason as in
    # sheet_for_beat: "Dan says: 'McKenna, put the cuffs on'" names McKenna in
    # dialogue only, and taking that as her being in the shot describes hardware
    # on somebody the text never put in the room -- which is exactly how a second
    # figure gets invented to own it.
    b = _outside_speech(beat or "")
    # The agent is whoever is named nearest BEFORE the applying verb, not whoever is
    # named first. "Mara runs for the door. Dan catches her and cuffs her wrists"
    # opens on the person being cuffed, and reading the first name as the agent put
    # the hardware on the wrong one -- which then silenced the hold in every shot she
    # was in, because the node thought she was not wearing anything.
    verb = None
    for pat in (_APPLY_NOW, _APPLY_PHRASE):
        for m in pat.finditer(b):
            verb = m.start() if verb is None else min(verb, m.start())
    if verb is None:
        return set(people)
    agent, at = None, -1
    for n in people:
        for m in re.finditer(r"\b" + re.escape(n) + r"\b", b, re.I):
            if at < m.start() < verb:
                agent, at = n, m.start()
    # No name in front of it -- "she is cuffed to the rail" -- so nothing here says
    # who is doing it. Everybody stays a candidate rather than nobody: a hold that
    # fires when it need not is a wasted sentence, one that fails to fire is hardware
    # that stops being described.
    return {n for n in people if n != agent} if agent else set(people)


# WHICH PART THE ANCHOR HOLDS. The clause used to say "holding the wrists" whatever
# the hardware was, so a steel collar chained to a wall came out as wrists held at
# the wall -- which describes a different restraint entirely, and leaves the neck
# free in the one shot whose point is that it is not. A model given wrists at the
# wall and a collar on the neck has two restraints to draw and reason to drop one.
_HELD_PART = (
    (r"\b(?:collars?|leash(?:es)?|leads?|chokers?|neck\s*(?:chain|iron)s?)\b", "neck"),
    (r"\b(?:leg\s*irons?|ankle\s*(?:cuffs?|chains?|straps?)|hobbles?|"
     r"shackles?)\b", "ankles"),
    (r"\b(?:harness(?:es)?|body\s*belts?)\b", "body"),
    (r"\b(?:waist\s*(?:chain|belt)s?)\b", "waist"),
)


def held_part(items):
    """The body part an anchored restraint holds, read from the hardware itself."""
    text = " ".join(items or [])
    for pat, part in _HELD_PART:
        if re.search(pat, text, re.I):
            return part
    return "wrists"          # cuffs, rope and tape, which is the common case


# THE POSE A LIMB POSITION MAKES, as a body rather than as a relation. Buried in
# the hardware sentence as "holding the wrists behind the back" it was reported as
# the wrists rendering in front on the next beat: the fact was there, in every
# shot, and it was a subordinate clause in the middle of thirty words about the
# metal. A pose is drawn from arms and shoulders.
# SHORT. At 28 words this outbid FALL_HOLD and the budget dropped the fall guard
# -- which exists because a fall grew a third leg to brace a landing nothing in
# the text was taking. Trading one reported bug for another is not a fix. Arms and
# wrists make the pose renderable; elbows, shoulders and chest were decoration.
_POSE_OF_POSITION = {
    "behind the back": ("Both arms are behind the body, wrists together at the "
                        "small of the back"),
    "above the head": ("Both arms are raised, wrists together above the head, "
                       "the body stretched long"),
    "in front of the body": ("Both arms are in front of the body, wrists "
                             "together at the waist"),
    "out to the sides": ("Both arms are held out level with the shoulders, one "
                         "hand to each side"),
    "at the waist": "Both arms are at the sides, wrists together at the waist",
}


# A BODY LYING DOWN NEEDS SOMETHING UNDER IT, and if the text does not say what,
# the model picks -- and what it picks for somebody on their side is the arm it has
# seen under every other body on its side: propped on the elbow, forearm out front.
# That is a hand in front of the body, which is the one place these wrists cannot be.
#
# Reported as her arm supporting her while the cuffs were meant to be holding her
# hands behind her back. The pose clause was already on that shot saying both arms
# are behind -- being told where the arms ARE does not settle what is BEARING THE
# WEIGHT, and between an arm it can see a use for and a sentence about wrists, the
# picture went with the arm.
#
# So name the contact. Positively, like everything else here: at cfg 1 nothing is
# negated, and "no arm under her" is the word "arm" next to the word "under". The
# shoulder and hip are what a bound body on its side actually rests on, and a
# shoulder taking the weight is an elbow with nothing to do.
#
# Only for wrists BEHIND THE BACK. Hands in front or above the head can prop a body
# up and it is not wrong that they do, so a clause forbidding it there would be
# taking away a shape the author may have wanted.
POSE_LYING_WEIGHT = "The shoulder and the hip take the weight of the body"


def pose_clause(position, lying=False):
    """One sentence describing the BODY a limb position makes. "" when unknown.

    `lying` adds what is under it -- see POSE_LYING_WEIGHT."""
    key = str(position or "").strip().lower()
    said = _POSE_OF_POSITION.get(key, "")
    if not said:
        return ""
    if lying and key == "behind the back":
        said = f"{said}. {POSE_LYING_WEIGHT}"
    return f" {said}."


def restraint_sentence(item, wearers, described, anchor="", rigid=False, posed=False,
                       part=""):
    """ONE sentence for the hardware: what it is, that it is closed, and where it holds.

    These used to be three, written at three different times for three different bug
    reports, and each of them names the same object again:

        Every restraint stays closed and fastened as it was put on, ... (29 w)
        The cuffs are still on her, in plain sight where they were put. (13 w)
        The fastened wrists stay behind the back, where they were locked. (11 w)

    53 words about one pair of handcuffs, beside a nine-word beat. Measured on a real
    scene the guards had reached 65% of the shot against a 12% beat -- the number this
    node was rebuilt to escape, arrived at again by adding a clause per report with no
    budget on the total. Merged, the same facts cost 25.

    Every guarantee survives: the thing is named so it gets drawn, it is closed, it is
    the same object in the same material, and it is where it was fastened."""
    # More than one piece of hardware reads as a list, and a list is plural however
    # its last word ends: "The cuffs, duct tape stays closed" was what a comma-joined
    # subject produced before this.
    items = [i.strip() for i in (item or "").split(",") if i.strip()]
    if len(items) > 1:
        item = ", ".join(items[:-1]) + " and " + items[-1]
        plural = True
    else:
        plural = bool(item) and item.endswith("s") and not item.endswith("ss")
    who = ""
    if wearers and len(described) >= 2:
        who = (wearers[0] if len(wearers) == 1
               else ", ".join(wearers[:-1]) + " and " + wearers[-1])
    if item:
        subject = f"The {item} on {who}" if who else f"The {item}"
        verb = "stay" if plural else "stays"
    else:
        subject = f"Every restraint on {who}" if who else "Every restraint"
        verb = "stays"
    it, was = ("they", "were") if plural else ("it", "was")
    # Rope is TIED. It is not closed and it is not fastened, and saying so of a cord
    # describes a mechanism that is not there -- the same class of error as telling a
    # strip of tape it sits in the mouth. Hardware closes; soft goods hold.
    # ALL of it, not any of it. Cuffs and tape together are still cuffs, and steel
    # that is only "tied and holding" is steel nobody has said is closed.
    _soft_word = re.compile(r"\b(?:rope|ropes|cord|cords|twine|string|strap|straps|"
                            r"tape|scarf|belt|stocking|stockings|zip\s*ties?|"
                            r"cable\s*ties?|laces?)\b", re.I)
    soft = bool(items) and all(_soft_word.search(i) for i in items)
    shut = "tied and holding as" if soft else "closed and fastened as"
    out = f" {subject} {verb} {shut} {it} {was} put on"
    if anchor:
        # `part` is passed in because the item NAME is dropped from this sentence
        # whenever the beat already says it -- and with the name went the only clue
        # to which part is held, so a collar the beat had just named came back
        # holding the wrists. The latch still knows what is on; ask it, not the
        # sentence being written.
        #
        # TWO RESTRAINTS, TWO ANCHORS. limb_anchor merges a limb POSITION with a
        # fixed POINT into one string, and with cuffs behind the back and a collar
        # chained to a wall that came out as "holding the neck behind the back, at
        # the wall" -- a neck behind a back, which is not a thing, in the sentence
        # whose whole job is to say plainly what is holding what. The position
        # always belongs to the wrists; the point belongs to whatever is chained.
        _m = re.match(r"^(.*?),?\s*(at the .+)$", anchor)
        _pos, _point = (_m.group(1).strip(), _m.group(2)) if _m else (anchor, "")
        _part = part or held_part(items)
        # The limb POSITION leaves this sentence and gets one of its own, in
        # pose_clause -- buried here it was the least prominent thing in thirty
        # words about the metal, and it was reported as the wrists rendering in
        # front. What stays is the anchor POINT, which is about the hardware and
        # belongs with it.
        # The PART and the POINT both stay: a collar holds the neck and the chain
        # holds it to the wall, and dropping either leaves a shot that does not
        # say what is attached to what. Naming the ITEM again here was worse than
        # both -- "The steel collar stays closed and fastened, the steel collar
        # fast at the wall" -- so the part carries it.
        if _point:
            out += f", holding the {_part} fast {_point}"
        elif not _pos:
            out += f", holding the {_part}"
    if posed:
        out += ("; the metal is already drawn to its full length, so the position it "
                "fixes is the position that keeps, and the body strains against it "
                "while the fastenings hold")
    elif rigid:
        out += (f", {'their' if plural else 'its'} links keeping their size and the run "
                f"between them taut")
    out += f", the same object in the same material."
    if who:
        out += " Everyone else in the shot has on exactly what their own entry lists."
    return out


def own_body(clause, who, described):
    """Say WHOSE body a bare-skin clause is about, when more than one is described.

    "Everything worn comes off during this shot" and "The legs are bare from the
    hip down" name nobody. With one person in the shot that is unambiguous; with
    two it is an instruction about whoever is on screen, and the second character
    undresses alongside the first. Reported as one character mimicking the other's
    actions -- and it is the same defect own_hold was written for, in the clause
    next door.

    Positively phrased, like own_hold: naming whose body it is excludes everyone
    else, where "nobody else undresses" asks the model to render an absence. The
    other people are pinned to their own entries in one short sentence rather than
    named individually, which costs a second mention of each."""
    if not clause or not who or len(described or []) < 2:
        return clause
    names = [n for n in (who if isinstance(who, (list, tuple)) else [who]) if n]
    if not names:
        return clause
    subject = names[0] if len(names) == 1 else \
        ", ".join(names[:-1]) + " and " + names[-1]
    body = clause.strip()
    # "The legs are bare" -> "McKenna's legs are bare". "Everything worn comes off"
    # -> "Everything McKenna is wearing comes off".
    body = re.sub(r"^The\s+", f"{subject}'s ", body)
    body = re.sub(r"^Everything worn\b", f"Everything {subject} is wearing", body)
    return (" " + body
            + f" Everyone else in the shot keeps on exactly what their own entry "
              f"lists.")


def own_hold(hold, wearers, described):
    """Attribute a hold to whoever actually wears the hardware.

    The holds say "every restraint stays fastened" and name nobody, which was fine
    while a shot meant one person. Put a second person in the frame and it becomes an
    instruction about whoever is on screen: the belt locked onto one character turned
    up on the other, over their clothes, because the sentence never said whose it was.

    Only when the shot describes more than one person -- with one there is no
    ambiguity, and the extra words are shot budget spent on nothing. Positively
    phrased: saying who wears it is what excludes everyone else, where "nobody else
    is wearing one" asks the model to render an absence."""
    if not hold or not wearers or len(described) < 2:
        return hold

    def _and(names):
        return names[0] if len(names) == 1 else \
            ", ".join(names[:-1]) + " and " + names[-1]

    who = _and(wearers)
    # ONE naming, not two. This used to add "The hardware is X's, worn on the body it
    # was locked to" on top of rewriting the clause to "Every restraint on X" -- which
    # says the same thing twice and costs a second mention of X in the shot.
    #
    # Reported as a second girl appearing at the moment of cuffing. A described person
    # is a person the model draws; that is the whole basis of character_guard, and it
    # does not stop applying because the describing sentence is a continuity guard.
    # The dropped sentence also put a bare "the body" into the text, unattached to
    # anybody, in the one shot where a second figure was turning up.
    #
    # What is KEPT is the half that does work the rewrite cannot: excluding everyone
    # else. That is what stopped one character's hardware appearing on another.
    tail = " Everyone else in the shot has on exactly what their own entry lists."
    return hold.replace("Every restraint", f"Every restraint on {who}", 1).rstrip() + tail

# Hardware that means restraint on its own.
_RESTRAINT_PLAIN = re.compile(
    r"\b(?:handcuff(?:s|ed)?|cuffed|shackle[sd]?|manacle[sd]?|hogtied|hog-?tied|"
    r"hogcuffed|hog-?cuffed|gag(?:ged|s)?|blindfold(?:ed|s)?|zip[- ]ties?|"
    r"cable[- ]ties?|restrain(?:t|ts|ed)|bound|bindings?|straitjacket|"
    r"collared|leashed|tethered|manacled|fettered|chained\s+up|"
    # PARTICIPLES are unambiguous and are not in the noun list, so they cannot
    # satisfy both halves of the MAYBE rule by themselves. "Ana is collared and
    # chained to the wall" matched nothing at all before this: "collared" is not
    # "collars?", so the noun half failed and the whole latch stayed down.
    #
    # A COLLAR MADE OF HARDWARE. Bare "collar" is genuinely ambiguous -- a shirt
    # has one -- so it needs a body part beside it, and "a collar at her throat"
    # supplies that. But a sheet entry reading "green dress, steel collar" has
    # neither a body part nor a binding verb anywhere in the line, so it was not
    # a restraint at all: no hold ever fired for it, and hardware nobody holds is
    # hardware the model drops. Reported as the collar missing from her neck.
    #
    # The material settles it. A shirt's collar is stiff or starched; it is not
    # steel and it does not lock, so "white shirt with a stiff collar" still
    # reads as clothing -- which is the distinction worth keeping.
    r"(?:steel|iron|metal|chrome|brass|leather|padded|locked|lockable|heavy|"
    r"thick|studded|spiked|posture|shock|bondage|slave)\s+collars?|"
    r"collars?\s+(?:and|with)\s+(?:a\s+)?(?:lock|padlock|leash|lead|chain|ring)|"
    r"spreader bar)\b", re.I)
# Hardware that is only a restraint in context -- a chain-link fence, a rope on a
# boat and a leather belt are none of the node's business.
# A clamp belongs here rather than in the list above: clamped to a bench it is a
# tool, clamped to a body it is hardware, and only the context tells them apart.
_RESTRAINT_MAYBE = re.compile(
    r"\b(?:chains?|ropes?|cords?|cuffs?|straps?|collars?|tapes?|taped|taping|"
    r"belts?|harness|hobble|clamps?|clips?)\b", re.I)
# VERB forms only. An earlier version listed "chain" and "cuff" here as well as in
# the noun list, so a chain-link fence matched both halves and armed the rule.
_BINDING_VERB = re.compile(
    r"\b(?:cuffed|chained|tied|tying|bound|binds?|binding|locked|locks|"
    r"strapped|taped|taping|gagged|shackled|fastened|fastens|secured|secures|"
    r"padlocked|trussed|lashed|wrapped|clamped|clamping|clipped|clipping|"
    r"pinned|attached|affixed)\b", re.I)
# NOTE the bare "clamps" and "clips" are deliberately absent above while "clamp" and
# "clip" are in the noun list. A word in BOTH lists satisfies both halves of the rule
# by itself, which is how "clamps the board to the workbench" armed the restraint
# hold -- the same way a chain-link fence did before "chain" was taken out of the
# verbs. Same reason "tapes" is a noun here and only "taped"/"taping" are verbs.
# _BODY_PART used to be defined twice at module level, here and again further down.
# Both readers sit below the second one, so the second has always been the one in
# force and this was dead -- but it read as the live definition from up here, and the
# restraint check below was written against this narrower vocabulary. Removed rather
# than merged: merging would change which shots read as restrained, and that is a
# behaviour change wearing a tidy-up's clothes.


# A turn shows a surface the shot has never shown. The keyframe pins the FRONT, so
# once the body rotates the model is filling in from its prior -- and its prior for
# an undescribed body is a CLOTHED one. That is a removed garment coming back, often
# stacked in the wrong order because nothing said which layer was where, and hardware
# on the far side being re-invented as it rotates into view.
#
# One sentence, only on shots that turn, and only once there is state worth holding.
# It names no garment and no person, so it summons neither.
TURN_HOLD = (" What is on the body now is all that is on it, front, side and behind, and "
             "whatever is fastened stays fastened and closed as the view comes round.")

_TURN_CUE = re.compile(
    r"\b(?:turn(?:s|ed|ing)?|rotat(?:es?|ed|ing)|spin(?:s|ning)?|swivel(?:s|led)?|"
    r"roll(?:s|ed|ing)?\s+(?:over|onto)|faces?\s+away|face[sd]?\s+the\s+other|"
    r"over\s+(?:her|his|their)\s+shoulder|from\s+behind|back\s+to\s+the\s+camera|"
    r"shows?\s+(?:her|his|their)\s+back|other\s+side)\b", re.I)


# Being MOVED does the same damage as turning, for the same reason: the keyframe
# pinned one pose seen from one side, and lifting, dragging or rolling someone puts
# the body somewhere that frame never showed. The verb needs a PERSON as its object
# -- "lifts her onto the table" moves her, "lifts the crate" does not, and
# "positions her legs" moves a limb, not the body.
_MOVE_VERB = re.compile(
    r"\b(?:lifts?|lifted|carr(?:ies|ied)|drags?|dragged|hauls?|hauled|hoists?|hoisted|"
    r"picks?\s+up|picked\s+up|sets?\s+down|set\s+down|lays?|laid|"
    r"lowers?|lowered|rolls?|rolled|flips?|flipped|props?|propped|"
    r"moves?|moved|repositions?|repositioned|pulls?|pulled|pushes|pushed|"
    r"shoves?|shoved|throws?|threw|drops?|dropped|turns?|turned)\s+", re.I)
_PERSON_OBJ = r"(?:the\s+|a\s+)?(?:her|him|them"


def body_moved(text, names=()):
    """Is a PERSON being moved in this beat, rather than an object or a limb?"""
    toks = [re.escape(n) for n in (names or []) if n]
    obj = re.compile(_PERSON_OBJ + (("|" + "|".join(toks)) if toks else "") + r")\b"
                     # ...not a possessive, and not a LIMB: "positions her legs" moves
                     # the legs, not the body. An earlier guard rejected any following
                     # word ending in "s", which threw out "drags her across the floor".
                     r"(?!\s*['’]s)"
                     r"(?!\s+(?:legs?|arms?|wrists?|ankles?|hands?|feet|foot|head|hair|"
                     r"hips?|shoulders?|knees?|elbows?|thighs?|face|chin)\b)"
                     # A moved BODY goes somewhere: the object is followed by a word
                     # of motion, or the clause simply ends. Without this, "pulls her
                     # shorts off" reads as moving her rather than the shorts.
                     r"(?=\s*(?:[.,;!?]|$)"
                     r"|\s+(?:onto|into|on|in|to|across|down|up|over|under|back|out|"
                     r"away|upright|off|against|toward|towards|through|round|around|"
                     r"beside|behind|clear)\b)", re.I)
    return any(obj.match(text[m.end():]) for m in _MOVE_VERB.finditer(text or ""))


def turns_in(text, names=()):
    """Does this beat rotate a body, move one, or bring the view around it?"""
    return bool(_TURN_CUE.search(text or "")) or body_moved(text, names)


# A falling body's reflex is to put its hands out. When the hands are fastened, the
# model has to resolve that conflict, and the cheapest resolution is to free them --
# which renders as the cuffs opening or the chain snapping mid-fall. Nothing in the
# restraint hold covers it, because the hold says the hardware is whole and says
# nothing about what the body does on the way down.
#
# So say what DOES take the landing. Positive, and it names no person: at cfg 1
# there is no negative prompt, and "does not catch itself" names catching.
FALL_HOLD = (" A bound body falls as one piece: the fastened limbs stay fastened and travel "
             "with it, the arms staying in the hold, the shoulder, hip or side takes "
             "the landing, and the legs fold together under the body.")

# The same shot without the hardware. A falling body is the frame where limbs are
# least determined -- fast motion, heavy occlusion, and a pose the model has to invent
# the middle of -- and the reported result is a third leg, grown to brace a landing
# nothing else was taking.
#
# Said as what the limbs DO, never as how many there are. Counting was tried in this
# node's first life and removed -- the old subject-counting sentence is one of the
# phrases test_verbatim still bans by name. A count is also a mention, and a mention
# is a presence cue: naming legs to ask for two of them is a way of asking for legs.
# Giving them a definite job is what stops the model inventing one.
FALL_HOLD_FREE = (" The body falls as one piece: the arms stay with it and the shoulder, "
                  "hip or side takes the landing, the legs folding together under it.")

_FALL_CUE = re.compile(
    r"\b(?:falls?|fell|falling|drops?\s+to|dropped\s+to|collapse[sd]?|collapsing|"
    r"topple[sd]?|topples|tips?\s+over|tipped\s+over|keels?\s+over|goes\s+down|"
    r"went\s+down|slumps?|slumped|stumbles?|stumbled|overbalance[sd]?|"
    r"loses?\s+(?:her|his|their)\s+balance|lost\s+(?:her|his|their)\s+balance|"
    # ...and being put down by someone else: "pushes her over", "knocked him down".
    #
    # WHAT GOES DOWN HAS TO BE A PERSON. The object here used to be optional, so the
    # verb and the direction could sit straight against each other -- and "pulls down
    # her shorts" is a verb and a direction. Every undressing beat written that way
    # was read as a body being put on the floor, and told what takes the landing and
    # how the legs fold. She stands up to take her shorts off and the shot drops her.
    #
    # Two shapes, both requiring more than the bare pair: somebody named and then the
    # direction, or a destination explicit enough to be nothing else ("pushed to the
    # floor"), which is how the passive gets in without an object.
    r"(?:push|knock|shove|pull|drag|throw|thr[eo]w)(?:es|s|ed|n)?\s+"
    r"(?:(?:her|him|them|herself|himself|themselves|[A-Z][\w-]+)\s+"
    r"(?:over|down|to\s+the\s+(?:floor|ground))|to\s+the\s+(?:floor|ground))|"
    r"hits?\s+the\s+(?:floor|ground|deck))\b", re.I)


# What can go down WITHOUT being a body. A garment let go of falls, and so does
# anything else the beat is holding -- and the fall guard exists to tell a shot what
# takes the landing and how the legs fold, so aiming it at a belt puts the person
# on the floor instead. Reported exactly that way: he took the belt off, it dropped
# to the ground, and she fell with it.
_OBJECT_FALLER = re.compile(
    r"\b(?:it|its|belt|belts|top|tops|shirt|shorts|jeans|trousers|skirt|dress|"
    r"coat|jacket|jumper|sweater|scarf|tie|boot|boots|shoe|shoes|sock|socks|"
    r"glove|gloves|hat|bag|towel|sheet|blanket|cuffs?|handcuffs?|chain|chains|"
    r"rope|ropes|tape|gag|collar|key|keys|phone|glass|bottle|cup|plate|book|"
    r"clothes|clothing|garment|garments|thing|things)\b", re.I)
# A person going down. A NAME, or a personal pronoun that is not "it".
_PERSON_FALLER = re.compile(
    r"\b(?:she|he|they|her|him|them|herself|himself|themselves|"
    r"[A-Z][\w-]{1,24})\b")


def falls_in(text):
    """Does a BODY go down in this beat? A dropped garment is not a fall.

    The fall guard tells the shot what takes the landing and what the legs do, so a
    match on something that is not a person aims all of that at the wrong subject
    and the shot puts a body on the floor to satisfy it.

    The subject is whatever sits between the start of the clause and the verb. An
    object there -- "it drops to the ground", "the belt falls to the floor" -- is
    the thing being let go of, not somebody going down."""
    t = text or ""
    for m in _FALL_CUE.finditer(t):
        # Back to the start of this clause: a subject does not reach across a full
        # stop, nor across a comma or conjunction joining two predicates.
        head = t[:m.start()]
        cut = max((c.end() for c in
                   re.finditer(r"[.;!?]\s+|,\s*|\s+(?:and|but|then|so)\s+", head)),
                  default=0)
        subject = head[cut:]
        if _OBJECT_FALLER.search(subject):
            continue                      # a thing came down, not a person
        if not subject.strip() or _PERSON_FALLER.search(subject):
            return True
        # Nothing recognisable as a subject: the passive and destination-only forms
        # ("pushed to the floor") are already narrow enough to mean a body.
        return True
    return False


# Steel does not behave like rope. A model with no reason to think otherwise draws a
# chain as a soft cord: it sags, stretches to wherever a limb is going, and lets the
# body move as if nothing were fastened. The restraint hold says the hardware stays
# WHOLE; it says nothing about how it behaves while whole.
#
# Positive and impersonal, like the other holds -- at cfg 1 there is no negative
# prompt, so "does not stretch" only names stretching.
# REPLACES the restraint hold rather than joining it -- the two said "stays whole and
# closed" twice, and two clauses saying the same thing is twice the stasis for one
# guarantee.
CHAIN_HOLD = (" Every restraint stays closed and fastened as it was put on, its links "
              "keeping their size and the run between them taut") + FORM_HOLD

# When hardware is what PUTS a body in a position, the length of that hardware is the
# whole reason the position holds. Saying the metal keeps its shape is not enough: a
# chain that keeps its shape can still be drawn as having slack, and slack is room to
# stand up out of a squat the chain was locked to enforce.
#
# It replaces the clause above rather than joining it, and it is careful to leave the
# body free to act: straining and pulling is exactly what should happen, and the last
# thing this should say is that anything holds still.
CHAIN_POSE_HOLD = (" Every restraint stays closed and fastened as it was put on; the metal "
                   "is already drawn to its full length, so the position it fixes is the "
                   "position that keeps, and the body strains against it while the "
                   "fastenings hold") + FORM_HOLD

# A position that hardware can be locked to enforce.
_FORCED_POSE = re.compile(
    r"\b(?:squat(?:s|ting|ted)?|kneel(?:s|ing)?|knelt|crouch(?:es|ing|ed)?|"
    r"hogtied|hog-?tied|hogcuffed|hog-?cuffed|trussed|"
    r"bent\s+(?:over|double)|doubled\s+over|folded\s+(?:up|forward)|"
    r"spread[- ]eagled?|curled\s+up|"
    r"on\s+(?:her|his|their)\s+(?:knees|haunches))\b", re.I)


# WHERE the fastened limbs are held. Distinct from _FORCED_POSE, which is what the
# whole body is doing -- kneeling, hogtied, bent over. Cuffed wrists above the head is
# not a pose in that sense: the body can be standing, sitting or lying and the arms are
# still fixed at one point.
#
# Reported: cuffs above the head in one shot, somewhere else in the next. The restraint
# hold kept them shut and said nothing about where they were, so the only thing
# carrying the position was the picture -- and the picture is the previous shot's last
# frame, which a close shot crops the anchor point straight out of. Text is the only
# thing that survives a tight frame.
# What makes a phrase describe a BODY rather than the room it is in: a limb, or a
# word for fastening one. Every entry below requires one of these within a few
# words of the position, because without it the table reads the set dressing --
# "one bulb overhead" put the wrists above the head, "crates stacked to the sides"
# put the arms out to the sides, and "her legs spread wide" moved the arms to
# wherever the legs were. limb_anchor takes the FIRST pattern that matches, so an
# unguarded entry does not merely add a wrong reading, it outranks the right one
# written in the same sentence.
_LIMB_EV = (r"\b(?:hands?|wrists?|arms?|cuffed|handcuffed|bound|tied|shackled|"
            r"manacled|strapped|secured|fastened|locked|pinned|chained|clasped|"
            r"held|clipped|hooked)")

_LIMB_ANCHOR = (
    # EVERY FORM HERE CARRIES ITS OWN EVIDENCE, the same rule the "behind" entries
    # below already follow. It did not, and bare "overhead" and "stretched up" are
    # scenery far more often than they are limbs: "one bulb overhead", "strip lights
    # overhead", "the cable is stretched up the wall". limb_anchor takes the FIRST
    # pattern that matches, and this is the first, so a light fitting in the scene
    # line beat the wrists written in the same sentence -- a woman cuffed behind her
    # back was told, in every shot, that both arms were raised above her head.
    # Reported as the cuffs breaking and the arms coming round to the front, which is
    # what a model does when the pose it is given contradicts the hardware.
    (r"(?:cuffed|handcuffed|bound|tied|shackled|manacled|strapped|secured|"
     r"fastened|locked|pinned|chained|clipped|hooked|suspended|hoisted)\s+"
     r"(?:\w+\s+){0,3}?(?:above|over)\s+(?:her|his|their|the)\s+head|"
     r"(?:hands?|wrists?|arms?)\s+(?:\w+\s+){0,4}?"
     r"(?:above|over)\s+(?:her|his|their|the)\s+head|"
     r"(?:hands?|wrists?|arms?)\s+(?:\w+\s+){0,3}?overhead|"
     r"(?:hands?|wrists?|arms?)\s+(?:\w+\s+){0,2}?stretched\s+(?:up|upward)",
     "above the head"),
    (_LIMB_EV + r"\s+(?:\w+\s+){0,3}?behind\s+(?:her|his|their|the)\s+back",
     "behind the back"),
    # THE SAME PLACE, WRITTEN THE WAYS PEOPLE WRITE IT. The line above needs the
    # literal word "back" after the possessive, so every one of these recorded
    # NOTHING -- and nothing here is not a smaller clause, it is pose_clause
    # returning "" and the shot never being told where the wrists are at all.
    # Reported as her hands simply not being bound together behind her.
    #
    #   "cuffed behind her"                  -- the back is implied and not typed
    #   "at the small of her back"            -- which is the phrase THIS NODE
    #                                            prints back in its own pose clause
    #   "hands behind back"                   -- no possessive, as stage directions
    #                                            are written
    #
    # "behind her" cannot be matched on its own: limb_anchor only runs on a shot
    # already holding a restraint, and in one of those "Dan stands behind her" is an
    # ordinary sentence that would anchor her wrists to his position. So each form
    # below carries its own evidence -- a limb, or a fastening participle, within a
    # few words of it.
    (r"(?:hands?|wrists?|arms?)\s+(?:\w+\s+){0,3}?behind\s+(?:her|his|their)\b",
     "behind the back"),
    (r"(?:cuffed|handcuffed|bound|tied|shackled|manacled|strapped|secured|"
     r"fastened|locked|pinned|clasped|held)\s+(?:\w+\s+){0,2}?"
     r"behind\s+(?:her|his|their)\b", "behind the back"),
    (r"at\s+the\s+small\s+of\s+(?:her|his|their|the)\s+back", "behind the back"),
    (r"\b(?:hands?|wrists?|arms?)\s+behind\s+back\b", "behind the back"),
    (_LIMB_EV + r"\s+(?:\w+\s+){0,3}?in\s+front\s+of\s+(?:her|his|their)\s+"
     r"(?:body|chest|waist)", "in front of the body"),
    # "Her legs spread wide" was anchoring her ARMS out to the sides, and
    # "crates stacked to the sides" did the same from the scenery. Legs are not
    # arms and a crate is not a limb.
    (_LIMB_EV + r"\s+(?:\w+\s+){0,3}?(?:(?:out\s+)?to\s+the\s+sides?|spread\s+wide)",
     "out to the sides"),
    (_LIMB_EV + r"\s+(?:\w+\s+){0,3}?at\s+(?:her|his|their|the)\s+waist",
     "at the waist"),
)
# What they are fastened TO. Named separately because a shot can state one, the other,
# or both, and the clause reads correctly with whichever it has.
#
# THE VERB IS REQUIRED, and it was not. "to the <noun>" alone read any movement as a
# fastening: "he walks to the table" came back anchored at the table and "she is
# dragged to the bed" anchored at the bed. That was survivable only because the noun
# list was short enough to miss most sentences -- and adding the missing nouns below
# without this would have made "she sinks to the floor" a chain.
#
# VERB FORMS ONLY -- the same rule _BINDING_VERB already documents, and the first
# version of this broke it. Written as bare stems with an optional suffix, "chain",
# "rope", "clip", "lock" and "bolt" are all NOUNS as well, so the pattern found its
# own hardware and called it a fastening: "she drops the rope to the floor" anchored
# at the floor, "the clip fell to the floor" anchored at the floor, and with the
# restraint gate now leaning on this, each of them latched a restraint hold over
# hardware lying on the ground for the rest of the film.
_FASTEN_PART = (r"(?:chained|cuffed|handcuffed|shackled|manacled|locked|padlocked|"
                r"fastened|secured|tethered|bound|tied|strapped|clipped|hooked|"
                r"bolted|attached|anchored|leashed|roped|affixed|fixed|pinned|"
                r"hitched|moored|lashed|chaining|cuffing|locking|fastening|"
                r"securing|tethering|tying|strapping|clipping|hooking|bolting|"
                r"attaching|anchoring|padlocking)")
# The -s forms are verbs or plural nouns depending on what sits in front of them.
# A determiner makes them nouns -- "the chains", "a clip", "those cuffs" -- and
# anything else makes them verbs: "the guard chains her collar", "...and clips the
# chain to a ring".
_FASTEN_S = (r"(?<!\bthe\s)(?<!\ba\s)(?<!\ban\s)(?<!\bthese\s)(?<!\bthose\s)"
             r"(?<!\btwo\s)(?<!\bsome\s)(?<!\bmore\s)"
             r"(?:chains|cuffs|handcuffs|shackles|manacles|locks|padlocks|fastens|"
             r"secures|tethers|ties|straps|clips|hooks|bolts|attaches|anchors|"
             r"leashes|ropes|pins)")
# WEAK verbs describe a chain's PATH rather than an act of fastening, and on their
# own they are ordinary motion -- "she runs to the wall" must not be an anchor. They
# count only with the hardware itself as the subject: "a chain runs from her collar
# to the wall" and "a short chain holds her collar to the wall" are both anchors and
# neither has a fastening verb anywhere in it.
_FASTEN_WEAK = (r"(?:chains?|ropes?|cords?|cables?|leash(?:es)?|leads?|straps?|"
                r"tethers?|links?|lines?)\s+(?:\S+\s+){0,4}?"
                r"(?:run|hold|lead|stretch|extend|go|reach|drop|hang)(?:s|es|ing)?")
# WHAT A CHAIN CAN BE FASTENED TO. The wall was not on this list, and that is the
# whole of the reported bug: "chains it to the wall" produced NO anchor, so the
# collar hold said the collar stays closed and nothing ever said she was tethered.
# A shot that then has her cross the room is a shot with a collar, no tether, and a
# beat saying she walks away -- and the cheapest way for the model to make that make
# sense is to take the collar off. Floor, ceiling and pillar were missing with it.
_ANCHOR_POINT = re.compile(
    r"\b(?:" + _FASTEN_PART + r"|" + _FASTEN_S + r"|" + _FASTEN_WEAK + r")"
    # One determiner list, in the engine, for the same reason the vocabularies
    # moved there: this one and the engine's drifted apart and a phrasing either
    # reader missed was a restraint the other half of the node did not know about.
    r"\b(?:\s+\S+){0,5}?\s+to\s+" + engine.ANCHOR_DET +
    r"((?:bed\s*frames?|bed\s*heads?|headboards?|bed\s*posts?|beds?|rails?|railings?|"
    r"bars?|posts?|rings?|hooks?|pipes?|radiators?|chairs?|tables?|beams?|frames?|"
    r"grates?|grilles?|fences?|walls?|floors?|grounds?|ceilings?|pillars?|columns?|"
    r"stakes?|eye\s*bolts?|bolts?|brackets?|cages?|bunks?|benches?|ladders?|"
    r"girders?|struts?|anchors?|loops?))\b", re.I)


def limb_anchor(text):
    """Where fastened limbs are being held, as a phrase. '' when the text says none."""
    t = text or ""
    where = next((phrase for pat, phrase in _LIMB_ANCHOR
                  if re.search(pat, t, re.I)), "")
    m = _ANCHOR_POINT.search(t)
    point = ("at the " + re.sub(r"\s+", " ", m.group(1).lower())) if m else ""
    if where and point:
        return f"{where}, {point}"
    return where or point


# Framing tight enough to crop an anchor point out of shot. Worth naming because the
# next shot starts from THIS shot's last frame: whatever a close shot cuts off, the
# next shot inherits a picture without it, and only the text still knows.
_TIGHT_FRAME = re.compile(
    r"\bclose[-\s]?up|\bclose\s+(?:shot|on)\b|\btight\s+(?:on|shot)\b|"
    r"\bfills?\s+the\s+frame\b|\bmacro\b", re.I)


# A WHOLE BODY DOING SOMETHING. Not a face acting, and not a hand: these are the
# verbs whose action does not fit inside a portrait.
_WHOLE_BODY = re.compile(
    r"\b(?:serves?|serving|throws?|throwing|kicks?|kicking|hits?|hitting|"
    r"swings?|swinging|spikes?|blocks?|blocking|jumps?|jumping|runs?|running|"
    r"sprints?|sprinting|dances?|dancing|plays?|playing|climbs?|climbing|"
    r"lifts?|lifting|carries|carrying|pushes|pushing|pulls?|pulling|"
    r"swims?|swimming|stretches|stretching|wrestles?|fights?|fighting)\b", re.I)
# A SHOT SIZE, which is the only kind of camera note that answers "how much of the
# person is in frame". This began as any camera word at all and that was wrong in the
# one way that mattered: the README tells you to put the camera in the anchor, so
# "Shot on 35mm, handheld" or "anamorphic lens" -- which say NOTHING about subject
# distance -- silenced the clause on every shot of a real script. Reported as the
# framing fix doing nothing whatsoever, and it was doing nothing: it never ran.
#
# Lens, stock, grade, mood and camera movement are not sizes and do not stand it
# down. A close-up does, because a close-up is a frame somebody asked for.
_FRAME_SIZE = re.compile(
    r"\bclose[-\s]?ups?|\bclose\s+(?:shots?|on)\b|\btight\s+(?:shots?|on)\b|\bmacro\b|"
    r"\bwide\s+(?:shots?|angle)\b|\bwide\b|\bestablishing\b|\blong\s+shots?\b|"
    r"\bfull\s+(?:shots?|body|figure|length)\b|\bmedium\s+shots?\b|\bmid\s+shots?\b|"
    r"\btwo[-\s]?shots?\b|\bover[-\s]the[-\s]shoulder\b|\bpov\b|"
    r"\bhead\s+and\s+shoulders\b|\bportrait\b|\bwaist[-\s]up\b|"
    r"\bknees?[-\s]up\b|\bhead\s+to\s+(?:toe|foot|feet)\b", re.I)


def frame_hold(beat, anchor="", people=1):
    """Say the frame holds a whole body, where nothing else says what the frame is.

    THE PORTRAIT IS WHAT AN UNSTATED FRAME BECOMES. This file already records the
    reason: "an attribute a prompt does not state is not LEFT to the model, it is
    left to the model's prior -- which for a named, described person is a PORTRAIT:
    facing the lens, pleasantly, because that is what photographs of people are."
    The sheet describes a face in every shot, because clothing continuity needs it,
    and the mouth guard describes a mouth in every silent shot, because babble needs
    it -- so the text is weighted towards a face and nothing in it says how much of
    the person to show. Measured on a volleyball beat: 15 words of appearance and 9
    of mouth against 8 of action. Reported as the camera staying fixated on one
    character, staring into the lens, with no reference image anywhere in the run.

    Only where the beat stages something a portrait cannot contain, and only where
    the author has said nothing about the camera -- in the beat or in the anchor.
    Their framing always wins, a close-up included, because a close-up is a frame
    somebody asked for. Impersonal, like the other picture guards, and positively
    phrased: it says what the frame holds, never what it is not."""
    b = str(beat or "")
    if _FRAME_SIZE.search(b) or _FRAME_SIZE.search(str(anchor or "")):
        return ""
    if tight_framing(b) or tight_framing(str(anchor or "")):
        return ""
    if not (_WHOLE_BODY.search(b) or _TRAVEL_VERB.search(b)):
        return ""
    if int(people or 1) > 1:
        return (" The frame holds every body in it whole, head to feet, with the room "
                "around them.")
    return (" The frame holds the whole body, head to feet, with the room around it.")


def tight_framing(text):
    """Does this beat call for a frame close enough to lose the anchor point?"""
    return bool(_TIGHT_FRAME.search(text or ""))


# WHAT THE CLOSE FRAME IS CLOSE **ON**, and therefore what it can hold.
#
# Reported: camera types written in the anchor did not take. They were reaching
# the model verbatim -- the anchor is 11-13% of a shot's conditioning -- but the
# other 87% asserted denim shorts, wrists at the small of the back and the weight
# on shoulder and hip. A close-up on a face contains none of those. The camera was
# not being ignored, it was being outvoted by the node's own continuity prose.
#
# Only WARDROBE is scoped away. The limb and restraint sentences STAY on a tight
# shot, and deliberately: the framing crops the anchor point out of the frame the
# next shot inherits, so the text is the only thing left that knows where the limbs
# are fastened. Dropping that is the exact drift the tight-frame warning exists to
# report.
_FRAME_ON = re.compile(
    r"\b(?:close[-\s]?up|close\s+shot|tight\s+shot|macro(?:\s+lens)?)\b[^.;]{0,24}?"
    r"\bon\s+(?:her|his|their|its|the)\s+([\w][\w\- ]{1,20})"
    r"|\b(?:close|tight)\s+on\s+(?:her|his|their|its|the)\s+([\w][\w\- ]{1,20})", re.I)
# Subject word -> the garment REGIONS that frame can still show. A face is read as
# head-and-shoulders, which is what a close-up on a face conventionally is, so a
# collar or neckline survives and the trousers do not.
_FRAME_HOLDS = (
    (r"face|eyes?|mouth|lips|head|hair|jaw|cheeks?|ears?|nose|expression",
     frozenset(("torso",))),
    (r"hands?|fingers?|wrists?|palms?|knuckles?", frozenset(("hands",))),
    (r"feet|foot|ankles?|toes?", frozenset(("feet",))),
    (r"chest|breasts?|torso|shoulders?|stomach|belly|waist|back",
     frozenset(("torso",))),
    (r"legs?|thighs?|hips?|knees?|calves|calf", frozenset(("legs", "feet"))),
)


def frame_holds(text):
    """The garment regions a named close frame can still contain.

    None when the text names no close frame, or names one without saying what it
    is close ON -- a bare "close-up" gives no way to know what is in it, and
    guessing would be the node cropping the author's wardrobe on a coin toss."""
    m = _FRAME_ON.search(text or "")
    if not m:
        return None
    subject = (m.group(1) or m.group(2) or "").strip().lower()
    for pat, regions in _FRAME_HOLDS:
        if re.search(r"\b(?:" + pat + r")\b", subject, re.I):
            return regions
    return None


def out_of_frame_garments(scene, holds):
    """Garments in `scene` whose region the frame cannot show.

    A garment that cannot be placed at all is KEPT: an unplaceable item is one this
    file does not recognise, and cropping what it does not understand is how a
    wardrobe quietly loses things the author wrote."""
    if not holds:
        return []
    out = []
    for g in garments_in(scene or ""):
        r = region_of(g)
        if r and r not in holds:
            out.append(g)
    return out


# WHERE SOMEBODY IS LOOKING.
#
# Reported: "she is looking at the TV" rendered her looking off to the side, posing
# for the camera. The beat says it once and nothing else in the shot agrees with it,
# while a near-clean reference is asking for the portrait's pose -- and the portrait
# looks at the lens, because photographs of people do. info already warned that a
# referenced person can hold the portrait's gaze; nothing in the TEXT argued back.
#
# The model's own prior pulls the same way: a person in frame faces the camera unless
# something says otherwise. So the target gets said a second time, as a physical fact
# about the eyes and the head rather than as an activity.
_GAZE_PREP = r"(?:at|to|towards?|into|onto|over\s+at)"
_GAZE_TAIL = (r"(?=[.,;:!?]|\s+(?:and|as|while|when|who|which|that|with|for|from|in|on|"
              r"before|after|until)\b|$)")
_GAZE_DET = r"(?:the|a|an|her|his|their|its|that|this)\s+"
_LOOK_AT = re.compile(
    r"\b(?:look(?:s|ed|ing)?|star(?:e|es|ed|ing)|gaz(?:e|es|ed|ing)|"
    r"glanc(?:e|es|ed|ing)|peer(?:s|ed|ing)?|squint(?:s|ed|ing)?)\s+"
    r"(?:back\s+|down\s+|up\s+|over\s+|round\s+|around\s+|straight\s+|right\s+)?"
    + _GAZE_PREP + r"\s+" + _GAZE_DET + r"([\w][\w\- ]{0,24}?)" + _GAZE_TAIL, re.I)
# Verbs that carry their object without a preposition. "Watching the TV" is a gaze
# instruction as much as "looking at the TV" is.
_WATCH = re.compile(
    r"\b(?:watch(?:es|ed|ing)?|stud(?:y|ies|ied|ying)|examin(?:e|es|ed|ing))\s+"
    + _GAZE_DET + r"([\w][\w\- ]{0,24}?)" + _GAZE_TAIL, re.I)
# Things that are not a place to look. "Looks at her" is a pronoun with no picture in
# it, and restating a pronoun as a target says nothing the beat did not.
_NOT_A_TARGET = frozenset(
    "him her them it me us you himself herself themselves one other others "
    "time moment thing things way".split())


# A LOOK AT A PERSON IS A LOOK.
#
# The old rule was "restating a pronoun says nothing the beat did not, and the other
# person is in frame to be looked at anyway". That is true against a neutral model
# and false against one whose prior is a portrait: the choice is not between the
# beat's word and a restatement of it, it is between the beat's word and the LENS.
# Reported as "she looks at the van in one beat and gazes at the camera in the
# next" -- and measured, "McKenna watches him" was the commonest way to lose it:
# the look moved, which cleared the latch correctly, and then nothing replaced it.
#
# Neither existing pattern can even see a person. Both require a determiner before
# the target -- the|a|an|her|his|their -- so "watches Dan" and "looks at Dan" match
# nothing at all.
_LOOK_AT_WHO = re.compile(
    r"\b(?:look(?:s|ed|ing)?|star(?:e|es|ed|ing)|gaz(?:e|es|ed|ing)|"
    r"glanc(?:e|es|ed|ing)|peer(?:s|ed|ing)?)\s+"
    r"(?:back\s+|down\s+|up\s+|over\s+|round\s+|around\s+|straight\s+|right\s+)?"
    + _GAZE_PREP + r"\s+([A-Z][\w-]+|him|her|them|he|she|they)\b"
    r"|\b(?:watch(?:es|ed|ing)?|stud(?:y|ies|ied|ying)|examin(?:e|es|ed|ing))\s+"
    r"([A-Z][\w-]+|him|her|them)\b", re.I)
# Which pronoun can be which. A gendered pronoun narrows the field, so a scene with
# a man and a woman resolves "him" without guessing; "them" does not narrow it and
# only lands where exactly one other person is there to land on.
_PRONOUN_SEX = {"him": "he", "he": "he", "her": "she", "she": "she"}


def _person_looked_at(beat, sheet="", described=()):
    """The PERSON this beat says somebody is watching. '' when it is not resolvable.

    Only where it is unambiguous. Three people and a bare "him" resolves to nobody,
    and guessing which one is worse than saying nothing: a shot told the wrong
    sightline is a shot that has to be reshot, while a shot told none is only back
    where it was."""
    m = _LOOK_AT_WHO.search(beat or "")
    if not m:
        return ""
    raw = (m.group(1) or m.group(2) or "").strip()
    if not raw:
        return ""
    rows = {n: ln for n, ln in sheet_lines(sheet or "") if n}
    # A NAME, spelled as the sheet spells it.
    for n in rows:
        if raw.lower() == n.lower():
            return n
    # A PRONOUN. Whoever else is in the shot, if that is one person -- and if the
    # pronoun is gendered, only the people whose entry agrees with it.
    want = _PRONOUN_SEX.get(raw.lower())
    here = [n for n in (described or []) if n in rows]
    # The looker is not the one being looked at.
    looker = subjects_for(beat, sheet, _LOOK_VERB_SRC)
    here = [n for n in here if n not in set(looker)]
    if want:
        here = [n for n in here
                if re.search(r"\b" + want + r"\b", rows.get(n, ""), re.I)]
    return here[0] if len(here) == 1 else ""


def look_target(beat, sheet="", described=()):
    """What this beat says somebody is looking at. '' when it names nothing."""
    for pat in (_LOOK_AT, _WATCH):
        m = pat.search(beat or "")
        if not m:
            continue
        target = re.sub(r"\s+", " ", m.group(1)).strip(" -")
        if not target or target.lower() in _NOT_A_TARGET:
            continue
        return target
    return _person_looked_at(beat, sheet, described)


# Going somewhere ends a look. Held across it, "the eyes are on the TV" follows
# somebody out of the room and into the next scene.
_MOVES_OFF_SRC = (r"walks?|walked|runs?|ran|steps?|stepped|moves?|moved|crosses|"
                  r"crossed|leaves?|left|exits?|exited|goes|went|heads?|headed|"
                  r"climbs?|climbed|follows?|followed")
_MOVES_OFF = re.compile(r"\b(?:" + _MOVES_OFF_SRC + r")\b", re.I)


# The look VERBS on their own, with no target required. _LOOK_AT needs a nameable
# object, so "looks at her" reads as no look at all -- and the latch then held a
# television she had just turned away from.
_LOOK_VERB_SRC = (r"look(?:s|ed|ing)?|star(?:e|es|ed|ing)|gaz(?:e|es|ed|ing)|"
                  r"glanc(?:e|es|ed|ing)|peer(?:s|ed|ing)?|watch(?:es|ed|ing)?|"
                  r"stud(?:y|ies|ied|ying)")
_LOOK_VERB = re.compile(r"\b(?:" + _LOOK_VERB_SRC + r")\b", re.I)


def subjects_for(beat, sheet, verbs):
    """Which people on the sheet this beat puts in front of one of these verbs.

    The shared shape behind speakers_in and vocal_sources_in, including the
    conjunction guard: "Dan holds the door and McKenna looks away" must not credit
    Dan, because `and` opens a new predicate with its own subject."""
    b, out = beat or "", []
    for n, _ in sheet_lines(sheet):
        if n and re.search(r"\b" + re.escape(n) + r"\b"
                           r"(?:\s+(?!and\b|but\b|then\b|who\b|,\s*who\b)[\w,']+){0,2}?"
                           r"\s+(?:" + verbs + r")\b", b, re.I):
            out.append(n)
    return out


def looks_somewhere(beat):
    """Does this beat stage a look at all, nameable target or not?

    "Mara looks at her" names no target this node can restate -- but it does
    move the look, and holding the previous target across it says her eyes are
    on a television she has just turned away from."""
    return bool(_LOOK_VERB.search(beat or ""))


def gaze_hold(target, who="", is_person=False):
    """One sentence putting the eyes and the head on the thing the beat named.

    NAMED when the caller says to, which it does once a second person is in the
    shot. Reported: "the girl is stuck gazing at a camera while the other character
    does his part" -- a look staged by one person went on being said impersonally in
    shots she was not in, so it landed on whoever was. Impersonal is still right with
    one person in frame: naming somebody is a second mention of them, and a described
    person is a person the model draws.

    Impersonal, like the hardware placement clause: naming the person again is one
    more mention of a person, and that has its own cost. Says nothing about where the
    camera is -- the shot may be looking straight down the line of sight -- only that
    the head is turned to face what the eyes are on."""
    if not target:
        return ""
    # SHORT. Nineteen words restating a nine-word beat is most of the shot spent
    # agreeing with it, and the guards crowding out the action is what "the
    # character did not do what I told it" looks like from the outside.
    what = target if is_person else f"the {target}"
    if who:
        # A PRONOUN WHERE THE NAME IS ALREADY SPENT. A person is named once in a
        # shot's guard text -- two clauses naming the same person is what put a
        # second girl in frame at the moment of cuffing. But the clause that
        # already named her is standing right beside this one, so "her eyes" has
        # its antecedent and costs no second naming. Used only where no one else
        # in the shot shares the pronoun.
        return f" {who[0].upper()}{who[1:]} eyes and head are turned to {what}."
    return f" The eyes and the head are turned to {what}."


def dialogue_gaze(n_people):
    """One impersonal sentence turning speakers and listeners to each other.

    gaze_hold restates a look the beat named. A dialogue beat that names none
    leaves both faces to the portrait prior, and the prior is the lens: reported
    as "it looks like they are talking to a camera and not to each other". A
    spoken line has an addressee whether or not the beat wrote one, and the
    addressee is in the shot, so turning the faces to each other is the one thing
    that can be said without inventing anything.

    Impersonal, like gaze_hold, and for the same reason: on a dialogue shot the
    speaker's name is spent by the mouth guard and the listener's by told_hold,
    and a third mention is a third person. Positively phrased -- at cfg 1 naming
    the lens would ask for it. Says nothing about where the camera is."""
    if n_people < 2:
        return ""
    if n_people == 2:
        return " They face each other, eyes on each other."
    return " Eyes on whoever is speaking, faces turned to them."


def forced_pose(text):
    """Does this text put a body into a position that hardware can enforce?"""
    return bool(_FORCED_POSE.search(text or ""))

# Hardware that is rigid by nature. Only consulted once a restraint is established,
# so a chain-link fence in the scenery cannot arm it on its own.
# Named hardware only. "steel" was in this list, which meant any steel object earned
# the chain clause -- and that clause talks about LINKS and the RUN between fastenings,
# which is nonsense said of a steel clamp. A clamp is rigid, but it is not a chain: it
# gets the plain restraint hold, which is what "it stays on" needs anyway.
_RIGID_HARDWARE = re.compile(
    r"\b(?:chain(?:s|ed|ing)?|padlock(?:s|ed|ing)?|shackle[sd]?|manacle[sd]?|"
    r"handcuff(?:s|ed)?|cuffs?|cuffed|irons|spreader\s+bar|"
    r"hogcuffed|hog-?cuffed)\b", re.I)


# Where each piece of hardware goes. Not a creative choice -- it is what the object
# IS. A collar without a neck is a band with no place to be, and a model handed a
# band-shaped object and no anatomy puts it where bands most often sit in its
# training data: on the head. That is the reported failure, and it happens whether
# the item is being fastened or merely held up and shown.
#
# (item pattern, the phrase that places it)
_TAPE_GAG = (r"(?:duct[\s-]*)?tape\s+gag|"
             r"gag(?:s|ged|ging)?\s+\w{0,12}\s*with\s+"
             r"(?:duct\s+|packing\s+|masking\s+)?tape|"
             r"tape\s+(?:over|across)\s+(?:her|his|their|the)\s+mouth")
_TAPE_GAG_CLAUSE = "a strip of tape lies flat across the mouth"
_GAG_CLAUSE = "a gag sits in the mouth"

_HARDWARE_ANCHOR = (
    (r"collar(?:s|ed)?",                 "a collar closes around the neck"),
    (r"leash(?:es)?|lead\b",             "a leash clips to the collar at the neck and hangs down from it"),
    # Tape is a gag that lies flat against the face. Told "a gag sits in the
    # mouth" it is given bulk it does not have, and bulk over the mouth,
    # re-encoded shot after shot, settles into a mask.
    (_TAPE_GAG,                          _TAPE_GAG_CLAUSE),
    (r"gag(?:s|ged)?|ball\s*gag",        _GAG_CLAUSE),
    (r"blindfold(?:s|ed)?",              "a blindfold covers the eyes"),
    (r"handcuff(?:s|ed)?",               "handcuffs close around the wrists"),
    (r"shackle[sd]?|leg\s+irons",        "shackles close around the ankles"),
    (r"harness(?:es)?",                  "a harness sits on the torso"),
    (r"spreader\s+bar",                  "a spreader bar holds the ankles apart"),
    # No entry for a chastity belt, and the lookbehind below keeps the plain belt off
    # it too, so it gets no placement clause at all. It is the item most likely to
    # arrive with its own <Picture N>, and a written description of where the shield
    # and the lock sit argues with the picture rather than adding to it. Where the
    # reference shows the object, the object is already placed; describe it in your
    # own words if you want it stated.
    (r"(?<!chastity\s)belt(?:s|ed)?",    "a belt closes around the waist and hips"),
)

# Anatomy that already places something, so the writer's own wording wins.
_BODY_PART = re.compile(
    r"\b(?:neck|throat|wrist|wrists|ankle|ankles|mouth|lips|jaw|eyes|face|head|"
    r"waist|hips?|chest|torso|stomach|belly|shoulders?|arms?|legs?|thighs?|"
    r"knees?|feet|foot|hands?)\b", re.I)


def unanchored_hardware(text):
    """Phrases placing any hardware that is named with no body part beside it.

    A window of 60 characters either side counts as 'beside'. If the text already
    says where the thing goes, nothing is added -- what you wrote wins."""
    out = []
    t = text or ""
    for pat, phrase in _HARDWARE_ANCHOR:
        placed = False
        found = False
        for m in re.finditer(r"\b(?:" + pat + r")", t, re.I):
            found = True
            window = t[max(0, m.start() - 60):m.end() + 60]
            if _BODY_PART.search(window):
                placed = True
                break
        if found and not placed and phrase not in out:
            out.append(phrase)
    # A tape gag answers the gag entry as well, and the two clauses disagree
    # about whether the thing has bulk. The flat one is the true one.
    if _TAPE_GAG_CLAUSE in out and _GAG_CLAUSE in out:
        out.remove(_GAG_CLAUSE)
    return out


def anchor_clause(phrases):
    """One sentence saying where the named hardware sits."""
    if not phrases:
        return ""
    return " Each piece of hardware sits where it belongs: " + "; ".join(phrases) + "."


# A state written down is a state the model can render by ARRIVING at it.
#
# Reported: "stand behind a van with its doors closed" put the doors open and the
# characters closing them. The text named a state and never said WHEN it was true,
# and a video model asked for a door renders the thing a door does. The state is
# the most interesting event in the sentence, so it gets performed.
#
# Fewer sampling steps make it worse rather than better. On a 4-step distill
# schedule the layout is committed almost immediately, so an opening frame that
# guessed wrong is never argued out of it by the later steps -- there are none.
# Saying the state is already true at the first frame costs one sentence and takes
# the event away.
#
# Scenery only, and only words that are not also something worn: no "boots", no
# "hood", no "bonnet". A character sheet lives in this same text.
_STATE_THING = (r"doors?|gates?|windows?|curtains?|blinds?|shutters?|"
                r"hatch(?:es)?|tailgates?|lids?|drawers?")
_STATE_WORD = r"closed|shut|open|locked|unlocked|latched|bolted|drawn|ajar|sealed"
# Verbs that CHANGE one of those states. Several are also the state word itself --
# "closed" is both -- which the reader below has to tell apart.
_STATE_ACTS = (r"opens?|opened|opening|closes?|closed|closing|shuts?|shutting|"
               r"slams?|slammed|slamming|slides?|slid|sliding|pulls?|pulled|pulling|"
               r"pushes?|pushed|pushing|draws?|drew|drawing|locks?|locked|locking|"
               r"unlocks?|unlocked|unlocking|lifts?|lifted|lifting|raises?|raised|"
               r"lowers?|lowered|swings?|swung|yanks?|yanked|wrenches|wrenched")
# The gap takes apostrophes: "closed the van's doors" is a determiner phrase, and a
# gap of bare \w+ does not match one, so the whole act went unseen.
_STATE_ACT = re.compile(r"\b(" + _STATE_ACTS + r")\s+((?:[\w']+\s+){0,3}?)(" +
                        _STATE_THING + r")\b", re.I)
# What tells "closed the rear doors" from "closed rear doors": a determiner. The verb
# reading needs one -- you close THE doors, ITS doors, THE VAN'S doors -- and the
# adjective reading cannot have one, because the determiner belongs in front of the
# whole phrase ("a van with closed rear doors").
#
# Getting this wrong is not a missed guard, it is an inverted one. Read as a verb,
# "a van with closed rear doors" earned the anchor "the doors are open at the first
# frame and shut by the last" -- the node itself asking for the doors to start open
# and be closed on camera, which is the bug it was written to fix.
_STATE_DET = re.compile(r"\b(?:the|a|an|its|his|her|their|our|my|your|this|that|these|"
                        r"those|both|all|each|every|another|one|two|three|\w+'s)\b", re.I)


def _adjectival(verb, gap):
    """Is this state word describing the noun rather than acting on it?

    Only words that are ALSO states can be adjectives: "opens" is a verb however it
    is placed. Modifiers may sit between -- "closed rear doors", "shut cargo doors" --
    so the test is the determiner, not the distance."""
    return (bool(re.fullmatch(_STATE_WORD, verb, re.I))
            and not _STATE_DET.search(gap or ""))


# "closed doors", and "closed rear doors" -- the state in front of its noun, with the
# modifiers a real sentence puts between them.
_STATE_ADJ = re.compile(r"\b(" + _STATE_WORD + r")\s+((?:[\w']+\s+){0,2}?)(" +
                        _STATE_THING + r")\b", re.I)
# "the doors are closed", "the doors closed", "the doors are still shut". The gap is
# copulas and nothing else, so a state word further off in the sentence -- belonging
# to some other object -- is not dragged onto this one.
_STATE_PRED = re.compile(r"\b(" + _STATE_THING + r")\s+" +
                         r"((?:(?:are|is|was|were|remains?|stay|stays|still|both|all)\s+){0,2})(" +
                         _STATE_WORD + r")\b", re.I)


# POSTURE. The one piece of continuity the scene-state reader never covered: it
# tracks scenery -- doors, windows, drawers -- and nothing about the body. A beat
# that sits somebody down establishes a pose the next shot is never told about, so
# the shot ends with them seated and the next one stands them back up. The keyframe
# does carry the pose as a picture, but the TEXT is what the model reconciles it
# against, and text that says nothing loses to a reference that says something.
#
# Deliberately coarse: four postures, no orientation, no limb detail. Naming more
# than the pose is how a continuity clause turns into an instruction to hold still.
# The posture vocabulary lives in the engine: there were two tables and they
# had diverged, with a crouch setting no posture at all here. This file keeps
# its own READER, because it answers a different question -- whose posture,
# clause by clause -- but off the same words.
_POSTURE_OF = engine._POSTURE_OF
_NOT_A_BODY = engine._NOT_A_BODY



# Words that follow a posture verb but are never its object: they are the
# direction the body goes, not the body.
_POSE_DIRECTIONS = frozenset(
    "down up back onto into on in over out away flat still there here".split())
# ...and the pronouns that ARE an object.
_OBJECT_PRONOUN = frozenset(
    "her him them herself himself themselves".split())


def posture_in(beat, cast):
    """{name: posture} this beat puts somebody into. {} when it stages none.

    Attributed by CLAUSE, so "Kate sits down and Sam stays by the door" does not
    seat them both. A clause with a posture verb and no name belongs to whoever the
    beat names first, which is the same reading the removal agent uses."""
    b = str(beat or "")
    people = [n for n in (cast or []) if n]
    if not b or not people:
        return {}
    out = {}
    # A CLOSING QUOTE ends a sentence too. 'Dana says: "Lie down." McKenna lies
    # down' has its full stop inside the quotes, so a plain full-stop split left it
    # as one clause -- and the subject span for the real "lies" reached back past
    # the quote and picked up Dana as well.
    #
    # But splitting there also cuts the quote in half, and a half-open quote is one
    # _in_a_request cannot see. So the clause gives the SUBJECT and the whole beat
    # gives the QUOTES: each part carries its offset, and the request test is asked
    # about the position in `b`.
    at0 = 0
    for part in re.split(r"(?<=[.;!?])[\"'”’]?\s+", b):
        base = b.find(part, at0)
        base = at0 if base < 0 else base
        at0 = base + len(part)
        if _NOT_A_BODY.search(part):
            continue
        # Every posture verb in the clause, in order, with the span of text that
        # precedes it. The SUBJECT is the names in that span: "Kate sits down and
        # Sam stays by the door" seated them both when the whole sentence was
        # searched, because Sam is in it -- but he is after the verb, doing
        # something else. "Kate and Sam sit down" still seats both, because both
        # names precede the one verb.
        # A posture ASKED FOR is not one taken. 'Dana says: "lie down on the
        # change table"' put McKenna down a beat early -- and both of them, since
        # both names precede the verb. _in_a_request is the reader the removal
        # side already uses: quoted speech, questions, and "tells her to sit"
        # alike. Asking for a garment stopped removing it for the same reason.
        # EVERY occurrence, not the first. Taking only the first meant that
        # skipping a quoted "Lie down." threw away the real "McKenna lies down"
        # that followed it, and the beat registered no posture at all.
        hits = sorted(((m.start(), pose) for pose, rx in _POSTURE_OF
                       for m in rx.finditer(part)
                       if not _in_a_request(b, base + m.start())
                       and not engine.denied_posture(part, m.start())),
                      key=lambda h: h[0])
        prev = 0
        for at, pose in hits:
            span = part[prev:at]
            # TRANSITIVE first. "Dana lies McKenna down" puts MCKENNA down -- the
            # person going into the pose is the object, and reading the subject
            # latched the wrong person, so every later shot said Dana was still
            # lying down while the beat had her up and working. Same subject/object
            # confusion as crediting an addressee with a line.
            #
            # Tight on purpose: the name has to sit between the verb and a
            # direction or preposition. "sits down and looks at Dana" has "down"
            # straight after the verb and Dana is nobody's object.
            tail = part[at:]
            # The object has to be a NAME on the sheet or a personal pronoun.
            # Matching [A-Z]\w+ under re.I matches any word at all, so "lying down
            # on the table" parsed as verb "lying", object "down", direction "on"
            # -- and the pose landed on whoever was not acting.
            obj = re.match(r"\w+\s+(?:the\s+)?([\w'-]+)\s+"
                           r"(?:down|up|back|onto|into|on|in)\b", tail, re.I)
            if obj and obj.group(1).lower() not in _POSE_DIRECTIONS:
                word = obj.group(1)
                named = [n for n in people
                         if n.lower() == word.lower()]
                if named:
                    for n in named:
                        out[n] = pose
                    prev = at
                    continue
                if word.lower() not in _OBJECT_PRONOUN:
                    obj = None          # not a name on the sheet, not a pronoun
                # A pronoun object: resolvable only when it points at exactly one
                # person who is NOT the one acting.
                actor = [n for n in people
                         if re.search(r"\b" + re.escape(n) + r"\b", span, re.I)]
                others = [n for n in people if n not in actor]
                if obj and len(others) == 1:
                    out[others[0]] = pose
                    prev = at
                    continue
            here = [n for n in people
                    if re.search(r"\b" + re.escape(n) + r"\b", span, re.I)]
            if not here:
                # No name before the verb: the beat's first-named person is acting,
                # and with one person in the shot there is nobody else it can be.
                first = next((n for n in people
                              if re.search(r"\b" + re.escape(n) + r"\b", b, re.I)), None)
                here = [first] if first else (people[:1] if len(people) == 1 else [])
            for n in here:
                out[n] = pose
            prev = at
    return out


# Actions that CONTRADICT a posture. A latched pose survives until another one is
# staged, and a beat can put somebody back on their feet without ever saying so:
# "Dana takes out a new nappy and places it on the change table" is not something
# anybody does lying down, but it names no posture, so "Dana is still lying down"
# went on being said in every later shot. Reported exactly that way.
#
# Split by how strong the contradiction is. TRAVEL is incompatible with every
# posture -- somebody walking across a room is not sitting, kneeling or lying.
# HANDLING at arm's length is only incompatible with LYING: it is perfectly
# possible to sit or kneel while picking something up.
_HANDLES = re.compile(
    r"\b(?:takes?|took|taking|picks?|picked|picking|places?|placed|placing|"
    r"puts?|putting|sets?|setting|lifts?|lifted|lifting|carries|carried|"
    r"carrying|fetch(?:es|ed|ing)?|hands?|handed|handing|passes|passed|"
    r"opens?|opened|opening|closes?|closed|closing|pours?|poured|pouring)\b",
    re.I)


def _real_travel(span):
    """Does this span actually move somebody, or does it just LOOK like it?

    "takes off her shirt" matched the travel list on "takes" -- the list has it
    for "takes her to the car" -- so undressing cleared the posture latch and the
    shot after was told nothing about how the body was left. Reported as a squat
    not being held: she stood up on her own.

    Only this one phrasal is excluded. "walks off" and "runs off" are locomotion
    and the particle does not change that; it is the HANDLING verb that makes
    "takes off" mean something else entirely."""
    for m in _TRAVEL_VERB.finditer(span or ""):
        if re.match(r"\s+off\b", (span or "")[m.end():m.end() + 6], re.I) \
                and re.match(r"(?:takes?|took|taking)$", m.group(0), re.I):
            continue
        return True
    return False


def posture_cleared(beat, poses):
    """{name} whose latched posture this beat contradicts without restating one.

    The beat is the author's own words and outranks a hold: where it puts somebody
    on their feet, the hold has to let go or it argues with the shot it is standing
    next to."""
    # WHAT SHE SAYS IS NOT WHAT SHE DOES. Read whole, this cleared the latch on
    # `Kate says: "Someone is coming."` -- "coming" is a travel verb, inside the
    # quoted line, about somebody else entirely. The pose was dropped and the next
    # shot was told nothing about how the body was left, so she stood up on her
    # own. Reported as a squat not being held.
    b = _outside_speech(str(beat or ""))
    out = set()
    if not b:
        return out
    for name, pose in (poses or {}).items():
        m = re.search(r"\b" + re.escape(name) + r"\b", b, re.I)
        if not m:
            continue
        # What this beat has them doing, up to the end of the clause.
        stop = re.search(r"[.;!?]", b[m.end():])
        span = b[m.end():m.end() + (stop.start() if stop else len(b))]
        if _real_travel(span):
            out.add(name)
        elif pose == "lying down" and _HANDLES.search(span):
            out.add(name)
    return out


def posture_hold(poses, described):
    """One short sentence keeping people in the pose an earlier beat put them in.

    Only for people this shot DESCRIBES -- a pose belonging to somebody the text
    does not mention is a pose for nobody, and the model draws the person that
    sentence implies. Short on purpose: this is latched, so it lands in every shot
    after the one that stages it, and a long clause repeated is the guard bloat
    this node was rebuilt to escape."""
    # STANDING is not held. It is the default pose -- a model draws a standing
    # person unless told otherwise -- so the clause buys nothing and costs a naming
    # of the person, and a described person is a person the model draws: naming
    # somebody twice in one shot is what put a second copy of them in frame.
    # Sitting, kneeling and lying down are the poses that need saying.
    who = [(n, p) for n, p in (poses or {}).items()
           if n in set(described or []) and p != "standing"]
    if not who:
        return ""
    # "is still lying down" -- the adverb meaning "as before", which is not how a
    # video model reads the token. This clause is LATCHED: it lands in every shot
    # after the one that stages the pose, so a scene where somebody sat down once
    # carried the word `still` beside their name for the rest of the film. Naming
    # the pose is the entire guarantee; "as before" was never part of it.
    if len(who) == 1:
        return f" {who[0][0]} is {who[0][1]}."
    said = "; ".join(f"{n} is {p}" for n, p in who[:2])
    return f" {said}."


def _state_key(thing):
    """One key for 'door' and 'doors', so a beat acting on either clears both."""
    t = (thing or "").lower()
    return t[:-2] if t.endswith("es") and t.startswith("hatch") else t.rstrip("s")


# Which way a verb runs. Reported: some distill LoRAs render an action BACKWARDS --
# the beat opens the doors and the shot closes them. A single staged action is
# direction-ambiguous to a model that has learned to treat a clip and its reverse as
# the same clip, which is what time-flip augmentation teaches. Naming the two ends
# settles it, and it is the same thing a removal already does: "off during this shot
# and away by the last frame".
#
# Only verbs that HAVE a direction. "pulls", "draws", "slides" and "swings" do not:
# drawing the curtains closes them and pulling a door can do either, and a wrong
# anchor is worse than none -- it asks for the reversal instead of merely allowing it.
_OPENS = re.compile(r"(?:opens?|opened|opening|unlocks?|unlocked|unlocking|"
                    r"lifts?|lifted|lifting|raises?|raised)\Z", re.I)
_SHUTS = re.compile(r"(?:closes?|closed|closing|shuts?|shutting|slams?|slammed|"
                    r"slamming|locks?|locked|locking|lowers?|lowered)\Z", re.I)


def state_changes(text):
    """[(thing, 'open'|'shut'|None)] for the scenery this text actually works.

    A state word sitting straight in front of its noun is an adjective describing
    the thing, not a verb acting on it: "the closed doors" says nothing happens.
    The direction is None where the verb does not carry one."""
    out, seen = [], set()
    for m in _STATE_ACT.finditer(text or ""):
        verb, gap, thing = m.group(1), m.group(2), m.group(3)
        if _adjectival(verb, gap):
            continue
        key = _state_key(thing)
        if key in seen:
            continue
        seen.add(key)
        way = ("open" if _OPENS.match(verb) else
               "shut" if _SHUTS.match(verb) else None)
        out.append((thing.lower(), way))
    return out


def state_acts(text):
    """Which of those things this text works, in either direction."""
    return [_state_key(t) for t, _ in state_changes(text)]


# TRAVEL between places. A beat that walks somebody from one room to another is a
# staged change with two ends, exactly like a door opening -- and told only where
# it finishes, the shot renders the destination and cuts straight to it. Reported
# as a scene starting in the living room and instantly being in the bedroom, with
# the hallway between them missing.
#
# The same fix direction_anchor uses for doors: name BOTH ends, and the middle if
# the beat gives one.
# The vocabulary lives in the engine, because there were two of these and they
# disagreed -- a cell and a warehouse were rooms to one reader and nowhere to the
# other. The READERS stay separate: this file's search free text with no
# preposition, which is what a scene paragraph gives them, and that is why they
# carry the also-a-verb guard the engine's does not need.
_PLACE = engine.PLACES
# "door" WAS IN THAT LIST and a door is not a room -- it is a thing inside one. So
# "Ana looks at the door", the most ordinary beat there is, moved the whole shot:
# "This shot is in the door, not the room the scene text names." The camera was
# relocated into a door by a character glancing at it. "doorway" stays, because
# standing in one is a real place to be.
# _PLACE is an alternation with no edges of its own, so searching it RAW matches
# inside words: "shallow depth of field" contains "hall", and every camera anchor
# ever written for this node says shallow. That put the film in a hallway it never
# had -- stated on each shot, used as the origin of the first journey, and handed
# to room_tone, which gave a lens setting the acoustic of a cathedral. The readers
# that sit behind a preposition were always safe, because the \s+ before them is
# already a boundary; the two that search free text were not.
_PLACE_WORD = engine._PLACE_WORD
# Place words that are also ordinary verbs. free-text readers cannot tell which
# sense is meant, and "she steps out", "he lands badly", "they study the map" are
# all commoner than the rooms they collide with.
_PLACE_ALSO_A_VERB = engine.PLACE_ALSO_A_VERB
# Words in front of "room" that do NOT make it a particular room -- an article or
# a preposition leaves it as "wherever we already are".
_NOT_A_ROOM_MODIFIER = {"the", "a", "an", "this", "that", "her", "his", "their",
                        "its", "my", "our", "your", "in", "into", "inside", "of",
                        "from", "to", "at", "on", "and", "or", "same", "other"}
# A room is usually DESCRIBED, not just named: "the tiled bathroom", "the long
# hallway", "the second-floor landing". Every reader below wanted the article and
# the room word to be adjacent, so one adjective made the whole journey invisible
# -- no ends named, `here` never updated, and the room hold that depends on it
# never fired. Silently: nothing warns, because nothing was found to warn about.
# That is a scene resetting at a cut with no explanation anywhere.
#
# NON-GREEDY on purpose. "at the kitchen door" should still read as the kitchen,
# so the FIRST place word wins and the modifiers are only tried when it does not
# match. Prepositions and articles are excluded, so a match cannot cross into
# "the door OF THE bedroom", and a comma ends it, so it cannot cross a clause.
_MOD = (r"(?:(?!(?:of|the|an?|and|or|to|in|into|from|with|on|at|by|for|her|his|"
        r"their|its|my|our|your)\b)[A-Za-z][A-Za-z-]*\s+){0,3}?")
# "to the bedroom", "into the kitchen" -- where it ENDS.
# A room can belong to somebody. "McKenna's bedroom" is the ordinary way to write
# whose room it is, and a determiner list of the/her/his/their/a did not match a
# possessive name -- so the destination of the journey was invisible.
# The determiner, possessive names included, lives in the engine -- two readers
# needed it and got narrower copies weeks apart. See engine.DET_POSS.
_DET_POSS = engine.DET_POSS
# ENTERING is how arrival is written, and it is a verb, not a preposition. The
# list here was prepositions only, so "Dana walks through the home and enters
# McKenna's bedroom" named no destination at all, travel_in returned ('','',''),
# the journey guard stayed silent and the walk rendered as a cut straight to the
# far end. That is the failure the guard exists to prevent, arriving through the
# one door it was not watching.
_GOES_TO = re.compile(r"\b(?:to|into|toward|towards|through\s+to|"
                      r"enters?|entered|entering|reaches|reached|arrives?\s+(?:at|in)|"
                      r"steps?\s+into|stepped\s+into)\s+"
                      + _DET_POSS + r"\s+" + _MOD
                      + r"(" + _PLACE + r")\b", re.I)
# "down the hallway", "along the corridor" -- what it passes THROUGH.
_GOES_VIA = re.compile(r"\b(?:down|along|across|through|up|via|past)\s+"
                       + _DET_POSS + r"\s+" + _MOD
                       + r"(" + _PLACE + r")\b", re.I)
# "from the living room", "out of the kitchen" -- where it STARTS.
_GOES_FROM = re.compile(r"\b(?:from|out\s+of|leaves?|leaving)\s+"
                        + _DET_POSS + r"?\s*" + _MOD
                        + r"(" + _PLACE + r")\b", re.I)
# A verb that actually MOVES somebody. "looks to the bedroom" is not travel.
_TRAVEL_VERB = re.compile(
    r"\b(?:walk|walks|walked|walking|lead|leads|led|leading|take|takes|took|taking|"
    r"go|goes|went|going|head|heads|headed|heading|move|moves|moved|moving|"
    r"carry|carries|carried|carrying|follow|follows|followed|following|"
    r"step|steps|stepped|stepping|climb|climbs|climbed|climbing|"
    r"run|runs|ran|running|come|comes|came|coming|"
    # _GOES_FROM already reads "leaves the bedroom" as the start of a journey, so
    # the two lists disagreed with each other: one saw an origin, the other did not
    # see a move, and the move was dropped. Entering and crossing are the same
    # omission from the other end.
    r"leave|leaves|left|leaving|enter|enters|entered|entering|"
    r"cross|crosses|crossed|crossing|exit|exits|exited|exiting|"
    r"return|returns|returned|returning)\b", re.I)


def travel_in(beat):
    """(from, via, to) for a beat that moves somebody between places.

    All three may be "". Only when a MOVEMENT verb is present: "she looks to the
    bedroom" names a place and goes nowhere."""
    b = str(beat or "")
    if not b or not _TRAVEL_VERB.search(b):
        return ("", "", "")

    def _one(rx):
        m = rx.search(b)
        return re.sub(r"\s+", " ", m.group(1)).strip().lower() if m else ""

    to, via, frm = _one(_GOES_TO), _one(_GOES_VIA), _one(_GOES_FROM)
    # A place cannot be two ends of the same journey.
    if via and via == to:
        via = ""
    if frm and frm in (to, via):
        frm = ""
    return (frm, via, to)


def travel_legs(beat):
    """(from, via, to) for a beat that moves somebody, with the one promotion that
    the render and the SIZING have to agree on.

    A BEAT THAT TRAVELS ALONG A PLACE ENDS IN IT. "walks down the hallway" reads as a
    via with no destination, and travel_anchor says nothing without one -- so that
    shot was told nothing about where it was, the tracked room kept the bedroom, and
    the NEXT beat opened "in the bedroom" on its way to the kitchen. Reported as a bed
    in the hallway. The place travelled along is where the beat arrives, so the shot
    opens where the last one left off and walks into it, in frame.

    Here rather than inline in the shot loop because travel_spaces reads the same
    fact to size the shot. Kept in two places it would drift, and a transit rendered
    as a walk while being sized as if it went nowhere is exactly the split that put a
    three-room walk in a three-second shot."""
    frm, via, to = travel_in(beat)
    if via and not to:
        via, to = "", via
    return frm, via, to


def where_hold(here, scene):
    """Say which room the shot is in, once the film has left the one in the scene.

    The scene paragraph is stamped into EVERY shot -- it has to be, or a removal
    has nothing to scrub -- so a script that walks from the living room to the
    bedroom goes on opening every later shot with "A living room." while the beat
    has them on the bed. The shot then holds two places at once, and the picture
    settles on whichever the model weighs more heavily, differently each time.
    That is a scene that keeps changing and resetting.

    The author's scene text is NOT edited. This states where the shot is now, and
    only where that disagrees with what the scene says, so a script that never
    moves is untouched and costs nothing."""
    here = (here or "").strip().lower()
    if not here:
        return ""
    txt = str(scene or "")
    if not txt.strip():
        return ""
    # Nothing to correct if the scene already names this room.
    if re.search(r"\b" + re.escape(here) + r"\b", txt, re.I):
        return ""
    # ...and nothing to correct unless the scene names a DIFFERENT place, since a
    # scene that names no room is not disagreeing with anything.
    if not _PLACE_WORD.search(txt):
        return ""
    # POSITIVELY PHRASED. This said "in the {here}, not the room the scene text
    # names" -- and at cfg 1 there is no negative to carry the "not", so the
    # clause pointed the model at the scene paragraph's room in the very shot
    # that has to override it. The scene text is already in the prompt saying
    # living room; this only has to assert the new one, harder, and let the
    # stronger, later, more specific statement win on its own terms.
    return (f" This shot takes place in the {here}: the walls, floor, light and "
            f"furniture are the {here}'s throughout.")


# THINGS THAT ARE IN A ROOM RATHER THAN BEING ONE.
#
# A closed list of place words cannot name every room a script invents -- a dungeon,
# a cargo bay, a stable, a chapel, a sauna, a morgue -- and a closed list is exactly
# why "they head to the locker room" was read as going nowhere and the set changed
# under the characters instead of being walked into. Adding room words one report at
# a time fixes one script each.
#
# So the generalisable side of the problem is the INVERSE: an unlisted destination is
# taken as a place unless it is one of these. Furniture, fittings, a body part, a
# position within a space, a vehicle, or a person. There are far fewer common object
# destinations in English than there are names for rooms, and this list does not have
# to grow when somebody writes a scene nobody has written before.
#
# "door" is the original of this whole failure and is named first: _PLACE once held
# it, so "Ana looks at the door" moved the camera into a door.
_NOT_A_DESTINATION = frozenset("""
door doors doorknob handle window windows curtain curtains blind blinds mirror
sink basin bath tap taps table desk counter worktop bench chair seat stool sofa
couch armchair bed mattress headboard pillow cushion duvet quilt sheets blanket
cupboard cabinet drawer drawers shelf shelves wardrobe closet locker lockers
fridge freezer oven stove hob kettle microwave dishwasher washer machine
floor ground ceiling wall walls rail railing bannister bars bar post pole fence
light lights lamp switch socket screen tv television phone radio speaker camera
bag bags box crate case suitcase trunk basket bin sack tray bottle glass cup
car van truck bike motorbike trailer boat seat
edge middle centre center side sides end front back rear top bottom corner corners
spot position point row line queue
girl boy man woman lady guy person stranger guard nurse doctor teacher
hand hands arm arms elbow shoulder shoulders knee knees foot feet leg legs lap
face mouth lips chin neck throat hair head chest breast breasts stomach belly
waist hip hips thigh thighs wrist wrists ankle ankles bum butt crotch
""".split())

# Same shape as _GOES_TO, with an OPEN noun where that one has the place list.
# The destination STOPS at a conjunction. Without that guard the capture ran
# straight through one -- "walks to the bench and picks up a towel" produced the
# destination "bench and picks", whose last word is a verb, so the blocklist never
# saw the bench it was there to catch and the shot was told to travel "to the bench
# and picks". Every word of the destination is checked, not only the head, for the
# same reason.
_MOVES_TO_ANY = re.compile(
    r"\b(?:to|into|toward|towards|inside|through\s+to|"
    r"enters?|entered|entering|steps?\s+into|stepped\s+into)\s+"
    + _DET_POSS + r"\s+((?:(?!(?:and|or|then|but|while|as|before|after|with|for|"
    r"to|into|onto|from|at|on|in|of)\b)[A-Za-z][\w-]*\s+){0,2}"
    r"(?!(?:and|or|then|but|while|as)\b)[A-Za-z][\w-]*)\b", re.I)


def moved_to(beat, people=()):
    """Where this beat MOVES somebody, whatever the place is called. "" if nowhere.

    The place-list readers answer first and more richly, because a known room can be
    named at both ends of the journey. This is what answers when they cannot: a
    travel verb, a destination, and a head noun that is not furniture, a body part,
    a vehicle or a person. It establishes NO room state -- it does not decide cuts,
    it does not feed where_hold or the acoustics, and it never claims to know what
    kind of space it is. All it does is make the arrival be PERFORMED, which is the
    one thing the reported failure was missing."""
    b = _DIALOGUE_TAG.sub(" ", _QUOTED.sub(" ", str(beat or "")))
    if not _TRAVEL_VERB.search(b):
        return ""
    names = {str(n).strip().lower() for n in (people or ()) if str(n).strip()}
    for m in _MOVES_TO_ANY.finditer(b):
        dest = re.sub(r"\s+", " ", m.group(1)).strip()
        # THE HEAD NOUN DECIDES, not every word in the phrase. A locker is furniture
        # and a locker ROOM is a room; so are an engine room and a boiler room. The
        # conjunction guard in the pattern is what stops a run-on phrase reaching
        # here with a verb for a head, which is what let "the bench and picks" past a
        # blocklist that holds "bench".
        head = dest.split()[-1].lower().strip("-")
        if (len(head) < 3 or head in _NOT_A_DESTINATION or head in names
                or dest.lower() in names or _EXTRA_PEOPLE.search(dest)):
            continue
        return dest.lower()
    return ""


def move_clause(dest):
    """Perform an arrival the place list cannot name. "" when there is nowhere."""
    if not dest:
        return ""
    return (f" The shot travels to the {dest} on screen, the whole move played out "
            f"from its first step to its last.")


def travel_anchor(frm, via, to, here=""):
    """Say where the shot starts, what it passes, and where it ends. "" if nowhere.

    `here` is the room an earlier beat established, used when the beat names no
    origin -- a journey with only a destination is what renders as a cut.

    Short on purpose: this lands on travel beats, which already carry an action,
    and the node's whole balance problem is continuity crowding the beat out."""
    start = frm or here
    if not to or start == to:
        return ""
    # POSITIVELY PHRASED, and it was not. Every version of this clause ended
    # "-- one continuous move, not a cut", which puts the word CUT in the prompt
    # of the one shot that must not cut. At cfg 1 H3 evaluates no negative, so a
    # negation in the positive prompt is just the thing it names: this file says
    # so in eleven other places -- "no leggings" would be read as leggings -- and
    # then asked for a cut in the clause written to prevent one.
    #
    # Reported twice as an instant cut across a house, once after the destination
    # reader was fixed and the clause was demonstrably in the prompt. Say what the
    # shot DOES: the walk happens, on screen, in frame, the whole way.
    walk = "the walk between them played out on screen, every step in frame."
    if via:
        return (f" The shot opens in the {start}, carries along the {via}, and "
                f"arrives in the {to}, {walk}") if start else (
                f" The shot carries along the {via} and arrives in the {to}, "
                f"{walk}")
    # AN UNKNOWN ORIGIN IS STILL A JOURNEY. This returned nothing when the room
    # they set out from was not on the list or had never been named -- so a beat
    # walking out of a gym was told to walk nowhere, and with the previous shot's
    # last frame as its keyframe the set simply changed under the characters.
    # Reported as the scene shifting to the locker room instead of them walking into
    # it. Naming no origin is fine; what the shot needs is that the arrival is
    # PERFORMED. Positively phrased, like everything else at cfg 1: the way in, then
    # the room.
    if not start:
        return (f" The shot enters the {to} on screen: the way in first, then the "
                f"{to} itself, the arrival played out and every step in frame.")
    return f" The shot opens in the {start} and arrives in the {to}, {walk}"


_IS_IN = re.compile(r"\b(?:in|inside|within|at)\s+(?:the|her|his|their|a)\s+"
                    + _MOD + r"(" + _PLACE + r")\b", re.I)


def first_place(text):
    """The first place this text names at all. "" when it names none.

    place_named wants "in the kitchen"; a scene paragraph is more often just "A
    living room." with no preposition to hang on."""
    for m in _PLACE_WORD.finditer(str(text or "")):
        got = re.sub(r"\s+", " ", m.group(0)).strip().lower()
        # WORDS THAT ARE ALSO VERBS ARE NOT PLACES HERE. "McKenna steps out of
        # the far room" read as the flight of STEPS -- the same collision as
        # "door", found the same way, in the same sentence. This reader searches
        # free text with no preposition in front of it, so it cannot tell a noun
        # from a verb; the ones that collide are simply not allowed to win here.
        # place_named still reads them, because "at the top of the steps" has a
        # preposition and is unambiguous.
        if got in _PLACE_ALSO_A_VERB:
            continue
        # ...and a bare "room" names nowhere. "Ana walks into the room" says she
        # goes inside, not which room, and taking it as a place produced "This
        # shot is in the room, not the room the scene text names" -- a sentence
        # that contradicts itself. QUALIFIED it is a real place, and the word in
        # front is what qualifies it: the back room, the far room, the next room
        # are each somewhere distinct from where we just were.
        if got == "room":
            before = re.search(r"(\w+)\s+$", str(text or "")[:m.start()])
            word = before.group(1).lower() if before else ""
            if word in _NOT_A_ROOM_MODIFIER or not word:
                continue
            return (word + " room")
        return got
    return ""


def place_named(text):
    """The place this text says somebody is IN, without travelling. "" if none."""
    m = _IS_IN.search(str(text or ""))
    return re.sub(r"\s+", " ", m.group(1)).strip().lower() if m else ""


def rooms_named(text):
    """Every room this text names, lowercased. "" -> [].

    first_place returns only the FIRST one, which is what a tracked position needs.
    A scene paragraph often names two -- "Her bedroom has an unmade bed. The kitchen
    is small." -- and deciding whether a SENTENCE is about the room we are in means
    accounting for all of them.

    The same two exclusions as first_place, for the same reasons: a word that is also
    an ordinary verb cannot win in free text with no preposition in front of it, and a
    bare "room" names nowhere unless the word in front qualifies it."""
    out = []
    s = str(text or "")
    for m in _PLACE_WORD.finditer(s):
        got = re.sub(r"\s+", " ", m.group(0)).strip().lower()
        if got in _PLACE_ALSO_A_VERB:
            continue
        if got == "room":
            before = re.search(r"(\w+)\s+$", s[:m.start()])
            word = before.group(1).lower() if before else ""
            if word in _NOT_A_ROOM_MODIFIER or not word:
                continue
            got = word + " room"
        if got not in out:
            out.append(got)
    return out


def split_sheet(scene, names=()):
    """(everything that is not a character sheet entry, the sheet entries).

    WHAT LEADS A PROMPT DECIDES ITS COMPOSITION. This file already recorded that --
    "anatomy in the opening tokens is what a distilled LoRA settles composition on",
    which is why the gaze clause was moved to follow the beat rather than lead it --
    and then left the biggest anatomy block in the prompt leading every shot: the
    character sheet. "McKenna: she, 22, tall, long blonde hair, blue eyes, freckles"
    is sixteen words of face, it has to be in every shot because clothing continuity
    needs it there, and it sat in front of the action.

    Measured on a volleyball beat: 69% of the shot's words were in sentences about a
    face, and turning off every face guard only took that to 63%, because the sheet is
    most of it. Reported across many attempts as the camera fixated on one character
    staring into the lens -- with no reference image, no LoRA and a pinned first frame,
    none of which touched it, because none of them were what was leading the prompt.

    Splitting lets the scene keep the front, the beat follow it, and the appearance
    come after the thing it is describing. The words are identical; only the order
    changes, which is the one thing about this that was never tried."""
    cast = [str(n).strip() for n in (names or ()) if str(n).strip()]
    rest, sheet = [], []
    for raw in str(scene or "").split("\n"):
        for unit in re.split(r"(?<=[.!?])\s+", raw):
            u = unit.strip()
            if not u:
                continue
            if cast:
                entry = any(re.match(r"^" + re.escape(n) + r"\s*:", u, re.I) for n in cast)
            else:
                entry = ":" in u
            (sheet if entry else rest).append(u)
    return " ".join(rest), " ".join(sheet)


def scene_for_here(scene, here, always="", names=(), beat=""):
    """(text to send, rooms held back, True if it declined to hold anything).

    THE SCENE PARAGRAPH IS STAMPED INTO EVERY SHOT, and it has to be -- a removal
    needs the text to have something to scrub, and where_hold's own comment says the
    paragraph "still names the room they started in and is stamped into every shot".
    But a paragraph that describes the opening ROOM describes its FURNITURE too, and
    furniture does not travel. Reported: a flat whose scene paragraph read "Her
    bedroom has an unmade bed and a lamp", a walk from the bedroom down the hallway
    to the living room, and then A BED IN THE LIVING ROOM. where_hold had the room's
    NAME right in every shot; the bed was in the text standing beside it, and at cfg 1
    there is no negative prompt that can take a named thing back.

    THE ROOM THE SHOT ENDS IN decides this, not every room it passes through, and the
    difference is the whole fix. A walk out of the bedroom genuinely shows the bedroom
    in its opening frames -- but that shot's LAST frame is the next shot's keyframe, so
    a bed drawn at the end of the walk is inherited by the shot after it, which is the
    second route the same bed took into the living room. Nothing is lost by holding it
    there: the opening room arrives as a PICTURE regardless, because the keyframe is
    the previous shot's last frame and that frame IS the room being left. So the words
    describe where the shot ends and the frame carries where it began.

    A WITHHOLDING, NOT AN EDIT, exactly like the covered-garment deferral: the
    author's paragraph is untouched, every reader inside this file still sees all of
    it, this is only what the model is told for THIS shot, and a beat that walks back
    into the bedroom gets the bed back in full.

    TWO GUARDS. A sentence carrying a LABEL -- "McKenna: she, 22, ..." -- is a
    character sheet entry and is never touched whatever it names, because losing a
    person's line is the failure hide_item exists to prevent. And if holding would
    leave the shot no scene sentence at all, nothing is held: a sentence that welds
    the film's own framing to one room's furniture ("A small flat at night, her
    bedroom with an unmade bed") would otherwise take the night away with the bedroom,
    and a shot with no scene is a bigger change than the bug. The caller reports that
    case so the author can split the sentence.

    `always` IS THE ANCHOR AND IS NEVER HELD. build_scene fuses the anchor and the
    scene paragraph into one string before either reaches a shot, and an anchor is
    documented as what belongs to the WHOLE film -- "look, camera, lighting,
    location". So an anchor reading "Shot on 35mm in a cramped kitchen" names a room,
    and without this it was held on every shot outside that kitchen: the film lost its
    stock and its lens to a rule about furniture. The anchor's sentences are spared by
    text, which survives terminate_lines adding a full stop to them.

    `names` IS THE DECLARED CAST, and it is what identifies a sheet entry. A bare
    colon test was the first guard and it had a hole both ways: "Her bedroom: an
    unmade bed and a lamp." is the author describing a room, not a person, and it was
    protected as though it were somebody's line -- so the bed survived in that
    phrasing. Matching the LABEL against a name the sheet actually declares closes it
    without ever risking a person: with no cast passed it falls back to protecting any
    colon, because losing somebody's line is worse than a bed in one shot.

    A ROOM THE BEAT ITSELF NAMES IS NEVER HELD. `here` goes stale whenever the beat's
    verb is not one the movement readers know -- "McKenna pads into the kitchen" moves
    nobody as far as place_in is concerned -- and a stale room would hold the
    description of the room the shot is actually IN. The beat's own words outrank
    anything inferred from them, which is this file's standing rule, so a sentence
    about a room the beat mentions stays whatever the tracked room says.

    Holding NOTHING returns the text unchanged, byte for byte, so a script that never
    leaves one room is untouched and costs nothing."""
    text = str(scene or "")
    room = (here or "").strip().lower()
    if not text.strip() or not room:
        return text, [], False, []
    cast = [str(n).strip() for n in (names or ()) if str(n).strip()]

    def _is_sheet_entry(unit):
        u = unit.strip()
        for n in cast:
            if re.match(r"^" + re.escape(n) + r"\s*:", u, re.I):
                return True
        return bool(not cast and ":" in u)

    beat_rooms = set(rooms_named(
        _DIALOGUE_TAG.sub(" ", _QUOTED.sub(" ", str(beat or "")))))
    spared = set()
    for unit in re.split(r"(?<=[.!?])\s+", str(always or "")):
        u = unit.strip().rstrip(".!? ").lower()
        if u:
            spared.add(u)
    lines, held, held_text, survived = [], [], [], False
    for raw in text.split("\n"):
        kept = []
        for unit in re.split(r"(?<=[.!?])\s+", raw):
            # A sheet entry. Never touched.
            if _is_sheet_entry(unit):
                kept.append(unit)
                continue
            # ...nor anything the anchor said. It frames the whole film.
            if unit.strip().rstrip(".!? ").lower() in spared:
                kept.append(unit)
                survived = True
                continue
            named = rooms_named(unit)
            # It names another room and not this one. A sentence naming BOTH stays --
            # it is partly about where we are, and keeping too much is the safe way to
            # be wrong here.
            if named and room not in named and not (beat_rooms & set(named)):
                for r in named:
                    if r not in held:
                        held.append(r)
                if unit.strip() not in held_text:
                    held_text.append(unit.strip())
                continue
            kept.append(unit)
            survived = True
        lines.append(" ".join(k for k in kept if k.strip()))
    if not held:
        return text, [], False, []
    if not survived:
        return text, held, True, held_text
    return "\n".join(l for l in (s.strip() for s in lines) if l), held, False, held_text


def direction_anchor(changes):
    """Say which end of a staged change is which, for the ones that have a direction.

    Two at most, and the caller trades these against the held states: a shot carrying
    four continuity sentences is a shot that has stopped being about its beat."""
    said = []
    for thing, way in changes:
        if not way:
            continue
        start, end = ("shut", "open") if way == "open" else ("open", "shut")
        said.append(f"The {thing} {'are' if thing.endswith('s') else 'is'} {start} at "
                    f"the first frame and {end} by the last.")
        if len(said) == 2:
            break
    return (" " + " ".join(said)) if said else ""


def stated_states(text):
    """(thing, state) for every scenery state this text asserts but does not stage."""
    # A thing this same text WORKS is not a thing standing in a state: "slams the
    # tailgate shut" reads as both, and the action is the true reading. Left to the
    # caller this came back twice, once held and once anchored, disagreeing.
    out, seen = [], set(state_acts(text))
    for pat, order in ((_STATE_ADJ, "sn"), (_STATE_PRED, "ns")):
        for m in pat.finditer(text or ""):
            if order == "sn":
                state, gap, thing = m.group(1), m.group(2), m.group(3)
                # The same determiner test, from the other side: with one, this is
                # somebody closing the doors, and the state is not standing at all.
                if not _adjectival(state, gap):
                    continue
            else:
                state, thing = m.group(3), m.group(1)
            key = _state_key(thing)
            if key in seen:
                continue
            seen.add(key)
            out.append((thing.lower(), state.lower()))
    return out


# Getting OUT of a vehicle. A person leaving a van opens a door to do it, so a beat
# staging an exit and a state saying the doors are shut are two instructions that
# cannot both be followed. The beat wins -- it stages an action, and an action beats
# a state -- and the hold is left arguing with the script it is supposed to serve.
#
# The node must not touch the wording either way: those are the author's words, and
# "out of the van" may be exactly what they mean. So it says so instead. Three rounds
# of this went by as a silent bad render when one line of info would have placed it.
_EXIT_VEHICLE = re.compile(
    r"\b(?:get|gets|got|climb(?:s|ed)?|step(?:s|ped)?|jump(?:s|ed)?|slid(?:e|es)|"
    r"come|comes|came|walk(?:s|ed)?|hop(?:s|ped)?|pile)\s+(?:down\s+|back\s+)?out\s+"
    r"of\s+(?:the\s+|a\s+|an\s+|his\s+|her\s+|their\s+|its\s+)?"
    r"(?:back\s+of\s+(?:the\s+|a\s+)?)?(?:van|car|truck|cab|vehicle|lorry|bus)\b"
    r"|\bexits?\s+(?:the\s+|a\s+)?(?:van|car|truck|cab|vehicle)\b"
    r"|\bout\s+of\s+(?:the\s+|a\s+)?(?:van|car|truck|cab)\b", re.I)


def exits_vehicle(text):
    """Does this beat stage somebody getting out of a vehicle?"""
    return bool(_EXIT_VEHICLE.search(text or ""))


_PICTURE_TAG = re.compile(r"<\s*picture[\s_\-]*(\d+)\s*>", re.I)


def renumber_reference_tags(text, wired):
    """Rewrite <Picture N> from INPUT number to position in the reference roster.

    The roster is packed dense -- the wired images become picture 1, 2, 3 in the
    order of their sockets -- but nobody writing a sheet knows that. They write the
    number on the socket, which is what the README documents. Wire ref_image_1 and
    ref_image_3 and the two conventions disagree: <Picture 3> names nothing in a
    roster of two, so the tag was stripped and the image was dropped in silence.

    `wired` is the socket numbers that actually have an image, in socket order. With
    no gaps this is the identity mapping and nothing changes, which is why the fault
    stayed hidden -- everybody fills the sockets from the top until they don't."""
    seat = {slot: i + 1 for i, slot in enumerate(wired)}
    if not text or all(k == v for k, v in seat.items()):
        return text
    return _PICTURE_TAG.sub(
        lambda m: (f"<Picture {seat[int(m.group(1))]}>"
                   if int(m.group(1)) in seat else m.group(0)), text)


def unwired_reference_tags(text, wired):
    """Tag numbers naming a socket with no image on it. Sorted, no repeats."""
    return sorted({int(m.group(1)) for m in _PICTURE_TAG.finditer(text or "")}
                  - set(wired or ()))


def handoff_rides_as_ref(handoff, refs, ref_noise_aug):
    """Is the shot handoff about to be encoded as a subject reference?

    Mirrors the demotion in build_conditioning. Read in the render loop as well,
    because the text has to claim the picture and the text is written up there."""
    return bool(handoff is not None and refs
                and not (ref_noise_aug is None
                         or float(ref_noise_aug) >= KEYFRAME_SAFE_AUG))


def handoff_claim(n):
    """Name the demoted handoff as this shot's opening frame.

    Below KEYFRAME_SAFE_AUG the handoff stops being a keyframe and is encoded as an
    extra reference -- and it was going in unclaimed, on the reasoning that a first
    frame is not a subject and needs no tag. It needs one HERE. In the reference
    rows it is not a first frame any more, it is picture N of N, and the rule that
    governs those is the node's oldest: a picture the prompt names is that subject,
    and a picture it never names is ANOTHER subject.

    So the last shot of a run carried a second person wearing the previous shot's
    clothes and face -- reported as a duplicate at the end of the video, and only
    ever below 0.99, which is why lowering the aug to strengthen identity was what
    produced the twin."""
    return (f" <Picture {n}> is the frame this shot opens on: the same place and the "
            f"same people, one moment earlier, carried forward rather than joined by "
            f"anybody new.")


def room_claim(n, present, joining):
    """Claim a handoff carried as a reference because somebody NEW is in the shot.

    The keyframe used to be thrown away here, and throwing it away is what the
    node's own note in build_conditioning warns about: with no handoff the VLM is
    never shown where the shot left off and re-imagines the scenery -- same place,
    new room. Reported as the scene not staying the same between shots.

    A keyframe and a reference are different instruments. A keyframe IS frame one,
    so a newcomer absent from it has to walk in from nowhere, which is the bug the
    fresh start was for. A reference only supplies appearance, so the same picture
    carries the room and the people already in it while the newcomer is simply
    there at the first frame.

    Claimed, and specifically. An unclaimed picture of somebody is another person
    who looks like them, and the standing claim is worse than nothing here: it says
    the shot is joined by nobody new, in the one case where it is."""
    said = (f" <Picture {n}> is this room a moment earlier: the same walls, floor, "
            f"furniture and light, from the same camera.")
    if present:
        said += (f" {' and '.join(present)} "
                 f"{'are the people' if len(present) > 1 else 'is the person'} there.")
    if joining:
        said += (f" {' and '.join(joining)} {'are' if len(joining) > 1 else 'is'} in "
                 f"this room too, already in place at the first frame.")
    return said


def _join_names(names):
    """"Nora", "Nora and Dan", "Nora, Dan and Mara" -- a list a reader can read."""
    names = [str(n) for n in (names or []) if str(n).strip()]
    if len(names) < 2:
        return names[0] if names else ""
    return ", ".join(names[:-1]) + " and " + names[-1]


def plate_claim(n):
    """Claim shot 1's first_frame when it is carried as the SET rather than frame one.

    room_claim cannot serve here and saying so is the point: it calls the picture
    "this room a moment earlier" and names who was standing in it, and on shot 1
    there is no earlier and nobody was. A plate is a picture of a place with no
    people in it, and the claim has to say exactly that -- an unclaimed picture is
    read as another subject, and a picture of an empty room claimed as a person is
    how a figure gets invented to stand in it."""
    return (f" <Picture {n}> is the set this shot takes place in: the same walls, "
            f"floor, furniture and light, from the same camera. It is a picture of "
            f"the place only, with nobody in it -- the people in this shot are the "
            f"ones named above, standing where the text puts them.")


def state_hold(pairs):
    """One sentence putting those states at the first frame instead of in the action.

    Two at most. These sentences are continuity, and continuity that outgrows the
    beat is what the beat stops being about."""
    said = []
    for thing, state in pairs[:2]:
        plural = thing.endswith("s")
        # BOUNDED, for the same reason a removal says "by the last frame": "stays
        # closed" has no end on it, and a state with time left over is a state
        # something can happen to before the shot is out.
        said.append(f"The {thing} {'are' if plural else 'is'} already {state} at the "
                    f"first frame and {'stay' if plural else 'stays'} {state} for the "
                    f"whole shot.")
    return (" " + " ".join(said)) if said else ""


def rigid_hardware(text):
    """Is the hardware here the kind that cannot flex?"""
    return bool(_RIGID_HARDWARE.search(text or ""))


def restraint_present(text):
    """Is a restraint being applied or worn, in this text?

    Plain hardware counts on its own. Ambiguous hardware needs a binding verb or a
    body part alongside it, so a chain-link fence and a leather belt do not arm a
    continuity rule about restraints."""
    t = text or ""
    if _RESTRAINT_PLAIN.search(t):
        return True
    # SAME CLAUSE. Both halves were searched across the whole text, however far
    # apart: a sheet listing a belt and a beat saying "she sits with her legs
    # crossed" satisfied both, so the belt became restraint hardware and the hold
    # latched from there -- every later shot told to keep fastened something that
    # was never a restraint. The qualifier has to be near the hardware to qualify it.
    for part in re.split(r"(?<=[.;!?])\s+", t):
        if _RESTRAINT_MAYBE.search(part) and (_BINDING_VERB.search(part)
                                              or _BODY_PART.search(part)):
            return True
        # ...OR the hardware is fastened to something that does not move. That is
        # what a restraint IS, and the verb list could not see it: _BINDING_VERB
        # holds participles only -- "chained", "clipped", "bolted" is not even in
        # it -- so "the guard chains her collar to the wall", "clips the chain to a
        # ring in the wall" and "a short chain holds her collar to the wall" all
        # came back with no restraint at all, and the anchor reader is only ever
        # consulted once one is latched. Six of fourteen ordinary phrasings.
        #
        # _ANCHOR_POINT is safe to lean on here because it now demands a fastening
        # verb of its own, so this cannot fire on somebody merely walking to a wall.
        if _RESTRAINT_MAYBE.search(part) and _ANCHOR_POINT.search(part):
            return True
    return False


def names_any(text, tokens):
    """Does `text` name any of these items?"""
    return any(re.search(r"\b" + re.escape(t) + r"\b", text or "", re.I)
               for t in (tokens or []) if t)


# Where a removal verb's object ENDS. "pulls off her coat and drops it, showing the
# jumper" takes off the coat; the jumper is what becomes visible. The old version of
# this node matched garment words anywhere in the beat and took both off, which is
# the failure that made prose inference untrustworthy.
def person_tags(text, objects=None):
    """The <Picture N> tags that belong to a PERSON rather than to an object.

    Decided by what stands immediately BEFORE the tag. A name -- capitalised, with or
    without its colon -- means the picture is of that person: "Nora: <Picture 1>",
    "Nora <Picture 1> in a grey coat". A lowercase noun means it is a picture OF the
    thing it is standing next to: "a silver locket <Picture 2>".

    That distinction is what lets an object's reference come off with the object. A
    person's tag has to survive a removal that shares its fragment, or the shot loses
    its identity reference; an object's tag has to go, or it keeps asserting the thing
    that was just taken off."""
    out = []
    for m in _PICTURE_TAG.finditer(text or ""):
        before = (text[:m.start()]).rstrip().rstrip(",").rstrip()
        w = re.search(r"([\w'’-]+)$", before)
        # An object owns the tag only when a lowercase NOUN stands immediately before
        # it -- "a silver locket <Picture 2>". Everything else is the person's: a
        # name, a colon, an age ("Kate is 20, <Picture 1> blonde crop top"), or
        # nothing at all. Erring this way on purpose, because losing a person's
        # identity reference costs the shot its face, while an object tag left behind
        # only keeps describing something already taken off.
        if not before.endswith(":") and w:
            head = w.group(1)[:1]
            if head.isalpha() and head.islower():
                continue                      # the picture belongs to the object
        # ...and the same claim written the other way round. "<Picture 2> a chastity
        # belt" puts the tag in FRONT, where there is nothing before it to read, so
        # the rule above called it the person's and the tag survived the belt going
        # under the jeans -- which kept sending the belt's picture into every covered
        # shot, to be drawn on top of them.
        #
        # Only ever for a tag standing directly in front of the thing being REMOVED,
        # which is the one case where the answer is not in doubt. A tag with nothing
        # before it and nothing of ours after it stays the person's, as it was.
        # Only when there is NOTHING in front of it. "Kate is 20, <Picture 1> blonde
        # crop top" also puts a tag before a garment, and that one is hers -- the age
        # standing in front is what says so. Reading ahead there would take her
        # identity reference off with the top.
        if objects and w is None and not before.endswith(":"):
            ahead = (text[m.end():]).lstrip()
            if any(re.match(r"(?:(?:a|an|the|her|his|their)\s+)?(?:[\w-]+\s+){0,2}"
                            + re.escape(o) + r"\b", ahead, re.I) for o in objects if o):
                continue
        out.append(m.group(1))
    return out


_OBJECT_END = re.compile(r"(?:,|;|\.|\bexposing\b|\brevealing\b|\bshowing\b|\bleaving\b|"
                         r"\bto\s+expose\b|\bto\s+reveal\b|\bthen\b|\buntil\b)", re.I)

# Words that sit in a removal's object span but are never the thing that comes off:
# grammar, the prepositions that place a garment, and the body it is placed on.
# "cuts the tight top away from her back" names ONE garment; the rest is syntax and
# anatomy. Without this, every word the beat happened to share with the scene was
# taken off -- "the tight and the her and the back come off during this shot".
_NOT_A_GARMENT = frozenset("""
the a an and or her his its their our your this that these those
"""
# PRONOUNS, AND ONE OF THEM DESTROYED THE CHARACTER ENTRY. The object span runs to
# the next clause boundary and "and" is not one -- deliberately, so that "unzips her
# jacket and pulls it off" reads as one removal -- which puts the SUBJECT of the
# next clause inside the span. "Kate pulls off the jumper and she sits down" offered
# "she", and a sheet declares its pronoun exactly the way it lists a garment
# ("Kate: she, 28, a wool jumper"), so the positional entry-head test said yes.
#
# The shot then said "The wool jumper and the she come off during this shot", and
# the scrub drops the whole comma-separated entry it matched -- so "Kate: she, 28, a
# wool jumper, a denim skirt." became "28, a denim skirt." in every later shot. Name
# gone, pronoun gone, person gone: shots with nobody described in them, which is the
# cost this file already records for scrubbing a sheet line.
             """
she he him them they us we you one both each either neither
herself himself themselves myself yourself itself
somebody someone anybody anyone nobody everybody everyone
off from over under onto into out down up away through across behind
front side left right rest way bit end edge
back neck chest waist hips hip wrist wrists ankle ankles arm arms hand hands
leg legs thigh thighs knee knees foot feet shoulder shoulders head face mouth
lips hair skin body torso stomach belly chin jaw eyes ear ears
floor ground wall room air
"""
# FIXTURES AND FURNITURE. A garment is recognised by POSITION here, not by
# vocabulary -- see the note above infer_removals, and the reason is good: an
# author writes garments this file has never heard of, and a vocabulary would
# drop them silently. The cost is that position cannot tell a shower from a
# shirt. "Kate steps out of the shower" is the same shape as "Kate steps out of
# the thong", and the scene paragraph lists the shower the same way a sheet
# lists a skirt -- so the shower was taken off her and scrubbed out of every
# later shot, in a bathroom scene, which is a room that quietly stops existing.
# "Kate kicks the stool away" took the stool.
#
# So the list is the other way round: not what a garment IS, but the handful of
# things a person can step out of, get off, drop onto or kick away that are
# plainly not worn. Anything not named here still reaches the positional test,
# which is what keeps an unheard-of garment working.
             """
shower showers bath baths bathtub tub tubs basin sink sinks toilet loo cubicle
stall stalls bed beds sofa sofas couch couches chair chairs stool stools bench
seat seats armchair table tables desk desks counter shelf shelves cupboard
cabinet drawer drawers door doors doorway window windows mirror curtain
car cars cab taxi van truck lift elevator stairs step steps
kitchen bathroom bedroom hallway corridor landing garden street pavement
water pool puddle steam tiles tile mat mats rug rugs carpet basket hamper
""".split())

# Where a scene's wardrobe entry ENDS. A garment word is the HEAD of its phrase --
# "black boots," "grey coat and", "wool scarf." -- while a modifier is followed by
# more of the phrase ("tight white crop top": tight, white and crop all fail this,
# top passes). Adjectives cannot be listed, so test position instead of vocabulary.
_ENTRY_END = re.compile(r"^\s*(?:[,;.!?]|$|(?:and|over|under|beneath|above|with|plus)\b)",
                        re.I)

# Hardware, not clothing. Inference never takes a restraint off: the standing rule is
# that once one goes on it stays on, and an explicit `remove:` is the only thing that
# clears it. A beat that cuts a rope must not silently unlock the cuffs as well.
_RESTRAINT_WORD = re.compile(
    r"^(?:handcuffs?|cuffs?|shackles?|manacles?|chains?|ropes?|cords?|straps?|"
    r"collars?|gags?|blindfolds?|restraints?|bindings?|tape|ties?|harness|"
    r"straitjacket|spreader|hogtie|clamps?|clips?)$", re.I)


# A <Picture N> immediately after a word, so the entry-end test can look past an
# object's own reference to the comma that actually ends its entry.
_LEADING_TAG = re.compile(r"^\s*<\s*picture[\s_\-]*\d+\s*>", re.I)


def _is_entry_head(word, scene):
    """Is `word` the head of a wardrobe entry in the scene, rather than a modifier
    inside one or a fragment of a hyphenated compound?"""
    for m in re.finditer(r"\b" + re.escape(word) + r"\b", scene, re.I):
        # "tight" inside "skin-tight" is half a word, not a garment.
        if m.start() and scene[m.start() - 1] == "-":
            continue
        if m.end() < len(scene) and scene[m.end()] == "-":
            continue
        # An object's own reference sits between the noun and the comma that ends its
        # entry -- "a silver locket <Picture 2>, green jacket" -- so the entry-end
        # test has to look past it. Without this, a tagged object is never the head of
        # anything, which means auto_remove can never take it off: it needed an
        # explicit `remove:` line while an untagged one came off from the prose.
        tail = _LEADING_TAG.sub("", scene[m.end():], count=1)
        if _ENTRY_END.match(tail):
            return True
    return False


def _modifier_of_a_named_entry(word, span, scene):
    """Is `word` a MODIFIER of a longer garment the same span already names?

    "Dan pulls off her jeans shorts" names one garment. But "jeans" is also the
    head of Dan's own entry, so the reader matched it against his line and took
    HIS jeans off as well -- in a beat that never mentions him. His trousers came
    off automatically, and stayed off.

    The span is what the beat says is coming off. If the word is immediately
    followed there by another garment word, it is describing that one, not naming
    a second: the phrase is "jeans shorts", and only "shorts" is the head.
    """
    m = re.search(r"\b" + re.escape(word) + r"\b\s+([\w-]{3,})", span or "", re.I)
    if not m:
        return False
    nxt = m.group(1).lower().strip("-")
    # ...and only when that following word is itself a garment the scene lists,
    # so "jeans and boots" -- two garments -- is not read as one.
    return bool(nxt and nxt not in _NOT_A_GARMENT and _is_entry_head(nxt, scene))


# A garment MOVED rather than taken off: pulled down, pushed up, shoved aside, left
# hanging open. It is still on the body and still in the picture, so it has to go on
# being described -- but described as it now is, or the next shot puts it back the way
# the sheet says it was worn.
# "back up" first, so it is matched whole. Written as two words it is the commonest
# way anybody says a garment is being put right, and matching only "back" left the
# trailing "up" outside the pattern -- so the restore looked like a new displacement.
# DISPLACEMENT AND RESTORE LIVE IN THE ENGINE, with the garment vocabulary
# they both read. Split across two files this pair went wrong three ways in
# one day; together they cannot disagree about what a garment is called.
_DISPLACE = engine._DISPLACE
scene_name_for = engine.scene_name_for
displaced_garments = engine.displaced_garments
puts_it_back = engine.puts_it_back
restored_garments = engine.restored_garments


def displaced_hold(items):
    """Say where a moved garment now sits, so the next shot does not put it back.

    Without this the garment is described by the sheet in the state it was WORN, and
    the sheet is re-stamped into every shot -- so shorts pulled down are pulled back
    up by the next beat, or come back looking like a different pair."""
    if not items:
        return ""
    said = ", ".join(f"the {thing} {how}" for thing, how in items[:2])
    # Sentence-initial "Still" meaning "nevertheless". Nothing else in the prompt
    # opens on that word, and "On the body" carries the fact by itself.
    return f" On the body and {said}, left exactly where the beat put them."


# A REQUEST is not the thing happening. "McKenna asks Dan to take the chastity belt
# off" contains a removal verb and a garment the scene says is worn, which is all
# infer_removals needs -- so asking for it stripped it, and the shot was then told the
# belt comes off and is away by the last frame. She asks, and it falls off.
#
# Worse where the answer is no: "she asks him to remove the belt. He shakes his head."
# took the belt off anyway, which is the script's meaning inverted.
#
# Only the verb inside the REQUEST is discounted. A beat that asks and is then obeyed
# in its own words -- "she asks him to unlock it, and he does" -- still has a removal
# in the second half, and that half is read normally.
_ASK_VERB = (r"asks?|asked|asking|begs?|begged|begging|pleads?|pleaded|pleading|"
             r"wants?|wanted|wishes|wished|tells?|told|orders?|ordered|demands?|"
             r"demanded|whispers?|whispered|says?|said|shouts?|shouted|screams?|"
             r"screamed")
_REQUEST = re.compile(
    # "asks him TO take it off"
    r"\b(?:" + _ASK_VERB + r")\b[^.;!?]{0,60}?\bto\s+(?=[a-z])"
    # "asks FOR the belt to come off"
    r"|\b(?:asks?|asked|begs?|begged|pleads?|pleaded)\b[^.;!?]{0,40}?\bfor\b"
    # "asks IF he will unlock it" / "asks WHETHER he can" -- an indirect question
    # has no "to" at all, so the first branch never saw it.
    r"|\b(?:asks?|asked|asking|wonders?|wondered)\b[^.;!?]{0,40}?\b(?:if|whether)\b",
    re.I)

# SPEECH is a request too. "McKenna approaches Dan. \"Will you take the chastity belt
# off?\"" has no asking verb before the removal at all -- the words are quoted, and a
# line of dialogue asking for a thing is not the thing happening. Nor is an imperative:
# "\"Take the chastity belt off.\"" is her telling him to, not him doing it.
#
# Only what is INSIDE the quotes. A beat that quotes a request and then narrates the
# act -- "\"Take it off.\" He unlocks the belt." -- still has a removal outside them.
_QUOTED = re.compile(r"\"[^\"]*\"|“[^”]*”|<d>.*?</d>", re.S)


def _in_quotes(text, at):
    """Is position `at` inside a span of dialogue?"""
    return any(m.start() <= at < m.end() for m in _QUOTED.finditer(text or ""))


# A question is a request whatever introduced it: "Will you take it off?" is asking,
# and so is "Can you", "Would you", "Could you". Judged by the question MARK, which
# is the one reliable mark of an interrogative in prose.
_QUESTION = re.compile(r"[^.;!?]*\?")


def _in_a_question(text, at):
    """Is the removal verb at `at` inside a sentence that ends in a question mark?"""
    return any(m.start() <= at < m.end() for m in _QUESTION.finditer(text or ""))


def _in_a_request(text, at):
    """Is the removal verb at `at` inside a request rather than an action?"""
    # Asked in someone's own words, or asked as a question: either way, not done.
    if _in_quotes(text, at) or _in_a_question(text, at):
        return True
    start = max((m.end() for m in _REQUEST.finditer(text or "") if m.end() <= at),
                default=None)
    if start is None:
        return False
    # Only up to the end of that clause: a request in one sentence does not reach
    # into the next, where the thing may actually be done. ", and he removes it"
    # is a new clause too, so a comma before a conjunction ends the request as
    # surely as a full stop does -- otherwise asking and then being obeyed inside
    # one sentence reads as pure request and the removal is lost.
    stop = re.search(r"[.;!?]|,\s*(?:and|then|so|but)\b", (text or "")[start:])
    return at <= (start + stop.start() if stop else len(text or ""))


# The object of a removal verb when the beat has already named the garment: "and
# steps out of it". Anchored at the start of the object span, so a pronoun further
# along the sentence is not mistaken for the object.
_PRONOUN_OBJECT = re.compile(r"\s*(?:it|them|these|those)\b", re.I)
_SENTENCE_BREAK = re.compile(r"[.;!?]\s+")


def _sentence_before(beat, at):
    """The sentence `at` is in, up to `at`. The pronoun's antecedent lives here.

    A beat is a paragraph and can hold several sentences. "It" reaches back across
    a comma or an "and", not across a full stop."""
    cut = max((m.end() for m in _SENTENCE_BREAK.finditer(beat[:at])), default=0)
    return beat[cut:at]


def infer_removals(beat, scene):
    """Garments this beat takes off, read from its own prose. [] when none.

    Two conditions, both required, because a wrong removal is worse than a missed
    one: the beat has to contain a REMOVAL verb, and the thing named has to be
    something the SCENE already says is worn. A beat cannot take off what the
    character was never described wearing.

    Only the verb's own object counts -- the span from the verb to the next clause
    boundary. That is what keeps "pulls off her coat, showing the jumper" to the
    coat."""
    if not beat or not scene:
        return []
    found = []
    for m in _REMOVAL_PROSE.finditer(beat):
        # Asked for is not done. See _in_a_request.
        if _in_a_request(beat, m.start()):
            continue
        _before = len(found)
        tail = beat[m.end():]
        cut = _OBJECT_END.search(tail)
        span = tail[:cut.start()] if cut else tail
        # In the TRAILING form the object sits between the verb and the particle --
        # "takes her jacket off" -- so the particle ends the object, and what comes
        # after it is a new clause: in "takes her jacket off and drops it on the
        # chair" the chair is furniture the beat mentions, not something worn.
        #
        # A verb before the particle means the particle is not ours. "kicks the
        # chair and Mike walks off" ends in "off", but it is the walking that is off,
        # and reading that as a removal deleted the chair from the scene.
        #
        # Neither test applies to a verb that already swallowed its particle
        # ("pulls off her coat") or needs none ("unzips her jacket and pulls it
        # off"), where the object follows the verb and the sentence runs on.
        if not (re.fullmatch(_UNDO_VERB, m.group(0), re.I)
                or re.search(r"\b(?:off|away|out\s+of|down)$", m.group(0), re.I)):
            part = re.search(r"\b(?:off|away)\b", span, re.I)
            if part:
                if _HAS_VERB.search(span[:part.start()]):
                    continue
                span = span[:part.start()]
        for word in re.findall(r"\b[\w-]{3,}\b", span):
            # THE TOKEN THE SHEET WROTE. "take off their skirts" gives "skirts" and the
            # sheet says "a denim skirt", so the entry-head test below found nothing and
            # the removal did nothing -- on every beat where more than one person
            # undressed. One owner for the normalisation, in the engine, because this
            # reader and garment_words both look the token up in the same sheet.
            low = engine.singular_garment(word)
            if not low or low in found:
                continue
            # Grammar, prepositions and anatomy are not garments.
            if low in _NOT_A_GARMENT:
                continue
            # Hardware is cleared by an explicit `remove:` and by nothing else.
            if _RESTRAINT_WORD.match(low):
                continue
            # It has to be worn: the HEAD of something the scene lists, not a
            # modifier inside it and not half of a hyphenated compound.
            if not _is_entry_head(low, scene):
                continue
            # "her jeans shorts" is ONE garment. "jeans" there is a modifier, but it
            # is also the head of Dan's own entry, so it matched his line and took
            # HIS trousers off in a beat that never mentions him -- and they stayed
            # off, because a removal is permanent.
            if _modifier_of_a_named_entry(word, span, scene):
                continue
            # ...and not a person or a place.
            if re.search(r"\b" + re.escape(word) + r"\b\s*(?:is|was|walks|stands|sits|=)",
                         scene, re.I):
                continue
            found.append(low)
        # "...AND STEPS OUT OF IT." The object is a pronoun, and the garment was
        # named one clause earlier -- which is how most undressing is actually
        # written: the hands arrive first ("hooks her thumbs in the thong"), the
        # removal second, and by then the thing has a pronoun. The word loop above
        # cannot see a pronoun at all; it skips anything under three letters. So the
        # removal verb matched, the span held nothing it recognised, and NOTHING came
        # off -- the sheet went on dressing her in the garment in every later shot,
        # which is the author's removal silently reversed.
        #
        # Resolved the way the unnamed restore is: only when there is exactly one
        # thing it can mean. The candidate has to be a garment by vocabulary AND an
        # entry the sheet dresses somebody in, and it is read from THIS SENTENCE only
        # -- a garment mentioned in an earlier sentence of the same beat is not what
        # "it" refers to, and guessing across a full stop is how a coat comes off in
        # a beat about a towel.
        if len(found) == _before and _PRONOUN_OBJECT.match(span):
            _near = []
            for _g in garments_in(_sentence_before(beat, m.start())):
                _low = engine.singular_garment(_g) or _g
                if (_low in _NOT_A_GARMENT or _RESTRAINT_WORD.match(_low)
                        or not _is_entry_head(_low, scene) or _low in _near):
                    continue
                _near.append(_low)
            # Compared on the garment KEY, not the word. The earlier clause names
            # the thing in full ("the chastity belt") while the word loop recorded
            # its head ("belt"), so a plain membership test read them as two
            # garments and took the same one off twice.
            if len(_near) == 1 and not any(engine._garment_key(x)
                                           == engine._garment_key(_near[0])
                                           for x in found):
                found.append(_near[0])
    # A garment the beat says is EXPOSED cannot also be one it takes off. "Pulls off
    # her coat to show the jumper underneath" ran the removal verb's object span past
    # "to show" and took the jumper with it -- so the one garment the beat exists to
    # reveal was scrubbed from the wardrobe, and every shot after it described bare
    # skin where the jumper was. Reported as a removal going straight past what the
    # sheet said was underneath.
    #
    # The comma form (", showing the jumper") already ended the span correctly, which
    # is why this only bit one phrasing of the two.
    shown_off = exposed_by(beat, scene)
    return [f for f in found if f not in shown_off]


# Clothing, for the one case that names no garment at all: "strips out of their
# clothes". A vocabulary is the wrong tool for reading a removal out of prose -- which
# is why infer_removals tests POSITION instead -- but here the beat says nothing about
# WHAT comes off, so the only place left to read it from is the wardrobe itself.
#
# Anything this misses stays described, and the note says which entries were cleared,
# so a gap is visible rather than silent.
# GARMENTS LIVE IN THE ENGINE. There were two vocabularies here and in
# engine.py and they disagreed -- this one had thong and no chastity belt,
# that one had chastity belt and matched the bare "belt" inside it. Both
# were fixed on the same day from opposite ends. One list now, and the two
# readers that need different answers are built on it rather than on each
# other: garment_words gives head words for tracking, garments_in keeps the
# adjectives for the text.
garments_in = engine.garment_words
# Which body region a garment covers -- read from the engine's own table so
# frame scoping and the bare/undress logic can never disagree about where a
# garment sits.
region_of = engine.region_of


# A beat that undresses somebody completely without naming one garment. Every other
# removal path needs the thing to be named; this is the case where the SCRIPT does not
# name it, so nothing came off and the scene went on listing the whole wardrobe --
# which is re-stamped into every later shot, so the clothes came back on.
#
# "naked eye" and "naked flame" are not people.
_NAKED_CUE = re.compile(
    r"\bnaked\b(?!\s+(?:eye|flame))"
    r"|\bnude\b|\bin\s+the\s+nude\b"
    r"|\bundress(?:es|ed|ing)?\b"
    r"|\bstrips?\s+(?:out\s+of|off|down|naked|bare)\b|\bstripp(?:ed|ing)\s+"
    r"(?:out\s+of|off|down|naked|bare)\b"
    r"|\btakes?\s+(?:everything|it\s+all|all\s+of\s+it|the\s+lot)\s+off\b"
    # A GENERIC garment word as the object. "Sam takes off his clothes" is the
    # commonest way anybody writes this, and it named no garment the sheet lists,
    # so every other path had nothing to remove: his wardrobe stayed in the scene
    # text and was re-stamped into every later shot, which is the clothes still
    # being on. Her named garments came off; his generic ones never did.
    r"|\b(?:takes?|took|taking|pulls?|pulled|peels?|peeled|sheds?|shed|"
    r"removes?|removed|gets?|got|slips?|slipped)\b"
    r"(?:\s+(?:off|out\s+of))?\s+(?:his|her|their|its|the|all\s+(?:his|her|their))?"
    r"\s*(?:clothes|clothing|garments|things|kit|outfit|gear)\b"
    r"(?:\s+off)?"
    # A bare "strips" only when it takes NO object: "Sam strips." undresses him,
    # "she strips the paint off the door" and "strips a length of tape" do not.
    # The object is what tells them apart, so anything but a clause end is out.
    r"|\bstrips?\b(?=\s*[.,;!?]|\s*$)"
    r"|\bstripp(?:ed|ing)\b(?=\s*[.,;!?]|\s*$)"
    r"|\bwearing\s+nothing\b|\bwith\s+no\s+clothes\b|\bbare\s+skin\b", re.I)


def strips_who(beat, cast):
    """Who this beat undresses. [] when it cannot tell.

    strips_bare only answers WHETHER somebody ends up with no clothes on. The
    wardrobe was then read off the whole shot sheet, so in a shot describing two
    people BOTH were stripped -- one character undressing made the other undress
    too. Reported as the second character mimicking the first.

    The subject is the name before the cue, the same reading posture_in uses. With
    one person in the shot there is nobody else it can be."""
    people = [n for n in (cast or []) if n]
    b = str(beat or "")
    if not people or not b:
        return []
    if len(people) == 1:
        return people[:1]
    m = _NAKED_CUE.search(b)
    if not m:
        return []
    # The SUBJECT is the span between the last clause boundary and the cue, not the
    # nearest name: "McKenna and Dan undress" is a compound subject and both are
    # stripped, while "McKenna watches as Dan undresses" is Dan alone.
    before = b[:m.start()]
    cut = max((c.end() for c in
               re.finditer(r"[.;!?]\s+|,\s*|\s+(?:as|while|and then|then|but)\s+",
                           before)), default=0)
    span = before[cut:]
    here = [n for n in people
            if re.search(r"\b" + re.escape(n) + r"\b", span, re.I)]
    if here:
        return here
    # No name before it: the beat's own first-named person is acting.
    first = next((n for n in people
                  if re.search(r"\b" + re.escape(n) + r"\b", b, re.I)), None)
    return [first] if first else []


def strips_bare(text):
    """Does this beat say somebody ends up with no clothes on?"""
    return bool(_NAKED_CUE.search(text or ""))


# Said once, in place of listing every garment separately. Restraints are named
# because they do NOT come off here, and a sentence about everything coming off would
# otherwise be read as including them.
BARE_HOLD = (" Everything worn comes off during this shot and is away by the last "
             "frame, leaving bare skin from the shoulders down; whatever is fastened "
             "to the body stays fastened exactly as it was.")


def missing_removals(beat, scene, already):
    """Garment words the SCENE still describes, in a beat whose prose takes
    something off and which carries no `remove:` line for them.

    Reports; never acts."""
    if not scene or not _REMOVAL_PROSE.search(beat or ""):
        return []
    hits = []
    for word in re.findall(r"\b[\w-]{4,}\b", beat or ""):
        low = word.lower().strip("-")
        if not low or low in already or low in hits or low in _NOT_A_GARMENT:
            continue
        # Same discipline as the inference: the head of an entry, not a modifier
        # inside one. Reporting "back" and "her" as unremoved garments is noise
        # that buries the one line that matters.
        if _is_entry_head(word, scene):
            hits.append(low)
    # Words that are in the scene because they are the PERSON or the place, not
    # something worn. A name or a room is not a garment.
    return [h for h in hits if not re.search(
        r"\b" + re.escape(h) + r"\b\s*(?:is|was|walks|stands|sits)", scene, re.I)]


def extract_directives(beat):
    """(beat text with directive lines taken out, [removed tokens], [added phrases]).

    `add:` is the other half of `remove:`, and it exists because of a specific
    failure: a scene that lists every layer at once -- coat, jumper, shirt --
    tells the model the character is wearing all of them simultaneously, with
    nothing saying which is hidden. The keyframe pins the first frame, so early
    frames look right; by the last frame only the text is governing, and the under
    layer starts showing through the top one.

    So describe what is VISIBLE, and add a layer when it becomes visible:

        Dan cuts off her jacket and throws it away.
        remove: jacket
        add: her white shirt underneath

    The added phrase is appended to the scene from that shot onward, in your words,
    unchanged."""
    removed, added = [], []

    def take_removed(m):
        removed.extend(t.strip() for t in m.group(1).split(",") if t.strip())
        return ""

    def take_added(m):
        phrase = m.group(1).strip()
        if phrase:
            added.append(phrase)
        return ""

    body = _ADD_LINE.sub(take_added, _REMOVE_LINE.sub(take_removed, beat or ""))
    return re.sub(r"\n{2,}", "\n", body).strip(), removed, added


def extract_removals(beat):
    """Back-compatible shim: (body, removed tokens)."""
    body, removed, _ = extract_directives(beat)
    return body, removed


# What makes a garment-less fragment read as CONTINUING the item before it. A
# print cue, a quoted span, a pronoun pointing back, a fragment that opens with
# the preposition that would have followed the noun -- or a capitalised word
# placed ON the garment. Capitals alone are not enough: "PVC mini-skirt" is a
# material, and a first version took the skirt with the belt in front of it.
# "red lipstick" or "a tattoo across the lower back" has none of these and
# stands on its own; "BRAT across the back" and "with a bow at the hip" do not.
_PRINT_WORDS = re.compile(
    r"\b(?:print(?:ed|s)?|lettering|letter(?:s|ed)?|text|reads?|reading|says|"
    r"written|writing|embroider(?:ed|y)|emblazoned|stitched|stamped|logo|slogan|"
    r"motto|monogram(?:med)?|words?|font|spell(?:s|ed|ing)?|its|it)\b", re.I)
_QUOTED_SPAN = re.compile(r'["“][^"”]+["”]')
_CAPS_WORD = re.compile(r"\b[A-Z]{2,}\b")                           # case matters
_ON_GARMENT = re.compile(
    r"\b(?:across|on|along|down|over)\s+(?:the|its|her|his|their)\s+"
    r"(?:front|back|chest|waistband|hem|seat|rear|crotch|straps?|cups?|hips?|"
    r"bum|butt)\b", re.I)
_CONTINUES = re.compile(r"^\s*(?:with|across|along|down|over|on|at|bearing|"
                        r"reading|printed|lettered|emblazoned)\b", re.I)


def _continues_item(unit):
    return bool(_PRINT_WORDS.search(unit) or _QUOTED_SPAN.search(unit)
                or _CONTINUES.search(unit)
                or (_CAPS_WORD.search(unit) and _ON_GARMENT.search(unit)))


def hide_item(text, items):
    """Take the named items out of a sheet line, keeping everything else.

    SURGICAL, unlike scrub_removed, which drops the whole comma-separated
    fragment -- that is right for a garment that has come off and wrong here: it
    took "green dress, steel collar" down to nothing and the person's line with
    it, leaving shots with nobody described in them.

    This removes the item and the adjectives attached to it, and stops. A
    fragment that held only that item disappears; a fragment holding anything
    else keeps the rest. A fragment carrying the person's LABEL ("McKenna: she,
    22") never disappears, whatever else is in it."""
    if not text or not items:
        return text
    pats = [re.compile(r"(?:\b\w+[\w-]*\s+){0,3}?\b" + re.escape(str(i).strip()) + r"\b",
                       re.I) for i in items if str(i).strip()]
    out_lines = []
    for line in str(text).split("\n"):
        frags, kept = line.split(","), []
        trailing = False        # the unit just before this one went with its garment
        entry = ":" in line     # a labelled sheet entry: where attribute lists live
        for n, frag in enumerate(frags):
            # A UNIT IS A SENTENCE, not only a comma-fragment. A fragment holding
            # "denim shorts. She wears a black thong. BRAT is printed across the
            # back." kept all of it because the shorts were still in it, and
            # shipped "She . BRAT is printed across the back." -- a stub and a
            # stranded print. Each sentence is judged alone, and the ones kept
            # are put back with the single space that separated them.
            units, kept_units = re.split(r"(?<=[.!?])\s+", frag), []
            for unit in units:
                new = unit
                for p in pats:
                    # The item, plus any adjectives sitting directly in front of it.
                    new = p.sub("", new)
                removed = new != unit
            # THE PRINT ON A COVERED GARMENT GOES WITH THE GARMENT.
            #
            # Reported: a thong under shorts, lettering on the thong, and the
            # lettering rendered ON THE SHORTS. The removal above takes the item
            # plus up to three words IN FRONT of it and nothing behind, so
            #     "denim shorts, a black thong with \"PRINCESS\" across the front."
            # became
            #     "denim shorts, with \"PRINCESS\" across the front."
            # -- the garment deleted out from under its own modifier, which then
            # sits in the list right after the shorts. A described print is a drawn
            # print and it is drawn on whatever garment is still there to carry it.
            # It survived the emptiness test below because that only strips
            # articles: the leftovers read as '"PRINCESS"acrossthefront'.
            #
            # So a fragment this removal EMPTIED OF GARMENTS goes whole. Narrowly:
            # only when something was actually removed from it, only when no
            # garment word is left -- "a thong and denim shorts" keeps the shorts,
            # which is the case hide_item exists to protect -- and never when the
            # fragment carries the person's LABEL, which would take their name out
            # of the sheet with it.
                if (removed and ":" not in unit
                        and not garments_in(new) and re.search(r"\w", new)):
                    trailing = True
                    continue
            # An article left standing alone ("a", "the") is not a garment,
            # so the fragment goes. A fragment carrying the person's LABEL
            # never reaches this test empty -- the removal takes the item and
            # leaves the name -- which is why there is no separate guard for
            # it. One was written; a disable-check showed it never fired, and
            # a guard that looks protective and is not is worse than none.
                if not re.sub(r"\b(?:a|an|the|and|with|in)\b|[\s,.;]", "", new):
                    if removed:
                        trailing = True
                    continue
                # THE PRINT IN ITS OWN FRAGMENT GOES TOO. The rule above catches a
                # print written inside the garment's fragment; one written after
                # the comma -- "a black thong, BRAT across the back, denim shorts"
                # -- had nothing removed from it, so it stayed, now sitting right
                # before the shorts with no garment to carry it. Reported as the
                # thong's lettering on the shorts, again. A garment-less unit that
                # reads as continuing the one just dropped goes with it; anything
                # else stands on its own and ends the chain.
                if (entry and not removed and trailing and ":" not in unit
                        and not garments_in(unit) and _continues_item(unit)):
                    continue
                kept_units.append(new)
                trailing = False
            if kept_units:
                kept.append(" ".join(kept_units))
        joined = ",".join(kept)
        # Tidy the seams the removal leaves: doubled commas and spaces.
        joined = re.sub(r"\s*,\s*,+", ",", joined)
        joined = re.sub(r"\s{2,}", " ", joined).strip()
        joined = re.sub(r",\s*([.;]|$)", r"\1", joined)
        # A dropped sentence can leave the next fragment's comma sitting right
        # after the previous full stop: "A bright beach., on the sand".
        joined = re.sub(r"([.!?])\s*,\s*", r"\1 ", joined)
        # The seams a removal leaves at the LABEL. "Ana: chastity belt, jeans"
        # becomes "Ana: , jeans" and "Ana: a chastity belt" becomes "Ana: ."
        # Both are malformed, and a sheet entry the reader cannot parse is
        # worse than one item missing from it.
        joined = re.sub(r":\s*,\s*", ": ", joined)
        joined = re.sub(r":\s*(?=[.;]|$)", "", joined)
        # The removed fragment may have carried the line's full stop away
        # with it. terminate_lines expects one, and without it the next
        # sheet line welds onto this one -- a name fused to the end of an
        # attribute list reads as one more item in it.
        if (line.rstrip().endswith((".", "!", "?")) and joined
                and not joined.endswith((".", "!", "?"))):
            joined += "."
        out_lines.append(joined)
    return "\n".join(out_lines)


def strippers_in(beat, sheet):
    """Who this beat says takes something off. [] when it does not say.

    subjects_for with the compound subject vocal_sources_in already needed: "McKenna
    and Tess take off their shirts" shares ONE verb between two names, and the
    conjunction guard -- right about "Dan holds the door and McKenna undresses", where
    `and` opens a new predicate -- cannot tell that apart on its own. Getting this
    wrong in the narrowing direction would leave a garment on somebody who took it
    off, so both names are kept."""
    verbs = engine._STRIP_VERB + "|" + engine._UNDO_VERB
    b = str(beat or "")
    out = list(subjects_for(b, sheet, verbs))
    for n, _ln in sheet_lines(sheet):
        if not n or n in out:
            continue
        if re.search(r"\b" + re.escape(n) + r"\b(?:\s*,\s*[\w'\u2019-]+)*"
                     r"\s+and\s+[\w'\u2019-]+\s+(?:" + verbs + r")\b", b, re.I):
            out.append(n)
    return out


def scrub_removed(text, tokens):
    """Drop the parts of `text` that name a removed item.

    Comma-separated fragments first, because that is how a scene lists what someone
    is wearing ("blonde, 20, grey jacket, black boots"). A sentence that is left
    with no words at all is dropped whole, so "She wears a red coat." disappears
    rather than becoming a stub."""
    if not text or not tokens:
        return text
    live = [t for t in tokens if t]
    pats = [re.compile(r"\b" + re.escape(t) + r"\b", re.I) for t in live]
    # A scene lists what someone wears as comma-separated NOUN PHRASES ("blonde,
    # pale blue cotton shirt, heavy black waxed canvas jacket"). For those, the
    # whole entry goes: trimming a fixed number of modifiers off the front left
    # orphans like "heavy black waxed" sitting in the list, and an orphan
    # description is read as some garment -- which is a garment coming back.
    #
    # A fragment with a VERB in it is prose, not a list entry, and there the entry
    # is only part of the sentence, so it gets the surgical treatment below.
    kept = []
    for sent in re.split(r"(?<=[.!?])\s+", text):
        # Ownership is decided on the WHOLE SENTENCE, then applied per fragment.
        #
        # person_tags reads what stands immediately before a tag, and splitting on
        # commas throws that away: " <Picture 2> a chastity belt" has nothing in front
        # of it once detached, so the tag fell back to "the person's" and survived the
        # belt being scrubbed -- an orphaned tag, which still fetches the picture. The
        # sentence has "blue jeans" in front of it and answers correctly.
        #
        # This also settles "Kate is 20, <Picture 1> blonde crop top" the same way and
        # without special-casing: in the full sentence the age stands before the tag,
        # so it is hers and stays.
        _person_tags = set(person_tags(sent))
        frags = sent.split(",")
        out_frags = []
        for frag in frags:
            if any(p.search(frag) for p in pats) and not _HAS_VERB.search(frag):
                # Restraint hardware is not clothing. An entry describing it goes
                # only when a token NAMES it: dropping "wrists handcuffed behind
                # her back" whole because a removal named "back" takes the cuffs
                # out of the prompt entirely, and hardware absent from the text
                # renders absent. Keep the fragment; the surgical pass below still
                # trims the token's own words out of it.
                if restraint_present(frag) and not any(_RESTRAINT_WORD.match(t)
                                                       for t in live):
                    out_frags.append(frag)
                    continue
                # One entry can carry two garments joined by "and" -- "a grey coat
                # and black boots". Dropping it whole takes the innocent one with
                # it, and an undescribed garment is one the model re-invents. So
                # drop only the side that names the removed item.
                sides = re.split(r"\s+\band\b\s+", frag, flags=re.I)
                gone = [s for s in sides if any(p.search(s) for p in pats)]
                keep = [s for s in sides if s not in gone] if len(sides) > 1 else []
                # A PERSON's tag must not leave with a garment that happened to share
                # its fragment -- losing it costs that shot its identity reference.
                # An OBJECT's tag is the opposite case: "a silver locket <Picture 2>"
                # is a picture OF the locket, so when the locket comes off the tag has
                # to come off with it. Left behind it kept asserting the thing that
                # was just removed, and a tag pointing at a picture nothing in the
                # text accounts for is also how a spare subject gets drawn.
                #
                # The person's tag is the one in the fragment carrying their LABEL --
                # "Nora: <Picture 1>" -- because that is where a sheet entry puts it.
                # Any other tag belongs to whatever it is standing next to.
                # `live` is what is being removed, so a tag standing in front of
                # one of those belongs to it and goes with it.
                # A leading tag is the person's or the object's depending on the
                # ENTRY, which a comma fragment cannot see. "Kate is 20,
                # <Picture 1> blonde crop top" and "<Picture 2> a chastity belt"
                # are the same shape once split. What tells them apart is whether
                # the person is ALREADY tagged at their label: if she is, a later
                # tag cannot be hers as well.
                tags = [n for s in (gone or [frag]) for n in picture_tags(s)
                        if str(n) in _person_tags]
                piece = " and ".join(k for k in keep if k.strip())
                if tags:
                    piece = ((piece + " ") if piece.strip() else "") + \
                            " ".join(f"<Picture {n}>" for n in tags)
                if piece.strip():
                    out_frags.append(piece)
                continue                      # the rest of the entry goes
            out_frags.append(frag)
        rebuilt = ",".join(out_frags)
        # A sentence's full stop lives on its LAST fragment. Dropping that fragment
        # -- which is exactly what removing the last-listed garment does -- takes the
        # full stop with it and runs the sentence into the next one: "blue eyes
        # Wrists cuffed behind back." Put the terminator back.
        end = re.search(r"([.!?])\s*$", sent)
        if end and rebuilt.strip() and not re.search(r"[.!?]\s*$", rebuilt):
            rebuilt = rebuilt.rstrip().rstrip(",;") + end.group(1)
        kept.append(rebuilt)
    out = " ".join(k for k in kept if k.strip())
    for t in live:
        # The item and the words that belong to it -- an article and up to two
        # modifiers -- and nothing else. Deleting the whole comma fragment took
        # neighbours with it: removing "jacket" from "a grey jacket over a white
        # shirt" deleted the shirt too, and an undescribed garment is one the model
        # re-invents, which looks like the clothing changing by itself.
        #
        # AND THE OBJECT'S OWN TAG WITH IT. "a chastity belt <Picture 2>" is a picture
        # OF the belt: take the words and leave the tag, and the shot carries a
        # reference with nothing in the text accounting for it. The comma-list path
        # above already knew this; this path did not, so any object written into a
        # fragment with a verb -- "wearing a chastity belt <Picture 2>" -- was scrubbed
        # to "wearing <Picture 2>". Reported as the object looking different when it
        # came back into view: the shots where it was covered still sent its picture,
        # unclaimed, and whatever those shots made of it is what the next shot
        # inherited as a keyframe.
        #
        # Only a tag STANDING ON the removed words. A person's tag sits after their
        # label -- "Mara: <Picture 1>" -- never after a garment, so it cannot be taken
        # by this: losing it would cost that shot its identity reference.
        #
        # A LEADING tag counts too. "<Picture 2> a chastity belt" is the same claim
        # written the other way round, and taking only the trailing form left the tag
        # standing when the belt went under the jeans -- so the image was still sent
        # on every covered shot and drawn on top of them. The words stopping is not
        # the same as the picture stopping.
        #
        # No comma may sit between: "Mara: <Picture 1>, blue jeans" has the person's
        # tag in front of a garment, and consuming across the comma would take her
        # identity reference with the jeans.
        out = re.sub(r"(?:<\s*picture[\s_\-]*\d+\s*>\s*)?"
                     r"\b(?:(?:a|an|the|her|his|their)\s+)?(?:[\w-]+\s+){0,2}"
                     + re.escape(t) + r"\b(?:\s*<\s*picture[\s_\-]*\d+\s*>)?",
                     "", out, flags=re.I)
    # Tidy what the deletion left behind, without touching anything it did not.
    # Twice: removing a stranded verb can strand the conjunction in front of it
    # ("Kate is 20 and wears a grey jacket" -> "... and wears" -> "... and").
    for _ in range(2):
        out = re.sub(r"\s{2,}", " ", out)
        # "wearing and black boots" / "wears over a white shirt"
        out = re.sub(r"\b(wearing|wears|in|dressed)\s+(?:and|over|under|with)\s+",
                     r"\1 ", out, flags=re.I)
        # a clothing verb with nothing left to govern
        out = re.sub(r"\s*\b(?:wearing|wears|dressed in)\s*(?=[.,;]|$)", "", out, flags=re.I)
        # a connector left hanging before punctuation or the end
        out = re.sub(r"\s+(?:and|over|under|with)\s*(?=[.,;]|$)", "", out, flags=re.I)
        out = re.sub(r",\s*(?=,)", "", out)
        out = re.sub(r"\s*,\s*(?=[.!?])", "", out)
        out = re.sub(r"\s+([.,;!?])", r"\1", out)
        # A dropped entry can leave its comma flush against the next one. Not
        # before a digit, so a thousands separator survives ("1,500").
        out = re.sub(r",(?=[^\s,\d])", ", ", out)
        # A dropped entry can leave the "and" that joined it to the next one
        # stranded at the front of the survivor: "30, and a long coat".
        out = re.sub(r"(,\s*)(?:and|or)\s+", lambda m: m.group(1), out, flags=re.I)
    out = re.sub(r"\s{2,}", " ", out)
    # Drop a sentence the deletion emptied, and one it reduced to a bare subject
    # ("She wears a red coat." -> "She.") -- which describes nobody and is one more
    # mention of a person, which is its own problem.
    kept = []
    for sent in re.split(r"(?<=[.!?])\s+", out):
        s = sent.strip()
        if not re.search(r"[A-Za-z0-9]", s):
            continue
        # ...including one left with only a copula: "She is wearing a belt." can
        # come down to "She is.", which is the same empty mention with a verb on
        # the end. The removal took everything the sentence was about.
        if re.fullmatch(r"(?:he|she|they|it|[A-Z][\w-]*)"
                        r"(?:\s+(?:is|are|was|were|has|have|had))?\s*[.!?]?",
                        s, re.I):
            continue
        kept.append(s if s[-1] in ".!?" else s + ".")
    return " ".join(kept).strip()


def _upscale_model_list():
    """Filenames in models/upscale_models, plus 'none'. Read fresh at INPUT_TYPES
    time so newly-added models show up on a graph reload."""
    try:
        import folder_paths
        return ["none"] + list(folder_paths.get_filename_list("upscale_models"))
    except Exception:
        return ["none"]


def upscale_video_latent(video, model_name, scale):
    """(upscaled_video_latent, note). Never raises -- a failure returns the input.

    Spatial only: the temporal length comes back unchanged, which is what lets this
    sit between sampling and decode without touching the audio half or the frame
    count the rest of the chain has already committed to."""
    if not model_name or model_name == "off" or float(scale) <= 1.0:
        return video, ""
    cls = latent_upscaler_node()
    if cls is None:
        return video, ("latent_upscale is set but the 'Minimax H3 Latent Upscaler' node pack is "
                       "not installed, so the shots were rendered at their sampled size. Install "
                       "Comfyui_Minimax_h3_latent_Upscaler, or set latent_upscale to 'off'")
    try:
        before = tuple(video.shape)
        # Its UpscaleMode is a str-Enum, so the literal VALUE compares equal without
        # importing the pack. Read the enum off the class when it is reachable, and
        # fall back to the literal -- hardcoding a foreign string is the fragile part
        # of this integration, so it is not the only path.
        mode_val = "scale by multiplier"
        try:
            mode_val = sys.modules[cls.__module__].UpscaleMode.SCALE_BY
        except Exception:
            pass
        out = _invoke_node(cls, latent={"samples": video},
                           model_name=model_name,
                           mode={"mode": mode_val, "scale": float(scale)},
                           align=32, device="cuda", precision="fp16")
        up = out["samples"] if isinstance(out, dict) else out
        if up is None or up.dim() != video.dim() or up.shape[2] != video.shape[2]:
            # A temporal change would desync the audio half and the frame count.
            return video, ("the latent upscaler returned an unexpected shape, so the shot was "
                           "left at its sampled size")
        return up.to(video.dtype), (f"latent upscale {model_name} x{float(scale):g}: "
                                    f"{before[-2]}x{before[-1]} -> {up.shape[-2]}x{up.shape[-1]} "
                                    f"latent cells per frame, sampled small and decoded large")
    except Exception as e:
        return video, (f"latent upscale failed ({type(e).__name__}), so the shots were rendered "
                       f"at their sampled size")


def _latent_upscale_model_list():
    """H3 latent-upscaler weights in models/latent_upscale_models, plus 'off'.

    Filtered to H3 builds: that folder also holds LTX spatial/temporal upscalers,
    and offering one here would let it be picked for a model it cannot take -- the
    first conv is [512, 24, 3, 3, 3] and 24 is H3's latents_dim specifically.

    Listed whether or not the node pack that RUNS them is installed. The widget has
    to exist unconditionally or a saved workflow would lose its widget positions the
    moment the pack was uninstalled; being unable to run is handled at render time."""
    try:
        import folder_paths
        d = os.path.join(folder_paths.models_dir, "latent_upscale_models")
        names = [f for f in sorted(os.listdir(d))
                 if f.lower().endswith((".pth", ".safetensors"))
                 and ("minimax" in f.lower() or "h3" in f.lower())]
    except Exception:
        names = []
    return ["off"] + names


def latent_upscaler_node():
    return _find_node(["minimaxh3latentupscaler", "3d"]) or _find_node(["minimaxh3latentupscaler"])


def landing_schedule(model, scheduler, steps, shift_video, shift_audio):
    """The sigmas this shot will run on, with the audio landing added. None if not.

    None means "take the ordinary path and change nothing", and it is returned for
    every reason there is: comfy not reachable, a schedule that already lands
    softly, anything unexpected. The schedule is built the way KSampler builds it --
    calculate_sigmas(model_sampling, scheduler, steps) at denoise 1.0 -- so what is
    handed back is the shot's own schedule with one step spliced into the end,
    never a different one."""
    try:
        import comfy.samplers as _cs
        _ms = model.get_model_object("model_sampling")
        base = [float(x) for x in _cs.calculate_sigmas(_ms, str(scheduler), int(steps))]
        landed = insert_audio_landing(base, shift_video, shift_audio)
        if len(landed) == len(base):
            return None
        return torch.tensor(landed, dtype=torch.float32)
    except Exception:
        return None


def sample_shot(model, cond, negative, latent, seed, steps, cfg, sampler_name,
                scheduler, sigmas=None, shift_video=None, shift_audio=None,
                soft_landing=False):
    """One sampling pass. denoise is fixed at 1.0: partial denoise desyncs the
    joint audio/video schedule."""
    if sigmas is not None and len(sigmas):
        return _sample_on_sigmas(model, seed, cfg, sampler_name, cond, negative,
                                 latent, sigmas)
    # THE AUDIO BRANCH'S LANDING. Only where the caller has established that this
    # schedule drops the audio from a height, and only when the node is the one
    # setting the shift -- with apply_model_sampling off, the shifts this is
    # computed from are not the shifts the model is using. See insert_audio_landing.
    if soft_landing:
        _own = landing_schedule(model, scheduler, steps, shift_video, shift_audio)
        if _own is not None:
            return _sample_on_sigmas(model, seed, cfg, sampler_name, cond, negative,
                                     latent, _own)
    (out,) = nodes.common_ksampler(model, seed, steps, cfg, sampler_name, scheduler,
                                   cond, negative, latent, denoise=1.0)
    return out


_HERE = os.path.dirname(os.path.abspath(__file__))


# (default, min, max, cast) for every numeric widget, so a value that cannot be used
# as a number can be replaced by the one the widget was built with.
_WIDGET_RANGE = {
    "megapixels": (1.0, 0.0, 2.0, float),
    "shot_seconds": (10.0, 1.0, 15.0, float),
    "steps": (8, 1, 100, int),
    "cfg": (1.0, 1.0, 20.0, float),
    "shift_video": (12.0, 1.0, 20.0, float),
    "shift_audio": (3.0, 1.0, 20.0, float),
    "ref_noise_aug": (0.999, 0.5, 1.0, float),
    "latent_upscale_scale": (2.0, 1.0, 4.0, float),
    "upscale_target_short_edge": (0, 0, 4096, int),
    "upscale_batch": (4, 1, 64, int),
    "pace": (1.0, 0.25, 2.0, float),
    "ambient_level": (0.25, 0.0, 1.0, float),
    "foley_level": (0.35, 0.0, 1.0, float),
    "speech_lead_seconds": (0.5, 0.0, 2.0, float),
    "speech_tail_seconds": (2.0, 0.0, 10.0, float),
    "hold_levels": (0.8, 0.0, 1.0, float),
}


def misaligned_widgets(values, options):
    """[(widget, value, what it should have been)] for choices that are not choices.

    sane_widgets repairs a NUMBER that arrives as NaN, and that is the visible symptom
    of a positional shift. It cannot see the cause, and it cannot help the widgets
    whose values are WORDS: a shift puts a scheduler's name into sampler_name and a
    seed into scheduler, and those pass straight through into the render.

    A combo holding a value that is not one of its own options is not a preference
    this node can honour. It is proof the list is out of step -- values are restored
    by POSITION, so converting one widget to an input, or adding or removing one,
    slides every value after it into the wrong slot."""
    bad = []
    for name, choices in (options or {}).items():
        if name not in values:
            continue
        got = values[name]
        if got not in choices:
            bad.append((name, got, choices))
    return bad


def combo_options(spec):
    """{widget: [options]} for every choice widget the node declares."""
    out = {}
    for section in ("required", "optional"):
        for name, decl in (spec or {}).get(section, {}).items():
            if decl and isinstance(decl[0], list):
                out[name] = list(decl[0])
    return out


def alignment_error(bad):
    """The message for a workflow whose widget values have slid out of position."""
    if not bad:
        return ""
    shown = "; ".join(f"{n} = {v!r}, which is not one of {c[:3]}"
                      + ("..." if len(c) > 3 else "") for n, v, c in bad[:3])
    return (
        "H3-LongVideos: this node's saved widget values are out of position. "
        + shown + ".\n\n"
        "Widget values are restored by POSITION, with no names stored, so converting "
        "a widget to an input -- or adding or removing one -- slides every value after "
        "it into the wrong slot. A scheduler's name lands in sampler_name, a seed in "
        "scheduler, and a number with nowhere to go reads as NaN.\n\n"
        "To fix it: right-click the node and choose 'Fix node (recreate)', or convert "
        "any widget you turned into an input back to a widget. Then set the values you "
        "want and save the workflow again. Nothing is wrong with the model or the "
        "prompt, and rendering with these values would use settings you did not pick.")


def sane_widgets(values):
    """(repaired values, notes) for the numeric widgets.

    Saved workflows restore widget values BY POSITION, with no names stored. Remove or
    reorder a widget and every later value shifts up one, so a boolean can land in a
    FLOAT slot -- which is where a widget reading NaN comes from, and a NaN pace makes
    NaN shot lengths and a render that never starts.

    A value that will not become a finite number falls back to the widget's built-in
    default; one that is merely out of range is clamped. Reported either way, because
    silently substituting a number the user did not choose is how a wrong render looks
    like a broken node."""
    out, notes, unusable = dict(values), [], []
    for name, (default, lo, hi, cast) in _WIDGET_RANGE.items():
        if name not in out:
            continue
        raw = out[name]
        try:
            if isinstance(raw, bool):
                raise TypeError("a boolean is not a setting for this widget")
            num = float(raw)
            if num != num or num in (float("inf"), float("-inf")):
                raise ValueError("not a finite number")
        except (TypeError, ValueError):
            out[name] = default
            unusable.append(f"{name} was {raw!r}, now {default}")
            continue
        clamped = min(max(num, lo), hi)
        if clamped != num:
            notes.append(f"{name} was {num:g}, outside {lo:g}..{hi:g}, so it was clamped "
                         f"to {clamped:g}")
        out[name] = cast(clamped)
    # ONE note for all of them. This used to emit a paragraph per widget, and a
    # workflow whose values have slid produces several at once -- the same
    # explanation three or four times, at the top of every run, which buries the
    # notes that are about the film. Said once, with the list.
    if unusable:
        notes.insert(0, "widget values that were not usable numbers, replaced with "
                        "their defaults: " + "; ".join(unusable)
                     + ". Values are restored BY POSITION with no names stored, so this "
                       "means the node's widget list and the saved one disagree -- "
                       "usually because a widget was converted to an input, or the node "
                       "gained one. It repairs itself for THIS run only: the graph still "
                       "holds the bad values, so it comes back every restart until the "
                       "node is fixed. Right-click the node and choose 'Fix node "
                       "(recreate)', set your values, and save the workflow")
    return out, notes


class H3LongVideos:
    """One prompt -> a chain of MiniMax-H3 shots, joined into one video."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "vae": ("VAE",),
                "audio_vae": ("VAE",),
                "prompt": ("STRING", {"multiline": True, "forceInput": True,
                    "tooltip": "Paragraph 1 is the SCENE, prepended to every shot verbatim. "
                               "Every paragraph after it is one beat = one shot.\n\n"
                               "Nothing is rewritten. What you type is what the shot is told, "
                               "plus the scene line. Put a quoted \"line of dialogue\" in a beat "
                               "and that shot keeps its audio; beats without one are silenced."}),
                "resolution": (list(NATIVE_RES), {"default": "16:9",
                    "tooltip": "Aspect ratio. megapixels sets the size."}),
                "megapixels": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05,
                    "tooltip": "1.0 = 1024x1024 worth of pixels, H3's native budget. Lower is "
                               "faster and leaner; 0 keeps the preset's own dimensions. Cost "
                               "scales with latent cells and attention is quadratic in them."}),
                "shot_seconds": ("FLOAT", {"default": 10.0, "min": 1.0, "max": 15.0, "step": 0.5,
                    "tooltip": "Maximum shot length. With 'from the beat', each shot is sized "
                               "independently up to this cap; with 'fixed', every shot uses this "
                               "length. Snapped to H3's 17k+5 frame grid."}),
                "steps": ("INT", {"default": 8, "min": 1, "max": 100,
                    "tooltip": "6-8 with a turbo/distill LoRA; 20+ without one."}),
                "cfg": ("FLOAT", {"default": 1.0, "min": 1.0, "max": 20.0, "step": 0.1,
                    "tooltip": "H3 is CFG-free. At 1.0 the negative prompt is never evaluated -- "
                               "which is why nothing here is phrased as a negation."}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS, {"default": "res_multistep"}),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, {"default": "simple"}),
                # control_after_generate DECLARED, not left implicit. The frontend adds
                # that control by itself for any INT named "seed", so it existed in the
                # panel while the backend knew nothing about it -- the UI's widget list
                # was one longer than this one, and widget values are restored BY
                # POSITION. Declaring it is what ComfyUI's own KSampler does
                # (nodes.py:1602), and it makes the two lists agree on where every
                # later value belongs.
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff,
                    "control_after_generate": True,
                    "tooltip": "One seed for the whole chain. Every shot is the same length, so "
                               "they share a noise field."}),
            },
            "optional": {
                "first_frame": ("IMAGE", {"tooltip":
                    "Pins the opening frame of shot 1 -- the only shot with no previous frame to "
                    "continue from.\n\n"
                    "It pins the WHOLE frame, so give it a composed frame of the shot you want: "
                    "subject, pose, framing, background. A head-and-shoulders portrait wired here "
                    "makes shot 1 a head-and-shoulders portrait. An identity portrait belongs on "
                    "ref_image_1, which says who the person is without dictating the frame.\n\n"
                    "OR GIVE IT THE SET, with nobody in it, and the node will read it that way: "
                    "when beat 1 PLACES the cast rather than staging an entrance, and every one of "
                    "them already has a <Picture N> reference of their own, this picture carries "
                    "the room, the light and the furniture as a reference and never becomes frame "
                    "one. That is the difference between a set and an opening frame -- pinned as "
                    "frame one, a picture with nobody in it makes the cast appear out of nothing "
                    "during shot 1, which is the same reason a later shot refuses the previous "
                    "frame when it introduces somebody in position.\n\n"
                    "Both readings are reported in info, so you can see which one you got. To "
                    "force the pinned reading, put the cast in the frame and drop their "
                    "<Picture N> tags, or write the entrance into beat 1."}),
                "ref_image_1": ("IMAGE", {"tooltip":
                    "Identity reference, applied to every shot unless the prompt places it with a "
                    "<Picture 1> tag. Kept on every shot on purpose: it is the only fixed anchor a "
                    "long chain has, and without it shot 11 is drift piled on drift."}),
                "ref_image_2": ("IMAGE",),
                "ref_image_3": ("IMAGE",),
                "ref_image_4": ("IMAGE",),
                "negative": ("CONDITIONING", {"tooltip":
                    "Ignored at cfg 1.0, which is where H3 runs. Wired for completeness."}),
                "sigmas": ("SIGMAS", {"tooltip":
                    "An external schedule (PDD Acc's Apply node). Drives the sampler directly; "
                    "steps and scheduler are then only for the progress bar."}),
                "shift_video": ("FLOAT", {"default": 12.0, "min": 1.0, "max": 20.0, "step": 0.1}),
                "shift_audio": ("FLOAT", {"default": 3.0, "min": 1.0, "max": 20.0, "step": 0.1,
                    "tooltip": "Keep video:audio near 4:1. H3 carries the audio latent on the "
                               "video schedule scaled by that ratio; flattening it breaks audio."}),
                "apply_model_sampling": ("BOOLEAN", {"default": True,
                    "tooltip": "Patch the dual video/audio schedule inside the node. Turn off only "
                               "if you patch it upstream yourself."}),
                "silence_nonspeech": ("BOOLEAN", {"default": True,
                    "tooltip": "Anchor the audio branch to real silence on any shot with no quoted "
                               "line. H3 is joint -- an unconditioned audio stream invents a voice "
                               "and the picture lip-syncs to it. This conditions the stream itself "
                               "rather than asking the prompt to stop it."}),
                "trim_seam": ("BOOLEAN", {"default": True,
                    "tooltip": "Drop the first frame of every shot after the first: it is the "
                               "model's own reproduction of the keyframe, so it is a duplicate."}),
                "ref_noise_aug": ("FLOAT", {"default": 0.999, "min": 0.5, "max": 1.0, "step": 0.005,
                    "tooltip": "How CLEAN a reference is shown. 0.999 (H3's default) hands over a "
                               "noise-free image, which invites the model to REPRODUCE it -- "
                               "including its background and pose -- in the opening frames. Lower "
                               "says approximate: try 0.95, then 0.90. One aug covers every "
                               "conditioning latent, so below 0.99 the keyframe rides as a "
                               "reference instead of an anchor."}),
                "tiled_decode": ("BOOLEAN", {"default": True,
                    "tooltip": "Decode in tiles. The whole-clip decode is the single largest "
                               "allocation in a run and the usual point a big checkpoint spills."}),
                "cleanup_between_shots": ("BOOLEAN", {"default": True,
                    "tooltip": "Move each finished shot to system RAM and purge VRAM between "
                               "shots, so a long chain does not accumulate on the card."}),
                "latent_upscale": (_latent_upscale_model_list(), {"default": "off",
                    "tooltip": "Upscale each shot in LATENT space, between sampling and decode, "
                               "so the shot is SAMPLED small and only DECODED large. That is the "
                               "cheap one: cost scales with latent cells and attention is "
                               "quadratic in them, so sampling 512x512 and upscaling 2x is far "
                               "less work than sampling 1024x1024.\n\n"
                               "Model and nodes by LBH-123-AI; needs the separate Minimax H3 "
                               "Latent Upscaler pack and its weights in "
                               "models/latent_upscale_models. Without the pack this does nothing "
                               "and info says so. Spatial only, so frame count and audio are "
                               "untouched, and tiled decode is forced while it is on."}),
                "latent_upscale_scale": ("FLOAT", {"default": 2.0, "min": 1.0, "max": 4.0,
                    "step": 0.05,
                    "tooltip": "Latent upscale factor on both axes. 2.0 doubles each side. "
                               "1.0 disables it as surely as 'off'."}),
                "upscale": (["off", "rtx", "model", "lanczos"], {"default": "off",
                    "tooltip": "Post-pass on the FINISHED frames, after the latent pass and after "
                               "the shots are joined. 'rtx' = NVIDIA RTX Video Super Resolution "
                               "(needs the Nvidia_RTX_Nodes_ComfyUI pack, falls back if absent); "
                               "'model' = an upscale model from upscale_models; 'lanczos' = a "
                               "plain resize. These ENLARGE; for real detail reconstruction from a "
                               "low-res render use a separate pass."}),
                # Explicit, not left to fall back to the list's first entry: the list
                # is built from what is installed, so leaving it implicit makes the
                # default depend on the machine.
                "upscale_model": (_upscale_model_list(), {"default": "none",
                    "tooltip": "Which model, when upscale = model. From models/upscale_models."}),
                "upscale_target_short_edge": ("INT", {"default": 0, "min": 0, "max": 4096,
                    "step": 32,
                    "tooltip": "Fit the result's short edge to this many pixels. 0 keeps the "
                               "model's own factor."}),
                "upscale_batch": ("INT", {"default": 4, "min": 1, "max": 64,
                    "tooltip": "Frames per chunk for the model upscale. Lower = less VRAM, "
                               "slower."}),
                "shot_length": (["from the beat", "fixed"], {"default": "from the beat",
                    "tooltip": "How long each shot is.\n\n"
                               "'from the beat' sizes every shot from what its own line "
                               "stages, capped by shot_seconds and floored at one action's "
                               "worth. A beat with one action stops getting a shot with room "
                               "for two -- which is what makes an action carry on past its "
                               "end, repeating itself on whatever is nearest once it has "
                               "run out of what it was given.\n\n"
                               "'fixed' gives every shot shot_seconds. Uniform lengths mean "
                               "uniform latent SHAPES, and noise is drawn to the shape -- so "
                               "one seed gives the whole chain one noise field and surface "
                               "detail does not reset at each cut. That consistency is what "
                               "you trade away for pacing.\n\n"
                               "The estimate leans short on purpose: a shot that ends before "
                               "its action does hands a mid-motion frame to the next shot, "
                               "which the chain continues from. A shot that outlasts its "
                               "action has to invent the rest."}),
                "auto_remove": ("BOOLEAN", {"default": True,
                    "tooltip": "Read removals out of the beat itself, so a garment comes "
                               "off without a 'remove:' line.\n\n"
                               "Two conditions, both required, because a wrong removal is "
                               "worse than a missed one: the beat has to contain a removal "
                               "verb, and the thing named has to be the HEAD of something "
                               "the SCENE already lists as worn -- not a modifier inside an "
                               "entry, not a body part, and never restraint hardware. Only "
                               "the verb's own object counts, the span up to the next clause "
                               "boundary, so 'pulls off her coat, showing the jumper' takes "
                               "off the coat and leaves the jumper.\n\n"
                               "info reports every removal it reads, by shot. An explicit "
                               "'remove:' line still works and is added to whatever is "
                               "inferred."}),
                "restart_after_removal": ("BOOLEAN", {"default": True,
                    "tooltip": "After a shot with a 'remove:', start the NEXT shot fresh "
                               "instead of continuing from that shot's last frame.\n\n"
                               "Every shot is anchored to the previous shot's last frame. If "
                               "the model does not finish taking the garment off inside its "
                               "own shot, that frame still shows it -- and a keyframe is a "
                               "PICTURE, which outvotes any sentence. Inherit it once and every "
                               "later shot inherits it too, with no wording able to undo it. "
                               "This breaks that inheritance at the one boundary where the "
                               "state changes.\n\n"
                               "The cost is a visible cut there, and that shot re-deriving its "
                               "pose and framing from the text. Turn it off if your removals do "
                               "complete on screen and you would rather keep the continuity."}),
                "hold_restraints": ("BOOLEAN", {"default": True,
                    "tooltip": "Once a restraint is put on, keep it whole. From the shot "
                               "that applies it onward, every shot carries one sentence: "
                               "every restraint stays whole and closed, fastened exactly as "
                               "it was put on. Cleared by a 'remove:' naming the hardware.\n\n"
                               "This is the ONE continuity fact the node asserts by itself, "
                               "because it is the one that cannot be recovered -- a cuff "
                               "that renders open is not a detail that drifted, it is the "
                               "scene ceasing to make sense. Everything else is yours to "
                               "write."}),
                "plan_only": ("BOOLEAN", {"default": False,
                    "tooltip": "Report the shot split, lengths and warnings without rendering."}),
                # Appended LAST on purpose. Saved workflows restore widget values by
                # POSITION, with no names stored, so inserting a widget anywhere above
                # this shifts every later value in every workflow already saved.
                "anchor": ("STRING", {"multiline": True, "default": "",
                    "tooltip": "Framing that belongs to the whole film -- look, camera, "
                               "lighting, location. Carried at the FRONT of every shot.\n\n"
                               "FILLING THIS IN MAKES EVERY PARAGRAPH OF THE PROMPT A "
                               "BEAT. The anchor is then the scene, so the prompt is "
                               "pure action and nothing is taken out of it to serve as "
                               "scene text.\n\n"
                               "Leave it empty and the first paragraph of the prompt is "
                               "the scene instead, as before. Use one or the other: with "
                               "both, put ALL the framing here, because the prompt's "
                               "first paragraph will be rendered as a shot."}),
                "character_memory": ("STRING", {"multiline": True, "default": "",
                    "tooltip": "Who is in the film and what they are wearing, re-stamped "
                               "into EVERY shot.\n\n"
                               "Write it as a sheet, one person per line:\n"
                               "  Maya: 27, silver hair, grey shorts, red jacket\n"
                               "  Jon: 34, navy overalls\n\n"
                               "This is what makes clothing hold across a chain. A "
                               "garment described in one beat is described in ONE shot; "
                               "every later shot then says nothing about it, and what "
                               "the model is not told, it invents -- which is a garment "
                               "changing colour, or coming back after it came off.\n\n"
                               "It is also what a removal scrubs. `remove:` and the "
                               "automatic inference take the item out of this sheet from "
                               "that shot onward, so the text stops describing what the "
                               "beat took off.\n\n"
                               "A `Name: ...` paragraph in the prompt itself is folded in "
                               "here automatically -- a sheet is not a beat, and spending "
                               "a shot rendering a description is the visible symptom."}),
                "character_guard": ("BOOLEAN", {"default": True,
                    "tooltip": "Describe only the people a beat actually involves.\n\n"
                               "The sheet has to be in every shot for clothing to hold. "
                               "But describing EVERYONE in every shot puts everyone in "
                               "every shot: a beat about one person renders two, because "
                               "the text standing beside it says the other one is there, "
                               "and a described person is a person the model draws.\n\n"
                               "A beat naming nobody keeps whoever the last one kept, so "
                               "'She lies still.' does not empty the frame. Off, every "
                               "sheet line goes into every shot. info names who each shot "
                               "kept."}),
                "pace": ("FLOAT", {"default": 1.0, "min": 0.25, "max": 2.0, "step": 0.05,
                    "tooltip": "Scales how much screen time each beat is given, when "
                               "shot_length is 'from the beat'.\n\n"
                               "A shot longer than its action does not get filled with "
                               "MORE action -- the model performs the same action more "
                               "slowly to reach the end of the shot. That is what "
                               "slow-looking footage is. Below 1.0 shortens every shot "
                               "and the motion in it quickens; above 1.0 lengthens and "
                               "slows.\n\n"
                               "Try 0.75 if the movement drags. Shots are still floored "
                               "at one action's worth and capped by shot_seconds, and "
                               "'fixed' ignores this entirely. info reports the seconds "
                               "each staged action ends up with."}),
                "auto_sound": ("BOOLEAN", {"default": True,
                    "tooltip": "Give each shot the sound its own action implies.\n\n"
                               "H3 is joint, so the same prose conditions the audio "
                               "branch -- and a beat that says what happens has already "
                               "said what it sounds like. Walking gets footsteps, a "
                               "chain gets links dragging, scissors get blades through "
                               "fabric, a lock gets a lock closing.\n\n"
                               "Read from the BEAT only, never the scene: a chain "
                               "standing in the scene does not rattle in a shot where "
                               "nobody moves. Three sounds at most, so the shot gets a "
                               "cue rather than an inventory.\n\n"
                               "The ambient bed and the room tone FOLLOW the "
                               "characters. Both are read from the scene, and a film "
                               "that walks into a tiled bathroom was going on being "
                               "told it sounds like the carpeted room it left -- the "
                               "picture in one room and the audio in another, in the "
                               "same conditioning. They are re-read at a move, but "
                               "only where the new room has a sound of its own.\n\n"
                               "NOTHING HERE CAN OPEN A SILENT SHOT. Ambience on "
                               "every shot was tried and does not work: the bed was "
                               "allowed to open the audio branch, and an open branch "
                               "on a joint model fills itself. At 4-8 steps the last "
                               "audio step clears 50%-30% of its denoising in one "
                               "jump, and what it invents there is a voice -- so "
                               "every wordless shot got ambience and a babbling mouth "
                               "with it. Ambience everywhere and silence cannot both "
                               "hold: the silence latent IS the audio, and there is "
                               "no room in it for a room tone. Score a silent shot by "
                               "writing the sound into that beat, or lay an ambient "
                               "track under the finished video outside the model.\n\n"
                               "A beat that already describes its own sound is left "
                               "alone -- what you wrote wins. A shot given sound is also "
                               "not silenced, since it is now asking for audio. info "
                               "lists which shots got one."}),
                # APPENDED, like every widget before it. Saved workflows restore these
                # positionally with no names stored, so inserting one shifts every
                # value after it into the wrong control.
                "hold_scene_state": ("BOOLEAN", {"default": True,
                    "tooltip": "Put a described state at the first frame instead of "
                               "leaving it to be performed.\n\n"
                               "'A van with its doors closed' names a state and never "
                               "says when it is true. A video model asked for a door "
                               "renders what a door does, so the shot opens on the doors "
                               "open and the characters close them -- the state arrives "
                               "as the action, because that is the most interesting "
                               "event in the sentence.\n\n"
                               "Doors, gates, windows, curtains, blinds, shutters, "
                               "hatches, tailgates, lids and drawers. Two at most per "
                               "shot.\n\n"
                               "A beat that WORKS the thing is not held -- 'Mara opens the "
                               "doors' is asking for exactly that motion. It is given the "
                               "two ENDS of the change instead: shut at the first frame, "
                               "open by the last. Some distill LoRAs render an action "
                               "backwards, and a beat naming one state names neither end, "
                               "so the reverse answers it just as well. Verbs that go "
                               "either way -- pulls, draws, slides, swings -- get no "
                               "anchor, since a wrong one asks for the reversal rather "
                               "than allowing it.\n\n"
                               "Once a beat has changed a state, no later shot is told "
                               "the old one, even though the scene paragraph still says "
                               "it. Two sentences per shot at most, the two kinds sharing "
                               "that budget. info lists which shots got which."}),
                # APPENDED. Saved workflows restore widgets by position.
                "mouths_shut_when_no_line": ("BOOLEAN", {"default": True,
                    "tooltip": "Keep mouths closed on shots where nobody speaks.\n\n"
                               "H3 is joint: the face follows the audio branch. A shot "
                               "with no line but a sound YOU wrote -- 'a low hum off the "
                               "strip light' -- kept its branch open, and an open branch "
                               "invents a voice the face lip-syncs to. Nobody is speaking "
                               "and the mouth moves anyway.\n\n"
                               "On, such a shot is conditioned on silence like any other "
                               "wordless shot, and every wordless shot is also told the "
                               "mouths are closed. Conditioning is what actually settles "
                               "it; the sentence alone loses to a stream that has already "
                               "decided somebody is talking.\n\n"
                               "THE COST: that shot gives up the sound you wrote for it. "
                               "info names those shots, so turn this off if you would "
                               "rather keep the ambience and risk the mouth.\n\n"
                               "EFFORT IS EXEMPT. Straining, thrashing, a body under load "
                               "is vocal and its mouth should be open, so those shots keep "
                               "their audio and are never told to close."}),
                # APPENDED. Saved workflows restore widgets by position.
                "hold_gaze": ("BOOLEAN", {"default": True,
                    "tooltip": "Put the eyes where the beat says they are looking.\n\n"
                               "'She is looking at the TV' says it once, and two things "
                               "pull the other way: the model's prior is that a person in "
                               "frame faces the camera, and a near-clean reference is "
                               "asking for the portrait's pose -- which looks at the lens, "
                               "because photographs of people do. The result is somebody "
                               "posing for the camera instead of watching what you "
                               "named.\n\n"
                               "On, a beat naming a thing to look at gets one more "
                               "sentence saying the eyes are on it and the head is turned "
                               "to face it. Stated as a physical fact rather than an "
                               "activity, and impersonally -- naming the person again is "
                               "one more mention of a person, which has its own cost.\n\n"
                               "Reads 'looks at', 'stares at', 'glances at', 'peers into', "
                               "'watching', 'studies'. It says nothing about where the "
                               "camera is, so a shot looking straight down the line of "
                               "sight is unaffected. Looking at a PERSON is left alone: "
                               "restating a pronoun says nothing the beat did not.\n\n"
                               "A LINE WITH NOBODY NAMED TO LOOK AT turns the faces to "
                               "each other. Reported: two people talking to the camera "
                               "instead of each other. With no look staged, both faces "
                               "fall to the same portrait prior, and a line has an "
                               "addressee whether or not the beat wrote one. Said once, "
                               "impersonally, only with two or more people in the shot; "
                               "a beat that names a look is never argued with.\n\n"
                               "IT ALSO SPEAKS FOR THE EXPRESSION, because that is the "
                               "same pull. Reported: she smiles at the camera in a "
                               "scene of duress. A four-shot scene of a woman "
                               "handcuffed in a van -- pulling at the cuffs, "
                               "struggling, going limp -- had not one word in it about "
                               "anybody's face, and an attribute the prompt leaves out "
                               "is not left to the model, it is left to the model's "
                               "prior: a portrait, facing the lens, pleasantly. So a "
                               "shot whose sheet lists BINDING hardware on somebody in "
                               "it, or whose beat uses your own distress verbs, gets "
                               "one sentence -- the face shows the strain of it, the "
                               "mouth set. A collar alone does not trigger it, a shot "
                               "staging neither gets nothing, and a beat that already "
                               "says what the face does is never argued with. Picture "
                               "only: it can never open the audio branch."}),
                # APPENDED, like every widget before it. Saved workflows restore
                # widget values by POSITION with no names stored.
                "ambient_audio": ("AUDIO", {"tooltip":
                    "Wire a recording to play UNDER the finished soundtrack. Empty "
                    "means no bed at all.\n\n"
                    "This used to be an override on a bed the node BUILT out of the "
                    "scene's own wording. That builder is gone -- reported as sounding "
                    "horrid -- so the soundtrack is the model's, and this is the one "
                    "way to put a room under it.\n\n"
                    "It is PLAYED, not conditioned on, and that is the point: ambience "
                    "needs no cooperation from a joint model, has nothing to lip-sync "
                    "to, and so cannot put a voice in a wordless shot. It is resampled "
                    "and looped with a crossfade to the length of the film. "
                    "ambient_level sets how loud."}),
                "ambient_level": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1.0,
                    "step": 0.01,
                    "tooltip": "How loud the recording wired to ambient_audio plays "
                               "under the finished soundtrack. With nothing wired this "
                               "does nothing -- the bed the node used to BUILD from "
                               "the scene is gone, reported as sounding horrid, and "
                               "the audio is the model's.\n\n"
                               "0.15-0.3 is a bed you notice only when it stops.\n\n"
                               "If the sum would clip, the whole mix is scaled down "
                               "rather than clipped, because clipping distorts the "
                               "line, which is the part worth keeping."}),
                # APPENDED. Saved workflows restore widget values by position.
                "foley_level": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 1.0,
                    "step": 0.01,
                    "tooltip": "DOES NOTHING. Kept only so saved workflows keep "
                               "loading: widget values are restored by POSITION with "
                               "no names stored, so deleting this one would load the "
                               "wrong number into the four widgets after it.\n\n"
                               "It used to set how loud the sounds this node BUILT "
                               "were -- a click, a rattle, a rustle, mixed into the "
                               "shots whose audio branch is pinned to silence, which "
                               "cannot get audio from the model at all because prompt "
                               "text never opens a branch. Removed on the report that "
                               "it sounded horrid; the soundtrack is the model's now, "
                               "whole.\n\n"
                               "WHAT THAT COSTS, said plainly: a shot with no line "
                               "and no sound you described is pinned to silence and "
                               "is SILENT. The pin stays -- it is what stops a free "
                               "branch filling itself with a voice and the face "
                               "lip-syncing to the babble. To put sound in such a "
                               "shot, write the sound into that beat, which opens its "
                               "branch on purpose and lets the model make it; or wire "
                               "a track to ambient_audio; or lay one under the "
                               "finished video outside the node."}),
                # APPENDED. Saved workflows restore widget values by position.
                "speech_lead_seconds": ("FLOAT", {"default": 0.5, "min": 0.0,
                    "max": 2.0, "step": 0.1,
                    "tooltip": "Pin generated audio to encoded silence at the start of each "
                               "dialogue shot. This stops pre-babble and keeps the joint "
                               "model's mouth still during that span. 0 disables it; a long "
                               "lead can trim the first word."}),
                "speech_tail_seconds": ("FLOAT", {"default": 2.0, "min": 0.0,
                    "max": 10.0, "step": 0.5,
                    "tooltip": "Free audio kept AFTER a dialogue shot's line, in seconds. The "
                               "line's length is estimated from its words; past lead + line + "
                               "this margin the audio is pinned to encoded silence, the way "
                               "the lead-in pins the opening. A short line in a long shot "
                               "otherwise leaves seconds of open branch the model fills with "
                               "more speech -- babble, or the line again. The model chooses "
                               "WHEN to speak, so a small margin can clip the last word: raise "
                               "it if it does. 0 disables it."}),
                # APPENDED. Saved workflows restore widget values by position.
                "beat_leads": ("BOOLEAN", {"default": True,
                    "tooltip": "Put the BEAT in front of the character sheet.\n\n"
                               "The sheet has to be in every shot, because clothing "
                               "continuity is read out of it. But it is a description of a "
                               "FACE -- 'she, 22, tall, long blonde hair, blue eyes' -- and "
                               "it was sitting in the opening tokens of every shot, ahead of "
                               "the action. Measured: 69% of a shot's words were in "
                               "sentences about a face, and turning every face guard off "
                               "only reached 63%, because the sheet is most of it.\n\n"
                               "What leads a prompt decides its composition: anatomy in the "
                               "opening tokens is what a distilled model settles the frame "
                               "on, which at cfg 1 no later sentence outvotes. On, the order "
                               "is scene, then what happens, then who it happens to. The "
                               "words are identical and none are rewritten -- only the "
                               "order changes.\n\n"
                               "Off restores the old order, so the two can be compared in "
                               "one render."}),
                # APPENDED. Saved workflows restore widget values by position.
                "hold_levels": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0,
                    "step": 0.05,
                    "tooltip": "Take the grade the chain adds to itself back out of each "
                               "handoff.\n\n"
                               "Every shot after the first is sampled from the previous "
                               "shot's last frame. The model reproduces that frame "
                               "faithfully -- which is what continuity needs -- so it "
                               "inherits whatever is already in it, and it SYNTHESISES the "
                               "opening frame rather than copying it, so its own bias lands "
                               "on top. The VAE then clamps every decode to 0..1, which "
                               "makes the expansion a ratchet: headroom spent is not given "
                               "back. Eleven shots of that is crushed blacks, blown "
                               "highlights and lurid colour, invisible shot to shot and "
                               "obvious end to end.\n\n"
                               "What makes this correctable without knowing anything about "
                               "your scene: at every boundary the render holds two pictures "
                               "that are supposed to be the SAME frame -- the handoff it "
                               "gave the shot, and the opening frame that came back. "
                               "Nothing was asked to change between them, so everything "
                               "separating them is the chain's doing and none of it is "
                               "yours. That difference is what is measured, per colour "
                               "channel, per boundary, and the median across boundaries is "
                               "what is taken back out.\n\n"
                               "It does NOT aim at a target and never compares a shot to "
                               "shot 1, so a beat that walks into a darker room stays "
                               "darker: measured, a deliberate lighting step keeps about "
                               "98% of its size. The correction is a capped fraction per "
                               "boundary rather than a reset, because shot N's frames reach "
                               "the video ungraded while N+1 is sampled from a corrected "
                               "keyframe -- an uncapped correction would trade burn-in for "
                               "a pop at every cut.\n\n"
                               "1.0 flattens the trend hardest; lower leaves more of the "
                               "look alone. 0 is off. Watch the contrast line in info: if "
                               "it still says UP, raise this. It cannot undo clipping that "
                               "earlier shots already baked in, and it corrects levels "
                               "only -- not softening, and nothing spatial."}),
            },
        }

    RETURN_TYPES = ("IMAGE", "AUDIO", "STRING", "STRING", "INT", "INT", "INT", "FLOAT")
    RETURN_NAMES = ("images", "audio", "info", "script", "frames_per_shot", "total_frames",
                    "shots", "video_seconds")
    FUNCTION = "run"
    CATEGORY = "sampling/minimax"
    DESCRIPTION = ("Chain MiniMax-H3 shots into one continuous video with synchronised audio. "
                   "One paragraph per shot; the first paragraph is the scene. Your text is "
                   "passed through verbatim.")

    def run(self, model, clip, vae, audio_vae, prompt, resolution, megapixels, shot_seconds,
            steps, cfg, sampler_name, scheduler, seed,
            first_frame=None, ref_image_1=None, ref_image_2=None, ref_image_3=None,
            ref_image_4=None, negative=None, sigmas=None,
            shift_video=12.0, shift_audio=3.0, apply_model_sampling=True,
            silence_nonspeech=True, trim_seam=True, ref_noise_aug=0.999,
            tiled_decode=True, cleanup_between_shots=True, plan_only=False,
            latent_upscale="off", latent_upscale_scale=2.0,
            upscale="off", upscale_model="none", upscale_target_short_edge=0,
            upscale_batch=4, shot_length="from the beat", hold_restraints=True,
            restart_after_removal=True, auto_remove=True, anchor="", character_memory="",
            character_guard=True, pace=1.0, auto_sound=True, hold_scene_state=True,
            mouths_shut_when_no_line=True, hold_gaze=True,
            ambient_audio=None, ambient_level=0.25, foley_level=0.35,
            speech_lead_seconds=0.5, speech_tail_seconds=2.0, beat_leads=True,
            hold_levels=0.8,
            **_removed):
        # **_removed: a workflow saved with the old `save_defaults` widget still sends
        # it. Swallowed rather than raising, so an existing workflow keeps loading.

        # An interrupt arrives as a BaseException (model_management.py:2133), so it is
        # NOT caught by the `except Exception` handlers in this file and must not be --
        # stopping a run has to stop it. What it does skip is every `del` in the render
        # loop, leaving a multi-gigabyte frame buffer to be freed by the collector in
        # its own order, after ComfyUI has already started unloading the models it was
        # sized against. Reported as an illegal memory access on stopping a run, thrown
        # from cuMemFreeAsync inside a tensor destructor rather than from any line of
        # Python. Dropping it here makes that free happen at a known point, before the
        # unwind; the interrupt is then re-raised untouched.
        self._frames = None
        prepared = self._prepare(
            model=model, clip=clip, vae=vae,
            audio_vae=audio_vae, prompt=prompt, resolution=resolution,
            megapixels=megapixels, shot_seconds=shot_seconds, steps=steps,
            cfg=cfg, sampler_name=sampler_name, scheduler=scheduler,
            seed=seed, first_frame=first_frame, ref_image_1=ref_image_1,
            ref_image_2=ref_image_2, ref_image_3=ref_image_3, ref_image_4=ref_image_4,
            negative=negative, sigmas=sigmas, shift_video=shift_video,
            shift_audio=shift_audio, apply_model_sampling=apply_model_sampling, silence_nonspeech=silence_nonspeech,
            trim_seam=trim_seam, ref_noise_aug=ref_noise_aug, tiled_decode=tiled_decode,
            cleanup_between_shots=cleanup_between_shots, plan_only=plan_only, latent_upscale=latent_upscale,
            latent_upscale_scale=latent_upscale_scale, upscale=upscale, upscale_model=upscale_model,
            upscale_target_short_edge=upscale_target_short_edge, upscale_batch=upscale_batch, shot_length=shot_length,
            hold_restraints=hold_restraints, restart_after_removal=restart_after_removal, auto_remove=auto_remove,
            anchor=anchor, character_memory=character_memory, character_guard=character_guard,
            pace=pace, auto_sound=auto_sound, hold_scene_state=hold_scene_state,
            mouths_shut_when_no_line=mouths_shut_when_no_line, hold_gaze=hold_gaze, ambient_audio=ambient_audio,
            ambient_level=ambient_level, foley_level=foley_level, speech_lead_seconds=speech_lead_seconds,
            speech_tail_seconds=speech_tail_seconds, beat_leads=beat_leads,
            hold_levels=hold_levels,
            **_removed)
        if isinstance(prepared, PreparedVideo):
            try:
                return self._render(prepared)
            except BaseException:
                _f, self._frames = self._frames, None
                if _f is not None:
                    try:
                        _f.release()
                    except Exception:
                        pass            # teardown must not mask the interrupt
                raise
            finally:
                self._frames = None
        return prepared

    def _prepare(self, model, clip, vae, audio_vae, prompt, resolution, megapixels, shot_seconds,
            steps, cfg, sampler_name, scheduler, seed,
            first_frame=None, ref_image_1=None, ref_image_2=None, ref_image_3=None,
            ref_image_4=None, negative=None, sigmas=None,
            shift_video=12.0, shift_audio=3.0, apply_model_sampling=True,
            silence_nonspeech=True, trim_seam=True, ref_noise_aug=0.999,
            tiled_decode=True, cleanup_between_shots=True, plan_only=False,
            latent_upscale="off", latent_upscale_scale=2.0,
            upscale="off", upscale_model="none", upscale_target_short_edge=0,
            upscale_batch=4, shot_length="from the beat", hold_restraints=True,
            restart_after_removal=True, auto_remove=True, anchor="", character_memory="",
            character_guard=True, pace=1.0, auto_sound=True, hold_scene_state=True,
            mouths_shut_when_no_line=True, hold_gaze=True,
            ambient_audio=None, ambient_level=0.25, foley_level=0.35,
            speech_lead_seconds=0.5, speech_tail_seconds=2.0, beat_leads=True,
            hold_levels=0.8,
            **_removed):
        # **_removed: a workflow saved with the old `save_defaults` widget still sends
        # it. Swallowed rather than raising, so an existing workflow keeps loading.

        notes = []
        # BEFORE the numbers are repaired, because the numbers are the symptom and
        # this is the cause. A combo holding something that is not one of its own
        # options cannot be honoured, and rendering anyway would use settings nobody
        # chose -- a scheduler's name in sampler_name, a seed in scheduler.
        _bad = misaligned_widgets(
            dict(resolution=resolution, sampler_name=sampler_name, scheduler=scheduler,
                 shot_length=shot_length, upscale=upscale, latent_upscale=latent_upscale,
                 upscale_model=upscale_model),
            combo_options(self.INPUT_TYPES()))
        if _bad:
            raise RuntimeError(alignment_error(_bad))
        # A widget value that arrives as NaN -- which is
        # what a positional shift in a saved workflow produces -- would otherwise flow
        # into the frame arithmetic and come out as a shot length of nan.
        _fixed, _fixnotes = sane_widgets(dict(
            megapixels=megapixels, shot_seconds=shot_seconds, steps=steps, cfg=cfg,
            shift_video=shift_video, shift_audio=shift_audio,
            ref_noise_aug=ref_noise_aug, latent_upscale_scale=latent_upscale_scale,
            upscale_target_short_edge=upscale_target_short_edge,
            upscale_batch=upscale_batch, pace=pace,
            ambient_level=ambient_level, foley_level=foley_level,
            speech_lead_seconds=speech_lead_seconds,
            speech_tail_seconds=speech_tail_seconds, hold_levels=hold_levels))
        megapixels, shot_seconds = _fixed["megapixels"], _fixed["shot_seconds"]
        steps, cfg = _fixed["steps"], _fixed["cfg"]
        shift_video, shift_audio = _fixed["shift_video"], _fixed["shift_audio"]
        ref_noise_aug = _fixed["ref_noise_aug"]
        latent_upscale_scale = _fixed["latent_upscale_scale"]
        upscale_target_short_edge = _fixed["upscale_target_short_edge"]
        upscale_batch, pace = _fixed["upscale_batch"], _fixed["pace"]
        ambient_level, foley_level = _fixed["ambient_level"], _fixed["foley_level"]
        speech_lead_seconds = _fixed["speech_lead_seconds"]
        speech_tail_seconds = _fixed["speech_tail_seconds"]
        hold_levels = _fixed["hold_levels"]
        notes.extend(_fixnotes)
        # <Picture N> means ref_image_N, the socket. Everything downstream works on
        # the packed roster instead, so translate once, here, before anything has
        # read a tag. With the sockets filled from the top this changes nothing.
        _wired = [n for n, r in enumerate((ref_image_1, ref_image_2, ref_image_3,
                                           ref_image_4), 1) if r is not None]
        _missing = unwired_reference_tags(f"{prompt}\n{character_memory}", _wired)
        if _wired and list(_wired) != list(range(1, len(_wired) + 1)):
            notes.append(
                f"reference sockets {', '.join('ref_image_' + str(n) for n in _wired)} "
                f"are wired with a gap, so <Picture N> has been read as the SOCKET "
                f"number and renumbered onto the packed roster "
                f"({', '.join(f'{n}->{i}' for i, n in enumerate(_wired, 1))}). Without "
                f"this a tag naming a socket past the end of the roster matched nothing, "
                f"and its image was dropped in silence")
        prompt = renumber_reference_tags(prompt, _wired)
        character_memory = renumber_reference_tags(character_memory, _wired)
        if _missing:
            notes.append(
                f"<Picture {'>, <Picture '.join(str(n) for n in _missing)}> "
                f"{'names a socket' if len(_missing) == 1 else 'name sockets'} with no "
                f"image on it: nothing is wired to "
                f"{', '.join('ref_image_' + str(n) for n in _missing)}. The tag is "
                f"dropped from the text, because a tag pointing at no picture is a "
                f"person the model is told to look up and cannot find. Wire the image, "
                f"or take the tag out")
        swap = flush_for_model_change(model)
        if swap:
            notes.append(swap)
        # Before anything is sampled, because the failure mode is an abort and an abort
        # cannot be reported from inside the render.
        _abort = sparse_attention_allocator_abort(model)
        if _abort:
            raise RuntimeError(_abort)
        check_vae_wiring(vae, audio_vae)
        # Before anything else reads the script, for the same reason the abort above is
        # here: the answer is a refusal, and a refusal has to happen before work does.
        _refuse = minor_with_sexual_staging(
            "\n".join([(character_memory or ""), (prompt or "")]), "\n".join(
                [(prompt or ""), (anchor or ""), (character_memory or "")]))
        if _refuse:
            raise RuntimeError(_refuse)

        prompt, n_legacy = strip_legacy_fields(prompt)
        if n_legacy:
            notes.append(f"dropped {n_legacy} field-label line(s) left over from an older "
                         f"version of this node (overall_soundscape:, [Generation N] and the "
                         f"like) -- your text now goes to the model verbatim, and a label like "
                         f"that is read as text to put ON the picture")
        if (anchor or "").strip():
            # The anchor IS the scene, so nothing has to be taken out of the prompt to
            # be one, and every paragraph is a beat. Otherwise the first ACTION becomes
            # the scene: prepended to every shot, repeated to the end of the film, and
            # never given a shot of its own. A removal written in it can never stick
            # either, because the scene restates the garment on every later shot.
            scene, beats = "", paragraphs(prompt)
        else:
            scene, beats = split_beats(prompt)
        # A character sheet is not a beat. Pulled out of the beat list and folded into
        # the scene, so it is re-stamped into EVERY shot -- which is what makes a
        # removal stick and what stops a later shot describing no clothing at all.
        beats, sheet = pull_character_sheets(beats)
        # The sheet is kept APART from the rest of the scene: it is the part that
        # varies per shot, because only the people a beat involves should be
        # described in it. Everything else is stamped on every shot unchanged.
        sheet, _dupes = merge_sheets((character_memory or "").strip(), sheet)
        # Read ONCE, over the whole script, and AFTER character_memory is merged in
        # -- that is where the wrists usually are. A shot of the captor alone is grim
        # on account of what the sheet says three beats ago, so this cannot be a
        # per-shot question. See film_stages_duress.
        # A DECLARED AGE UNDER 18 GETS NO BODY DESCRIBED FOR IT, and the author is told
        # so rather than left to wonder why one entry reads differently from the rest.
        # The scene itself renders: children are in films. What is withheld is this
        # node's own anatomy clauses, every one of them. See body_of and figure_of, and
        # minor_with_sexual_staging for the case that does not render at all.
        _minors = sorted({_n for _n, _ln in sheet_lines(sheet)
                          if _n and 0 < age_in(_ln) < ADULT_AGE})
        if _minors:
            notes.append(
                f"{_join_names(_minors)} "
                f"{'are' if len(_minors) > 1 else 'is'} declared under {ADULT_AGE} on "
                f"the sheet, so NO body is described for "
                f"{'them' if len(_minors) > 1 else _minors[0]} by this node -- not a "
                f"softer description, none. Every clause that would name a body, a bare "
                f"region's anatomy or a figure stays silent for that entry, and the rest "
                f"of the film is unaffected. The scene renders. Had the script also "
                f"staged nudity or sex anywhere in it, nothing would have rendered at "
                f"all. If the age is a typo, fix it and the entry behaves like any other"
            )
        _film_mood = mood_declared(anchor)
        _film_duress = film_stages_duress(beats, sheet, anchor)
        if _dupes:
            notes.append(
                f"{', '.join(_dupes)} described more than once -- character_memory and a "
                f"'Name:' paragraph in the prompt are the same channel by two routes, and "
                f"using both put the person in every shot twice. A model told about one "
                f"person twice renders two of them. Kept the character_memory entry and "
                f"dropped the duplicate")
        static = build_scene(anchor, scene, "", "")
        scene = build_scene(anchor, scene, "", sheet)      # the whole of it, for inference
        # Which rooms the author actually DESCRIBES. A room the text only names is a
        # room the model invents; this is what the warning below is read from.
        _described_rooms = set(rooms_named(static))
        if sheet:
            notes.append(f"folded {sheet.count(chr(10)) + 1} character-sheet line(s) into "
                         f"the scene instead of spending a shot on them -- a sheet "
                         f"describes people, it does not stage anything, and it has to "
                         f"be in EVERY shot for a removal to have something to scrub")
        # Somebody the beats stage and the sheet never describes. Nothing in the shot
        # says who they are, so the model invents them -- and a beat whose only person
        # is undescribed falls back to the previous beat's cast, which describes
        # someone who is not in the shot and says nothing about the one who is.
        for _who, _in in unknown_people([extract_directives(b)[0] for b in beats],
                                        sheet).items():
            notes.append(
                f"shot(s) {', '.join(str(n) for n in _in)} name {_who}, who has no entry "
                f"in the character sheet. {_who} is IN those shots and nothing describes "
                f"them -- no age, no clothes, no face -- so the model invents them, "
                f"differently each time. Where that is the ONLY person a beat names, the "
                f"shot falls back to the previous beat's people, and then it describes "
                f"someone who is not in it and nobody who is. If {_who} is already on the "
                f"sheet under another name, use one name throughout; otherwise add "
                f"'{_who}: ...' to character_memory")
        # Account for every paragraph, so a beat that quietly went somewhere else is
        # visible. Two ways one disappears: it reads as a character sheet and is folded
        # into the scene, or it was never a separate paragraph to begin with.
        _given = len(paragraphs(prompt))
        _sheets = len(sheet_lines(sheet)) if sheet else 0
        notes.append(f"{_given} paragraph(s) in the prompt: {len(beats)} rendered as "
                     f"shots" + (f", {_sheets} folded in as character sheet(s)"
                                 if _sheets else "")
                     + ("" if (anchor or "").strip() else ", 1 kept as the scene"))
        # THE SCENE PARAGRAPH THAT QUIETLY BECAME A SHOT.
        #
        # Reported as a van changing direction between shots. Filling in `anchor`
        # makes every paragraph a beat -- the anchor is then the scene -- and that is
        # deliberate, and the widget's tooltip says so. What it MEANS is that a
        # prompt whose first paragraph is scene text loses that text after shot 1:
        #
        #     no anchor    4 shots   van direction carried: yes yes yes yes
        #     with anchor  5 shots   van direction carried: yes NO  NO  NO  NO
        #
        # The heading of a vehicle, the location, the time of night -- stated once,
        # spent on a shot of their own, and never said again. Nothing reported it at
        # runtime, so what the author sees is the van turning round between takes,
        # with no way to connect that to a widget they filled in for the camera.
        #
        # Read as "this paragraph puts nobody on screen", which is this file's own
        # test for whether there is a person in a beat. An opening ACTION is a real
        # beat and is left alone: telling somebody to move it into the anchor would
        # be wrong, and the anchor branch exists precisely because the no-anchor path
        # was making scenes out of actions.
        _first_para = beats[0] if beats else ""
        if ((anchor or "").strip() and len(beats) > 1 and _first_para
                and not beat_puts_somebody_on_screen(_first_para, sheet)):
            _quoted = " ".join(_first_para.split())
            notes.append(
                f"the prompt's first paragraph describes a PLACE rather than staging "
                f"anything, and with `anchor` filled in every paragraph is a beat -- so "
                f"it is being spent as shot 1 and is not carried into any other shot: "
                f"\"{_quoted[:100]}{'...' if len(_quoted) > 100 else ''}\". Whatever it "
                f"establishes -- which way a vehicle faces, the room, the time of night "
                f"-- is said once and then gone, and the shots after it are free to put "
                f"it back differently. That is what a van changing direction between "
                f"shots looks like from the outside. Move it into `anchor`, which is "
                f"carried at the front of EVERY shot, or clear `anchor` and let the "
                f"first paragraph be the scene as it is without one. Use one or the "
                f"other: with both, all of the standing description belongs in `anchor`")
        # Paragraphs are separated by a BLANK line. Lines joined by a single newline
        # are ONE beat, so three actions written on three lines become one shot with
        # three actions in it, and two of them look like they were absorbed.
        _multi = [i for i, b in enumerate(beats, 1) if "\n" in b]
        if _multi:
            notes.append(
                f"shot(s) {', '.join(str(i) for i in _multi)} carry more than one line. "
                f"Paragraphs are separated by a BLANK line, so lines with only a single "
                f"newline between them are one beat and share one shot. If those were "
                f"meant to be separate shots, put an empty line between them")
        if not beats:
            raise RuntimeError("H3-LongVideos: no beat to render. Every paragraph after "
                               "the first is one shot; a character sheet ('Name: ...') "
                               "is folded into the scene and does not count as one.")

        w, h = scale_to_megapixels(*parse_resolution(resolution), megapixels)
        ceiling = align_frame_count(int(round(float(shot_seconds) * H3_FPS)))
        # 'remove:' lines take their item out of the SCENE from that shot onward, so
        # the scene stops describing a garment a beat has taken off. It applies to
        # the removing shot too: the keyframe already shows the garment on at the
        # start, and a description saying it is still worn is what puts it back.
        plan = ShotPlan()
        gone, shown = [], []
        # token -> who took it off, so the scrub reaches their entry and nobody
        # else's. A token with nobody recorded stays unscoped. See scrub_removed.
        gone_by = {}
        _extras_seen = False        # the film has staged people the sheet does not name
        untracked_strip = []        # (shot, items) a group removal the sheet cannot hold
        # Of those, the ones open ONLY because the beat stages effort. The branch
        # is open on both, but for opposite reasons, and built sound has to tell
        # them apart. (It used to matter for the foley mix as well, which is gone.)
        inferred_sound = []         # shots given one derived from their action
        restrained = posed = rigid_latched = False
        # Has any BEAT stated a posture yet? The scene fallback for the weight
        # clause stops the moment one does. Film-level on purpose: it is the
        # degraded path, taken when the per-person latch cannot fill because the
        # script names no cast, and in that state there is nobody to attribute a
        # posture to either. It errs towards saying nothing, which is the old
        # behaviour.
        beat_said_posture = False
        restrained_who = set()    # who is actually in the hardware
        anchored = ""             # where fastened limbs are held
        worn_item = ""            # the hardware, in the author's words
        worn_items = []           # ...each piece of it, in order
        displaced = {}            # garment -> how it was moved
        moved_shots = []          # shots reminded of it
        revealed_shots = []       # shots that uncover a layer
        unattributed = []         # shots whose line names no speaker
        mouth_named = []          # shots with a line, holding the OTHER mouths
        language_shots = []       # shots told which language the line is in
        _spoken_words = {}        # shot -> words actually inside the quotes
        _breath_shots = []        # shots whose only sound was a breath
        _langs_used = []          # ...and which languages those turned out to be
        # THE WHOLE SCRIPT'S language, as the per-shot fallback. A single short
        # line -- "Si." -- carries no evidence on its own, and reading it alone
        # would call it English inside a Spanish script.
        # ...and where the vote abstains on the whole script, the author's own
        # statement anywhere in it settles the fallback rather than English:
        # a script with ONE line in it, carrying one function word, is a script
        # whose language nothing could vote for.
        _script_voted = engine.language_of(engine.spoken_text(prompt or ""),
                                           fallback="")
        _script_lang = (_script_voted or engine.language_named(prompt or "")
                        or "English")
        told_shots = []           # shots whose line orders somebody about
        dialogue_marked = []      # shots whose quotes became <d>...</d>
        poses = {}                # name -> the posture a beat put them in
        # Seeded from the SCENE, so the first journey has somewhere to start
        # from. Without it "walks him down the hallway to the bedroom" had a
        # destination and no origin, and a journey stated as a destination
        # alone is the one that renders as a cut.
        #
        # READ FROM THE WHOLE SCENE, ANCHOR INCLUDED, and that was checked rather
        # than assumed. Excluding the anchor looks right -- an anchor is the camera,
        # and a lens line was being read for a location -- but the anchor is also
        # DOCUMENTED to carry the location, and with one set there is no scene
        # paragraph for the room to live in instead. Excluded, "A carpeted living
        # room. Shot on 35mm" lost the origin of its first journey, which is the
        # destination-with-no-origin case that renders as a cut: the bug this seed
        # exists to fix, reintroduced through the widget meant to prevent it.
        #
        # The lens was never the anchor's fault. _PLACE matched INSIDE "shallow",
        # and the word boundary in _PLACE_WORD is the whole of the fix.
        here = place_named(scene) or first_place(scene)
        # The film's ambient bed, read from the anchor and the scene rather
        # than typed into every beat. See scene_ambient.
        # THE OPENING BEAT IS THE FALLBACK, the same one room_tone has had all along
        # and for the same stated reason: with `anchor` set there is no scene
        # PARAGRAPH, and an anchor describes the camera rather than the room. Without
        # it the film's bed died for everybody who filled in the widget the tooltips
        # tell them to fill in -- and the bed is what this file's own answer to
        # lead-in babble depends on: "a branch with a bed to lay down does not need
        # to invent a voice to fill the space." Reported as babble at the opening of
        # the beat, measured as three speaking shots with no sound clause at all.
        #
        # Read only when the scene names nothing, exactly as room_tone reads it, so
        # a film whose scene DOES name a space is unchanged.
        _opening = extract_directives(beats[0])[0] if beats else ""
        ambient_bed = (scene_ambient(anchor, scene)
                       or scene_ambient(anchor, _opening)) if auto_sound else ""
        _bed_src = ("the anchor and the scene" if scene_ambient(anchor, scene)
                    else "the opening beat")
        ambient_shots = []        # shots given the bed
        posture_shots = []        # shots told to keep a standing posture
        travel_shots = []         # shots that move between places
        where_shots = []          # shots in a room the scene does not name
        acoustic_shots = []       # ...and the ones whose sound followed them there
        paced_shots = []          # shots told to spread their action
        staging_shots = set()     # shots that MOVE a garment on screen
        bared_shots = []          # ...and shots that uncover skin
        bare_held = []            # ...and shots told a region is STILL bare
        crowded = []              # (shot, clauses dropped for room)
        absent_hold = []          # shots where the wearer is not on screen
        exposed_by_beat = []      # (shot, garments the beat names while covered)
        named_shots = []          # shots reminded the thing is still there
        anchored_shots = []       # shots reminded of it
        gaze_shots = []           # shots told where the look goes
        dialogue_gaze_shots = []  # dialogue shots turned to face each other
        scene_held = []           # (shot, rooms) whose scene description waited
        scene_welded = []         # ...and shots where it could not be held
        looking_at = {}           # {name: target}, each held until it changes
        fall_shots = []           # shots told what takes the landing
        device_shots = []         # shots whose line belongs to a machine
        applied_shots = []        # shots that put the hardware on
        early_hardware = []       # ...where the sheet already claimed it
        tight_shots = []          # ...where the framing also crops it
        cropped_wardrobe = []     # garments a named close frame stopped describing
        # FILM-WIDE FRAMING LIVES IN THE ANCHOR, which is where the tooltip sends
        # it: "Framing that belongs to the whole film -- look, camera, lighting,
        # location." tight_framing was only ever handed the BEAT, so a film shot
        # entirely in close-up -- declared once, in the documented place -- read as
        # no close framing at all, and the warning below never fired for anybody who
        # put their camera where they were told to. It fired only for people who
        # wrote "close-up" into a beat, which the tooltip does not ask them to do.
        # Computed once: the anchor is the same on every shot by definition.
        _anchor_tight = tight_framing(anchor)
        # Scenery whose state a beat has CHANGED. After that the node stops asserting
        # the state it was written with, because it is no longer the state: a van
        # opened in shot 2 must not be told it is shut in shot 3, and the scene
        # paragraph goes into every shot still saying "doors closed".
        state_acted = set()
        stated_shots = []           # shots given a state put at the first frame
        turned_shots = []           # shots given both ends of a staged change
        mouth_shut = []             # shots told every mouth is closed
        mouth_acting = []           # ...and the ones whose beat works the mouth
        duress_shots = []           # shots told what the face is doing
        vocal_shots = []            # shots where a vocal was given an owner
        muted_sound = []            # shots whose written sound was given up for it
        stripped_shots = set()      # 0-based shots that took something off
        cut_shots = set()           # 0-based shots opening in a room the keyframe is not in
        _undescribed = []           # rooms the film enters that the prompt never describes
        open_moves = []             # (shot, where) moves to a place the list cannot name
        frame_shots = []            # shots told what the frame holds
        contact_shots = []          # shots told which body is with which
        led_shots = []              # shots whose beat was put ahead of the sheet
        restarted = []              # shots started fresh after a removal
        restored = []               # garments an add: put back on
        wearing_shots = []          # shots that put one back on, given both ends
        # Names, so "lifts Kate onto the table" reads as moving a person rather than
        # an object. A sheet LABELS them, which beats scanning prose for capitals --
        # that way "Medium shadows" is not a member of the cast, and a name with an
        # inner capital (McKenna) is not missed.
        cast = re.findall(r"^\s*([A-Z][\w'’-]{1,24})\s*:", sheet or "", re.M)
        if not cast:
            cast = re.findall(r"\b[A-Z][a-z]{2,}\b", scene or "")
        # Which garment is under which, read from the script's own "takes A off to
        # expose B". A sheet lists every layer at once, and a layer the model is told
        # about is a layer it draws -- through the one on top of it.
        # What the script states wins over what the categories imply: a beat saying
        # "takes the shorts off to expose the belt" is the author telling us directly,
        # and it may pair things the lists opposite know nothing about.
        # LAYERING IS OPTIONAL. On, a covered garment is left out of the shot
        # text until the thing over it comes off, because a described thing is a
        # drawn thing and it would be drawn over its cover. Off, nothing is ever
        # held back from the character memory -- which is what somebody wants who
        # has attached a <Picture N> to the item and expects to see it.
        # PER PERSON. Read off the whole sheet at once, layering has no idea whose
        # garments it is pairing: a sheet with Dana in jeans and McKenna in a skirt
        # and a chastity belt produced {chastity belt: skirt} with no owner on it,
        # and the under-clause was then written into a shot describing only Dana.
        # The belt does not go on Dana. A described garment is a drawn garment, and
        # it is drawn on whoever is in the frame.
        #
        # Each sheet line is one person, so the layers are read line by line and
        # the owner is kept. Anything the SCENE paragraph implies has no owner and
        # is left unattributed, which is right: it belongs to the set, not a body.
        deferred_shots = []       # (shot, items whose picture waits this shot)
        covers, cover_owner = {}, {}
        for _who, _line in sheet_lines(sheet):
            for _u, _o in implied_layers(_line).items():
                covers[_u] = _o
                if _who:
                    cover_owner[_u] = _who
        for _u, _o in implied_layers(static or "").items():
            covers.setdefault(_u, _o)
        covers.update(infer_layers([extract_directives(b)[0] for b in beats], scene))
        if covers:
            notes.append("read as layers -- underwear goes under whatever the sheet "
                         "also puts over it, and anything the script itself pairs by "
                         "taking one off to expose the other: "
                         + "; ".join(f"{u} under {o}" for u, o in covers.items())
                         + " -- each is left out of the scene text until the thing "
                           "over it comes off, so it is not described as visible "
                           "while it is covered")
        _pose = posture_note(scene, first_frame is not None)
        if _pose:
            notes.append(_pose)
        _ref = reference_note(len([r for r in (ref_image_1, ref_image_2, ref_image_3,
                                               ref_image_4) if r is not None]),
                              ref_noise_aug, first_frame is not None)
        if _ref:
            notes.append(_ref)
        # The acoustic of the space, read once: it is the same room in every shot.
        # The opening beat is the fallback: with `anchor` set there is no scene
        # paragraph, and an anchor describes the camera rather than the room.
        _room = room_tone(scene, _opening) if auto_sound else ""
        _room_src = "the scene" if room_tone(scene) else "the opening beat"
        # The two readings that used to be kept for the MIX as well -- ungated by
        # auto_sound, because the built bed conditioned nothing -- are gone with the
        # builder that consumed them. The readings themselves still run for the
        # PROMPT, a few lines down, which is the conditioning side and the only side
        # left. See the note at the top of audio.py.
        if _room:
            notes.append(f"room tone read from {_room_src}: {_room}. It goes under the "
                         f"shots whose audio branch is already open -- ones with a line, "
                         f"or with a sound you described yourself -- so those are not "
                         f"conditioned on digital silence, and nothing real is that "
                         f"quiet. It can never OPEN a branch: a shot with no line and no "
                         f"sound of your own stays pinned to silence and carries no room "
                         f"tone either, because the clause would describe an acoustic the "
                         f"conditioning says is not there. That is what stops the mouth "
                         f"moving. H3 is joint, so a free branch fills itself with a "
                         f"voice and the face lip-syncs to the babble, and no wording "
                         f"suppresses that -- only the audio denoise mask does, and it pins "
                         f"the whole shot rather than just its opening")
        active = []                 # the people the previous beat involved
        _seen_before = set()        # everyone a shot has described so far
        _returns = []               # (shot, names back after a shot away)
        _placed_shots = {}          # 0-based shot -> who it introduces in position
        # WHOSE FACE IS ALREADY COVERED BY A PICTURE OF THEIR OWN. A sheet line
        # carrying <Picture N> for a slot that actually has an image connected -- a
        # tag pointing at an empty socket covers nobody.
        _have_slot = {_i + 1 for _i, _r in enumerate(
            (ref_image_1, ref_image_2, ref_image_3, ref_image_4)) if _r is not None}
        _portrait_of = {_n for _n, _ln in sheet_lines(sheet)
                        if _n and (set(picture_tags(_ln)) & _have_slot)}
        # ...and whether shot 1's first_frame is a SET rather than an opening frame.
        # See the decision below.
        _first_is_plate = False
        guard_words = beat_words = total_words = sound_words = 0
        # THE PROMPT ENGINE. One state, read beat by beat, rendered once per shot.
        # It replaces the continuity guards that used to be derived independently
        # -- hold, anchors, posture, where, removal, wearing, moved -- each of
        # which searched the beat for its own thing and appended its own sentence
        # with no way to see the others. That is what emitted "holding the neck
        # behind the back", dropped the handcuffs from a beat that applied two
        # things, and moved the camera into a door. See engine.py.
        _state = engine.SceneState(place=engine.place_in(scene or ""))
        # Which beat first puts each thing on, read before anything renders. The
        # sheet cannot say when; the script can, and where it does it wins.
        _staged_at = engine.staged_applications(
            [extract_directives(b)[0] for b in beats])
        # What the SHEET names, so the two can be told apart: hardware this node
        # held back out of the sheet is a conflict it created, and hardware the
        # sheet never mentioned is not.
        _sheet_hw = {c for c, _p, _w, _a in engine.hardware_spans(sheet or "")}
        for b in beats:
            body, toks, adds = extract_directives(b)
            # Quoted speech becomes H3'S OWN dialogue marker before anything else
            # reads it. <d> and </d> are special tokens the model was trained with,
            # and they say "this is spoken" where quotation marks say nothing at
            # all -- a quoted imperative is just an imperative sentence in the
            # prompt, and the model performed it. Every word is kept in order; only
            # the quotation marks are exchanged. Reported below.
            _marked = mark_dialogue(body)
            if _marked != body:
                dialogue_marked.append(len(plan) + 1)
                body = _marked
            # THE ENGINE READS FIRST, before anything downstream asks it what is
            # true. It was reading further down at one point, after the hardware
            # latch had already consulted it, so every shot was answered with the
            # PREVIOUS shot's state -- and shot 1 with an empty one.
            # The sheet first: what it already says is true before any beat runs.
            # ...except anything the SCRIPT stages later. A sheet says what
            # somebody has and never says when, so "McKenna: she, 27, green
            # dress, handcuffs" beside a script that cuffs her in beat 3 put the
            # cuffs on her from shot 1 -- reported as a handcuff on her arm
            # before she is handcuffed.
            _later_for_state = {c for c, at in _staged_at.items()
                                if at > len(plan) + 1}
            for _n, _line in sheet_lines(sheet):
                if _n:
                    _state.declare(_n, _line, staged_later=_later_for_state)
            _ch = _state.read(body, cast=[n for n, _ in sheet_lines(sheet) if n],
                              shot=len(plan) + 1)
            # Who this beat involves, decided BEFORE the removals: a beat that
            # undresses somebody names no garment, so the wardrobe to clear is read
            # off their sheet entries -- and only theirs. Undressing one person must
            # not take the other one's clothes off.
            # Bound whether or not the guard runs: the previous shot's cast is read
            # further down, to keep saying what is bare about somebody the keyframe
            # still carries, and that has nothing to do with the guard being on.
            _was = list(active)
            if character_guard:
                shot_sheet, active = sheet_for_beat(sheet, body, active)
                if len(sheet_lines(sheet)) > len(sheet_lines(shot_sheet)):
                    notes.append(f"shot {len(plan) + 1} describes only "
                                 f"{', '.join(active) or 'the scene'} -- the rest of the "
                                 f"sheet is held back, because a person the text "
                                 f"describes is a person the model draws")
                # Somebody back after a shot away. The keyframe is the PREVIOUS shot's
                # last frame, so a person who was not in that shot is not in the
                # picture this one starts from -- their appearance is carried by the
                # sheet text and nothing else, and text drifts where a picture does
                # not. This is what "walks out of frame and comes back looking
                # different" is.
                for _grp, _who_all in unresolved_pronouns(sheet, body, _was):
                    notes.append(
                        f"shot {len(plan) + 1} says '{_grp}' and "
                        f"{' and '.join(_who_all)} all answer to it, so the guard could "
                        f"not tell which -- and it describes NEITHER rather than both, "
                        f"because naming somebody the beat did not is how an extra "
                        f"character walks into a shot. Write the name instead of the "
                        f"pronoun in that beat and it resolves")
                # First appearance, with the beat saying where they ARE rather than
                # staging them arriving. See the handoff decision in the render loop.
                _new = [n for n in active if n not in _seen_before]
                # SHOT 1 HAS THE SAME PROBLEM AND COULD NOT REACH THE SAME ANSWER.
                #
                # The branch below is the one that matters here, and for years it
                # carried `and plan` -- which excludes the FIRST shot, because there
                # is no previous frame to demote. True, until first_frame exists: wire
                # one and shot 1 has a keyframe like any other, and if that picture is
                # a SET rather than a composed opening frame then nobody in the script
                # is in it. Which is this branch's whole subject: "that frame does not
                # have them in it, and a keyframe is a picture, so they would have to
                # appear out of nothing and travel to the spot the beat describes".
                #
                # Measured, with a plate wired and beat 1 placing her in position:
                # shot 1 took it as a HARD KEYFRAME, while the identical case one beat
                # later was correctly refused. Reported as the girl not looking the
                # same in the first beat and fine in the rest -- she is inserted into a
                # frame that lacks her during shot 1, and shot 2 onward inherits the
                # settled version from its handoff, which is why only the first is off.
                #
                # A PLATE IS TOLD FROM AN OPENING FRAME BY THE SCRIPT, not by looking
                # at the pixels. Two conditions, both required:
                #   * the beat PLACES the cast rather than staging an entrance. An
                #     entrance genuinely wants a frame they are absent from.
                #   * every one of them already has a portrait of their own. Their
                #     appearance is carried by that picture, so this one has nothing
                #     left to contribute but the room -- and an author who gives both a
                #     composed opening frame AND a portrait of the same person has
                #     described that person twice, which is its own hazard here.
                # Without portraits the frame is the only picture of them there is, and
                # it stays frame one.
                if (_new and not arrives_in(body) and not plan
                        and first_frame is not None
                        and all(_n in _portrait_of for _n in _new)):
                    _first_is_plate = True
                    notes.append(
                        f"first_frame is being read as the SET, not as shot 1's opening "
                        f"frame, so it carries the room while "
                        f"{_join_names(_new)} {'are' if len(_new) > 1 else 'is'} placed by "
                        f"the text and held by "
                        f"{'their own reference images' if len(_new) > 1 else 'a reference image of their own'}"
                        f". Beat 1 puts "
                        f"{'them' if len(_new) > 1 else _new[0]} "
                        f"in position rather than staging an entrance, and every one of "
                        f"them already has a <Picture N> of their own -- so this picture "
                        f"has nothing left to say about who they are, only about where "
                        f"they are. Pinned as frame one it would be a picture they are "
                        f"not in, and they would have to appear out of nothing during "
                        f"shot 1: that is the same reason a later shot refuses the "
                        f"previous frame when it introduces somebody in position. It is "
                        f"NOT discarded -- the room, the light and the furniture come "
                        f"with it as a reference. To pin frame one exactly instead, put "
                        f"the cast IN that frame and take their <Picture N> tags off the "
                        f"sheet, or write the entrance into beat 1")
                if _new and not arrives_in(body) and plan:
                    _placed_shots[len(plan)] = list(_new)
                    notes.append(
                        f"shot {len(plan) + 1} introduces {', '.join(_new)} in "
                        f"position rather than arriving, so the previous shot's last "
                        f"frame stops being this shot's FIRST frame -- that frame does "
                        f"not have them in it, and a keyframe is a picture, so they "
                        f"would have to appear out of nothing and travel to the spot "
                        f"the beat describes. The frame is still carried, as a "
                        f"reference, so the room comes with it. Write the entrance -- "
                        f"'walks in', 'steps through' -- if you would rather they "
                        f"arrive on screen and keep the frame as the anchor")
                _back = [n for n in active if n not in _was and n in _seen_before]
                if _back:
                    _returns.append((len(plan) + 1, list(_back)))
                _seen_before.update(active)
            else:
                shot_sheet = sheet
            # Read the removal out of the beat itself. Explicit 'remove:' lines still
            # win and are added to whatever is inferred.
            if auto_remove:
                inferred = [t for t in infer_removals(body, scene)
                            if t not in toks and t not in gone]
                # HARDWARE the beat itself unlocks. infer_removals filters restraint
                # words out on purpose -- a cuff must not come off because a beat
                # mentions it -- so a script that unlocks the cuffs in its prose and
                # writes no remove: line left them in the sheet for ever. Clearing
                # the hold was not enough: the sheet still listed them, so the next
                # shot re-detected the restraint from the scene text and latched it
                # again, over hardware the beat had put on the floor.
                if hold_restraints and restraint_coming_off(body):
                    # The WHOLE sheet, not this shot's. A shot that describes only
                    # the person doing the unlocking has no entry for the person
                    # wearing it, so nothing was found to remove and the next shot
                    # read the hardware straight back out of her sheet.
                    for _n, _ln in sheet_lines(sheet if sheet_lines(sheet) else scene):
                        for _hw in restraint_words(_ln):
                            # Only hardware THIS BEAT names, or one it refers to by
                            # pronoun when the wearer has just one piece. "Sam cuts
                            # the rope free" must not unlock her handcuffs.
                            _named = re.search(r"\b" + re.escape(_hw) + r"\b",
                                               body or "", re.I)
                            _pron = (len(restraint_words(_ln)) == 1
                                     and re.search(r"\b(?:them|it)\b", body or "", re.I))
                            if (_named or _pron) and _hw not in toks and _hw not in gone:
                                inferred.append(_hw)
                if inferred:
                    toks = list(toks) + inferred
                    notes.append(f"shot {len(plan) + 1}: read '{', '.join(inferred)}' as "
                                 f"coming off, from the beat's own wording")
            # "...strip out of their clothes, becoming naked" names nothing, so every
            # other path had nothing to take off and the scene went on listing the
            # whole wardrobe -- in every later shot, which is how the clothes came
            # back on. Here the garments are read off the sheet instead of the beat.
            bare = auto_remove and strips_bare(body)
            if bare:
                # ...off THEIR OWN entry. Read off the whole shot sheet, a shot
                # describing two people stripped both wardrobes, so one character
                # undressing undressed the other as well.
                _strippers = strips_who(body, active if character_guard and active
                                        else [n for n, _ in sheet_lines(shot_sheet) if n])
                _their_sheet = "\n".join(
                    ln for n, ln in sheet_lines(shot_sheet) if n in set(_strippers)
                ) or shot_sheet
                stripped = [g for g in garments_in(_their_sheet)
                            if g not in toks and g not in gone]
                if stripped:
                    toks = list(toks) + stripped
                    notes.append(
                        f"shot {len(plan) + 1} reads as undressing "
                        f"{', '.join(active) if character_guard and active else 'the cast'}"
                        f" completely, and the beat names no garment -- so the wardrobe was "
                        f"read off the character sheet and all of it taken off: "
                        f"{', '.join(stripped)}. Anything worn that is not in that list is "
                        f"still described as on; name it in a 'remove:' line if so")
                elif not gone:
                    notes.append(
                        f"shot {len(plan) + 1} reads as undressing completely, but no "
                        f"garment was recognised in the character sheet, so nothing was "
                        f"taken off and every later shot still describes the clothes. Add "
                        f"a 'remove:' line naming them")
            # A beat's own words go to the model verbatim. Naming a garment that came
            # off in an EARLIER beat puts it back -- the scene is clean, the removal
            # was honoured, and then the beat itself asks for it. The removing beat
            # names it legitimately, so only later ones are reported.
            revived = [t for t in gone if names_any(body, [t])]
            if revived:
                notes.append(
                    f"shot {len(plan) + 1} names {', '.join(revived)} in its own text, and "
                    f"that came off earlier. Beats are sent to the model word for word, so "
                    f"naming it puts it back on -- the scene no longer mentions it, but this "
                    f"beat does. Reword the beat if it should stay off")
            if toks:
                stripped_shots.add(len(plan))
                gone.extend(t for t in toks if t not in gone)
                # WHOSE garment it was. Without this the scrub took "shirt" out of
                # every entry that had one, so a second woman in the same shirt lost
                # hers while still wearing it -- the text and the keyframe then
                # disagree, which renders as a garment half present.
                #
                # THE WEARER, NOT THE REMOVER, and the difference is the whole rule:
                # "Dan unlocks the chastity belt" is Dan removing McKenna's, so scoping
                # to whoever the beat names would strand it on her for ever. The
                # candidates are the entries that LIST the garment; the beat only picks
                # between them when it names one of them, which is what tells
                # self-undressing ("McKenna takes off her shirt", two shirts on the
                # sheet) from somebody being undressed.
                #
                # Read off the WHOLE sheet, not this shot's: a shot describing only the
                # person doing the unlocking has no entry for the one wearing it, which
                # is the case the hardware path already had to solve.
                # A REMOVAL THAT INCLUDES PEOPLE THE SHEET DOES NOT NAME reaches only
                # the ones it does. There is no entry to scrub for an unnamed woman and
                # no state to carry her bare region, so her skirt persists on the
                # keyframe alone and comes back the moment the keyframe stops showing
                # it off. Reported as some of the skirts still being on when all of
                # them should have come off. Said rather than left to be discovered.
                if extras_in(body):
                    untracked_strip.append((len(plan) + 1, list(toks)))
                _took = strippers_in(body, shot_sheet if shot_sheet else sheet)
                for _t in toks:
                    _wears = [n for n, _wl in sheet_lines(sheet)
                              if n and re.search(r"\b" + re.escape(_t) + r"\b",
                                                 _wl or "", re.I)]
                    # No entry lists it -- it came out of the scene paragraph, and the
                    # scrub stays unscoped, exactly as it was.
                    gone_by.setdefault(_t, set()).update(
                        [n for n in _took if n in _wears] or _wears)
                # An added layer is subject to removal too: once the shirt comes off,
                # the phrase that introduced it goes with it, or the scene keeps
                # describing a garment that is no longer there. Retired HERE, at the
                # moment of removal, so it retires the phrases that exist NOW -- an
                # add written later is putting the thing back on and must survive.
                _retired = [a for a in shown if names_any(a, toks)]
                if _retired:
                    shown = [a for a in shown if a not in _retired]
                    notes.append(f"shot {len(plan) + 1} takes off something an earlier "
                                 f"'add:' had put on, so that line retires with it: "
                                 + "; ".join(_retired))
                # Reported with the SHEET's words, not the head-noun keys. The
                # reader checks this line to see what the shot was told, and a
                # bare "shorts" here for a sheet saying "blue jeans shorts" reads
                # as the node having lost the description -- which is exactly the
                # bug it had, so the report has to be able to show it is gone.
                notes.append(f"removed from the scene from shot {len(plan) + 1} on: "
                             + ", ".join(scene_name_for(t, scene) or t for t in toks))
            maybe = missing_removals(body, scene, gone) if not auto_remove else []
            if maybe:
                notes.append(f"shot {len(plan) + 1} reads as taking something off, but the "
                             f"scene still describes {', '.join(maybe)} and there is no "
                             f"'remove:' line for it -- so every shot keeps saying it is worn. "
                             f"Add 'remove: {maybe[0]}' to that beat")
            _wearing = ""             # the both-ends clause for a garment going on
            _staged_add = []          # ...and the phrases it covers, held out of
                                      #    this shot's static wardrobe
            if adds:
                shown.extend(a for a in adds if a not in shown)
                # An `add:` that names something previously removed is putting it
                # back ON. `gone` only ever grew, so the layering could never
                # re-cover what it uncovered: shorts taken off and then added back
                # left the thong described for the rest of the film.
                # NOT removed from `gone`. The scene stays scrubbed, or the sheet
                # describes the thing again alongside the add: line that put it
                # back -- two mentions, and with a tagged object two copies of its
                # <Picture N>, which is the duplicate-reference hazard.
                #
                # Layering is told separately: for covering purposes the garment is
                # back on, so what is under it is hidden again.
                _back = [g for g in gone
                         if any(names_any(a, [g]) for a in adds)
                         and g not in restored]
                if _back:
                    restored.extend(_back)
                    notes.append(
                        f"shot {len(plan) + 1} puts " + ", ".join(_back)
                        + " back on, so anything it covers is hidden again from "
                          "here. A garment coming back has to un-cover as well as "
                          "re-cover, or the layer under it stays described for the "
                          "rest of the run")
                    # ...and it goes on DURING this shot, which nothing said. The
                    # phrase went straight into the scene block as a worn item, so a
                    # shot inheriting a last frame without the garment was told flatly
                    # that it has it. That is a disagreement rather than a change, and
                    # the model settles it in the opening frames by turning whatever
                    # is on the body into the garment.
                    #
                    # Only where the BEAT stages the dressing. An `add:` revealing a
                    # layer that was underneath all along describes something already
                    # worn, and staging it would invent a dressing that never happens.
                    _worn_now = [a for a in adds
                                 if any(beat_stages_wearing(body, g) for g in _back)
                                 and any(names_any(a, [g]) for g in _back)]
                    if _worn_now:
                        _wearing = wearing_clause(_worn_now)
                        _staged_add = list(_worn_now)
                        wearing_shots.append(len(plan) + 1)
                notes.append(f"added to the scene from shot {len(plan) + 1} on: "
                             + "; ".join(adds))
            # The scrub applies to the removing shot too -- but only because that
            # shot's KEYFRAME already shows the garment on at the start, so the text
            # saying it is worn would put it back at the end.
            #
            # A shot with no keyframe has no such picture. Scrubbing there deletes the
            # only statement that the garment was ever on, and the shot then says: it
            # is not worn, take it off, and the thing under it is already showing.
            # The model renders that contradiction as a garment half present -- open,
            # or partly cut -- with the layer beneath it on display.
            #
            # ...and a keyframe that is not ANCHORING is no such picture either.
            # Below KEYFRAME_SAFE_AUG the handoff stops being a keyframe and rides
            # as an extra reference: it says who somebody is, not what the opening
            # frame holds. Scrubbing on that assumption took the belt out of the
            # text of the very shot that removes it, so the shot said it is not
            # worn AND to take it off, and it was gone a beat early with nothing
            # anchoring it on. Reported exactly that way.
            i_shot = len(plan)
            _anchoring = (ref_noise_aug is None
                          or float(ref_noise_aug) >= KEYFRAME_SAFE_AUG)
            has_keyframe = ((i_shot > 0 or first_frame is not None)
                            and _anchoring
                            and not (restart_after_removal
                                     and (i_shot - 1) in stripped_shots))
            visible = gone if has_keyframe else [g for g in gone if g not in toks]
            # Whether the chain actually broke was invisible. restart_after_removal
            # costs a visible cut, so it should be possible to confirm it happened
            # without reading the code -- and to see it did NOT when it should have.
            if (i_shot > 0 and restart_after_removal
                    and (i_shot - 1) in stripped_shots):
                restarted.append(i_shot + 1)
            if toks and not has_keyframe:
                _why = ("its opening frame is not anchored -- ref_noise_aug "
                        f"{float(ref_noise_aug):g} is below {KEYFRAME_SAFE_AUG:g}, so "
                        "the handoff rides as a reference rather than holding the "
                        "first frame" if not _anchoring else
                        "it has no keyframe")
                notes.append(f"shot {i_shot + 1} takes something off and {_why}, "
                             f"so {', '.join(toks)} stays described as worn HERE -- the "
                             f"text is the only thing saying it was on to start with. It "
                             f"is scrubbed from the next shot on")
            # A garment still underneath something stays out of the text: described,
            # it gets drawn, and it is drawn through whatever is over it.
            # A displaced outer garment is still WORN, so `gone` never hears about
            # it -- but it is no longer covering what is under it. Without this a
            # beat pulling the shorts down to show the thong described the thong
            # in that shot only, and the layering hid it again in the next.
            # THIS BEAT'S displacements, read here rather than 500 lines further
            # down where the latch is updated. The layering consumed 
            # before the beat had been added to it, so the shot that LIFTS the
            # skirt still saw it covering, and the belt came out from under it one
            # shot late. The latch below is unchanged; this only looks ahead.
            _moved_now = {g for g, _h in displaced_garments(body, shot_sheet or sheet)}
            # ...minus anything this beat puts BACK. Without it the shot that
            # lets the skirt fall still counted the skirt as moved, so what
            # was under it stayed uncovered for one shot too many -- the
            # mirror of the off-by-one that made it uncover one shot late.
            _back_now = set(restored_garments(body, shot_sheet or sheet))
            if puts_it_back(body) and len(displaced) == 1:
                _back_now |= set(displaced)
            _heads_back = {str(g).lower().split()[-1] for g in _back_now}
            _moved_now = {g for g in _moved_now
                          if str(g).lower().split()[-1] not in _heads_back}
            covered = hidden_layers(covers,
                                    [g for g in visible if g not in restored],
                                    (set(displaced) | _moved_now)
                                    - {g for g in (set(displaced) | _moved_now)
                                       if str(g).lower().split()[-1]
                                       in _heads_back})
            # A BEAT that names a covered garment. Beats are passed through word for
            # word and never scrubbed -- that is the node's oldest promise -- so the
            # layering can take the belt out of the sheet and the beat can put it
            # straight back. The words win, the thing is drawn over what is on top of
            # it, and from there the keyframe carries it into every later shot, which
            # is why it looks permanent rather than like one bad shot.
            #
            # Not edited, ever. Reported, because from the outside it is
            # indistinguishable from the layering being broken.
            _said = [g for g in covered
                     if re.search(r"\b" + re.escape(g) + r"\b", body or "", re.I)]
            if _said:
                exposed_by_beat.append((len(plan) + 1, _said))
            # The shot that UNCOVERS one says so. Reported: the shorts come off and
            # the render goes straight to bare skin, past the underwear the sheet
            # named. The removal clause is emphatic and specific -- off the body,
            # dropped out of frame -- while the layer beneath is one entry in an
            # attribute list, and against a model whose prior for trousers coming
            # off is nudity, a list entry does not compete. Only on the shot that
            # takes the cover off; after that it is simply worn.
            # ...and not when the under-layer is coming off in the same breath. A full
            # strip takes the cover AND what was under it, and "the panties underneath
            # are what shows there now" would put back the one garment the beat was
            # most explicit about removing.
            _revealed = reveal_clause([u for u in revealed_by(covers, toks)
                                       if u not in visible and not names_any(u, toks)])
            if _revealed:
                revealed_shots.append(len(plan) + 1)
            # ...and when the sheet names NOTHING underneath, say the region is bare.
            # Otherwise the shot says a garment is gone and leaves the space it left
            # unspecified, which is where the model's own prior fills in -- legwear
            # the prompt never asked for, carried on by the keyframe from there.
            # Never both: reveal_clause speaks when something is under, this when
            # nothing is.
            # ...and not beside BARE_HOLD, which already says everything comes off.
            # Both firing said it twice and attributed it twice.
            # The body is named from the sheet's own declared pronoun, and only where
            # one person is described -- with two, bare_hold's per-person path below
            # carries it and naming it here would attach it to whichever of them the
            # reader reached first.
            # ...and the AGE off the same entry, so the body named is the age the sheet
            # states rather than whatever the prior supplies. See body_of and figure_of.
            _one_line = dict(sheet_lines(shot_sheet)).get((active or [""])[0], "")
            _one_pron = sheet_pronoun(_one_line)
            _one_age = age_in(_one_line)
            _one_body = (body_of(_one_pron, _one_age) if len(active or []) == 1 else "")
            _one_fig = (figure_of(_one_pron, _one_age) if len(active or []) == 1 else "")
            _bare = ("" if (_revealed or bare)
                     else bare_clause(toks, covers, shot_sheet, body=_one_body,
                                      figure=_one_fig))
            # ...and on EVERY shot after it, from state, for as long as the
            # region has nothing on it. Said only on the uncovering beat, the
            # region went unspecified from the next shot on -- and the model
            # fills an unspecified region from its own prior. Reported as a bra
            # coming back on a topless character whose sheet never had one.
            #
            # Only for people this shot describes: a region belonging to nobody
            # in the frame is the sentence that draws the body to own it.
            if not _bare and not bare and not _revealed:
                # The same people _described names further down; that is computed
                # after this clause, so the expression is repeated rather than
                # moved -- moving it ahead of the sheet work it depends on is how
                # a shot ends up guarding the previous shot's cast.
                _who_here = (active if character_guard else
                             [n for n, _ in sheet_lines(shot_sheet) if n])
                # ...and ALSO for anybody the keyframe still carries. A beat that
                # names only the other person -- "Sam watches from the doorway" --
                # left her out of the shot's cast, so nothing said what was on her
                # chest for that one beat, and the model filled it in. Reported as
                # a bra popping into ONE beat: this is the beat.
                #
                # The previous shot's cast, because that is the frame this shot
                # opens on. It is one continuity sentence, not a sheet entry --
                # no face, no wardrobe, nothing that would stage a person who is
                # not there. She is already in the picture; the words only have to
                # stop contradicting it.
                # NOT gated on the shot starting fresh, though the frame is the
                # reason this exists. That gate was written and reverted: the shot
                # that uncovers a region is a removal shot, restart_after_removal
                # makes the NEXT one fresh, and that next one is exactly the shot
                # this clause is for -- so the gate disabled the fix in every real
                # case while the reported bug stayed.
                #
                # The residual risk is real and stated: on a fresh shot nothing
                # pictorial carries her, so naming her is one sentence about
                # somebody the beat did not stage. It is one clause, not a sheet
                # entry, and the alternative is the region the model fills in by
                # itself. If a duplicate of the UNDRESSED character ever shows up,
                # this is the first thing to look at.
                _carried_on = [n for n in (_was or []) if n not in (_who_here or [])]
                _rows = []
                for _n in list(_who_here or []) + _carried_on:
                    _q = _state.people.get(_n)
                    if _q and _q.bare:
                        # WHAT IS ACTUALLY ON, from the state -- not the sheet.
                        # The sheet still lists the shirt, because the character
                        # memory is never edited, so passing it here suppressed
                        # every region the sheet ever mentioned and the clause
                        # could only ever speak about feet.
                        _rows.append((_n, list(_q.bare), ", ".join(_q.worn)))
                _name_it = (len(_rows) > 1 or len(_who_here or []) > 1
                            or any(_n in _carried_on for _n, _r, _o in _rows))
                _bare = "".join(
                    bare_hold(_rg, covers, _on,
                              # WHAT HAS COME OFF, so a layer the sheet puts
                              # underneath stops suppressing the clause once it
                              # has come off too. Cumulative, not this beat's:
                              # this path exists to speak on the shots AFTER the
                              # removal. Minus anything an `add:` put back on,
                              # which is worn again and is covering again.
                              [g for g in gone if g not in restored],
                              whose=(_n if _name_it else ""),
                              # PER PERSON here, unlike the single-cast path above:
                              # this loop already runs once for each of them, so each
                              # body and each figure is read off that person's OWN
                              # entry. One age applied to two people is the bug the
                              # `whose` argument exists to prevent, one attribute over.
                              body=body_of(*_pron_age(shot_sheet, _n)),
                              figure=figure_of(*_pron_age(shot_sheet, _n)))
                    for _n, _rg, _on in _rows)
                if _bare:
                    bare_held.append(len(plan) + 1)
            if _bare:
                bared_shots.append(len(plan) + 1)
            # Terminated, or the last sheet line welds onto the beat -- "grey coat
            # Maya lies still" -- and a name fused to the end of an attribute list is
            # read as one more item in it.
            # THE CHARACTER MEMORY IS NOT EDITED. This briefly scrubbed hardware
            # out of the sheet before the beat that stages it, to stop a cuff
            # appearing on a wrist before the cuffing. It was the wrong lever and
            # it was told so: "Stop removing items from the character memory!"
            #
            # It was also worse than it looked. scrub_removed drops the whole
            # comma-separated entry, so "green dress, steel collar" lost the line
            # -- and with the line gone the person went with it, leaving shots
            # with nobody described in them at all.
            #
            # The sheet is the author's. Where it disagrees with the script the
            # node says so in the report and holds ITS OWN clause back, which is
            # the half that was actually asserting a lie. Only removals the
            # author staged still scrub, which is what that mechanism is for.
            # COVERED IS NOT REMOVED. `covered` used to go in here beside
            # `visible`, so a garment read as under something came out of the
            # sheet entirely and took its <Picture N> with it. Reported three
            # times as items disappearing out of the character memory. Only
            # removals the AUTHOR staged scrub now; being underneath is said, in
            # under_clause, not enacted by deletion.
            # UNDERWEAR IS PLACED, NOT DELETED. Everything else that is covered
            # still waits: a locket under a coat cannot be seen, nothing is lost
            # by holding it until the coat comes off, and its picture would ask
            # the model to draw a thing that is not visible.
            # ...and only for people this shot actually describes. A garment
            # whose owner is not in the frame is a garment drawn on whoever is.
            # NOT `_described` -- that is assigned further down the loop, so
            # reading it here would answer with the PREVIOUS shot's cast. Same
            # expression, evaluated where it is needed.
            _here = set(active if character_guard
                        else [n for n, _ in sheet_lines(shot_sheet) if n])
            _worn_under = [u for u in covered
                           if is_undergarment(u)
                           and (cover_owner.get(u) in _here
                                or u not in cover_owner)]
            _hidden = [u for u in covered if u not in _worn_under]
            # A REMOVAL TAKES THE GARMENT OFF THE PERSON WHO REMOVED IT, AND NOBODY
            # ELSE.
            #
            # Reported: shirts looking half missing. Two women in white shirts, one
            # takes hers off, and the token "shirt" was scrubbed from the WHOLE text --
            # so the other one's entry lost her shirt while the keyframe still showed
            # her wearing it. That is the contradiction this file already describes:
            # "the shot then says it is not worn, take it off, and the thing under it
            # is already showing. The model renders that contradiction as a garment
            # half present -- open, or partly cut". The strip-bare path has been scoped
            # to "THEIR OWN entry" since it was written; the prose-removal path was not.
            #
            # Scoped HERE rather than inside scrub_removed: that function applies a
            # second, whole-text sweep after its per-sentence pass, and threading an
            # owner through both would mean restructuring 186 lines whose comments
            # record a dozen separate fixes. Calling it once per entry gets the same
            # answer and leaves it untouched.
            #
            # gone_by maps a token to who took it off. A token nobody is recorded for
            # stays unscoped, which is what keeps every other removal behaving as it did.
            _toks_all = visible + _hidden
            # Scoped to this shot's people first: see static_for_shot.
            _static_here = static_for_shot(static, sheet, shot_sheet)
            _scrubbed = ([scrub_removed(terminate_lines(_static_here), _toks_all)]
                         if _static_here.strip() else [])
            for _ln in (terminate_lines(shot_sheet).split("\n")
                        if shot_sheet.strip() else []):
                _m = re.match(r"\s*([A-Za-z][\w'\u2019-]*)\s*:", _ln)
                _who = _m.group(1).lower() if _m else ""
                _allow = [t for t in _toks_all
                          if not (_who and gone_by.get(t) and _who not in
                                  {str(x).lower() for x in gone_by[t]})]
                _scrubbed.append(scrub_removed(_ln, _allow))
            shot_scene = "\n".join(p for p in _scrubbed if p.strip())
            # A NAMED CLOSE FRAME STOPS DESCRIBING WHAT IT CANNOT HOLD.
            #
            # Applied HERE, last, on the finished text: the layer, removal and
            # restore logic has already run and its bookkeeping is untouched, so
            # this only changes what is SAID on this shot and can never be mistaken
            # by anything downstream for a garment coming off. The next shot builds
            # from the sheet again, so nothing accumulates.
            _holds = frame_holds(anchor) or frame_holds(body)
            _cropped = out_of_frame_garments(shot_scene, _holds)
            if _cropped:
                shot_scene = hide_item(shot_scene, _cropped)
                for _c in _cropped:
                    if _c not in cropped_wardrobe:
                        cropped_wardrobe.append(_c)
            # THE REFERENCE STAYS. It was taken off for one commit, on the
            # reasoning that a near-clean reference reproduces its picture and so
            # draws the belt over the jeans -- which is true as far as it goes,
            # but I changed the occlusion clause in the SAME commit and so never
            # tested the combination that matters: the picture present AND the
            # cover described as a whole opaque surface. The version that poked
            # through had the picture with the weak clause.
            #
            # An author who attaches a <Picture N> to an item wants that item to
            # look like that picture, and dropping the tag drops the reference
            # entirely -- there is no weaker setting for one image, only
            # ref_noise_aug for all of them. So it stays, and the cover carries
            # the weight. See under_clause.
            # The words stay in every shot; the PICTURE waits for the cover
            # to come off. See defer_tag_for -- a reference reproduces its
            # image and draws the thing, whatever the text says is over it.
            _deferred = list(_worn_under)
            # THE WORDS WAIT WITH THE PICTURE. At cfg 1 there is no negative
            # prompt, so naming a thing draws it -- and with the picture
            # already withheld and the occlusion clause no longer naming the
            # belt, the sheet's own mention was the last one standing and it
            # was enough on its own. Text cannot take itself back; every
            # wording added to suppress it made it worse.
            #
            # Held back, not deleted, and the difference is what made this
            # feel like deletion the first time: lifting a skirt was not read
            # as a displacement, so the cover never came off and the item
            # never returned. That is fixed, the restore verbs are in, and
            # the report below names the item and the shots. hide_item is
            # surgical where scrub_removed is not: it takes the phrase and
            # leaves the entry, so a person's line cannot go with it.
            shot_scene = defer_tag_for(shot_scene, _worn_under)
            shot_scene = hide_item(shot_scene, _worn_under)
            if _deferred and len(shot_scene) >= 0:
                deferred_shots.append((len(plan) + 1, list(_worn_under)))
            _under = under_clause(
                [(u, covers.get(u, ""),
                  cover_owner.get(u, "") if len(_here) > 1 else "")
                 for u in _worn_under])
            # A READING COPY, never emitted. The sheet is sent to the model exactly
            # as written; this is only what the node consults when deciding whether
            # to assert hardware is FASTENED, and it leaves out anything the script
            # stages later. Without it the sheet's own mention latched the standing
            # hold from shot 1 -- "the handcuffs stay closed and fastened as they
            # were put on", two shots before anybody put them on -- which is the
            # assertion that had to stop, as against the author's description,
            # which did not.
            _sheet_says_early = [c for c, at in _staged_at.items()
                                 if c in _sheet_hw and at > len(plan) + 1]
            _scene_for_state = (scrub_removed(shot_scene, _sheet_says_early)
                                if _sheet_says_early else shot_scene)
            # "ALREADY ON" MEANS BEFORE THIS SHOT. The applying test asks whether
            # the hardware was on before the beat that puts it on, so the item
            # being staged HERE has to be out of the answer as well -- otherwise
            # the sheet's own mention vetoes the both-ends clause on exactly the
            # shot that stages the fastening, and it gets the standing hold: a lie
            # about its first frame.
            _sheet_says_now_or_later = [c for c, at in _staged_at.items()
                                        if c in _sheet_hw and at >= len(plan) + 1]
            _scene_before_now = (
                scrub_removed(shot_scene, _sheet_says_now_or_later)
                if _sheet_says_now_or_later else shot_scene)
            # Retirement is handled at the moment of removal, above, so this is just
            # what is currently on. Filtering here against the whole history of `gone`
            # meant an add could never put anything BACK: the token stays in `gone`
            # for the rest of the film, so "add: her locket is back on" was suppressed
            # by the removal that took it off in the first place.
            # A garment going ON in THIS shot is described by the wearing clause,
            # which gives it both ends. Listing it here as well would say it is
            # already worn while the clause says it is being put on -- the same
            # shot holding the garment in two states, which is the disagreement
            # that made it appear at the first frame. It joins the static wardrobe
            # from the NEXT shot on, exactly as a removal scrubs from its own.
            live = [a for a in shown if a not in _staged_add]
            if live:
                tail = ". ".join(a.rstrip(".") for a in live) + "."
                tail = tail[0].upper() + tail[1:]
                shot_scene = f"{shot_scene} {tail}".strip() if shot_scene else tail
            # The removal has to FINISH inside this shot, because its last frame is
            # the next shot's keyframe. Stated only here; naming the garment again
            # later would put it back.
            #
            # A full strip says it once rather than reciting the wardrobe: listing
            # eight garments coming off is eight more mentions of clothing in a shot
            # whose point is that there is none.
            # WHOSE HANDS. Without an agent the clause says a garment comes off by
            # itself, and a belt nobody is touching drops to the floor -- reported on
            # a beat where she ASKS to have it taken off, which the clause turned into
            # it removing itself. The wearer is read from the sheet where the item is
            # listed, so "she asks Dan" gives the hands to Dan and not to her.
            # The sheet, or the SCENE when the sheet is empty. A sheet paragraph
            # that was folded into the scene never reaches pull_character_sheets --
            # it only ever sees the beat -- so `sheet` is "" for the whole run and
            # shot_sheet with it. Both the wearer and the cast then came back empty
            # and EVERY removal clause went out agentless: the beat says she takes
            # the shorts off, the clause says they come off with no hands named, and
            # with a second person in the shot the model gives that second removal to
            # him. The action happens twice, once by each of them.
            _who_sheet = shot_sheet if sheet_lines(shot_sheet) else scene
            # ...AND WHEN TWO ENTRIES LIST THE SAME KIND OF GARMENT, the first entry
            # was taken. "Lena takes off her sweater" with Maya also in a sweater put
            # the bare chest on Maya -- "Maya's chest, shoulders and arms are bare
            # skin" beside Maya's own entry still listing her green sweater, one woman
            # described both clothed and bare, which is a woman drawn twice. The sheet
            # stays the answer when it is unambiguous (it is what gets "she asks Dan"
            # right); a tie goes to whoever the scene state recorded taking it off
            # this beat, then to whoever the beat names.
            _listed = [n for n, ln in sheet_lines(_who_sheet)
                       if n and names_any(ln, toks)]
            if len(_listed) > 1:
                _by_state = [w for w, _g in (_ch.get("removed") or []) if w in _listed]
                _by_beat = engine.names_in(body, _listed)
                _wearer = (_by_state or _by_beat or _listed)[0]
            else:
                _wearer = _listed[0] if _listed else None
            # WHOSE body is bare. Unattributed in a two-person shot this reads as
            # an instruction about everyone on screen, and the second character
            # undresses alongside the first. Done HERE because _wearer is what
            # answers it, and `active` rather than `_described` because that is
            # assigned further down the loop -- reading it here would get the
            # PREVIOUS shot's cast.
            _bare = own_body(_bare, _wearer or (active[:1] if active else []),
                             active if character_guard else
                             [n for n, _ in sheet_lines(_who_sheet) if n])
            # `active`, not `_described`: that is assigned further down the loop, so
            # reading it here gets the PREVIOUS shot's cast -- which on this shot meant
            # Dan was not in it, the "asks" rule never applied, and the clause gave the
            # hands back to the person doing the asking.
            _cast_here = (active if (character_guard and active) else
                          [n for n, _ in sheet_lines(_who_sheet) if n])
            # PER GARMENT. One agent for the whole beat meant a beat that takes a
            # coat off and then asks about a scarf gave BOTH to the other person --
            # her own coat came off by his hands. Each garment is attributed on its
            # own clause, and garments sharing an agent are said in one sentence.
            _by_agent = {}
            for _t in (toks if not bare else []):
                _w = next((n for n, ln in sheet_lines(_who_sheet)
                           if n and names_any(ln, [_t])), _wearer)
                _a = removal_agent(body, _cast_here, _w, _t)
                _by_agent.setdefault(_a, []).append(_t)
            tail = (own_body(BARE_HOLD, _wearer or (active[:1] if active else []),
                             active if character_guard else
                             [n for n, _ in sheet_lines(_who_sheet) if n])
                    if (bare and toks)
                    else "".join(off_by_last_frame(_items, _a, scene, body)
                                 for _a, _items in _by_agent.items()))
            # Once hardware is on, it stays on. Latched, not re-detected: a beat that
            # does not mention the cuffs does not mean they came off, and a cuff that
            # renders open is not a detail that drifts -- it is the scene ceasing to
            # make sense. Cleared only by a `remove:` that names the hardware.
            _was_restrained = restrained
            if hold_restraints:
                if (names_any(RESTRAINT_HOLD_KEY, toks)
                        or any(restraint_present(t) for t in toks)
                        # ...or the BEAT itself says the hardware comes off. Without
                        # this the latch could only ever be cleared by a remove:
                        # line, and a script that unlocks the cuffs in its own prose
                        # kept being told they stay fastened -- for the rest of the
                        # film, over hardware lying on the floor.
                        # ...and only when this beat's undoing actually took a
                        # piece of hardware out of the sheet. "Sam cuts the rope
                        # free" reads as an undoing, but she wears handcuffs, and
                        # clearing on the verb alone unlocked them.
                        or (restraint_coming_off(body)
                            and any(_RESTRAINT_WORD.match(str(t)) for t in toks))):
                    restrained = posed = rigid_latched = False
                    anchored = ""
                    worn_item = ""
                    worn_items = []
                    restrained_who = set()
                elif restraint_present(body) or restraint_present(_scene_for_state):
                    restrained = True
                    # At the moment hardware GOES ON -- every time, not only the
                    # first. Latching once meant a second person cuffed in a later
                    # beat never joined the set, so their hardware was applied and
                    # then never described again for the rest of the film.
                    #
                    # Still not re-read on shots that merely MENTION restraints:
                    # that was the original fault, where the man alone checking the
                    # cuffs was marked as wearing them.
                    if not _was_restrained or restraint_going_on(body):
                        _new = restrained_by_beat(body, active)
                        restrained_who |= (_new if _new else set(active))
            # The shot where the hardware GOES ON. Newly restrained -- so it was not on
            # before -- and the beat stages the act rather than describing it worn. On
            # that one shot the standing hold is a lie about the first frame, and a
            # first frame that already has the cuffs closed leaves the struggle to
            # happen in whatever order is left over. That is being caught after being
            # restrained instead of before.
            #
            # "Already on" has to include what the SCENE says, not only the latch.
            # On shot 1 the latch is empty by definition, so a sheet reading "wrists
            # cuffed behind back" would otherwise let a beat that locks a SECOND item
            # on declare the first one off at the first frame.
            # Every item, not just the newest. worn_item was a single string, so
            # "cuffs her wrists" then "gags her with duct tape" overwrote the
            # cuffs -- and from that shot on the cuffs were never named again,
            # which is hardware that stops being drawn.
            _named_item = hardware_named(body) if restrained else ""
            # EVERY item this beat names, not just the most specific one. One beat
            # that cuffs the wrists AND locks on a collar used to record whichever
            # phrase was longer and drop the other for the rest of the film.
            for _hw in (hardware_all_named(body) if restrained else []):
                # Substring-aware, because the beats name the same thing differently
                # from shot to shot: "handcuffs" in shot 1 and "the cuffs" in shot 4
                # is ONE pair of handcuffs, and an exact-match check listed both --
                # "The handcuffs, steel collar, chain and cuffs stay closed", which
                # reads as four things and invites the model to draw a spare set.
                _same = next((k for k, p in enumerate(worn_items)
                              if p in _hw or _hw in p), None)
                if _same is None:
                    worn_items.append(_hw)
                elif len(_hw) > len(worn_items[_same]):
                    worn_items[_same] = _hw
            # THE ENGINE IS THE AUTHORITY ON WHAT IS ON WHOM, and this is the
            # only place the answer comes from now. The old derivation ran here
            # too, in parallel, and a disable-check showed the engine was not
            # load-bearing at all: pull it out and nothing changed, because both
            # paths were computing the same thing and the old one won by being
            # first. A second implementation nothing depends on is not a port.
            #
            # So the old accumulation is gone. What is on somebody is what the
            # state says is on them -- read once per beat, every item recorded
            # rather than the longest, each modifier bound to its own item.
            _eng_hw = [r for p in _state.people.values()
                       for r in p.hardware.values()]
            worn_items = []
            for _r in _eng_hw:
                _same = next((k for k, p_ in enumerate(worn_items)
                              if p_ in _r.item or _r.item in p_), None)
                if _same is None:
                    worn_items.append(_r.item)
                elif len(_r.item) > len(worn_items[_same]):
                    worn_items[_same] = _r.item
            worn_item = ", ".join(worn_items)
            # THE SCRIPT DECIDES THE MOMENT, and the sheet check must not veto it.
            # Blocking on restraint_present(shot_scene) is right when the sheet
            # says somebody is ALREADY restrained and the beat merely mentions
            # it. It is wrong once the sheet's own hardware has been held back
            # until this beat: the sheet then names the cuffs in exactly the shot
            # that applies them, which suppressed the both-ends clause and left
            # the applying shot with a standing hold -- a lie about its first
            # frame, and the cuffing happening in whatever order was left over.
            # ...and ONLY for hardware the sheet itself named and this held back
            # until now. A sheet that says "wrists cuffed behind back" beside a
            # beat that locks a CHAIN on is a different situation: the cuffs are
            # genuinely already on, the sheet check is doing its job, and
            # overriding it there cost the cuffs their standing hold. The veto is
            # lifted only where this node created the conflict.
            _stages_now = any(at == len(plan) + 1 and canon in _sheet_hw
                              for canon, at in _staged_at.items())
            _applying = bool(restrained and not _was_restrained
                             and not restraint_present(_scene_before_now)
                             and restraint_going_on(body))
            # The sheet claiming hardware the beat is only now putting on. The sheet
            # goes into EVERY shot, so it is on her in the shots before it happens,
            # and this shot is told it is already fastened rather than going on.
            # Not the node's to resolve -- the sheet is the author's standing
            # description and the beat is the author's action -- but it is exactly
            # the shape that renders as being restrained first and caught after.
            # Not gated on the latch: the sheet has already made her restrained
            # from shot 1, which is the whole problem being reported. Recorded
            # once -- it is one authoring decision, not one per shot.
            if (not early_hardware and restraint_going_on(body)
                    and restraint_present(_scene_for_state)):
                early_hardware.append(len(plan) + 1)
            # Rigidity latches like the hardware itself. Steel locked on in shot 1 is
            # still steel in shot 5, and a beat that does not happen to say "chain"
            # does not mean the chain became rope -- but tested per shot, that is
            # exactly what happened: the shot naming it got the rigid clause and every
            # shot after it fell back to the soft one. Which is where the slack came
            # back from.
            if restrained and rigid_hardware(f"{body} {shot_scene}"):
                rigid_latched = True
            # And a position that hardware was locked to enforce latches too: the chain
            # that put a body in a squat is still that length three shots later, so the
            # squat is still the position.
            if rigid_latched and forced_pose(f"{body} {shot_scene}"):
                posed = True
            # WHERE the fastened limbs are held latches the same way, and for the
            # same reason the pose does. Cuffs above the head are above the head
            # three shots later: nothing let go of them. The restraint hold keeps
            # them SHUT and says nothing about position, so the only thing carrying
            # it was the picture -- and a close shot crops the anchor point straight
            # out of frame, which is the reported failure exactly.
            # POSTURE, latched the way the gaze is. A beat that sits somebody down
            # ends its shot with them seated; the next beat says nothing about it,
            # so the shot was free to stand them back up -- reported as the end of
            # one beat and the start of the next not matching. The keyframe does
            # carry the pose as a picture, but the text is what the model
            # reconciles it against, and text saying nothing loses to a reference
            # saying something.
            #
            # Said only on the shots AFTER the one that stages it: the staging beat
            # has the author's own words and does not need a sentence arguing
            # beside them. Cleared by whatever the new beat stages instead.
            # WHERE the shot goes. A beat that walks somebody from one room to
            # another is a staged change with two ends -- told only where it
            # finishes, the shot renders the destination and cuts straight to it,
            # with the hallway between them missing. Named both ends, the way a
            # door's direction is.
            # A short action in a long shot is performed at once and then carried
            # on to fill the rest. Give it the whole shot to happen in.
            # This beat's own length. plan_lengths sizes each beat independently,
            # so asking it for one gives the same answer the whole run will --
            # and `lens` itself is not computed until after this loop.
            _have = plan_lengths([body], ceiling,
                                 shot_length == "from the beat", pace)[0][0] / H3_FPS
            _pace = pace_clause(beat_seconds(body), _have)
            if _pace:
                paced_shots.append(len(plan) + 1)
            # travel_legs, not travel_in: the promotion of a bare via to the
            # destination is read by the SIZING too, and a transit rendered as a walk
            # while sized as if it went nowhere is how a three-room walk ended up in a
            # three-second shot. See travel_legs and travel_spaces.
            _frm, _via, _to = travel_legs(body)
            _travel = travel_anchor(_frm, _via, _to, here)
            if _travel:
                travel_shots.append(len(plan) + 1)
            else:
                # The place list could not name either end. Perform the arrival
                # anyway: a move nobody is told to make is a move the model cuts to.
                # See moved_to -- this establishes no room state at all.
                _open_to = moved_to(body, active)
                _travel = move_clause(_open_to)
                if _travel:
                    open_moves.append((len(plan) + 1, _open_to))
            # The room the next beat starts from: where this one ended, or where it
            # simply says everyone is.
            _room_before = here
            here = _to or _frm or place_named(body) or here
            # A ROOM THE KEYFRAME IS NOT IN IS A CUT.
            #
            # Every shot is anchored to the previous shot's last frame, and a keyframe
            # is a PICTURE, which outvotes any sentence -- the reasoning
            # restart_after_removal is already built on. So a shot that OPENS in a
            # different room from the one the shot before ended in has a first frame
            # showing the wrong room, and the model reconciles the two by blending
            # them. Reported as a living room turning into a bathroom, which is exactly
            # what a kitchen frame and the words "living room" have in common: tiles, a
            # sink, cabinets. Breaking the chain costs a cut where a cut belongs.
            #
            # A WALK IS NOT THIS. A travel beat opens in the room it is leaving, so that
            # frame is the right one -- which is why the test is on where the shot
            # OPENS, not on whether the room changed. A beat naming an origin of its own
            # is judged on that origin, so "walks from the bedroom to the bathroom"
            # after a kitchen shot is still a cut.
            _opens_in = _frm or (_room_before if _travel else here)
            if (len(plan) and _opens_in and _room_before and _opens_in != _room_before):
                cut_shots.add(len(plan))
            if here and here not in _described_rooms and here not in _undescribed:
                _undescribed.append(here)
            # ...and say so on later shots, because the scene paragraph still
            # names the room they started in and is stamped into every shot.
            _where = where_hold(here, scene) if not _travel else ""
            if _where:
                where_shots.append(len(plan) + 1)
            # ...and the ACOUSTIC follows them. Both were read ONCE, before the
            # loop, out of the scene -- so a film that walks into a tiled bathroom
            # went on being told it sounds like the carpeted living room it left.
            # H3 is joint, so that is the picture told one room and the audio told
            # another, inside the same conditioning: the contradiction the room
            # hold was written to end, arriving through the other branch.
            #
            # Only where the room has actually changed, and only when the new room
            # has a sound of its own -- otherwise the film's own bed stands, since
            # one bed across a chain is part of what makes it one film. A travel
            # shot keeps the origin's acoustic, because that is where it begins.
            _room_now = (room_tone(here) or _room) if (auto_sound and _where) else _room
            _bed_now = ((scene_ambient(here) or ambient_bed)
                        if (auto_sound and _where) else ambient_bed)
            if _where and auto_sound and (_room_now != _room or _bed_now != ambient_bed):
                acoustic_shots.append((len(plan) + 1, here))
            _pose_now = posture_in(body, active if character_guard and active
                                   else [n for n, _ in sheet_lines(_who_sheet) if n])
            # ...and let go of any the beat contradicts. A pose that survives an
            # action nobody performs in it is a hold arguing with its own shot.
            for _gone_pose in posture_cleared(body, poses):
                poses.pop(_gone_pose, None)
            _posture = ("" if not hold_scene_state
                        else posture_hold({n: p for n, p in poses.items()
                                           if n not in _pose_now},
                                          # `active`, not `_described`: that is
                                          # assigned further down this loop, so
                                          # reading it here gets the PREVIOUS
                                          # shot's cast.
                                          active if character_guard else
                                          [n for n, _ in sheet_lines(_who_sheet) if n]))
            if _posture:
                posture_shots.append(len(plan) + 1)
            poses.update(_pose_now)
            _anchor_now = limb_anchor(body) if restrained else ""
            if _anchor_now:
                anchored = _anchor_now
            # Said only on the shots AFTER the one that staged it. The staging shot
            # has the author's own words for this and does not need a second
            # sentence arguing beside them.
            # Where the limbs are held is inside the restraint sentence now. What is
            # still worth reporting is that it is being held, and where the framing
            # is tight enough to crop the anchor out of the picture the chain hands
            # on -- so those key off the latch rather than off a clause.
            _holding = bool(restrained and anchored and not _anchor_now)
            if _holding:
                anchored_shots.append(len(plan) + 1)
            if _holding and (_anchor_tight or tight_framing(body)):
                tight_shots.append(len(plan) + 1)
            # A turn shows a surface the keyframe never pinned, and the model fills
            # it from a clothed prior. Only on shots that turn, and only once there
            # is something to hold -- a removal already made, or hardware on.
            turn = TURN_HOLD if (turns_in(body, cast)
                                 and (gone or shown or restrained)) else ""
            # Going down with the hands fastened: say what takes the landing, or the
            # model frees the hands to break the fall and the hardware gives way.
            #
            # A FREE body needs the landing named too, for a different reason. Reported:
            # a third leg on the shot where she fell, grown to brace a landing nothing
            # in the text was taking. A fall is the frame where limbs are least
            # determined -- fast motion, heavy occlusion, and a middle the model has to
            # invent -- so leaving it to work out what catches the body is leaving it
            # free to add something that can.
            _falls = falls_in(body)
            fall = (FALL_HOLD if (restrained and _falls)
                    else FALL_HOLD_FREE if _falls else "")
            if fall:
                fall_shots.append(len(plan) + 1)
            # Steel is not rope. Without being told, the model draws a chain slack --
            # sagging, stretching to wherever a limb is going, allowing movement the
            # hardware does not allow. Only where such hardware is actually named.
            rigid = restrained and rigid_latched
            # Where the hardware is holding a POSITION, its length is the reason the
            # position holds -- and a chain drawn with slack is room to stand out of it.
            chain = (CHAIN_POSE_HOLD if (rigid and posed)
                     else CHAIN_HOLD if rigid else "")
            # Hardware named with nowhere to sit. A collar with no neck beside it is a
            # band with no place to be, and it ends up on the head. Only where this
            # beat itself raises the item, and only when the text has not already put
            # it somewhere -- what you wrote wins.
            anchors = anchor_clause(unanchored_hardware(body))
            if anchors:
                notes.append(f"shot {i_shot + 1} names hardware with no body part beside "
                             f"it, so the shot says where it sits: "
                             f"{anchors.split(': ', 1)[1].rstrip('.')}")
            # The scene's description of a room this shot does not END in waits here.
            # The paragraph is stamped into every shot, and a paragraph that describes
            # a room describes its furniture too -- which is how a bed reached a living
            # room two beats after she left the bedroom. See scene_for_here. Only the
            # text SENT changes: shot_scene itself is left alone, so every reader above
            # and below this line keeps its full view of the scene.
            _scene_sent, _held_rooms, _held_blocked, _held_text = scene_for_here(
                shot_scene, here, anchor,
                [n for n, _ in sheet_lines(shot_sheet) if n], body)
            if _held_rooms and _held_blocked:
                scene_welded.append((len(plan) + 1, list(_held_rooms)))
            elif _held_rooms:
                scene_held.append((len(plan) + 1, list(_held_rooms), list(_held_text)))
            # WHAT LEADS DECIDES THE FRAME. See split_sheet: the appearance block goes
            # after the action it describes, so the opening tokens are the place and
            # what happens in it rather than sixteen words of face.
            # NOT ON A SHOT CARRYING A PICTURE TAG. <Picture N> is numbered by the
            # order the tags APPEAR in the shot, and the number is the image's place in
            # that shot's reference list -- so moving the sheet past the beat renumbers
            # them, and a renumbered reference is the wrong face on the wrong person.
            # That is the oldest and worst bug in this file and it is not worth a
            # composition gain. A shot with no tag has no numbering to disturb.
            if beat_leads and _scene_sent and not picture_tags(f"{_scene_sent} {body}"):
                _scene_part, _sheet_part = split_sheet(
                    _scene_sent, [n for n, _ in sheet_lines(shot_sheet) if n])
                line = " ".join(p for p in (_scene_part, body, _sheet_part) if p).strip()
                if _sheet_part:
                    led_shots.append(len(plan) + 1)
            else:
                line = f"{_scene_sent} {body}".strip() if _scene_sent else body
            # A state the text asserts but does not stage. Read from the whole line,
            # because the van usually stands in the scene paragraph rather than in
            # the beat -- and suppressed for anything this beat is actually working,
            # since a shot that opens the doors is a shot about the doors opening.
            _pairs, _moves = [], []
            if hold_scene_state:
                _moves = state_changes(body)
                _acting = [_state_key(t) for t, _ in _moves]
                _pairs = [(t, s) for t, s in stated_states(line)
                          if _state_key(t) not in state_acted and _state_key(t) not in _acting]
            # Which end of the action is which. Some distill LoRAs render a staged
            # change backwards, and a beat that names one state names neither end.
            _turn = direction_anchor(_moves)
            # The two share a budget. Holding a state and anchoring a change are both
            # continuity, and four such sentences is a shot about its own continuity.
            _state_clause = state_hold(_pairs[:max(0, 2 - _turn.count("first frame"))]) + _turn
            if _pairs:
                stated_shots.append(len(plan) + 1)
            if _turn:
                turned_shots.append(len(plan) + 1)
            # The beat and the hold asking for opposite things. Reported three times
            # running as "the doors keep opening", and every time the node text was
            # by then correct -- it was the beat staging an exit the doors have to
            # open for. Say it; do not touch the wording.
            if _pairs and exits_vehicle(body) and any(
                    _state_key(t) in ("door",) for t, _ in _pairs):
                notes.append(
                    f"shot {len(plan) + 1} says somebody gets OUT of a vehicle and also "
                    f"says the doors are closed. Those are opposite instructions and the "
                    f"beat wins: a person leaving a van opens a door to do it, so the "
                    f"doors open however firmly the text says they are shut. If they are "
                    f"meant to be shut the whole shot, the people cannot be leaving the "
                    f"vehicle in it -- write them already out and standing ('Mara and Dom "
                    f"stand behind the van, its rear doors closed'), or put the exit in "
                    f"its own earlier shot. Your wording is never edited, so this is "
                    f"yours to resolve.")
            # Latch what this beat changed, so no later shot re-asserts the old state.
            state_acted.update(_state_key(t) for t, _ in _moves)
            # The chain clause SUBSUMES the restraint hold -- it says "whole and closed"
            # itself. Emitting both said it twice, which is twice the stasis for one
            # guarantee.
            # On the shot that PUTS the hardware on, both ends instead of the standing
            # hold: the chain clause is about a chain that is already taut, and the
            # restraint hold asserts a first frame that has not happened yet.
            # ...and where the limbs finish, so the NEXT shot's keyframe has them
            # in the right place. See RESTRAINT_ENDS_AT.
            _ends_at = ""
            if _applying and _anchor_now:
                _pos = _anchor_now.split(", at the")[0].strip()
                if _pos and not _pos.startswith("at the "):
                    _ends_at = RESTRAINT_ENDS_AT.format(
                        part=engine.held_part_of(worn_items) or "wrists",
                        where=_pos)
            # The limb pose, said as a body, on every shot the position holds --
            # the applying shot included, where it says where they FINISH.
            # ...and whether anybody holding that position is off their feet, which
            # is what decides if the weight needs naming. Read off the poses this
            # shot is carrying, not off the beat: the beat that lays her down is
            # rarely the shot the propped arm shows up in.
            _lying_now = any(_p == "lying down" for _p in poses.values())
            # ...and the posture may only be written once in the scene too, which is
            # the same asymmetry the anchor below had: a scene reading "McKenna lies
            # in the back" put nobody in a posture, because posture_in reads the
            # beat. So a restrained body the script never lays down ON SCREEN was
            # never known to be off its feet, and the weight clause -- the whole
            # point of which is bodies that are -- could not fire for it.
            #
            # ONLY for somebody the shot holds in hardware, and only while NOTHING
            # is latched for them. A beat that stands her up latches standing and
            # this stops: the scene paragraph still says she lies in the back, and
            # believing it over the beat would hold her down for the rest of the
            # film. The author's beat outranks the author's scene, always.
            #
            # The posture HOLD is deliberately not given this. That sentence exists
            # to carry a pose the scene text does not, and the scene is stamped into
            # every shot verbatim -- "McKenna is still lying down" beside a scene
            # that just said she is lying in the back is the node repeating the
            # author back to the author. What was missing was the physics, not the
            # restatement.
            # restrained_who can be EMPTY while restrained is True -- it is filled
            # from the beat that applies the hardware, and a script whose restraint
            # is only ever stated in the scene never has such a beat. So "nothing
            # latched for the people in the hardware" has to degrade to "nothing
            # latched at all" rather than refusing to answer, or this misses exactly
            # the scripts it was written for.
            if engine.posture_in(body):
                beat_said_posture = True
            if not _lying_now and restrained and not beat_said_posture:
                _watch = restrained_who or set()
                _free = (not any(n in poses for n in _watch)) if _watch else (not poses)
                if _free and engine.posture_in(_scene_for_state) == "lying down":
                    _lying_now = True
            # WHERE THE WRISTS ARE MAY ONLY EVER BE SAID ONCE, IN THE SCENE.
            #
            # restrained is set by `restraint_present(body) or
            # restraint_present(_scene_for_state)` -- the beat OR the scene. The
            # anchor was read from the beat alone. So the ordinary way of writing
            # this -- "McKenna: ..., handcuffed behind her back" on the sheet, or a
            # scene paragraph saying it once, and beats that never repeat it --
            # marked her restrained and left the position empty for the whole film.
            #
            # pose_clause looks its argument up in a dict, so empty is not a shorter
            # sentence, it is NO sentence: never told the wrists are together, never
            # told the arms are behind the body, and never told what takes the weight
            # when she lies down, because that clause reads this same anchor. Where
            # the text says nothing the model puts the hands where the picture wants
            # them, which is under her, propping her up.
            #
            # Reported twice, and neither the weight clause nor the wider anchor
            # vocabulary could reach it: both fixed readers that were never being
            # shown the text the position was written in.
            #
            # The BEAT still wins where it says one -- a beat that moves the wrists
            # is the author changing them -- and the latch still wins over the
            # scene, so this is only the fallback for a position that was stated
            # once and never repeated. _anchor_now itself is left alone: it is what
            # _holding keys off to tell a staging shot from the ones after it.
            _pose_pos = (_anchor_now or anchored
                         or (limb_anchor(_scene_for_state) if restrained else ""))
            _pose = pose_clause(_pose_pos.split(", at the")[0].strip(),
                                lying=_lying_now)
            hold = (RESTRAINT_GOING_ON + (CHAIN_RIGID_TAIL if rigid else "") + _ends_at
                    if _applying
                    else chain if chain else (RESTRAINT_HOLD if restrained else ""))
            if _applying:
                applied_shots.append(len(plan) + 1)
            # Name the thing on shots that do not. The hold says a restraint stays
            # fastened and never says WHAT, so a shot after the applying one is told
            # a restraint exists with no object to draw -- which renders as the
            # behaviour without the hardware. Skipped where the text already names
            # it, and where nothing has been seen to name.

            # A garment MOVED rather than removed. It stays in the scene text, so
            # the sheet keeps describing it the way it was WORN -- and the sheet is
            # re-stamped into every shot, which pulls it back up. Latch the state the
            # beat left it in and restate that instead.
            _staged_here = displaced_garments(body, shot_scene)
            if _staged_here:
                # The shot that STAGES a displacement -- the garment is being moved
                # on screen in it. Recorded because the render loop must not capture
                # a subject reference from it: moved_shots starts the shot AFTER.
                staging_shots.add(len(plan) + 1)
            for _g, _how in _staged_here:
                _was = displaced.get(_g, "")
                # Put back up again is a restore, not a new displacement.
                if _was == "pulled down" and _how in ("pulled up", "pulled back"):
                    displaced.pop(_g, None)
                else:
                    displaced[_g] = _how
            # A real removal takes the garment out of the scene, so there is nothing
            # left to describe as displaced.
            for _g in [g for g in displaced if names_any(g, toks)]:
                displaced.pop(_g, None)
            # "pulls them back up" names nothing, and a pronoun cannot be matched
            # against the wardrobe -- but with one garment displaced there is only
            # one thing it can mean, and leaving it displaced is the error that shows.
            # ...and a restore that NAMES the garment clears that one, however
            # many are displaced. "Lets the skirt fall" is not a pronoun and
            # does not need the one-garment guess.
            for _g in restored_garments(body, shot_scene):
                _head = str(_g).lower().split()[-1]
                for _k in [k for k in displaced
                           if str(k).lower().split()[-1] == _head]:
                    displaced.pop(_k, None)
            if len(displaced) == 1 and puts_it_back(body):
                displaced.clear()
            # The shot that STAGES the displacement already says so in the beat, and
            # saying it again is telling it twice. Matched on the HEAD NOUN: the key
            # is the sheet's full name ("blue denim shorts") while the beat says
            # "her shorts", so comparing whole names stopped recognising the beat
            # that was staging it and the staging shot got the guard as well.
            _body_low = (body or "").lower()
            _moved = displaced_hold([(g, h) for g, h in displaced.items()
                                     if not re.search(r"\b" + re.escape(g.split()[-1])
                                                      + r"\b", _body_low)])
            if _moved:
                moved_shots.append(len(plan) + 1)

            # ...and say WHOSE. Unattributed, "every restraint stays fastened" is an
            # instruction about whoever is on screen, so hardware locked onto one
            # character turned up on the other, over their clothes. Read from the sheet
            # entries, which are what say who is wearing it.
            _wearers = [n for n in restraint_wearers(shot_sheet)
                        if not character_guard or n in active]
            _described = (active if character_guard else
                         [n for n, _ in sheet_lines(shot_sheet) if n])
            # LATCHED, WITH AN EXPLICIT WAY OUT.
            #
            # This went both ways before settling here. Latched on any plural word it
            # stood the body count down for the whole film -- "the others have gone"
            # included -- and that clause is what keeps a duplicate or a stranger out
            # of the frame. Read per beat instead, a shot whose beat simply stops
            # mentioning the extras got "There is one person in the shot: one body,
            # one face" while five women were standing in it, which asserts four of
            # them out of existence.
            #
            # Both faults were the same missing piece: extras are STATE, and state
            # needs a transition out. extras_in is now absence-aware, so the latch no
            # longer fires on a sentence saying they left, and extras_dismissed is the
            # way out -- "she is alone now", "the others have gone". Background people
            # do not leave because a sentence stopped mentioning them, and they do not
            # stay for ever either.
            if extras_in(body):
                _extras_seen = True
            elif extras_dismissed(body):
                _extras_seen = False
            _cast_hold = cast_hold(_described, body, _extras_seen)

            # Where the beat says somebody is looking, said once more as a fact
            # about the eyes and the head. One mention in the beat loses to a
            # near-clean reference asking for the portrait's pose, and the
            # portrait looks at the lens because photographs of people do.
            # LATCHED, like every other state here. A look was stated once and then
            # dropped, so somebody watching a screen across four shots was told
            # where their eyes were in the first one only -- and the portrait pull
            # that made this necessary does not stop after one shot.
            #
            # Cleared by a beat that moves the look somewhere else, or one that
            # moves the person: walking away ends it, and holding a stale target
            # across that would be worse than saying nothing.
            #
            # PER PERSON, because a look is one. It was a single string with no
            # owner, held across shots and said impersonally, so a look staged by
            # one character went on being said in shots she was not in -- and
            # landed on whoever was: "McKenna looks at the lane" in shot 3, and
            # shot 4, describing only Dan, was told "the eyes and the head are
            # turned to the lane behind them." That is his head on her sightline,
            # and it is the same defect as the vocal flag -- a per-person fact kept
            # in a shot-level variable.
            _look_now = (look_target(body, shot_sheet, _described)
                         if hold_gaze else "")
            _lookers = (subjects_for(body, shot_sheet, _LOOK_VERB_SRC)
                        if hold_gaze else [])
            _look_is_person = bool(_look_now) and any(
                _look_now == _n for _n, _ in sheet_lines(shot_sheet))
            if _look_now:
                # Whoever the beat says is looking. If it names nobody, everybody it
                # describes -- which is what the single string did for everyone.
                for _n in (_lookers or (_described or [])):
                    looking_at[_n] = (_look_now, _look_is_person)
            elif (looks_somewhere(body) or arrives_in(body) or falls_in(body)
                  or turns_in(body, cast) or _MOVES_OFF.search(body or "")):
                # Clear only the people this beat actually moved or turned. When it
                # cannot be pinned on anybody, clear all of it: a stale target is
                # worse than none, which is why this branch exists at all.
                _ends = (_lookers
                         or subjects_for(body, shot_sheet, _MOVES_OFF_SRC))
                for _n in (_ends or list(looking_at)):
                    looking_at.pop(_n, None)
            # ...and said for people this shot describes, PLUS anybody who was in
            # the previous shot and whom this beat has not moved off. She is still
            # in the van when the beat is about him: the next shot starts from a
            # picture with her in it, and dropping her from the text is what leaves
            # her with nothing to do but face the lens. ONE shot of memory -- the
            # node knows she was in the last picture, not where she is now.
            _carried = [n for n in _was
                        if n not in set(_described or []) and looking_at.get(n)
                        and n not in set(subjects_for(body, sheet, _MOVES_OFF_SRC))]
            _gazers = [n for n in (_described or []) if looking_at.get(n)] + _carried
            _gaze = ""
            _faces = ""     # the eye-line inferred for a dialogue shot
            # What the frame holds, where the beat and the anchor both leave it open.
            # An unstated frame becomes the prior, and the prior for a described
            # person is a portrait facing the lens. See frame_hold.
            # WHO IS WITH WHOM. Only where it could be read wrong: with two people in
            # the shot there is nobody else to pair with, and naming them again costs a
            # mention each. Three or more and an unnamed pairing is the model's to
            # choose -- reported as girls kissing each other instead of the boys.
            _contact = (contact_hold(contact_pairs(body, _described))
                        if len(_described or []) > 2 else "")
            if _contact:
                contact_shots.append(len(plan) + 1)
            _frame = frame_hold(body, anchor, len(_described or []) or 1)
            if _frame:
                frame_shots.append(len(plan) + 1)
            # ONE sentence for the hardware. The hold, the name of the thing and
            # where it holds were three separate clauses written for three separate
            # reports, each naming the same object again -- 53 words about one pair
            # of cuffs beside a nine-word beat. Merged they cost 25 and every
            # guarantee survives.
            #
            # The applying shot keeps its own wording: it is the one shot where the
            # hardware is NOT already closed, and that is the whole point of it.
            # ONLY where somebody wearing it is in this shot. Otherwise the hold
            # describes cuffs on wrists belonging to nobody the text mentions,
            # and the model draws the person that sentence implies.
            _wearer_here = (not restrained_who
                            or not character_guard
                            or bool(restrained_who & set(_described or [])))
            if not _wearer_here:
                # Nobody in this shot is wearing it. The hold would describe cuffs
                # on wrists belonging to nobody the text mentions, and the model
                # draws the person that sentence implies -- which is the duplicate.
                # It latches, so the shot they come back in has it again.
                hold = ""
                absent_hold.append(len(plan) + 1)
            elif not _applying and restrained:
                hold = restraint_sentence(
                    worn_item if not _named_item else "",
                    # Not on the shot that STAGES the anchor: the author's own
                    # words are right there, and a second sentence saying it back
                    # is the redundancy this merge exists to remove.
                    _wearers, _described, anchor=("" if _anchor_now else anchored),
                    rigid=bool(rigid), posed=bool(posed),
                    part=held_part(worn_items or ([worn_item] if worn_item else [])))
                if worn_item and not _named_item:
                    named_shots.append(len(plan) + 1)
            else:
                hold = own_hold(hold, _wearers, _described)
            # What you wrote wins: a beat that already describes its own sound is left
            # alone, and only one that describes none gets the sound its action implies.
            # ONLY WHAT THE AUTHOR WROTE OPENS THE AUDIO BRANCH.
            #
            # H3 is joint: the mouth follows the audio. Leave that branch free on a
            # shot with no line and it fills itself with a voice, and the face
            # lip-syncs to the babble. Text cannot stop it -- "the only sounds are
            # footsteps" was tried and the mouth still moved -- because the only thing
            # that actually settles the branch is CONDITIONING it, and the silent
            # keyframe pins the whole shot, not just its opening.
            #
            # So nothing this node infers may unsilence a shot. A quoted line is a
            # request for audio; a sound the AUTHOR described is a request for audio;
            # footsteps this file worked out from "walks in" is not, and neither is
            # room tone. That is the whole rule, and it is the only one that holds --
            # every version that let an inference open the branch babbled.
            _speaks = has_speech(body)
            _own = sound_described(body)
            # A breath before a line no longer holds the branch open. Recorded so
            # the trade is visible: a breath that will not be heard is a change to
            # what was written, and finding that out from the render is worse than
            # reading it here.
            if not _own and not _speaks and _BREATH_PREP.search(body):
                _breath_shots.append(len(plan) + 1)
            # A beat staging EFFORT or vocal reaction is asking for a voice, and that
            # is read from the author's own verbs -- "thrashes", "writhes", "moans" --
            # so it belongs with a quoted line and a written sound, not with the things
            # this file infers. Silencing it says the person makes no sound, and a
            # person making no sound is rendered still: it is the flat, unreacting
            # face, and it is why a body under effort came out mute.
            # A VOCAL THE BEAT NAMES IS ASKING FOR AUDIO, as much as an effort verb is.
            # This was exertion_in alone, and four of the six vocals passed only by
            # ACCIDENT -- whimper, sob, moan and scream happen to sit in the effort table
            # too. groan and whine do not, so "She groans." was sound_described with
            # nothing to keep it open: _mute_written fired, _will_silence fired, and the
            # shot was pinned to silence. The groan the author wrote never happened, and
            # the clause naming it was never emitted either. Reading the vocal directly
            # makes all six behave the way the four already did.
            _voiced = bool(exertion_in(body) or named_vocals_in(body))
            # A shot where nobody speaks but the author wrote a SOUND kept its branch
            # open, and an open branch invents a voice the face lip-syncs to. That is
            # the hole: "a low hum off the strip light" is nobody talking, and it was
            # enough to leave the mouth free for the whole shot.
            #
            # Effort is different and stays out of this. Straining, thrashing, a body
            # under load -- those are vocal, the mouth SHOULD be open, and silencing
            # them was a bug once already: a person making no sound renders as a flat,
            # unreacting face.
            # THE BED. A shot with no line and no written sound was pinned to real
            # silence -- not "no speech" but no footsteps, no room tone, nothing,
            # which is what makes a scene sound staged. Read from the anchor and
            # the scene, the ambience no longer has to be typed into every beat.
            #
            # This DOES open the audio branch, which derived sound was never
            # allowed to do before. The rule it replaces was written when nothing
            # held the mouth on such a shot; the mouths-shut guard now lands on
            # exactly these shots, so the picture half is covered. It is still a
            # trade -- an open branch can put a voice in the gap -- and it is off
            # with auto_sound.
            # EVERY shot, including ones silence would otherwise close. Chosen
            # deliberately on 2026-09-06, with the trade stated: this is the
            # mechanism that babbled before and was reported twice, and nothing
            # this file infers was allowed to open a branch because of it.
            #
            # What has changed since is the picture half -- the mouths-shut guard
            # now lands on exactly these shots, and the language clause keeps a
            # spoken shot in one language. Neither can outvote an audio stream
            # that has decided somebody is talking, so if babble comes back on
            # wordless shots, THIS is the first thing to turn off: auto_sound.
            _bed = _bed_now if auto_sound and _bed_now else ""
            if _bed:
                ambient_shots.append(len(plan) + 1)
            _mute_written = bool(mouths_shut_when_no_line and _own and not _speaks
                                 and not _voiced)
            # The bed no longer defeats this. It is the one thing this file infers
            # that was allowed to open a branch, and opening a branch is what puts a
            # voice in a wordless shot.
            _will_silence = bool(silence_nonspeech and not _speaks and not _voiced
                                 and (not _own or _mute_written))
            if _mute_written and _will_silence:
                muted_sound.append(len(plan) + 1)
            # The picture side -- and ONLY where the shot actually describes somebody.
            # A mouth sentence on a scenery beat describes a person who is not there,
            # and the one way to satisfy it is to draw a face in an empty frame. That
            # is ca75672's bug and it must not come back.
            # Read from the BEAT, not from the carried cast. The guard keeps the
            # previous shot's people in the text so a wordless beat does not empty the
            # frame, and it falls back to the sole sheet entry when there is no
            # previous -- so "Rain on the corrugated roof", before anybody has walked
            # in, still has a person described beside it. Taking that as "somebody is
            # here" puts a mouth sentence on an empty yard, which is the whole of
            # ca75672. If the beat itself does not put a person in the shot, say
            # nothing about mouths and let the audio half do the work.
            _has_people = beat_puts_somebody_on_screen(body, sheet)
            # A line that belongs to a MACHINE is not this shot's people speaking.
            # Reported as somebody mouthing what was on the television: the quote
            # made it a speaking shot, which opened the branch and turned the mouth
            # guard off, so the only face in frame was handed the line. The branch
            # still opens -- the set is meant to be heard -- but the mouths close and
            # the voice is given back to the thing it came out of.
            _device_line = (mouths_shut_when_no_line
                            and speech_is_a_devices(body, sheet))
            # The beat's own mouth. Read here, used ONLY on the picture guard
            # below -- never on the audio decision, which is what keeps a smile
            # silent. See mouth_performs.
            # THE FACE. Built here because _wearers and _described are what say who
            # is under duress and who else is in the frame; used only on the picture
            # side, like the mouth guard beside it. See duress_face.
            _duress = (duress_face(
                body,
                [(n, ln) for n, ln in sheet_lines(shot_sheet) if n in set(_wearers)],
                _described, _film_duress) if hold_gaze else "")
            if _duress:
                duress_shots.append(len(plan) + 1)
            # A STATED EMOTION PUTS THE MOUTH TO WORK. Delight is a smile, terror is
            # an open mouth; holding it closed holds the performance. See _EMOTION.
            _mouth_busy = bool(mouth_performs(body) or emotion_in(body))
            _mouth = MOUTH_HOLD if (mouths_shut_when_no_line and _has_people
                                    and (not _speaks or _device_line)
                                    and not _voiced and not _mouth_busy) else ""
            # One of two people speaking still leaves the OTHER one's mouth free. The
            # shot is a speaking shot, so the guard stood down for everybody in it --
            # and the listener is exactly who the invented lip-sync lands on. Name the
            # speaker and close the rest, which needs the speaker to be identifiable:
            # an unattributed line could belong to either of them.
            # WHICH hold this shot got. Both end up in _mouth, and reporting them
            # together said a shot with a line had "no scripted line" -- the reader
            # then cannot tell a silenced shot from one where the speaker is named,
            # which are opposite situations.
            _mouth_from_silence = bool(_mouth)
            # WHOSE VOICE IS WHOSE. A vocal used to switch this whole block off --
            # _voiced is a shot-level flag and both guards stood down on it, for
            # everybody -- so her sob opened the branch and freed his mouth with it.
            # The vocal gets an owner instead, and only the mouths that own neither a
            # line nor a sound are closed. See voice_sources.
            _vocal_src = vocal_sources_in(body, shot_sheet) if _voiced else []
            _voicers = [n for n, _ in _vocal_src]
            _vocal_word = _vocal_src[0][1] if _vocal_src else ""
            # A vocal this file cannot pin on anybody leaves every mouth alone, the
            # way an unattributed line does: closing mouths on a guess could close
            # the mouth of whoever is making the noise, and muting a real sound is
            # worse than a mouth moving.
            if (not _mouth and mouths_shut_when_no_line and (_speaks or _voicers)
                    and not _mouth_busy and not _device_line
                    and not (_voiced and not _voicers)):
                _talkers = speakers_in(body, shot_sheet) if _speaks else []
                _open = set(_talkers) | set(_voicers)
                # WHOSE MOUTH THERE IS TO HOLD. Not only the people this beat
                # names: a beat naming just the speaker does not empty the room,
                # and the person it leaves out is standing in the picture this
                # shot starts from. That is the mouth an invented voice lands on,
                # and "Dan says: ..." on its own is the commonest beat there is.
                # One shot of memory, and not for anybody the beat walks off.
                _here_too = [n for n in _was
                             if n not in set(_described or [])
                             and n not in set(subjects_for(body, sheet,
                                                           _MOVES_OFF_SRC))]
                _silent = [n for n in list(_described or []) + _here_too
                           if n not in _open]
                _mouth = voice_sources(_talkers, _vocal_word, _voicers, _silent)
                if _mouth and _voicers:
                    vocal_shots.append(len(plan) + 1)
                if (not _mouth and _speaks and not _talkers and not _voicers
                        and len(_described or []) > 1):
                    # A line with no name on it, and more than one person who could
                    # be saying it. Whose mouth to hold is unknowable, but how many
                    # voices there are is not -- and leaving it unsaid is what let
                    # the listener talk too.
                    _mouth = ONE_VOICE
                    unattributed.append(len(plan) + 1)
            if _mouth:
                (mouth_shut if _mouth_from_silence
                 else mouth_named).append(len(plan) + 1)
            elif _mouth_busy and mouths_shut_when_no_line and _has_people:
                mouth_acting.append(len(plan) + 1)
            # A shot with a line is told what language it is in. Every shot with a
            # line, not only the ones with a listener to hold: a single speaker can
            # deliver the line in whatever language the model picks.
            # ...in the language THIS shot's line is written in. Read from the
            # line itself, falling back to the language the script as a whole is
            # in, so one short line ("Si.") in a Spanish script is not called
            # English on a technicality.
            # ...and THIS beat's own stage direction outranks the script-wide
            # fallback, so one German line inside an English script is not told it
            # is English -- which is what a fallback alone does to it.
            _shot_lang = engine.language_of(engine.spoken_text(body),
                                            fallback=_script_lang,
                                            named=engine.language_named(body))
            _lang = (LANGUAGE_HOLD.format(lang=_shot_lang)
                     if (_speaks and not _voiced) else "")
            if _lang and _shot_lang not in _langs_used:
                _langs_used.append(_shot_lang)
            # A quoted ORDER is still in the shot's words, and a model renders what
            # the words describe. Give the listener something to be doing, so the
            # instruction is not the only thing in the frame about their body.
            _told = told_hold(told_to_act(
                body, speakers_in(body, _who_sheet),
                _described if character_guard else
                [n for n, _ in sheet_lines(_who_sheet) if n])) if _speaks else ""
            if _told:
                told_shots.append(len(plan) + 1)
            if _lang:
                language_shots.append(len(plan) + 1)
            # How much of this shot the line actually fills. A short line in a long
            # shot leaves the audio branch with time and nothing to put in it, and
            # what it puts there is more speech -- the line again. Counted here
            # where the beat is; judged against the shot length further down.
            _said_words = len(engine.spoken_text(body).split())
            if _said_words:
                _spoken_words[len(plan) + 1] = _said_words
            _device = device_voice_clause(body) if (_device_line and _has_people) else ""
            if _device:
                device_shots.append(len(plan) + 1)
            # The held scenery goes in, so the shot is not asked to keep the doors
            # shut and to sound like a door swinging in the same breath.
            heard = ([] if (not auto_sound or _own)
                     else sounds_for(body, held=[_state_key(t) for t, _ in _pairs]))
            # ...AND THE AUTHOR'S OWN VOCAL GOES BACK IN, because the sentence below
            # is EXCLUSIVE. The zeroing above is right in intent -- nothing this node
            # infers may claim to be the sound of a shot the author already scored --
            # but it drops the author's word along with the inferences, and what is
            # appended next is the ambient bed. On a shot kept open by _voiced the
            # result was an exclusive claim naming only the bed:
            #
            #   "She screams."          -> "The only sound is an engine idling."
            #   "She sobs quietly."     -> "The only sound is an engine idling."
            #   "...starts whimpering"  -> "The only sound is an engine idling."
            #
            # Reproduced on all three. That is the node telling the model the scream
            # is not happening, on precisely the shots whose branch is open and which
            # therefore must fill themselves with something.
            #
            # Only the six vocals, matched literally in the beat -- the author's own
            # words, not an inference -- so "nothing inferred may unsilence a shot"
            # still holds. A beat whose written sound is NOT a vocal is muted
            # outright by _mute_written and reaches no clause at all, which is a
            # different path and reports itself.
            if _own:
                heard = [v for v in named_vocals_in(body) if v not in heard] + heard
                # AND WHAT IS HAPPENING BETWEEN THEM. A vocal is intermittent and the
                # branch is open for the whole shot, so a list naming nothing but vocals
                # describes the peaks and leaves the troughs blank -- and a blank trough on
                # a joint model, next to a face, fills itself with speech. Reported as
                # babble between the moans.
                #
                # Room tone does not answer it, even though the bed appends two continuous
                # phrases below: the gap is a PERSON's audio presence, and a soft room with
                # little echo is not a person. Breath is -- non-verbal, continuous where
                # the vocal is not, and true of anybody making any of these six sounds.
                #
                # Only when the list is ALL vocal. A beat whose sound is already part
                # non-vocal has something in the troughs, and the sound budget exists to
                # stop inventories.
                if heard and all(v in _NAMED_VOCALS for v in heard):
                    heard = heard + [_VOCAL_BETWEEN]
            if _will_silence:
                # The audio is pinned to silence for this shot's whole length, so a
                # sentence saying what it sounds like would describe an acoustic the
                # conditioning says is not there.
                heard = []
            elif _bed:
                heard = heard + [_bed] + ([_room_now] if _room_now else [])
            elif auto_sound and _room_now:
                heard = heard + [_room_now]
            if heard:
                inferred_sound.append(len(plan) + 1)
            # The branch is free on this shot, so SOMETHING fills it. Naming the sound
            # as the only thing heard leaves nothing for a voice to be -- it is not
            # the guard, the silence is, but it is what shapes a branch that is
            # legitimately open. Positively phrased: "the only sound is X" says what
            # IS there, where "nobody speaks" asks the model to render an absence.
            _sound = sound_clause(heard, only=not _speaks)
            # Gaze is resolved after the other guards so a character is never named
            # twice. Person targets need no looker's name; object targets do when
            # several people are present.
            if hold_gaze and _gazers:
                _g = _gazers[0]
                _target, _is_person = looking_at[_g]
                _elsewhere = " ".join([
                    hold, _posture, _pose, _travel, _where, _told, turn, _duress,
                    _mouth, _revealed, _under, _bare, _wearing, tail, _moved,
                    anchors, _state_clause, _device, _sound, _pace, fall])

                def _named_already(_n):
                    return bool(re.search(r"\b" + re.escape(_n) + r"\b", _elsewhere))

                def _their_pronoun(_n):
                    """'her'/'his'/'their', if nobody else in the shot shares it."""
                    _rows = {a: b for a, b in sheet_lines(shot_sheet) if a}
                    _m = re.search(r"\b(she|he|they)\b", _rows.get(_n, ""), re.I)
                    if not _m:
                        return ""
                    _sex = _m.group(1).lower()
                    for _o in (_described or []):
                        if _o != _n and re.search(r"\b" + _sex + r"\b",
                                                  _rows.get(_o, ""), re.I):
                            return ""
                    return {"she": "her", "he": "his", "they": "their"}[_sex]

                if _is_person:
                    # Nobody turns their eyes to themselves, so an impersonal
                    # sentence naming the TARGET can only be the other person's
                    # eyes. It spends no naming on the looker at all.
                    if not _named_already(_target):
                        _gaze = gaze_hold(_target, "", True)
                elif len(_described or []) >= 2 or _g not in set(_described or []):
                    _who = "" if _named_already(_g) else f"{_g}'s"
                    if not _who:
                        _who = _their_pronoun(_g)
                    if _who:
                        _gaze = gaze_hold(_target, _who)
                else:
                    _gaze = gaze_hold(_target)
            if _gaze:
                gaze_shots.append(len(plan) + 1)
            # A LINE SAID, NO LOOK STAGED. gaze_hold restates what the beat named;
            # a dialogue beat that names no look leaves both faces to the portrait
            # prior, which is the lens. Reported as two people talking to the
            # camera instead of each other. The addressee is in the shot, so the
            # faces are turned to each other -- impersonally, both names here
            # being already spent. Not for a voice from a device: somebody on the
            # phone is not facing the room. A look the beat stages, even a pronoun
            # one gaze_hold declines to restate, is never argued with.
            if (hold_gaze and not _gaze and _speaks and not _look_now
                    and not _device_line and len(_described or []) >= 2):
                _faces = dialogue_gaze(len(_described))
                if _faces:
                    dialogue_gaze_shots.append(len(plan) + 1)
            _guards = [
                (1, "removal", tail),        # the beat's own action, completing
                (1, "wearing", _wearing),    # ...and its mirror, a garment going on
                (2, "revealed", _revealed),  # what shows where it was
                (2, "under", _under),         # ...and what is underneath, still on
                (2, "bare", _bare),          # ...or that nothing does
                (3, "hold", hold),           # hardware coming open is not a drift
                (4, "fall", fall),           # a body going down needs a landing
                (4, "travel", _travel),      # a journey needs both its ends
                (4, "where", _where),        # ...and later shots need the new room
                (5, "pace", _pace),          # ...and a short action needs the whole shot
                (5, "device", _device),      # a voice that is not hers
                (6, "moved", _moved),        # a garment left where it was put
                (7, "anchors", anchors),     # hardware with nowhere to sit
                (10, "state", _state_clause),
                (9, "posture", _posture),   # where the last beat left the body
                # ...and the pose the hardware holds them in, as a BODY. Ranked
                # beside posture because that is what it is: an arm position, not
                # a fact about metal. See pose_clause.
                (3, "pose", _pose),
                (11, "gaze", _gaze),
                # Beside the gaze, because they answer the same pull: with nothing
                # said about the eyes or the face, both come from the portrait prior.
                #
                # RANKED BELOW IT, though, and measured. At 11 it tied the gaze and
                # won on list position, and on a short beat the budget then dropped
                # the gaze clause from the very shot that staged the look while a
                # stale copy survived on the shot after. Where the two compete, the
                # spatial fact the beat itself stated goes first.
                (12, "duress", _duress),
                (12, "mouth", _mouth),
                (12, "language", _lang),   # ...and in which language
                # The eye-line INFERRED for a dialogue shot. Reads after the mouth
                # guard it belongs with. Rank 15, below even sound: it is a guess
                # about where the eyes go, and at rank 11 -- the staged look's rank
                # -- it took the budget from "Only Dan speaks" on a seven-word beat.
                # An inference is cut before anything the author's own words imply.
                # WHO IS WITH WHOM, ranked with the holds rather than the inferences:
                # it restates a pairing the author WROTE, the way the gaze clause
                # restates a look they wrote, and a wrong pairing is a gross error
                # rather than a missing nicety.
                (3, "contact", _contact),
                (15, "faces", _faces),
                # The frame, where nothing else says what it is. Ranked with the
                # other inferred picture guards and below everything the author's
                # own words imply: it is a guess about the camera, and the camera is
                # the author's to state. See frame_hold.
                (15, "frame", _frame),
                (6, "told", _told),          # a listener given an order to ignore
                (13, "turn", turn),
                # LAST in the list, and last in the ranking of anything the author's
                # words imply -- only the inferred eye-line (15) is cut before it.
                # Both on purpose.
                #
                # This was appended after fit_guards and so was the one piece of
                # node-written text no budget could reach -- unranked, uncuttable,
                # and measured as the largest single contributor: 97 words of 420
                # across a six-shot script, double the budgeted guard on shot 1.
                # Counting it APART for the balance report is right, because it asks
                # for something to HAPPEN rather than to stay as it is. Exempting it
                # from the CAP was a different thing, and not intended.
                #
                # Rank 14 was measured, not assumed. Ranked high it wins its words
                # from the continuity holds, and the suites caught exactly that: at
                # the SAME budget, ranking it 2 cost the fall/landing guard. So it
                # goes last -- above nothing, cut before anything that traces to a
                # report. At the shipped floor the budget never binds, so nothing
                # about a current render changes; what changes is that sound can no
                # longer grow the pile without the cap noticing.
                #
                # Last in the LIST keeps it at the end of the sentence where it
                # already sat: fit_guards orders its output by list position, not by
                # priority. Being last to survive is not the same as being last to
                # read, and only the ranking was in question.
                (14, "sound", _sound),
            ]
            _kept, _dropped = fit_guards(_guards, len(body.split()))
            if _dropped:
                crowded.append((len(plan) + 1, _dropped))
            # Body count is a composition invariant, not a continuity detail. It
            # must not evict speaker, gaze, or ownership clauses from the bounded
            # guard budget; doing so fixed the extra body by breaking who spoke.
            shot_text = (line + _cast_hold + _kept).strip()
            # Sound direction is not a continuity guard -- it asks for something to
            # HAPPEN rather than for something to stay as it is -- so it is counted
            # apart, or the balance report blames the wrong text for crowding the beat.
            _sound_kept = "" if "sound" in _dropped else _sound
            sound_words += len(_sound_kept.split())
            guard_words += (len(shot_text.split()) - len(_sound_kept.split())
                            - len(f"{_scene_sent} {body}".split()))
            beat_words += len(body.split())
            total_words += len(shot_text.split())
            # The event sounds this beat implies, kept per shot so they can be
            # BUILT and mixed into that shot's span later. `heard` is not it:
            # that one has the bed and the room tone folded in and is emptied
            # on a silenced shot, which is precisely the shot this is for.
            _events = (list(sounds_for(body, held=[_state_key(t)
                                                   for t, _ in _pairs]))
                       if auto_sound else [])
            # What the AUTHOR wrote, and nothing this file worked out. See above --
            # effort counts, because the verb staging it is theirs.
            #
            # `_bed` USED TO BE IN HERE, against the comment above it. The ambient
            # bed is inferred, and putting it in this list left the audio branch
            # open on every shot that got one -- which is every wordless shot, which
            # is what the bed was for. An open branch on a joint model fills itself,
            # and at 4-8 steps the final audio step clears 50%-30% of the denoising
            # in one jump, so what it fills with is a voice. Ambience everywhere and
            # silence are mutually exclusive by construction: the silence latent IS
            # the audio, and there is no room in it for a room tone.
            plan.add(shot_text,
                     list(active) if character_guard else [],
                     _speaks, _own or _voiced, _voiced and not _own, _events)

        # What share of a shot is the node talking rather than the script. Continuity
        # clauses all say some version of "this stays as it is", and enough of them
        # drown the one sentence describing what HAPPENS -- which renders as a shot
        # where nothing does. The previous node reached 96%; this is here so the creep
        # is visible before it gets there again.
        # Somebody back after a shot away, with nothing pictorial carrying them.
        if _returns:
            _lines = "; ".join(f"shot {n}: {', '.join(w)}" for n, w in _returns)
            _tagged_back = {w for _, ws in _returns for w in ws
                            if re.search(r"^\s*" + re.escape(w) + r"\s*:.*<\s*picture",
                                         sheet or "", re.I | re.M)}
            _bare = sorted({w for _, ws in _returns for w in ws} - _tagged_back)
            notes.append(
                f"back after a shot away -- {_lines}. Each shot starts from the PREVIOUS "
                f"shot's last frame, so somebody who was not in that shot is not in the "
                f"picture this one begins from: their appearance comes from the sheet "
                f"text and nothing else, and text drifts where a picture does not. That "
                f"is a character walking out of frame and coming back looking different"
                + (f". {', '.join(_bare)} " + ("has" if len(_bare) == 1 else "have")
                   + " no <Picture N> tag, so there is no picture of them anywhere in the "
                     "run -- tag a reference to them and it is carried into every shot "
                     "they are named in, this one included"
                   if _bare else
                   ". All of them carry a reference tag, which is what pins them here"))
        if total_words:
            notes.append(
                f"prompt balance: the beat is {100 * beat_words / total_words:.0f}% of "
                f"what each shot is told, continuity clauses "
                f"{100 * guard_words / total_words:.0f}%, sound "
                f"{100 * sound_words / total_words:.0f}%, scene and sheet the rest"
                + (" -- the guards are outweighing the action, which reads as a shot "
                   "where nothing happens. Fewer restraints named, or a beat with more "
                   "in it, shifts the balance back"
                   if guard_words > beat_words * 3 else ""))
        refs_all = [r for r in (ref_image_1, ref_image_2, ref_image_3, ref_image_4)
                    if r is not None]
        # A reference nothing tags rides EVERY shot -- including the ones where a
        # garment it may depict is covered. Layering can hide the words; it cannot
        # hide a picture, and the picture wins. Reported as a chastity belt drawn on
        # top of the jeans while the text had correctly stopped mentioning it.
        #
        # The node cannot know what an untagged image shows, so it cannot withhold it
        # on its own. Tagging is what puts it under the layering's control, and that
        # is the one thing that fixes this.
        if refs_all and covers and not _PICTURE_TAG.search(f"{scene}\n" + "\n".join(beats)):
            notes.append(
                f"{len(refs_all)} reference image(s) and not one <Picture N> tag anywhere, "
                f"while the wardrobe has layers in it ("
                + "; ".join(f"{u} under {o}" for u, o in list(covers.items())[:3])
                + "). An untagged reference goes into EVERY shot, so a picture of "
                  "something that is currently underneath something else is still sent "
                  "on the shots where it is covered -- and the text having stopped "
                  "describing it does not stop the model drawing it. That is an under "
                  "layer rendered on top. Tag the image onto the thing it shows -- "
                  "'a chastity belt <Picture 2>' -- and it is sent only where that "
                  "thing is actually visible")

        lens, len_note = plan_lengths(beats, ceiling, shot_length == "from the beat", pace)
        plan.set_frame_counts(lens)
        plan.validate()
        # How much of a SPEAKING shot the line does not cover. The branch is free for
        # the whole shot, so whatever the line does not fill is unconditioned audio in
        # a shot the model knows somebody is talking in -- which is where invented
        # speech after the line comes from. Reported per shot, because the fix is the
        # author's: a longer line, or a shorter shot.
        _tail = []
        _tailpin = []               # (shot, seconds) pinned past the line's end
        for _i, _b in enumerate(beats):
            if _i >= len(lens) or not has_speech(_b):
                continue
            _words = (sum(len(q.split()) for q in _QUOTED.findall(_b))
                      + sum(len(q.split()) for q in _DIALOGUE_TAG.findall(_b)))
            _say = _words / WORDS_PER_SEC
            plan.shots[_i].line_seconds = _say
            _tf = ShotAudio(True, True, False, bool(silence_nonspeech), speech_lead_seconds,
                            AUDIO_LATENT_FPS, _say, speech_tail_seconds, lens[_i]).tail_frames
            if _tf:
                _tailpin.append((_i + 1, _tf / AUDIO_LATENT_FPS))
            _shot = lens[_i] / H3_FPS
            if _shot - _say >= 3.0:
                _tail.append((_i + 1, _words, _say, _shot))
        if _tail:
            notes.append(
                "dialogue headroom -- "
                + "; ".join(f"shot {n}: {w} word(s), about {s:.1f}s of a {t:.1f}s shot"
                            for n, w, s, t in _tail)
                + ". The audio branch is open for the whole shot, so the seconds the "
                  "line does not fill are unconditioned in a shot the model already "
                  "knows has a voice in it -- that is where speech carries on after the "
                  "line, or turns into babble. Give the beat a longer line, or a "
                  "shorter shot: shot_length 'from the beat' sizes to the line, while "
                  "'fixed' gives every shot shot_seconds whatever the line needs")
        # Seconds of shot per staged action -- the number that decides whether the
        # motion looks brisk or stretched. A shot longer than its action is filled by
        # performing the action more slowly, not by inventing more of it.
        _clauses = sum(max(1, len([p for p in _CLAUSE_SPLIT.split(b)
                                   if p and len(p.split()) >= 2])) for b in beats)
        if _clauses and lens:
            _per = sum(lens) / H3_FPS / _clauses
            notes.append(
                f"pacing: {_per:.1f}s of shot per staged action across {len(beats)} "
                f"beat(s), at pace {float(pace):.2f}"
                + (" -- a staged action is usually 2 to 3 seconds on screen, and a shot "
                   "longer than its action is filled by performing it more slowly. Lower "
                   "pace for brisker movement" if _per > 3.5 else ""))
        if len(set(lens)) == 1:
            notes.append(f"{len(plan)} shot(s) x {lens[0]}f (~{lens[0] / H3_FPS:.1f}s) "
                         f"at {w}x{h} = ~{sum(lens) / H3_FPS:.1f}s total")
        else:
            notes.append(f"{len(plan)} shot(s) at {w}x{h}, sized per beat: "
                         + ", ".join(f"{n}f/{n / H3_FPS:.1f}s" for n in lens)
                         + f" = ~{sum(lens) / H3_FPS:.1f}s total")
        if len_note:
            notes.append(len_note)
        if stated_shots:
            notes.append(
                f"shot(s) {', '.join(str(n) for n in stated_shots)} describe scenery in a "
                f"state -- doors closed, curtains drawn -- so the shot is told that state "
                f"is already true at the first frame. A state written down and not placed "
                f"in time is a state the model can render by arriving at it, which is a "
                f"van whose doors open so somebody can close them. A beat that works the "
                f"thing itself is left alone, and once a beat has changed a state no "
                f"later shot is told the old one. Off with hold_scene_state.")
        if paced_shots:
            notes.append(
                f"shot(s) {', '.join(str(n) for n in paced_shots)} stage less than "
                f"their length, so each is told its action runs across the whole "
                f"shot. A shot told WHAT happens and nothing about WHEN performs it "
                f"at once, and the cheapest way to fill the seconds left is to carry "
                f"on -- the same movement repeated on whatever is nearest. It names "
                f"when, never how fast: 'slowly' is a style instruction and this is "
                f"not one. Give the beat more to do, or shorten the shot, and it "
                f"stops being needed")
        if scene_held:
            _rooms = sorted({r for _, rs, _ in scene_held for r in rs})
            _waited = sorted({t for _, _, ts in scene_held for t in ts})
            notes.append(
                f"the scene paragraph describes {', '.join(_rooms)}, and a paragraph that "
                f"describes a room describes its FURNITURE too -- so that description WAITS "
                f"OUTSIDE it, on shot(s) {', '.join(str(n) for n, _, _ in scene_held)}. The "
                f"paragraph is stamped into every shot, which is what gives a removal "
                f"something to scrub, and the room's NAME was already right in every shot -- "
                f"but the bed was in the text standing beside it, and at cfg 1 there is no "
                f"negative prompt that can take a named thing back. Reported as a bed in the "
                f"living room two beats after she left the bedroom. The room a shot ENDS in "
                f"decides this, not every room it passes through: a walk out of the bedroom "
                f"does show it in the opening frames, but that shot's LAST frame is the next "
                f"shot's keyframe, so a bed drawn at the end of the walk is inherited by the "
                f"shot after it -- and the room being left arrives as a PICTURE anyway, "
                f"because the keyframe IS the previous shot's last frame. So the words say "
                f"where the shot ends and the frame carries where it began. Your paragraph is "
                f"not edited: this is per shot, and a beat that walks back in gets it back in "
                f"full. A character sheet line is never touched, whatever it names, and "
                f"neither is the anchor or a room your beat mentions. What waited: "
                + "; ".join(f'"{t}"' for t in _waited[:3])
                + ". If one of those also carried the film's own hour or light, split it: "
                  "one sentence for the film, one for the room")
        if scene_welded:
            notes.append(
                f"shot(s) {', '.join(str(n) for n, _ in scene_welded)} are not in the room the "
                f"scene paragraph describes, and that description was KEPT anyway, because "
                f"holding it would have left those shots no scene sentence at all. The "
                f"paragraph welds the film's own framing to one room's furniture in a single "
                f"sentence, so taking the room would take the lighting and the hour with it, "
                f"and a shot with no scene is a bigger change than a bed in the wrong room. "
                f"Split it in two -- one sentence for the film ('A small flat at night.') and "
                f"one for the room ('Her bedroom has an unmade bed and a lamp.') -- and the "
                f"room's half will wait outside that room on its own")
        if cut_shots:
            notes.append(
                f"shot(s) {', '.join(str(n + 1) for n in sorted(cut_shots))} START FRESH, "
                f"because they OPEN IN A DIFFERENT ROOM from the one the shot before ended "
                f"in. Every shot is anchored to the previous shot's last frame, and a "
                f"keyframe is a PICTURE, which outvotes any sentence -- so a living-room shot "
                f"opening on a frame of the kitchen renders neither of them, it renders a "
                f"blend, and a kitchen blended with the words 'living room' is a bathroom: "
                f"tiles, a sink, cabinets. Breaking the chain costs a cut exactly where a cut "
                f"belongs, which is the same trade restart_after_removal makes. A WALK IS NOT "
                f"THIS: a travel beat opens in the room it is leaving, so that frame is the "
                f"right one and the shot keeps its keyframe -- write the move as a journey "
                f"('she walks through to the kitchen') and you get the walk instead of a cut")
        if led_shots:
            notes.append(
                f"shot(s) {', '.join(str(n) for n in led_shots)} put the BEAT in front of "
                f"the character sheet. The sheet has to be in every shot -- clothing "
                f"continuity is read out of it -- but it is a description of a FACE, and it "
                f"was leading every prompt ahead of the action. Measured: 69% of a shot's "
                f"words sat in sentences about a face, and turning every face guard off only "
                f"reached 63%, because the sheet is most of it. What LEADS a prompt decides "
                f"its composition -- anatomy in the opening tokens is what a distilled model "
                f"settles the frame on, and at cfg 1 no later sentence outvotes it. Your "
                f"words are identical and none are rewritten; only the order changed, which "
                f"is the one thing about this that had never been tried. Off with beat_leads "
                f"to compare the two in one render")
        if contact_shots:
            notes.append(
                f"shot(s) {', '.join(str(n) for n in contact_shots)} have three or more "
                f"people and a beat that puts two of them in contact, so the shot is told "
                f"WHICH body is with which. Your beat already says it, and it was the only "
                f"thing that did: one sentence among everyone's appearance, and at cfg 1 "
                f"the model reads the prompt as a bag of words and pairs by its own prior. "
                f"Reported as girls kissing each other when they should have been kissing "
                f"the boys. Both sides are named, because an unnamed pairing in a shot with "
                f"four people is the sentence that let it choose. Read from your own words "
                f"only -- a beat that pairs nobody by name ('they kiss') gets nothing, "
                f"because guessing which two is the bug. With two people in the shot "
                f"nothing is said: there is nobody else to pair with")
        if frame_shots:
            notes.append(
                f"shot(s) {', '.join(str(n) for n in frame_shots)} stage something a "
                f"portrait cannot contain and say nothing about the camera, so they are "
                f"told what the frame HOLDS: the whole body, head to feet, with the room "
                f"around it. An attribute a prompt does not state is not left to the model, "
                f"it is left to the model's PRIOR -- and the prior for a named, described "
                f"person is a portrait facing the lens. The sheet describes a face in every "
                f"shot because clothing continuity needs it there, and the mouth guard "
                f"describes a mouth in every silent shot because babble needs it, so the "
                f"text leans towards a face and nothing in it said how much of the person to "
                f"show. Reported as the camera fixated on one character staring into the "
                f"lens, with no reference image in the run at all. Your camera always wins: "
                f"write any framing in the beat or the anchor -- a close-up included, since "
                f"a close-up is a frame somebody asked for -- and this stands down. It is "
                f"ranked below everything your own words imply, so a crowded shot drops it "
                f"first")
        if open_moves:
            notes.append(
                "shot(s) " + ", ".join(f"{n} (to the {w})" for n, w in open_moves[:6])
                + " move somewhere the place list cannot name, so the arrival is told to be "
                  "PERFORMED -- the whole move on screen, first step to last. A closed list "
                  "of room words can never cover a script nobody has written yet, and a move "
                  "nobody is told to make is a move the model CUTS to: reported as the set "
                  "changing under the characters instead of them walking into it. This reads "
                  "the destination from your own words and claims nothing else about it -- no "
                  "room state, no acoustic, no cut decision -- so a dungeon, a cargo bay or a "
                  "stable all work without being listed anywhere. Anything that is furniture, "
                  "a body part, a vehicle or a person is left alone")
        if untracked_strip:
            _items = sorted({t for _n, ts in untracked_strip for t in ts})
            notes.append(
                f"shot(s) {', '.join(str(n) for n, _ in untracked_strip)} take "
                f"{', '.join(_items)} off PEOPLE THE SHEET DOES NOT NAME, and the "
                f"removal reaches only the ones it does name. A character sheet entry is "
                f"what a removal scrubs and what carries the bare region into every later "
                f"shot; an unnamed woman has neither, so her own words come off in the "
                f"beat that says so and nothing holds them off afterwards -- the next shot "
                f"says nothing about her, and what it opens on is a keyframe taken while "
                f"she was still half in them. Reported as some of the skirts still being "
                f"on when all of them should have come off. Give each of them an entry, "
                f"however short -- 'Girl 1: she, 20, a denim skirt.' -- and their removals "
                f"hold exactly like the named character's. Your words are never rewritten "
                f"either way; this is about what the node can keep saying after the beat "
                f"that said it")
        if _undescribed:
            _them = "them" if len(_undescribed) > 1 else "it"
            notes.append(
                f"the film enters {', '.join(_undescribed)}, and your prompt never "
                f"describes {_them} -- a room the text only NAMES is a room the model "
                f"invents, and what it invents from is the frame the shot opened on plus "
                f"whatever the other rooms suggest. Reported as a living room turning into "
                f"a bathroom: a kitchen frame and the words 'living room' share tiles, a "
                f"sink and cabinets, and nothing in the text said otherwise. Give each "
                f"room a sentence of its own -- 'The living room has a green sofa and a low "
                f"table.' -- either in the beat that enters it or in the scene paragraph. A "
                f"scene sentence that names a room is carried ONLY in the shots that are in "
                f"that room, so it costs every other shot nothing")
        if where_shots:
            notes.append(
                f"shot(s) {', '.join(str(n) for n in where_shots)} are in a room the "
                f"scene text does not name, so each is told which one. The scene "
                f"paragraph is stamped into EVERY shot -- it has to be, or a removal "
                f"has nothing to scrub -- so a script that walks from one room to "
                f"another goes on opening every later shot with the room it started "
                f"in, while the beat has them somewhere else. The shot then holds two "
                f"places at once and settles on whichever the model weighs more, "
                f"differently each time. Your scene text is not edited: move the "
                f"location into the beats, or keep the scene general, and this stops "
                f"being needed")
        if wearing_shots:
            notes.append(
                f"shot(s) {', '.join(str(n) for n in wearing_shots)} put a garment back "
                f"ON, so each is told both ends: off the body as the shot opens, fully "
                f"on by the last frame. An 'add:' used to go straight into the scene "
                f"block as a worn item, which told a shot inheriting a last frame "
                f"WITHOUT the garment that it flatly has it -- a disagreement rather "
                f"than a change, and the model settles those in the opening frames by "
                f"turning whatever is on the body into the garment. That reads as one "
                f"thing instantly becoming another, a beat before the beat that puts it "
                f"on, which is what those opening frames are. The garment joins the "
                f"static wardrobe from the NEXT shot, the way a removal scrubs from its "
                f"own. An 'add:' that merely reveals a layer already underneath is left "
                f"alone: nothing is being put on there")
        if crowded:
            # This was collected and never reported. The budget rarely binds, so the
            # one time it did there was nothing in info saying a guard had been cut
            # -- the shot simply stopped holding something, with no way to tell that
            # from the guard having failed to fire. A dropped clause is exactly the
            # case worth reading, being the one place the node knowingly stops
            # answering a bug it knows about.
            notes.append(
                "guard clauses dropped for room -- "
                + "; ".join(f"shot {n}: {', '.join(d)}" for n, d in crowded)
                + f". Each shot's continuity text is capped at "
                  f"{GUARD_WORDS_PER_BEAT_WORD} words per word of beat, floored at "
                  f"{GUARD_FLOOR_WORDS}, and the lowest-ranked clauses give way "
                  f"first. The cap is set to catch a runaway rather than to trim "
                  f"routinely, so this firing at all means one shot is carrying far "
                  f"more continuity than its beat -- usually a one-line beat in a "
                  f"scene holding a lot of state. Giving that beat more to do buys "
                  f"back the room, and is better than raising the cap: every clause "
                  f"below the line is answering something")
        if acoustic_shots:
            notes.append(
                "the sound followed them into the new room on "
                + "; ".join(f"shot {n}: {r}" for n, r in acoustic_shots)
                + ". The ambient bed and the room tone were read once, before the "
                  "first shot, out of the scene -- so a film that walked into a "
                  "tiled bathroom went on being told it sounds like the carpeted "
                  "room it left. H3 is joint, so that is the picture told one room "
                  "and the audio told another inside the same conditioning, which is "
                  "the contradiction the room hold exists to end, arriving through "
                  "the other branch. Only where the room actually changed and only "
                  "where the new room has a sound of its own: otherwise the film's "
                  "own bed stands, because one bed across a chain is part of what "
                  "makes it one film. Off with auto_sound")
        if travel_shots:
            notes.append(
                f"shot(s) {', '.join(str(n) for n in travel_shots)} move between "
                f"places, so the shot is told where it BEGINS as well as where it "
                f"ends. A journey given only its destination is a journey the model "
                f"can satisfy by starting there -- the living room becomes the "
                f"bedroom at the first frame and the hallway between them is never "
                f"seen. Named both ends, it has to travel. The starting place is "
                f"read from the beat, or from wherever the last one left everybody")
        if posture_shots:
            notes.append(
                f"shot(s) {', '.join(str(n) for n in posture_shots)} are told to keep "
                f"the posture an earlier beat put somebody in -- seated, kneeling, "
                f"lying down. The scene-state reader tracks scenery and nothing about "
                f"the body, so a shot that ended with somebody seated was followed by "
                f"one free to stand them up: the keyframe carries the pose as a "
                f"picture, but the text is what the model reconciles it against, and "
                f"text that says nothing loses to a reference that says something. "
                f"Standing is never held -- it is the default pose, so the clause "
                f"would cost a naming of the person and buy nothing. Off with "
                f"hold_scene_state")
        if unattributed:
            notes.append(
                f"shot(s) {', '.join(str(n) for n in unattributed)} carry a line that "
                f"names no speaker, and more than one person is in them -- so which "
                f"mouth to hold is unknowable and the shot is told only that there is "
                f"ONE voice. H3 is joint, so an unheld mouth beside an open audio "
                f"branch is where a second voice comes from, and that voice is the "
                f"babble. Attribute the line -- 'Nora says: \"...\"' -- and the "
                f"listener's mouth is held shut by name instead")
        if revealed_shots:
            notes.append(
                f"shot(s) {', '.join(str(n) for n in revealed_shots)} take off a "
                f"garment that was covering another, so the shot is told what shows "
                f"there now. The removal clause is emphatic and specific -- off the "
                f"body, dropped out of frame -- while the layer underneath is one "
                f"entry in an attribute list, and against a prior that says trousers "
                f"coming off means bare skin, a list entry does not compete. Said only "
                f"on the shot that uncovers it; after that it is simply worn")
        if bared_shots:
            notes.append(
                f"shot(s) {', '.join(str(n) for n in bared_shots)} take off a garment "
                f"with nothing named underneath it, so the shot is told that region is "
                f"BARE. Left unsaid, the space a garment leaves is unspecified, and an "
                f"unspecified region is filled by the model's own prior -- for legs "
                f"that prior is legwear, so leggings or tights appear that the prompt "
                f"never asked for, and the keyframe carries them into every later shot. "
                f"It names a body part and never a garment: at cfg 1 there is no "
                f"negative prompt, so naming the unwanted thing would summon it. Name "
                f"an under-layer in the sheet and this gives way to that instead")
        if restarted:
            notes.append(
                f"shot(s) {', '.join(str(n) for n in restarted)} start FRESH rather "
                f"than from the previous shot's last frame, because the shot before "
                f"took something off -- that is restart_after_removal, and it is what "
                f"stops a garment being inherited back through the keyframe. It costs "
                f"a visible cut at each of those points. Turn it off to keep the "
                f"chain unbroken and accept the risk")
        if exposed_by_beat:
            notes.append(
                "a beat NAMES something the wardrobe says is covered: "
                + "; ".join(f"shot {n}: {', '.join(g)}" for n, g in exposed_by_beat)
                + ". Your beats are passed through word for word and are never "
                  "scrubbed, so the layering can take it out of the sheet and the "
                  "beat puts it straight back -- and a described thing is a drawn "
                  "thing, drawn over whatever is on top of it. Worse, the next shot "
                  "starts from this one's last frame, so once it is rendered on top "
                  "it is carried forward and looks permanent. Take the name out of "
                  "the beat while it is underneath, or take the outer garment off "
                  "first. Nothing here edits your wording")
        if absent_hold:
            notes.append(
                f"shot(s) {', '.join(str(n) for n in absent_hold)} describe nobody who "
                f"is wearing the hardware, so the restraint hold is left out of them. It "
                f"says cuffs are closed on wrists, and in a shot where the person wearing "
                f"them is not described those wrists belong to nobody the text mentions -- "
                f"so the model draws the person the sentence implies, which is a duplicate "
                f"nobody asked for. The hold latches, so the shot they come back in has it "
                f"again")
        if moved_shots:
            notes.append(
                f"shot(s) {', '.join(str(n) for n in moved_shots)} carry a garment "
                f"MOVED rather than taken off -- pulled down, pushed up, shoved aside. "
                f"It is still on the body, so it stays in the scene and is described "
                f"where the beat left it. Counted as a removal it would be scrubbed "
                f"instead, and every later shot would describe nothing where something "
                f"still is -- which is the garment coming back looking like a "
                f"different one. Putting it back ('pulls them back up') releases it, "
                f"and a real removal or a `remove:` empties it for good")
        if named_shots:
            notes.append(
                f"shot(s) {', '.join(str(n) for n in named_shots)} name the hardware "
                f"itself, because their own text does not. The holds say a restraint "
                f"stays whole and closed and never say WHAT it is, so a shot after the "
                f"one that applied it is told a restraint exists with no object to "
                f"draw -- and what renders is the consequence without the hardware: "
                f"held hands and a restrained posture, bare wrists. Taken from your own "
                f"wording at the shot that put it on, and released by a `remove:` "
                f"naming it")
        if early_hardware:
            notes.append(
                f"shot(s) {', '.join(str(n) for n in early_hardware)} stage hardware "
                f"going ON, but the character sheet already lists it as worn. The sheet "
                f"goes into every shot, so it DESCRIBES the hardware in the shots "
                f"BEFORE this happens, and a described item is a drawn item. What the "
                f"node will not do is assert it: no shot before this one is told the "
                f"restraint is fastened, and this one is told both ends rather than "
                f"the standing hold. Take the hardware off the sheet entry and let the "
                f"beat put it on, or drop the beat if she wears it throughout. Your "
                f"wording is never edited, so this one is yours")
        if deferred_shots:
            _items = sorted({i for _n, its in deferred_shots for i in its})
            notes.append(
                f"{', '.join(_items)} WAITS on shot(s) "
                f"{', '.join(str(n) for n, _ in deferred_shots)}, where it is under "
                f"something else -- BOTH the words and the picture. It is deferred, "
                f"never removed: your character memory is not edited, and the item "
                f"comes back in full on the shot that lifts, moves or removes what "
                f"covers it. A "
                f"reference is an instruction to REPRODUCE an image, so handing the "
                f"model a picture of a thing that is under a skirt draws it through "
                f"the skirt -- measured twice, including with the cover described as "
                f"whole and opaque. Reference strength is ref_noise_aug and it is one "
                f"number for every image, so this one cannot be weakened without "
                f"weakening the face")
        if applied_shots:
            notes.append(
                f"shot(s) {', '.join(str(n) for n in applied_shots)} put the hardware "
                f"ON, so they are told both ends -- open and off at the first frame, "
                f"closed on the body by the last -- instead of the standing hold. The "
                f"standing hold says the restraint is fastened as it was put on and "
                f"still fastened at the last frame, which read at frame 1 means it is "
                f"already closed. A first frame that already has the cuffs on leaves "
                f"the catching and the struggling to happen in whatever order is left, "
                f"which is being restrained and THEN caught. From the next shot the "
                f"standing hold is correct again, because by then it is on")
        if device_shots:
            notes.append(
                f"shot(s) {', '.join(str(n) for n in device_shots)} have a spoken "
                f"line that belongs to a machine, not to anybody in the room. H3 is "
                f"joint and the audio branch has no idea a voice came out of a set, so "
                f"a quote made the shot a speaking one and the only face in frame was "
                f"handed the line. The branch still opens -- the set is meant to be "
                f"heard -- but the mouths are held closed and the voice is given back "
                f"to the thing it came out of. A line anybody in the room might have "
                f"stays theirs: an unattributed quote is a person talking")
        if fall_shots:
            notes.append(
                f"shot(s) {', '.join(str(n) for n in fall_shots)} put a body down, so "
                f"the shot is told what takes the landing and what the legs do. A fall "
                f"is the frame where limbs are least determined -- fast motion, heavy "
                f"occlusion, and a middle the model has to invent -- and leaving it to "
                f"work out what catches the body leaves it free to add something that "
                f"can, which is where a spare limb comes from. Said as what the limbs "
                f"DO, never as how many there are: a count is also a mention, and "
                f"naming legs to ask for two is a way of asking for legs")
        if gaze_shots:
            notes.append(
                f"shot(s) {', '.join(str(n) for n in gaze_shots)} name something to "
                f"look at, so the eyes and the head are put on it in so many words. "
                f"The beat says it once and two things pull the other way: a person in "
                f"frame faces the camera unless something says otherwise, and a "
                f"near-clean reference asks for the portrait's pose -- which looks at "
                f"the lens, because photographs of people do. Nothing is said about "
                f"where the camera is. It is HELD until something moves it, and a "
                f"look belongs to whoever is doing the looking -- so it is said only "
                f"in shots that describe that person, and named once a second person "
                f"is in frame with them. Reported as one character stuck gazing at the "
                f"camera while the other does his part: the target was one string with "
                f"no owner, said impersonally, so a look she staged went on being said "
                f"in shots she was not in and landed on whoever was. Off with hold_gaze")
        if dialogue_gaze_shots:
            notes.append(
                f"shot(s) {', '.join(str(n) for n in dialogue_gaze_shots)} carry a line "
                f"and two or more people, and the beat names nothing to look at, so the "
                f"faces are turned to each other. Reported as two people talking to the "
                f"camera instead of each other: with no look staged, both faces fall to "
                f"the model's prior -- a portrait, facing the lens -- and a near-clean "
                f"reference asks for exactly that pose. A line has an addressee whether "
                f"or not the beat wrote one, and the addressee is in the shot, so this is "
                f"the one thing that can be said without inventing. One impersonal "
                f"sentence: both names in the shot are already spent, and a third "
                f"mention is a third person. Write 'looks at' or 'turns to' in the beat "
                f"and that is said instead. Off with hold_gaze")
        if anchored_shots:
            notes.append(
                f"fastened limbs held in place on shot(s) {', '.join(str(n) for n in anchored_shots)}"
                f" -- the shot that staged it said where, and every shot after it is "
                f"told the same, because the restraint hold keeps the hardware SHUT and "
                f"says nothing about where it is. Position was being carried by the "
                f"picture alone, and the picture is the previous shot's last frame. "
                f"Cleared by a `remove:` naming the hardware, like the hold itself")
        if cropped_wardrobe:
            notes.append(
                "the anchor names a close frame and says what it is close ON, so the "
                "wardrobe that frame cannot hold stopped being described: "
                + ", ".join(cropped_wardrobe)
                + ". The camera was always reaching the model -- it is a tenth of a "
                  "shot's text -- and the rest of the shot was asserting clothes the "
                  "frame has no room for, which is a wider frame said at length. "
                  "ONLY WARDROBE GOES. Where the limbs are held and what is fastened "
                  "to them are still said, because a close frame crops the anchor "
                  "point out of the picture the NEXT shot inherits and the text is "
                  "then the only thing that knows. Write the frame without naming a "
                  "subject -- \"close-up\" and no more -- and nothing is cropped, "
                  "because there is no way to know what it is close on")
        if tight_shots:
            notes.append(
                f"shot(s) {', '.join(str(n) for n in tight_shots)} frame tight enough to "
                f"crop the anchor point out. That matters past this shot: the next one "
                f"starts from THIS one's last frame, so whatever the close framing cut "
                f"off is missing from the picture the next shot inherits, and the text is "
                f"the only thing that still knows where the limbs are fastened. It is "
                f"being said. If the position still drifts, give the beat a wider frame "
                f"so the anchor is in the picture the chain hands on")
        if ambient_shots:
            notes.append(
                f"shot(s) {', '.join(str(n) for n in ambient_shots)} were given an "
                f"ambient bed read from {_bed_src} -- \"{ambient_bed}\". "
                f"It goes under shots whose audio branch is ALREADY open: ones with a "
                f"line, or with a sound you wrote yourself. It can never open one. "
                f"AMBIENCE ON EVERY SHOT WAS TRIED AND DOES NOT WORK: the bed was "
                f"allowed to open a branch, which is the one thing nothing inferred "
                f"here may do, and an open branch on a joint model fills itself. At "
                f"4-8 steps the final audio step clears 50%-30% of its denoising in "
                f"one jump, and what a branch resolving that much at once invents is "
                f"a voice -- so every wordless shot got ambience and a babbling mouth "
                f"with it. Ambience everywhere and silence are mutually exclusive by "
                f"construction: the silence latent IS the audio, and there is no room "
                f"in it for a room tone. To score a silent shot, write the sound into "
                f"that beat -- that is you asking for audio on purpose -- or lay an "
                f"ambient track under the finished video outside the model, where it "
                f"costs nothing and cannot speak")
        if dialogue_marked:
            notes.append(
                f"shot(s) {', '.join(str(n) for n in dialogue_marked)} had their "
                f"quoted speech wrapped in H3's own dialogue marker, <d>...</d>. "
                f"Those are special tokens the model was trained with, and they say "
                f"a span is SPOKEN; quotation marks say nothing at all, so a quoted "
                f"instruction reached the model as an imperative sentence and was "
                f"performed -- often a beat before anybody said it. Every word you "
                f"wrote is kept in order; only the quotation marks are exchanged. "
                f"Mark them yourself and this leaves them alone")
        if told_shots:
            notes.append(
                f"shot(s) {', '.join(str(n) for n in told_shots)} carry a line that "
                f"ORDERS somebody to do something, so the listener is given "
                f"something to be doing while it is said. The node does not stage "
                f"what a quoted line asks for -- the readers refuse speech -- but "
                f"the words are still in the shot, because beats go to the model "
                f"verbatim, and a video model does not tell a quoted instruction "
                f"from a stage direction: it renders what the words describe, and "
                f"the action lands a beat early. The words cannot be removed "
                f"without breaking the one promise this node makes about your text. "
                f"If it still happens, put the order in narration instead -- 'Dana "
                f"tells her to lie down' -- and keep the quoted line for something "
                f"that is not an instruction")
        if language_shots:
            notes.append(
                f"shot(s) {', '.join(str(n) for n in language_shots)} carry a line, "
                f"so each is told which language it is spoken in -- "
                f"{', '.join(_langs_used) or SPOKEN_LANGUAGE}, read from the line "
                f"itself rather than fixed. H3 is joint and multilingual: the prose "
                f"conditions the audio branch, and a branch told a line is spoken "
                f"but never told in WHAT will pick a language -- fluent delivery in "
                f"one nobody asked for sounds like babble to anybody expecting the "
                f"one they wrote. Said positively, because at cfg 1 there is no "
                f"negative prompt and naming the unwanted language would ask for it. "
                f"Write the dialogue in the language you want spoken; a line too "
                f"short to tell falls back to the rest of the script, then to "
                f"{SPOKEN_LANGUAGE}")
        # A LINE THAT DOES NOT FILL ITS SHOT. H3 is joint: the audio branch runs
        # for the whole shot, and a short line in a long one leaves it with time
        # and nothing to say. What it does with that time is say the line again.
        # Reported as dialogue duplication.
        #
        # A REPORT, not a clause. "the line said once" was tried as prompt text
        # and made it worse -- more speech words on a shot is more reason for the
        # branch to make speech -- so this says it to YOU instead, where the fix
        # is to shorten the shot or write more line.
        _roomy = []
        for _n, _w in sorted(_spoken_words.items()):
            _sec = (lens[_n - 1] / H3_FPS) if _n - 1 < len(lens) else 0.0
            _need = _w / 2.5 + 0.5      # ~150 words a minute, plus a breath
            if _sec > 0 and _sec > _need * 2:
                _roomy.append((_n, _w, _sec, _need))
        if _roomy:
            notes.append(
                "shot(s) " + ", ".join(
                    f"{n} ({w} word{'s' if w != 1 else ''} of line, about "
                    f"{need:.1f}s, in a {sec:.1f}s shot)"
                    for n, w, sec, need in _roomy)
                + " leave more than half their length with no line in it. The audio "
                  "branch runs for the whole shot and fills what is left, and what "
                  "it fills it with is the line again -- that is where doubled "
                  "dialogue comes from. Shorten those shots (shot_length 'from the "
                  "beat', or a lower shot_seconds), or give the beat more to say. "
                  "Room tone is already laid under them, which is what makes the "
                  "silence survivable at all")
        # WHAT A LINE CANNOT BE READ ALOUD FROM. Digits, times and abbreviations
        # have no single pronunciation -- "7:30" is "seven thirty" and also "seven
        # three zero", "Dr." is "doctor" and also "dee arr", "1985" is a year and
        # also four digits -- so the model picks, and picking wrong is what
        # mispronunciation sounds like. Written out, there is nothing to pick.
        #
        # Reported, never rewritten: the one promise this node makes about your
        # text is that it goes to the model as you wrote it.
        if _breath_shots:
            notes.append(
                f"shot(s) {', '.join(str(n) for n in _breath_shots)} stage a "
                f"breath and nothing else audible, so they are conditioned on "
                f"silence and the breath is NOT heard. A single indrawn breath "
                f"is half a second; holding the audio branch open for a whole "
                f"shot to render it leaves the rest of that shot open, and an "
                f"open branch on a joint model fills itself with a voice -- "
                f"which is the babble that arrives just before somebody speaks. "
                f"To hear it, put the breath in the same beat as the line, or "
                f"give the shot a sound that lasts: breathing hard, a chain, "
                f"footsteps")
        _hard = []
        _said_all = engine.spoken_text(prompt or "")
        for _m in _HARD_TO_SAY.finditer(_said_all):
            _t = _m.group(0).strip()
            if _t and _t not in _hard:
                _hard.append(_t)
        if _hard:
            notes.append(
                f"the dialogue contains {len(_hard)} thing(s) with no single way "
                f"to say them out loud: {', '.join(_hard[:10])}. A joint model "
                f"reads the line as text and chooses a pronunciation -- \"7:30\" is "
                f"\"seven thirty\" and equally \"seven three zero\", \"Dr.\" is "
                f"\"doctor\" and equally \"dee arr\" -- and the choice is where "
                f"mispronounced dialogue comes from. Spell them the way they should "
                f"be SPOKEN and there is nothing left to choose. They are NOT "
                f"rewritten: your words go to the model as you wrote them")
        _odd = non_latin_in(prompt) + non_latin_in(character_memory or "") \
            + non_latin_in(anchor or "")
        _odd = list(dict.fromkeys(_odd))
        if _odd:
            notes.append(
                f"the prompt contains {len(_odd)} character(s) that are not Latin "
                f"text: {' '.join(_odd[:12])}. A multilingual model reads those as a "
                f"strong signal about which language to speak, and one pasted glyph "
                f"is easy to miss by eye. They are NOT removed -- the node passes "
                f"your words through -- so retype them if the delivery is coming out "
                f"in a language you did not ask for"
                # ...and when the script IS in that language, this is not a warning
                # at all. Reporting a Cyrillic script as a stray glyph would be the
                # node telling somebody their own dialogue looks like a mistake.
                + (f". Your dialogue reads as {_script_lang}, though, so these are "
                   f"most likely meant to be here -- the lines are told they are "
                   f"spoken in {_script_lang}"
                   if _script_lang != SPOKEN_LANGUAGE else ""))
        if mouth_named:
            notes.append(
                f"shot(s) {', '.join(str(n) for n in mouth_named)} have a line, so "
                f"the shot is told who is speaking and every other mouth in it is "
                f"held closed. One of two people speaking still leaves the OTHER "
                f"one's mouth free, and the listener is exactly who invented "
                f"lip-sync lands on")
        if mouth_shut:
            notes.append(
                f"mouths held closed on shot(s) {', '.join(str(n) for n in mouth_shut)} -- "
                f"no scripted line and no effort staged in them. H3 is joint, so the face "
                f"follows the audio branch: the sentence is the picture half and the "
                f"silent conditioning is the half that actually settles it, since a "
                f"lips-closed line loses to a stream that has decided somebody is "
                f"talking. Shots staging effort are left out on purpose -- straining is "
                f"vocal and that mouth should be open. Off with mouths_shut_when_no_line")
        # WHICH OF THE THREE HAPPENED, every run, because the inference is weak and
        # the author needs to know when it decided nothing.
        if _film_mood == "grim":
            notes.append(
                f"the anchor declares the film's tone, so every shot carries \"The mood "
                f"is grim.\" and no guessing is done. That is the reliable way to set "
                f"it: swept over 512 beats, inferring a mood from the beats alone "
                f"called an ORDINARY film grim as often as a duress one, because the "
                f"words overlap -- screams is a waterslide, tied is a boat, bound is a "
                f"flight to Lisbon, chained is a desk job")
        elif _film_mood == "light":
            notes.append(
                f"the anchor declares a light tone, so no grim mood is applied to any "
                f"shot whatever the beats say. That is the override for a wrong "
                f"reading, and it wins outright")
        elif _film_duress:
            notes.append(
                f"no tone is declared in the anchor, and the beats or the character "
                f"sheet carry UNAMBIGUOUS duress -- hardware on a body, a captor, an "
                f"abduction, being locked in -- so every shot carries \"The mood is "
                f"grim.\" Only unambiguous evidence counts here: ordinary coercion "
                f"verbs and distress words are not enough on their own, because "
                f"grabbing, dragging and screaming are as much a garden centre and a "
                f"waterslide as an abduction. Write the tone into the anchor to settle "
                f"it either way")
        elif any(beat_duress_strength(b) for b in beats):
            notes.append(
                f"some beats read as though they MIGHT stage duress -- coercion or "
                f"distress verbs -- but nothing unambiguous, so no mood was applied and "
                f"every face is left to the model, whose prior for a described person "
                f"is a pleasant posed portrait. If this film has a tone, write it into "
                f"the anchor: 'grim', 'tense', 'a kidnapping' and the like turn it on "
                f"for every shot, and 'warm' or 'comic' turn it off for good. Measured "
                f"over 512 beats, guessing from the beats alone is no better than a "
                f"coin toss, so it does not guess")
        if vocal_shots:
            notes.append(
                f"shot(s) {', '.join(str(n) for n in vocal_shots)} have a vocal that "
                f"belongs to somebody -- a whimper, a sob, a moan -- so the shot is "
                f"told whose it is, and the mouths owning neither a line nor a sound "
                f"are closed. Reported as one character's whimpering opening up "
                f"another's ability to babble. A vocal opens the audio branch, which "
                f"is right -- it is meant to be heard -- but the flag saying so was "
                f"shot-level with no owner, and BOTH mouth guards stood down on it "
                f"for everybody in the shot. The person straining should have an open "
                f"mouth; the person watching them should not, and theirs was the face "
                f"an invented voice landed on. Two sources in one shot are named "
                f"separately for the same reason, so the line and the vocal cannot be "
                f"swapped between them. A vocal the beat does not pin on anybody "
                f"holds nobody: closing mouths on a guess could close the mouth "
                f"making the noise. This changes which faces move and never what the "
                f"audio is conditioned on. Off with mouths_shut_when_no_line")
        if duress_shots:
            notes.append(
                f"shot(s) {', '.join(str(n) for n in duress_shots)} are told what the "
                f"face is doing, because the scene already stages duress -- restraint "
                f"hardware the sheet lists on somebody in the shot, or your own "
                f"distress verbs in the beat. Reported as somebody smiling at the "
                f"camera in a scene of duress: a four-shot scene of a woman handcuffed "
                f"in a van had not one word in it about anybody's face, and an "
                f"attribute a prompt does not state is not LEFT to the model, it is "
                f"left to the model's prior -- which for a named, described person is "
                f"a portrait, facing the lens, pleasantly, because that is what "
                f"photographs of people are. The eyes have had a clause since "
                f"hold_gaze; the expression never had one. It is one sentence, it "
                f"names no camera, and it reads the staging rather than inventing a "
                f"feeling -- a shot staging neither gets nothing, and a beat that "
                f"already says what the face does is never argued with. Picture only: "
                f"it can never open the audio branch")
        if mouth_acting:
            notes.append(
                f"shot(s) {', '.join(str(n) for n in mouth_acting)} kept their mouths "
                f"because the beat itself puts the mouth to work -- a grin, a yawn, a "
                f"bitten lip, a jaw dropping. The guard holds mouths closed on every "
                f"shot with nobody speaking, and against a face doing nothing that is "
                f"right; against these it was countermanding the only performance "
                f"direction the shot has, in one case word for word. The beat has said "
                f"what the mouth does, so nothing is added over the top. This frees the "
                f"PICTURE only: a smile is silent, and the audio branch is left exactly "
                f"where it was, because a silent expression is the commonest beat there "
                f"is and letting one open a branch would be the invented voice back at "
                f"its widest point. A stare or a wince is a face acting with its mouth "
                f"shut and is still held")
        if muted_sound:
            notes.append(
                f"shot(s) {', '.join(str(n) for n in muted_sound)} gave up the sound you "
                f"wrote for them so the mouths could be held shut. Those shots have no "
                f"line, and a sound alone was enough to leave the audio branch open -- "
                f"which is where the invented voice and the lip-sync came from. This is "
                f"the trade and it is the only one available: the ambience cannot be kept "
                f"while the branch is conditioned to silence. Turn off "
                f"mouths_shut_when_no_line to keep the sound and accept the mouth")
        if turned_shots:
            notes.append(
                f"shot(s) {', '.join(str(n) for n in turned_shots)} stage a change with a "
                f"direction -- something opened or shut -- so the shot is told both ends: "
                f"what is true at the first frame and what is true by the last. Some "
                f"distill LoRAs render an action backwards, and a beat naming one state "
                f"names neither end, so the reverse reads as an equally good answer. Verbs "
                f"that genuinely go either way -- pulls, draws, slides, swings -- get no "
                f"anchor, because a wrong one asks for the reversal instead of allowing "
                f"it. Reversal is likeliest in shot 1, which has no previous last frame "
                f"pinning where it starts; first_frame pins it. Off with hold_scene_state.")
        if inferred_sound:
            notes.append(
                f"shot(s) {', '.join(str(n) for n in inferred_sound)} were given the "
                f"sound their own action implies -- H3 is joint, so the same prose "
                f"conditions the audio branch, and a beat that says what happens has "
                f"said what it sounds like. Read from the beat, never the scene, so a "
                f"chain standing in the scene does not rattle where nobody moves. A beat "
                f"that describes its own sound is left alone. This is TEXT ONLY and can "
                f"never unsilence a shot: it is added to shots whose audio branch is "
                f"already open, meaning ones with a line or with a sound you wrote "
                f"yourself. A shot with neither stays pinned to silence and gets no "
                f"sound sentence, because the mouth follows the audio and an inference "
                f"is not a good enough reason to let it move")
        # WHICH shots, not how many. "2 shot(s) have an open branch" told a reader
        # that two of eleven can babble and gave them no way to find out which two
        # -- and the whole point of the note is that the beat's own sound wording is
        # what opened it, which cannot be acted on without knowing the beat.
        _open_br = [i + 1 for i, (s_, snd) in enumerate((shot.speech, shot.sounded) for shot in plan.shots)
                    if not s_ and snd]
        _pinned = [i + 1 for i, (s_, snd) in enumerate((shot.speech, shot.sounded) for shot in plan.shots)
                   if not s_ and not snd]
        n_silent, n_kept = len(_pinned), len(_open_br)
        if silence_nonspeech and n_kept:
            notes.append(
                f"shot(s) {', '.join(str(n) for n in _open_br)} have no line but either "
                f"describe a sound IN THE BEAT or "
                f"stage EFFORT, so their audio is left free to make it -- writing the "
                f"sound, or the verb that produces one, is asking for audio on purpose. "
                f"Those are the only shots without a line "
                f"where the branch is open, and an open branch on a joint model can "
                f"still put a voice in the gap. If one of them babbles, that beat's own "
                f"sound wording is what opened it")
        if silence_nonspeech and n_silent:
            notes.append(
                f"shot(s) {', '.join(str(n) for n in _pinned)} have no quoted line and no "
                f"sound described, so they "
                f"are conditioned on real silence -- which is not 'no speech', it is 'no "
                f"sound at all': no footsteps, no room tone, nothing. H3 is joint, so the "
                f"way to score a scene is to DESCRIBE it in the prose: 'boots on concrete, "
                f"a chain dragging, a low hum off the strip light'. Write it into a beat "
                f"for that shot, or into the anchor to carry it through the film. Do not "
                f"use a label like 'sound:' -- a labelled line is read as text to draw")
            # The note that used to sit here warned that a beat staging effort was
            # being silenced, which read as a flat, unreacting face. It cannot happen
            # any more: effort opens the audio branch, because the verb staging it is
            # the author's. See _voiced in the shot loop.

        if first_frame is None:
            # SAID AS AN ASYMMETRY, because that is what it is and the old wording hid
            # it. This used to read "shot 1 has nothing pinning its opening frame, so
            # its starting pose and framing come from the text and any reference" --
            # true, and it left out the half that matters: every OTHER shot IS pinned,
            # by the previous shot's last frame, so shot 1 is the only shot in the film
            # that is free. What that looks like from outside is not shot 1 drifting,
            # it is shot 1 disagreeing with a chain that agrees with itself.
            #
            # Reported exactly that way -- "doesn't look the same from the first to
            # last beat", "the remaining beats are fine" -- together with a hardware
            # artefact in beat 1 alone, hair caught in a collar. Both are the same
            # thing: an arrangement no picture settles is settled by the model, and
            # from shot 2 on the keyframe settles it.
            #
            # And the note has to say which dial is NOT this one, because the report
            # came with "I even have image reference strength set to 0.999". It cannot
            # work. build_conditioning's own comment is plain about it: "the keyframe
            # ANCHORS the first frame, which is what continuity needs, while a
            # reference only supplies identity. They are not alternatives." Raising
            # ref_noise_aug makes the reference cleaner; it does not give shot 1 a
            # first frame, because there is no frame there to clean.
            notes.append(
                "NO first_frame IS WIRED, so shot 1 is the only shot in this film whose "
                "opening frame is pinned by NOTHING. Every other shot opens on the "
                "previous shot's last frame, which fixes its pose, its framing and the "
                "arrangement of everything on the body; shot 1 has only the text and "
                "any reference. So the chain agrees with itself and shot 1 is the one "
                "that can disagree -- which from outside looks like the person changing "
                "between the first beat and the rest, and is also where a one-shot-only "
                "oddity comes from: hair sitting differently against a collar, a "
                "garment hanging differently, a pose the beat did not ask for. "
                "ref_noise_aug IS NOT THE DIAL FOR THIS and raising it cannot help: a "
                "reference says WHO somebody is and a keyframe says what the opening "
                "frame HOLDS, and they are not alternatives -- there is no frame on "
                "shot 1 for a cleaner reference to sharpen. Wire first_frame to fix it"
                + (", and see the ref_noise_aug note above for what to put in it -- it "
                   "pins the WHOLE frame, so a composed frame of the shot you want and "
                   "not an identity portrait"
                   if refs_all else
                   ". It pins the WHOLE frame, so give it a composed frame of the shot "
                   "you want: subject, pose, framing, background. The last frame of a "
                   "previous run, or any still matching how beat 1 should open")
                + ". Leaving it empty is fine when beat 1 is meant to establish the "
                  "look and the rest follow it -- which is what is happening now")
        # Text in the frame. H3 draws letterforms when the prompt names them, and at
        # cfg 1 there is no negative prompt to take them back -- adding "no watermark"
        # to the positive only names it again, which is how a mention becomes a
        # presence cue. So: point at the words, and leave the decision to the author.
        # H3 has a caption channel of its own. A prompt carrying those tokens is
        # ASKING for text on the picture.
        if any(_CAPTION_TOKEN.search(s) for s in plan.prompts):
            notes.append("the prompt contains H3's caption/lyrics tokens "
                         "(<|caption_start|> and friends) -- those request text ON the "
                         "picture. Remove them unless you want subtitles burned in")
        # Quoted dialogue with no <d> marker. H3 distinguishes speech, captions and
        # lyrics with explicit tokens; unmarked quoted text is not identified as any
        # of them, and a model with a caption channel may render it rather than say
        # it. Worth trying if subtitles are appearing under spoken lines.
        # Only what the marker did NOT catch: a quote with no terminal punctuation
        # and no speech cue in front of it, which is a scare quote far more often
        # than a line. The note used to tell the reader to wrap their dialogue by
        # hand; the node does that now, so this is what is left over.
        n_bare = sum(1 for b in beats
                     if _QUOTED.search(b) and not _DIALOGUE_TAG.search(b)
                     and mark_dialogue(b) == b)
        if n_bare:
            notes.append(f"{n_bare} beat(s) carry quotes that were NOT read as speech: "
                         f"no full stop, question mark or exclamation inside them, and "
                         f"no speech verb in front. A quote like that is usually "
                         f"emphasis or a title, so it was left exactly as written. If "
                         f"one of them IS a line, end it with punctuation or mark it "
                         f"yourself with <d>...</d> and it will be spoken rather than "
                         f"drawn")
        cued = sorted({m.group(0).lower() for s in plan.prompts for m in _TEXT_CUE.finditer(s)})
        if cued:
            notes.append(f"the prompt names on-screen text ({', '.join(cued)}) -- H3 draws "
                         f"letterforms when asked, and at cfg 1 no negative prompt can take "
                         f"them back. Remove the words if you do not want the text")
        # Each beat against ITS OWN shot length; thin_beats numbers from 1, so the
        # shot number is restored here.
        thin = [t.replace("shot 1:", f"shot {i + 1}:")
                for i, b in enumerate(beats)
                for t in thin_beats([b], lens[i] / H3_FPS)]
        if thin:
            notes.append(
                "THIN BEATS -- the shot outlasts what the beat gives it to do, and the "
                "cheapest way for the model to fill the rest is to CARRY ON with the "
                "action, repeating it on whatever is nearest: "
                + "; ".join(thin)
                + ". Give the beat a second action -- what happens after it -- or lower "
                "shot_seconds")
        if float(cfg) != 1.0:
            notes.append(f"cfg is {float(cfg):g}; H3 is CFG-free and expects 1.0")

        # Resolve the <Picture N> tags before `script` is written, so what you read is
        # what the model is given. Which roster they resolve against depends entirely
        # on the format -- see build_conditioning.
        # Tags PLACE the references. With none written anywhere, placing by tag would
        # place them nowhere -- a connected reference that silently does nothing at
        # all. The old node fell back rather than no-op, and so does this.
        # Judged on what was WRITTEN, not on what survives scrubbing.
        #
        # A tag on a covered garment is removed from every shot that hides it --
        # correctly, since the tag has to leave with the thing it names. But if
        # that was the only tag in the sheet, the check below then saw no tags
        # anywhere and fell back to "untagged references ride EVERY shot", which
        # sent the picture straight back into the shots that had just hidden it.
        # Reported as a chastity belt drawn over the shorts by somebody whose only
        # reference was the belt.
        #
        # The author tagged something. That the layering consumed it later is not
        # a reason to start placing pictures everywhere.
        _written = "\n".join([scene or ""] + list(beats))
        _tagged = bool(picture_tags(_written)
                       or any(picture_tags(s) for s in plan.prompts))
        _tagged_names = {n for n, ln in sheet_lines(sheet) if n and picture_tags(ln)}
        if refs_all and not _tagged:
            notes.append(
                f"{len(refs_all)} reference image(s) connected and no <Picture N> tag "
                f"anywhere, so they go on EVERY shot -- placing by tag would place them "
                f"nowhere. To aim them, write the tag on the person they depict: 'Nora: "
                f"<Picture 1>, 34, she, ...'. Each then travels with that person into "
                f"the shots she is in, and only those")
        for _i, _s in enumerate(plan.prompts):
            # The tag is the BINDING between a picture and the subject the prompt
            # describes, and it stays IN the text -- comfy_extras/nodes_minimax_h3.py:
            # "the prompt refers to them as <Picture i>", "Use the same tags when
            # prompting". Renumbered per shot, because the encoder numbers by the
            # order it receives images and a shot carrying only slot 2 receives that
            # image as <Picture 1>.
            if not _tagged:
                plan.shots[_i].refs = list(refs_all)
                continue
            _s, _r, _missing = resolve_tags(_s, refs_all)
            plan.shots[_i].prompt = _s
            plan.shots[_i].refs = _r
            for _n in _missing:
                _msg = f"<Picture {_n}> names a slot with no image connected"
                if _msg not in notes:
                    notes.append(_msg)
        # ONE FACE, TWO PEOPLE. A shot that carries a picture for somebody AND
        # describes somebody else who has none gives the model a photographed face
        # and two faces to draw. A reference is the strongest identity signal in the
        # prompt -- far stronger than "38, dark hair" -- so the one that exists gets
        # used twice, and the second character arrives as a copy of the first.
        #
        # Reported as two of the same woman in a scene written for two people. The
        # node cannot stop it: it is the model resolving a shot that has more
        # subjects than pictures, and there is no sentence that outranks a photo.
        # What it can do is say which shots are in that state, and say it in terms
        # of the fix -- a second reference, tagged onto the other person.
        _twinned = []
        if refs_all and _tagged_names:
            for _i, _s in enumerate(plan.prompts):
                if not picture_tags(_s):
                    continue
                _cast_here = plan.shots[_i].cast
                _cast_here = [n for n in _cast_here if n] or [
                    n for n, _ in sheet_lines(sheet) if n]
                _bare = [n for n in _cast_here if n not in _tagged_names]
                if _bare and any(n in _tagged_names for n in _cast_here):
                    _twinned.append((_i + 1, _bare))
        if _twinned:
            _who = sorted({n for _, ns in _twinned for n in ns})
            notes.append(
                f"shot(s) {', '.join(str(n) for n, _ in _twinned)} carry a reference "
                f"picture for one person and also describe "
                f"{', '.join(_who)}, who {'has' if len(_who) == 1 else 'have'} no "
                f"<Picture N> of their own. That "
                f"is one photographed face and two people to draw, and a reference is "
                f"the strongest identity signal in the prompt -- much stronger than a "
                f"line of description -- so the face that exists tends to be used "
                f"twice and the second character arrives as a copy of the first. Wire "
                f"a picture of {', '.join(_who)} to a free ref_image slot and tag it "
                f"on their sheet "
                f"{'entries' if len(_who) > 1 else 'entry'} -- "
                f"'{_who[0]}: <Picture 2>, ...' -- so every shot with both of them "
                f"carries both faces. No wording fixes this: nothing in the text "
                f"outranks a photograph")
        if not character_guard and len([n for n, _ in sheet_lines(sheet) if n]) > 1:
            _wardrobes = [f"{n} ({', '.join(garments_in(ln)[:3])})"
                          for n, ln in sheet_lines(sheet) if n and garments_in(ln)]
            notes.append(
                f"character_guard is OFF, so EVERY sheet line is in EVERY shot -- "
                f"including the wardrobe of everyone the beat does not involve. "
                + ("With " + "; ".join(_wardrobes[:4]) + ", " if _wardrobes else "")
                + f"a shot about one person is also describing what the others have on, "
                f"and at cfg 1 the model reads the prompt as a bag of words before it "
                f"reads a label: a garment listed for one character lands on whichever "
                f"body is in frame. Reported as boys wearing stockings. Measured: with "
                f"the guard ON, a shot that names only the men carries no word of the "
                f"women's clothing at all, because only the people a beat involves are "
                f"described. Turn it on. For extras nobody has an entry for, write them "
                f"into the beat instead -- your words reach the model verbatim and an "
                f"unnamed person needs no entry, though a removal cannot be held for one")
        if refs_all and _tagged and not character_guard:
            notes.append(
                f"character_guard is OFF and {len(refs_all)} reference image(s) are tagged -- "
                f"the combination that fixes the camera on one person. Off, EVERY sheet line "
                f"goes into every shot, including the line carrying <Picture N>, so the "
                f"reference is named in every shot and rides all of them. At ref_noise_aug "
                f"{float(ref_noise_aug):g} that asks the model to reproduce the PICTURE -- "
                f"pose and framing, not only the face -- so the portrait's composition "
                f"becomes every shot's composition, and anyone without a reference is placed "
                f"relative to it. AND TURNING THE GUARD OFF ADDS NOBODY: it describes the "
                f"people your SHEET already names, in every shot, whether the beat involves "
                f"them or not. For extras nobody has a sheet entry for, leave the guard ON "
                f"and write them into the beat -- your words reach the model verbatim and an "
                f"unnamed person needs no entry. To loosen the framing instead, lower "
                f"ref_noise_aug (try 0.95, then 0.90) or crop the reference to head and "
                f"shoulders, so there is less composition in it to reproduce")
        if refs_all:
            _named = sum(1 for s in plan.prompts if picture_tags(s))
            notes.append(
                f"{len(refs_all)} reference image(s) supply IDENTITY, and they go WHERE "
                f"TAGGED: every shot whose text names <Picture N> carries the image on "
                f"ref_image_N, which is what holds a face across beats instead of "
                f"letting it drift down the "
                f"keyframe chain. Put the tag on the person -- 'Nora: <Picture 1>, 34, "
                f"she, ...' -- and it travels with her. {_named} shot(s) claim one here. "
                f"References ride alongside the keyframe rather than instead of it: the "
                f"keyframe anchors the first frame, a reference only says who somebody "
                f"is, and ComfyUI packs both (keyframe rows then ref rows, in the same "
                f"order model_base builds the latents). References keep slots 1..N so the "
                f"tag points at the right image; the handoff is appended after them and "
                f"disturbs no numbering. Expect the NUMBER in script to differ from the "
                f"one you wrote: it is the picture's place in THAT shot's reference "
                f"list, not a name for the image, so a shot carrying one reference "
                f"always says <Picture 1> whichever socket it came from. The image is "
                f"still that person's -- what would be wrong is a shot carrying two "
                f"references and naming only one, since a picture the text never names "
                f"is read as another subject")
            if _named > 1 and float(ref_noise_aug) >= KEYFRAME_SAFE_AUG:
                notes.append(
                    f"a reference on {_named} shots at ref_noise_aug "
                    f"{float(ref_noise_aug):g} is the trade this makes. Near-clean, a "
                    f"reference asks the model to reproduce the PICTURE -- pose and "
                    f"framing, not only the face -- and on a shot that is not introducing "
                    f"the character that competes with the staging the beat describes: "
                    f"the referenced person can hold the portrait's gaze while anyone "
                    f"without a reference is placed relative to that composition and then "
                    f"travels to where the text put them. It is the price of the face "
                    f"holding. A hybrid fl2va/ref2va checkpoint is trained for reference "
                    f"conditioning and does not make this trade; on a plain fl2va one, "
                    f"lowering ref_noise_aug is the dial")
            if _named < len(plan):
                notes.append(
                    f"{len(plan) - _named} shot(s) name no <Picture N> at all, so they "
                    f"carry no reference. Claim it on the person it depicts -- 'Nora: "
                    f"<Picture 1>, 34, she, ...' -- and it travels with her into the shots "
                    f"she is in, and only those. A picture the prompt never refers to is "
                    f"read as ANOTHER subject")


        # What each shot was SENT. Built here so plan_only has it, and corrected in
        # the render loop for the one sentence that is added down there.
        #
        # It used to be built here and never touched again, while the recovered-face
        # claim was written onto the loop's own copy of the prompt -- so the model got
        # "Dom: <Picture 1>, he, 41" and this said "Dom: he, 41". The output documented
        # as the exact per-shot text was wrong about the one shot most likely to be
        # under investigation, and it is the output the reader is told to check when a
        # shot renders somebody they did not ask for.
        # CAN the silence conditioning actually be built? Every failure inside
        # _silent_audio_latent returns None on purpose so a render never dies for a
        # nicety -- which means a wrong VAE on the audio_vae input costs nothing at
        # load time and silently unpins every line-free shot, and the first anybody
        # knows of it is a shot with no dialogue that babbles.
        #
        # Probed HERE, before the plan is returned, because finding out should not
        # cost a full render. The unit is cached, so a real render pays nothing for
        # this and the answer is the same one the render would get.
        # The audio branch's own last step. Reported whenever it is steep, because
        # shift_video is the dial people reach for and it does not touch this.
        _last_a = last_audio_sigma(steps, shift_audio, scheduler, shift_video)
        # A SCHEDULER CAN END THIS OUTRIGHT, and this note used to deny it.
        _alt_sched = scheduler_that_finishes_audio(steps, shift_audio, shift_video,
                                                   scheduler)
        # Never advise RAISING it: the target is a ceiling on the last step, not a
        # setting to move towards from below.
        _fix_a = min(shift_audio_for(steps), float(shift_audio or 0.0) or 1.0)
        # ...AND THE NODE CAN SHORTEN THE FALL ITSELF, without taking the scheduler
        # away from the picture. The audio branch has no schedule of its own -- it is
        # derived from the video sigma at every step -- so choosing a scheduler for
        # the audio means giving up the one chosen for the video. One extra step does
        # not: it splits the final jump and leaves every earlier sigma alone.
        #
        # Only when the node is the one setting the shift. With apply_model_sampling
        # off, the shifts this is computed from are not the shifts the model uses,
        # and a schedule built on the wrong ones would be worse than none. A wired
        # `sigmas` input is the author's own schedule and is never touched.
        _soft_landing = bool(apply_model_sampling
                             and not (sigmas is not None and len(sigmas)))
        # WHETHER THE LANDING ACTUALLY FIRES. Used twice: to describe it, and to stop
        # the older warning sending the reader off to do by hand the thing that has
        # already been done for them.
        _landing_on = bool(_soft_landing and _last_a > 0.10)
        if _landing_on:
            notes.append(
                f"the audio branch was landing from sigma {_last_a:.3f} on its final "
                f"step, so ONE extra step is spliced into the end of the schedule to "
                f"put it down at about 0.030 instead. That step costs one model "
                f"evaluation per shot and nothing else: every earlier sigma is exactly "
                f"where '{scheduler}' put it, so the picture keeps the schedule you "
                f"chose. This is the one lever prose cannot reach -- every clause in "
                f"this node changes what the branch is TOLD, and none of them changes "
                f"how much noise it still has to clear when it stops. A branch "
                f"resolving 43% of its denoising in one jump invents whatever is "
                f"easiest, which on a branch told somebody speaks is a voice, and it "
                f"lands at the OPENING of the shot because that is where there is "
                f"least conditioning to anchor it. Choosing a scheduler that already "
                f"finishes the audio"
                + (f" -- '{_alt_sched[0]}' leaves {_alt_sched[1]:.3f}" if _alt_sched
                   else "")
                + f" is still the better fix and costs no step; this one fires only "
                f"while the tail is steep. It lands at 0.030 whatever shift_audio is "
                f"set to -- 1, 3 and 5 all end up there -- so shift_audio does NOT "
                f"need tuning by hand for this any more, and the older advice to "
                f"lower it does not apply while this is on. Off by wiring your own "
                f"`sigmas`, or with apply_model_sampling")
        # ...and NOT where the landing has already dealt with it. Both notes fired
        # together at the shipped defaults and the second was false the moment the
        # first was true: it said the branch "still has sigma 0.43 to clear on its
        # FINAL step" when that final step had just been replaced, and then sent the
        # reader off to lower shift_audio by hand. Asked directly whether the manual
        # shift was still needed, which is the confusion this caused.
        if _last_a > 0.4 and not _landing_on:
            notes.append(
                f"the audio branch still has sigma {_last_a:.2f} to clear on its FINAL "
                f"step at {int(steps)} steps with shift_audio {float(shift_audio):g} -- "
                f"about {_last_a * 100:.0f}% of its denoising in one jump, and a branch "
                f"resolving that much at once invents whatever is easiest, which is a "
                f"voice. It is the step where babble appears. shift_VIDEO does not "
                f"change this: time_shift_sigma inverts the video shift and re-applies "
                f"the audio one. "
                + (f"'{_alt_sched[0]}' at these same {int(steps)} steps and the same "
                   f"shift_audio leaves {_alt_sched[1]:.3f} instead of {_last_a:.2f}, "
                   f"and it honours shift_video, so the picture keeps the schedule "
                   f"shape you asked for. DO NOT reach for kl_optimal, exponential or "
                   f"karras for this: comfy grades schedulers by use_ms, those three "
                   f"are called with sigma_min and sigma_max ONLY and never see the "
                   f"shift at all, so the video schedule collapses off its high-sigma "
                   f"steps and the picture comes out watery. Reported exactly that "
                   f"way. "
                   if _alt_sched else "")
                + f"Otherwise LOWER shift_audio or raise steps -- sigma rises with "
                f"shift_audio, so raising it makes this worse. shift_audio "
                f"{_fix_a:.2f} at {int(steps)} steps leaves "
                f"{last_audio_sigma(steps, _fix_a, scheduler, shift_video):.2f}, "
                f"against the {DEFAULT_LAST_AUDIO_SIGMA:.2f} the default 3.0 leaves "
                f"at 8 steps on 'simple'.")
        # Probed whenever silencing is ON, not only when a shot is silent today:
        # an ambient bed can cover every shot, and the answer still matters for
        # the moment one is not covered -- and for knowing the wiring is sound.
        if silence_nonspeech or speech_lead_seconds > 0 or speech_tail_seconds > 0:
            if audio_vae is None:
                notes.append(
                    "SILENCE CANNOT BE APPLIED: no audio VAE is wired to the node's "
                    "audio_vae input, so every shot listed above as conditioned on "
                    "real silence has an audio branch that is NOT pinned. H3 is "
                    "joint, so an unconditioned branch invents a voice and the "
                    "picture lip-syncs to it -- a shot babbling with nothing "
                    "scripted to say")
            elif _silent_audio_latent(audio_vae, lens[0], H3_FPS) is None:
                notes.append(
                    "SILENCE CANNOT BE APPLIED: the VAE on the audio_vae input would "
                    "not encode a silent second, so every shot listed above as "
                    "conditioned on real silence has an audio branch that is NOT "
                    "pinned -- and an unconditioned branch on a joint model invents "
                    "a voice the picture then lip-syncs to. That input wants the "
                    "MiniMax H3 AUDIO vae (minimax_h3_audio_vae.safetensors) in its "
                    "own VAELoader. Every VAE carries an audio_sample_rate "
                    "attribute, so a video VAE wired here passes every check "
                    "until the encode itself fails -- which is caught and "
                    "turned into no conditioning at all")
            else:
                notes.append(
                    f"silence can be applied: the audio VAE encodes silence, so the "
                    f"{n_silent} line-free shot(s) above can be pinned to it rather "
                    f"than merely told to be quiet"
                    + (f", and dialogue gets a {speech_lead_seconds:g}s silent lead-in"
                       if speech_lead_seconds > 0 else "")
                    + ((", and a silent tail past the line on shot(s) "
                        + ", ".join(f"{n} (last {s:.1f}s)" for n, s in _tailpin)
                        + f" -- everything after lead + the line's estimate + "
                        f"{speech_tail_seconds:g}s is pinned, so the branch cannot carry on "
                        f"talking into the seconds the line does not fill. The model chooses "
                        f"when to speak: if a last word is clipped, raise speech_tail_seconds")
                       if _tailpin else ""))
        script = "\n---\n".join(f"[Shot {i}] {s}" for i, s in enumerate(plan.prompts, 1))
        info = " | ".join(notes)
        if plan_only:
            empty = torch.zeros((1, h, w, 3))
            return (empty, {"waveform": torch.zeros((1, 2, 1)), "sample_rate": 44100},
                    "PLAN ONLY -- nothing rendered. " + info, script,
                    lens[0], 0, len(plan), 0.0)

        return PreparedVideo(
            _placed_shots=_placed_shots, _first_is_plate=_first_is_plate,
            _returns=_returns, _soft_landing=_soft_landing, _tagged_names=_tagged_names,
            ambient_audio=ambient_audio, ambient_level=ambient_level, apply_model_sampling=apply_model_sampling,
            audio_vae=audio_vae, auto_sound=auto_sound, bared_shots=bared_shots,
            cfg=cfg, cleanup_between_shots=cleanup_between_shots, clip=clip,
            first_frame=first_frame, foley_level=foley_level, h=h,
            latent_upscale=latent_upscale, latent_upscale_scale=latent_upscale_scale,
            megapixels=megapixels, model=model, moved_shots=moved_shots,
            negative=negative, notes=notes, plan=plan,
            ref_noise_aug=ref_noise_aug, restart_after_removal=restart_after_removal, revealed_shots=revealed_shots,
            sampler_name=sampler_name, scheduler=scheduler, seed=seed,
            shift_audio=shift_audio, shift_video=shift_video,
            sigmas=sigmas, silence_nonspeech=silence_nonspeech,
            speech_lead_seconds=speech_lead_seconds, speech_tail_seconds=speech_tail_seconds, hold_levels=hold_levels, staging_shots=staging_shots, steps=steps,
            stripped_shots=stripped_shots, cut_shots=cut_shots,
            tiled_decode=tiled_decode, trim_seam=trim_seam,
            upscale=upscale, upscale_batch=upscale_batch, upscale_model=upscale_model,
            upscale_target_short_edge=upscale_target_short_edge, vae=vae, w=w,
        )

    def _render(self, prepared):
        """Execute the prepared shots and assemble the video and soundtrack."""
        _placed_shots = prepared._placed_shots
        _first_is_plate = prepared._first_is_plate
        _returns = prepared._returns
        _soft_landing = prepared._soft_landing
        _tagged_names = prepared._tagged_names
        ambient_audio = prepared.ambient_audio
        ambient_level = prepared.ambient_level
        apply_model_sampling = prepared.apply_model_sampling
        audio_vae = prepared.audio_vae
        auto_sound = prepared.auto_sound
        bared_shots = prepared.bared_shots
        cfg = prepared.cfg
        cleanup_between_shots = prepared.cleanup_between_shots
        clip = prepared.clip
        first_frame = prepared.first_frame
        foley_level = prepared.foley_level
        h = prepared.h
        latent_upscale = prepared.latent_upscale
        latent_upscale_scale = prepared.latent_upscale_scale
        megapixels = prepared.megapixels
        model = prepared.model
        moved_shots = prepared.moved_shots
        negative = prepared.negative
        notes = prepared.notes
        plan = prepared.plan
        ref_noise_aug = prepared.ref_noise_aug
        restart_after_removal = prepared.restart_after_removal
        revealed_shots = prepared.revealed_shots
        sampler_name = prepared.sampler_name
        scheduler = prepared.scheduler
        seed = prepared.seed
        shift_audio = prepared.shift_audio
        shift_video = prepared.shift_video
        sigmas = prepared.sigmas
        silence_nonspeech = prepared.silence_nonspeech
        speech_lead_seconds = prepared.speech_lead_seconds
        speech_tail_seconds = prepared.speech_tail_seconds
        hold_levels = prepared.hold_levels
        staging_shots = prepared.staging_shots
        steps = prepared.steps
        stripped_shots = prepared.stripped_shots
        cut_shots = prepared.cut_shots
        tiled_decode = prepared.tiled_decode
        trim_seam = prepared.trim_seam
        upscale = prepared.upscale
        upscale_batch = prepared.upscale_batch
        upscale_model = prepared.upscale_model
        upscale_target_short_edge = prepared.upscale_target_short_edge
        vae = prepared.vae
        w = prepared.w

        if apply_model_sampling:
            model, ms_note = apply_h3_model_sampling(model, shift_video, shift_audio)
            notes.append(ms_note)
        if negative is None:
            negative = clip.encode_from_tokens_scheduled(clip.tokenize(""))

        handoff = first_frame
        # Where the time actually goes. Sampling and decode trade off against each
        # other -- latent_upscale buys cheaper sampling and pays for it at decode,
        # and which side wins depends on `steps`. Reported so the trade is a
        # measurement rather than an argument.
        t_sample = t_decode = 0.0
        _aug_warned = False
        fresh = []
        t_start = time.perf_counter()
        aud_out, sr = [], 44100
        # AN UPPER BOUND, NOT AN ESTIMATE. This used to subtract one frame per seam on
        # the assumption that trim_seam drops one from every shot after the first. It no
        # longer does: a shot that opens on no keyframe keeps its first frame, and a
        # room change or a removal makes such shots on purpose. The buffer then filled
        # and FrameAccumulator fell through to its overflow list -- which, with
        # cleanup_between_shots off, retains each shot's decoded frames ON THE GPU,
        # uncopied, for the rest of the run. Higher peak VRAM under dynamic VRAM loading
        # is where a bad free turns into an illegal access, and the whole point of the
        # accumulator is that the final tensor is allocated ONCE.
        #
        # Over-allocating by at most one frame per seam is a rounding error against a
        # chain of hundreds, and it is the difference between a bounded allocation and
        # an unbounded list of live GPU tensors.
        frame_capacity = sum(shot.frame_count for shot in plan.shots)
        frames = FrameAccumulator(frame_capacity, _image_out_dtype(), cleanup_between_shots)
        # Reachable from run(), so an interrupt can drop it before unwinding. See there.
        self._frames = frames
        av_fix = 0                  # samples of A/V drift corrected across the chain
        _captured = {}              # name -> a frame from the last shot they were in
        _captured_from = {}         # name -> which shot that frame came from
        _recovered = []             # (shot, name, source shot) actually pinned
        _handoff_claimed = []       # shots whose demoted handoff was named in the text
        _untrimmed = []             # shots that opened on no keyframe, so kept frame one
        fresh_room = []             # shots cut because they open in another room
        _plate_on = 0               # the shot whose first_frame rides as the SET
        _carried = []               # (shot, who was there, who joins) room carried on
        shot_detail = []            # (detail, contrast) per shot, on its last frame
        # One per run, never reset at a chain break: the grade belongs to the FILM, and
        # restarting it per segment would give a film one grade per segment, which is a
        # worse-looking version of the same complaint.
        _levels = HandoffLevels()
        _SILENCE_STATUS.update(asked=0, applied=0, why="")
        _deep_cleanup()

        for i, shot in enumerate(plan.shots):
            shot_prompt = shot.prompt
            _audio = ShotAudio(plan.shots[i].speech, plan.shots[i].sounded, plan.shots[i].voiced_only,
                               bool(silence_nonspeech), speech_lead_seconds,
                               AUDIO_LATENT_FPS, shot.line_seconds, speech_tail_seconds,
                               shot.frame_count)
            silent = _audio.pinned

            # A shot that follows a removal starts FRESH. Every shot is anchored to
            # the previous one's last frame, so if the model did not finish taking
            # the garment off inside its own shot, that frame still shows it -- and a
            # keyframe is a PICTURE, which outvotes any sentence. Inherit it once and
            # every later shot inherits it too, with no wording able to undo it.
            # Breaking the chain at the one boundary where the state changes costs a
            # cut exactly where a cut belongs.
            shot_handoff = handoff
            _handoff_ref = False
            if restart_after_removal and (i - 1) in stripped_shots:
                shot_handoff = None
                fresh.append(i + 1)
            # ...and so does a shot that OPENS IN A DIFFERENT ROOM. Before the
            # _placed_shots branch on purpose: that one DEMOTES the frame to a
            # reference claiming "this room a moment earlier", which is a lie when the
            # room has changed. See cut_shots.
            elif i in cut_shots:
                shot_handoff = None
                fresh_room.append(i + 1)
            # SHOT 1'S first_frame, READ AS THE SET. Same answer as the branch below
            # and for the same reason -- a keyframe is a picture, and the people the
            # beat places are not in this one -- reached separately because shot 1 has
            # no previous shot to ask about. See where _first_is_plate is decided.
            elif i == 0 and _first_is_plate and shot_handoff is not None:
                _handoff_ref = True
                _plate_on = i + 1
            # ...and so does a shot that INTRODUCES somebody already in position.
            #
            # Same reasoning, same evidence. The keyframe is the previous shot's last
            # frame, and a character appearing for the first time is not in it. The
            # beat says where they are; the picture says they are nowhere. The picture
            # wins, so the model starts from a frame without them and has to put them
            # there during the shot -- which renders as the person arriving out of
            # nothing and then travelling to the spot the beat described.
            #
            # Only when the beat does NOT stage an entrance. "Dan walks in through the
            # side door" is a person who SHOULD arrive, and continuing from the frame
            # before is exactly right there. "Dan is already sitting on the crate" is
            # a person who should be there at the first frame, and there is no frame to
            # inherit that has him in it.
            # The frame is still the right picture of the ROOM, though, and throwing
            # it away is what build_conditioning's own note warns about: with no
            # handoff the VLM is never shown where the shot left off and re-imagines
            # the scenery -- same place, new room. So it is DEMOTED rather than
            # dropped. As a reference it carries the walls, the light and the people
            # already there without being frame one, so the newcomer is simply in
            # place instead of walking in from nowhere.
            #
            # Only when everybody in that frame is named in this shot. The picture
            # contains whoever was on screen when it was taken, and one the prompt
            # cannot account for is the node's oldest bug: a picture nobody claims is
            # another person. When it cannot be claimed, the old fresh start stands.
            elif i in _placed_shots:
                _was_here = [n for n in (plan.shots[i - 1].cast) if n]
                _here_now = plan.shots[i].cast
                # ...and NOT when somebody in that frame already has a portrait of
                # their own in this shot. Their identity is carried by that
                # picture; the carried frame would be a SECOND picture of the same
                # person, and two pictures of one person is how a duplicate of her
                # gets drawn. Reported as a duplicate Mistress: her sheet portrait
                # went in as <Picture 1> and this frame as <Picture 2>, both of
                # her. The recovered-frame path below skips tagged people for the
                # same reason and this was written without that skip.
                #
                # The room is lost on those shots, back to the fresh start it was
                # before. A re-imagined set is a smaller bug than a second person.
                if _cond_module.may_carry_room(_was_here, _here_now, _tagged_names):
                    _handoff_ref = True
                    _carried.append((i + 1, list(_was_here),
                                     list(_placed_shots[i])))
                else:
                    shot_handoff = None
                    fresh.append(i + 1)

            # SOMEBODY BACK AFTER A SHOT AWAY, with no picture of them anywhere.
            #
            # This shot starts from the previous shot's last frame, and they were not
            # in that shot -- so nothing pictorial carries their appearance and the
            # sheet text is on its own. A frame from the last shot they WERE in fixes
            # that, and the node has one: it rendered it.
            #
            # Narrow on purpose. Only when this shot describes that person ALONE,
            # because the recovered frame contains whoever else was on screen when it
            # was taken, and an unexplained person in a reference is how a second one
            # gets drawn. A multi-character return is reported and left alone.
            #
            # Skipped for anyone with a <Picture N> tag: their own reference already
            # travels into every shot they are named in, and a second picture of the
            # same person is just a second picture.
            _extra = []
            _cast = plan.shots[i].cast
            _returning = {w for n, ws in _returns if n == i + 1 for w in ws}
            _who = _cond_module.recoverable_subject(
                _cast, _tagged_names, _returning, _captured)
            if _who:
                _extra = [_captured[_who]]
                _recovered.append((i + 1, _who, _captured_from.get(_who, 0)))
                    # CLAIM IT IN THE PROSE. A picture the prompt refers to is that
                    # subject; one it never mentions is ANOTHER subject. Sent
                    # unclaimed, a recovered frame of somebody is read as a second
                    # person who looks exactly like them -- same face, same clothes --
                    # standing beside the one the beat asked for.
                    #
                    # Its number is its place in the roster: the shot's own references
                    # first, this after them. The handoff follows and stays unclaimed,
                    # which is H3's own first-frame shape.
                _n = len(shot.refs) + 1
                _tag = f"<Picture {_n}>"
                if f"{_who}:" in shot_prompt:
                    shot_prompt = shot_prompt.replace(
                        f"{_who}:", f"{_who}: {_tag},", 1)
                else:
                    shot_prompt = f"{shot_prompt} {_who} is the person in {_tag}."
            # The handoff, when it is demoted to a reference, is a picture like any
            # other and has to be claimed or it reads as a second person. Decided
            # here rather than inside build_conditioning because the claim is text,
            # and the text is assembled up here.
            _shot_refs = list(shot.refs) + _extra
            if _handoff_ref and _plate_on == i + 1:
                # A SET, not a room a moment earlier. See plate_claim.
                shot_prompt = shot_prompt + plate_claim(len(_shot_refs) + 1)
                _handoff_claimed.append(i + 1)
            elif _handoff_ref:
                # Carried for the ROOM, with somebody new in the shot -- so the
                # standing claim is exactly wrong here ("joined by anybody new") and
                # this one names the room, who was in it, and who is also here.
                _was, _join = next(((w, j) for s, w, j in _carried if s == i + 1),
                                   ([], []))
                shot_prompt = shot_prompt + room_claim(len(_shot_refs) + 1, _was, _join)
                _handoff_claimed.append(i + 1)
            elif handoff_rides_as_ref(shot_handoff, _shot_refs, ref_noise_aug):
                shot_prompt = shot_prompt + handoff_claim(len(_shot_refs) + 1)
                _handoff_claimed.append(i + 1)
            # Whatever this shot ends up being, that is what `script` reports.
            shot.prompt = shot_prompt
            cond, latent, fc, demoted = build_conditioning(
                clip, vae, audio_vae, shot_prompt, w, h, shot.frame_count,
                handoff=shot_handoff, refs=list(shot.refs) + _extra,
                ref_noise_aug=ref_noise_aug, silent=silent,
                handoff_as_ref=_handoff_ref,
                speech_lead_seconds=(_audio.lead_frames / AUDIO_LATENT_FPS),
                speech_tail_frames=_audio.tail_frames)
            if (demoted and not _aug_warned and ref_noise_aug is not None
                    and float(ref_noise_aug) < KEYFRAME_SAFE_AUG):
                _aug_warned = True
                notes.append(
                    f"ref_noise_aug is {float(ref_noise_aug):g}, below {KEYFRAME_SAFE_AUG:g} -- "
                    f"one aug covers references AND the keyframe, so at this value the "
                    f"anchor would be noised and mis-timestepped, and every shot after the "
                    f"first degrades while sampling. The handoff is riding as an extra "
                    f"reference instead: continuity is weaker but nothing is corrupted. "
                    f"Raise it to {KEYFRAME_SAFE_AUG:g}+ for a real keyframe")
            _evict_all_but(model, latent)
            try:
                _t0 = time.perf_counter()
                out = sample_shot(model, cond, negative, latent, seed, steps, cfg,
                                  sampler_name, scheduler, sigmas,
                                  shift_video, shift_audio, _soft_landing)
                t_sample += time.perf_counter() - _t0
            except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                if not _is_oom(e):
                    raise
                raise RuntimeError(
                    f"H3-LongVideos: shot {i + 1} of {len(plan)} ran out of VRAM while "
                    f"sampling. " + sampling_oom_help(w, h, fc, H3_FPS, megapixels)) from e

            # The video latent, for the latent upscale below. NOT used as the next
            # shot's keyframe -- see _keyframe_latent for why that failed.
            try:
                parts = out["samples"].unbind() if hasattr(out["samples"], "unbind") else None
            except Exception:
                parts = None

            # LATENT upscale, between sampling and decode: the shot is SAMPLED small
            # and only DECODED large, which is where the saving is -- cost scales with
            # latent cells and attention is quadratic in them. Note the handoff latent
            # was taken ABOVE, before this: the chain must inherit the sampled latent,
            # not the upscaler's reinterpretation of it, or that guess compounds.
            shot_tiled = tiled_decode
            pre_up = None            # the SAMPLED video latent, when upscaling ran
            if latent_upscale and latent_upscale != "off" and parts and len(parts) == 2:
                vid_up, up_note = upscale_video_latent(parts[0], latent_upscale,
                                                       latent_upscale_scale)
                if vid_up is not parts[0]:
                    pre_up = parts[0]
                    out["samples"] = comfy.nested_tensor.NestedTensor((vid_up, parts[1]))
                    shot_tiled = True      # a 2x latent is ~4x the decode memory
                if up_note and up_note not in notes:
                    notes.append(up_note)

            _t0 = time.perf_counter()
            # The DiT goes so the decode fits; the two VAEs stay, because both are
            # used in the next two lines and evicting them only buys a reload.
            imgs = _decode_video(vae, out, shot_tiled, free_first=model,
                                 keep=(vae, audio_vae))
            wav = _decode_audio(audio_vae, out)
            t_decode += time.perf_counter() - _t0
            sr = wav["sample_rate"]
            del out

            # The chain must not inherit the UPSCALER's reinterpretation. The shot's
            # own frames stay upscaled, but the handoff comes from the sampled latent
            # -- otherwise every boundary hands on an upscaled-then-downscaled frame,
            # and eleven shots of that compounds into colour cast and mush.
            hand_src = imgs
            if pre_up is not None:
                try:
                    n = min(int(pre_up.shape[2]), HANDOFF_LATENT_TAIL)
                    tail = _decode_video(vae, {"samples": pre_up[:, :, -n:].contiguous()},
                                         True)
                    if tail is not None and tail.shape[0] > 0:
                        hand_src = tail
                except Exception:
                    pass                  # fall back to the upscaled frames
            # MEASURE FIRST, on the uncorrected frames. shot_handoff is the keyframe this
            # shot was given and imgs[0] is what came back in its place -- two pictures of
            # the same frame, so what separates them is the chain and not the author. The
            # last two arguments put the pre-upscale handoff and the post-upscale output in
            # one frame of reference; with latent_upscale off they are the same frame and
            # the term is zero. A demoted handoff is skipped: it rode as a reference, so
            # imgs[0] was never asked to reproduce it.
            try:
                if (shot_handoff is not None and not demoted and imgs is not None
                        and imgs.shape[0] > 1 and hand_src is not None and hand_src.shape[0]):
                    _levels.observe(shot_handoff, imgs[0], imgs[-1], hand_src[-1])
            except Exception:
                pass
            # Then correct, on a REBINDING -- imgs itself is untouched, so the frames the
            # viewer sees are the ones the model made. Everything that leaves this shot for
            # a later one comes off hand_src, so the handoff and any captured face take the
            # same grade from the same call.
            _lv_note = ""
            try:
                if hold_levels > 0 and hand_src is not None and hand_src.shape[0]:
                    _lg, _lo = _levels.gains(hold_levels)
                    if _lg is not None:
                        hand_src = apply_levels(hand_src, _lg, _lo)
                        _lv_note = _levels.note(_lg, _lo)
            except Exception:
                pass
            # Clamp before it becomes a keyframe. A decode can land slightly outside
            # 0..1, and feeding that back in to be re-encoded every boundary is a
            # drift that accumulates rather than cancels.
            handoff = hand_src[-1:].detach().clamp(0.0, 1.0).to("cpu", copy=True)
            # Keep a frame for the shot they come back on -- but ONLY from a shot that
            # was theirs alone.
            #
            # A frame is a picture of everyone who was in it. Captured from a shot with
            # two people and sent later as a reference, it brings the other one back
            # into a shot that does not call for them. That is the second character
            # turning up uninvited, and it was this code: the destination was guarded
            # (the return shot has to describe one person) and the SOURCE was not.
            #
            # The MIDDLE frame, not the last: somebody walking out during the shot is
            # gone by the last frame -- which is the whole failure -- and somebody
            # walking in is missing from the first.
            #
            # ...and not from a shot whose WARDROBE is unusual. A captured frame is
            # sent later as a subject reference, and a reference outranks the sheet:
            # it is a picture of what the person looks like. Captured where a garment
            # was displaced, removed, or newly uncovered, it is a picture of them
            # dressed differently from the sheet -- and the shot that receives it
            # renders the garment the way the PICTURE has it, which is a garment the
            # prompt never described. Reported as clothing invented several shots in,
            # because that is exactly when a recovery first fires.
            # Read from the per-shot records, NOT from the text loop's own variables:
            # that loop finished long before this one started, so its `toks` and
            # `displaced` hold the last shot's values for every shot down here, and
            # `_bare` has since been reused for something else entirely.
            #
            # moved_shots holds the shots that CARRY the displacement guard, which
            # starts the shot AFTER the one that stages it -- and the staging shot is
            # the worst one to capture from, since the garment is being moved on
            # screen in it. Its own beat is what says so.
            _n = i + 1
            _wardrobe_normal = not (i in stripped_shots
                                    or _n in moved_shots
                                    or _n in revealed_shots
                                    or _n in bared_shots
                                    or _n in staging_shots)
            try:
                if (hand_src.shape[0] and len(plan.shots[i].cast) == 1 and _wardrobe_normal):
                    _mid = hand_src.shape[0] // 2
                    _keep = hand_src[_mid:_mid + 1].detach().clamp(0.0, 1.0).to(
                        "cpu", copy=True)
                    for _who in plan.shots[i].cast:
                        _captured[_who] = _keep
                        _captured_from[_who] = i + 1
            except Exception:
                pass                       # a recovered frame is a nicety, not the render
            del hand_src
            # TRIM ONLY WHERE THERE WAS A KEYFRAME TO DUPLICATE.
            #
            # The first frame of a shot is dropped because it is "the model's own
            # reproduction of the keyframe, so it is a duplicate" -- and that is true
            # only of a shot that OPENED on one. Three paths above leave a shot with
            # no keyframe: restart_after_removal breaks the chain after a garment
            # comes off, a character introduced already in position demotes the
            # handoff to a reference, and the same case unclaimable drops it. On those
            # shots the first frame is not a reproduction of anything -- it is the
            # genuine opening frame of a deliberate cut -- and trimming it threw away
            # real footage AND removed the one frame nearest the shot before it.
            # Reported as the last frame and the first frame of the next beat not
            # matching up, which is exactly what it looks like: the bridge frame is
            # gone and what meets the cut is frame two.
            #
            # `demoted` is build_conditioning's own answer to "did the handoff ride as
            # a reference instead of anchoring frame one", so this asks the question
            # of the code that decided it rather than re-deriving the three cases and
            # drifting from them. The audio trim moves with the video trim or the two
            # come apart by a frame.
            if trim_seam and i > 0 and shot_handoff is not None and not demoted:
                imgs = imgs[1:]
                wav["waveform"] = wav["waveform"][..., max(0, round(sr / H3_FPS)):]
            elif trim_seam and i > 0:
                _untrimmed.append(i + 1)
            # Make the sound exactly as long as the picture it belongs to.
            #
            # The audio latent count is round(frames / 24 * 40), which lands exactly
            # only when the frame count divides by 3 -- so most of H3's 17k+5 grid
            # leaves a shot's audio 8.3 ms longer or shorter than its video. On its own
            # that is inaudible. Concatenated it is not: with shots of equal length the
            # error carries the same sign every time and adds up, and eleven 73-frame
            # shots finish 92 ms out, which is plainly visible on a mouth.
            #
            # Correcting per shot rather than once at the end keeps every cut aligned
            # too, instead of only the final duration.
            want = int(round(imgs.shape[0] * sr / H3_FPS))
            have = int(wav["waveform"].shape[-1])
            if have > want:
                wav["waveform"] = wav["waveform"][..., :want]
            elif have < want:
                shape = list(wav["waveform"].shape)
                shape[-1] = want - have
                wav["waveform"] = torch.cat(
                    [wav["waveform"], torch.zeros(shape, dtype=wav["waveform"].dtype,
                                                  device=wav["waveform"].device)], dim=-1)
            av_fix += have - want
            # Measured on the frame that becomes the next shot's keyframe, because that is
            # the one whose losses are inherited -- which means the CORRECTED handoff, not
            # imgs[-1]. Measured on imgs[-1] the line would report the defect for ever and
            # never show whether the correction worked.
            try:
                if handoff is not None and handoff.shape[0]:
                    shot_detail.append(frame_detail(handoff[0]))
                elif imgs is not None and imgs.shape[0]:
                    shot_detail.append(frame_detail(imgs[-1]))
            except Exception:
                pass
            frames.add(imgs)
            aud_out.append(wav["waveform"].to("cpu", copy=True) if cleanup_between_shots
                           else wav["waveform"])
            del imgs, wav
            if cleanup_between_shots:
                _deep_cleanup()

        if _untrimmed:
            notes.append(
                f"shot(s) {', '.join(str(n) for n in _untrimmed)} kept their FIRST frame "
                f"even though trim_seam is on, because they did not open on a keyframe. "
                f"The trim exists to drop a duplicate -- the model's own reproduction of "
                f"the frame it was handed -- and these shots were handed none: the chain "
                f"is broken deliberately after a removal, and a character introduced "
                f"already in position gets the previous frame as a REFERENCE rather than "
                f"as frame one. Their first frame is the real opening frame of a cut, so "
                f"trimming it threw away footage and left frame TWO meeting the shot "
                f"before -- reported as the last frame and the first frame of the next "
                f"beat not matching up. The audio is trimmed with the picture or not at "
                f"all, so the two cannot come apart")
        if _handoff_claimed:
            notes.append(
                f"ref_noise_aug is below {KEYFRAME_SAFE_AUG:g}, so on shot(s) "
                f"{', '.join(str(n) for n in _handoff_claimed)} the handoff is encoded as "
                f"a reference rather than a keyframe, and the text now NAMES it as the "
                f"frame the shot opens on. Unnamed it was a picture of the previous shot "
                f"-- the same people, a moment earlier -- sitting in the reference rows "
                f"with nothing claiming it, and a picture the prompt never names is read "
                f"as another subject. That is a duplicate of whoever was on screen, "
                f"appearing on the later shots because those are the ones with both a "
                f"handoff and a reference. Raising ref_noise_aug to {KEYFRAME_SAFE_AUG:g} "
                f"or above keeps the handoff a keyframe and the question does not arise")
        if _recovered:
            notes.append(
                "recovered a face for "
                + "; ".join(f"{who} on shot {n}, from shot {src}"
                            for n, who, src in _recovered)
                + ". They were back after a shot away with no picture of them anywhere "
                  "-- the keyframe is the previous shot's last frame and they were not "
                  "in it -- so a frame from the middle of the last shot that was THEIRS "
                  "ALONE was sent as a reference. The middle, because somebody walking "
                  "out is gone by the last frame and somebody walking in is missing from "
                  "the first. Both ends have to be solo: a frame is a picture of "
                  "everyone in it, so one taken from a shared shot would carry the other "
                  "person into a shot that does not call for them. A character never on "
                  "screen alone gets nothing, which beats importing somebody. Skipped "
                  "for anyone with a <Picture N> tag of their own. The frame is "
                  "CLAIMED on their sheet entry for that shot -- a picture the "
                  "prompt never refers to is read as another subject, so an "
                  "unclaimed one would arrive as a second person with the same "
                  "face and the same clothes. `script` is written before the render, so it does not show that tag")
        # FrameAccumulator writes each decoded shot directly into the finished chain.
        # Its overflow path covers malformed VAE output without making the normal
        # path allocate and concatenate a second full copy.
        video = frames.finish()
        # PIXEL upscale, once, on the finished chain. After the latent pass and after
        # the join, so a model-based upscaler sees whole frames and the seam is not
        # upscaled twice.
        if upscale and upscale != "off":
            video, up_note = _upscale_frames(video, upscale, upscale_model,
                                             upscale_target_short_edge, upscale_batch)
            if up_note:
                notes.append(up_note)
        # AUDIO IS FLOAT32 WHATEVER THE FRAMES ARE, and this is the one place the two
        # branches must not follow the same rule. --fp16-intermediates is a good trade
        # on pixels and a bad one on a waveform, because what each is quantised to at
        # the end is not the same:
        #
        #   images  0..1, out at 8 bits : fp16 step 2.4e-04 against 3.9e-03 -- 16x finer
        #                                 than the output can show. Invisible.
        #   audio  -1..1, out at 16 bits: fp16 step 2.4e-04 against 3.1e-05 -- 8x
        #                                 COARSER than the format. ~12 effective bits.
        #
        # And it buys nothing: the frames are 9.3GB of the chain and the whole
        # soundtrack is 0.018GB, so holding it at full width costs 18MB of the 58.9GB
        # that made this render fit. The bed is mixed onto this AFTER the join and the
        # levelling runs over the joined track, so a narrow accumulator is not merely
        # stored coarse, it is added up coarse.
        audio = torch.cat(aud_out, dim=-1)
        if audio.dtype != torch.float32:
            audio = audio.float()
        # ...and a wired ambient file goes on last, over the joined soundtrack rather
        # than per shot, so it runs continuously through the cuts instead of
        # restarting at each one. A bed that resets every shot is a bed you can hear.
        #
        # A WIRED FILE IS ALL THAT GOES ON NOW. The node used to BUILD this, out of
        # the scene's own wording, and build the shot-by-shot foley too -- see the
        # note at the top of audio.py for what that was and why it is gone. Removed
        # on the report: "Just get rid of the ambient sounds all together. They sound
        # horrid. Go back to the model's natural audio."
        #
        # What is played here is the user's own recording, which is a different thing
        # from ambience the node invents: it is not synthesised, it conditions
        # nothing, and it cannot put a voice in a wordless shot.
        _bed_in, _built = ambient_audio, ""
        if _bed_in is None and float(ambient_level or 0.0) > 0.0:
            notes.append(
                f"ambient_level is {float(ambient_level):.2f} and nothing is wired to "
                f"ambient_audio, so no bed went under the soundtrack -- and that is "
                f"now the only way to get one. The node used to BUILD a room tone out "
                f"of the scene's wording, and a layer of foley into every shot pinned "
                f"to silence; both are gone, because they were reported as sounding "
                f"horrid and synthesis that measures right and sounds wrong is the end "
                f"of that road. The audio is the model's, whole. This widget still "
                f"sets the level for a recording you wire yourself, which is played "
                f"under the finished track and conditions nothing")
        if auto_sound and float(foley_level or 0.0) > 0.0:
            notes.append(
                f"foley_level is {float(foley_level):.2f} and does nothing any more. "
                f"It set how loud the sounds this node BUILT were -- a click, a "
                f"rattle, a rustle, mixed into the shots whose audio branch is pinned "
                f"to silence, because prompt text can never open a branch and those "
                f"shots could not make their own. That is removed: the soundtrack is "
                f"the model's. The widget stays at this position because saved "
                f"workflows restore values by position and shifting it would load the "
                f"wrong number into every widget after it. "
                f"THE CONSEQUENCE, said rather than left to be found: a shot with no "
                f"line and no sound you described is pinned to silence and is SILENT. "
                f"The pin is deliberate and untouched -- it is what stops a free "
                f"branch filling itself with a voice and the face lip-syncing to the "
                f"babble. To put sound in such a shot, write the sound into that beat, "
                f"which opens its branch on purpose and lets the model make it; or "
                f"wire a track to ambient_audio; or lay one under the finished video "
                f"outside the node")
        audio, _bed_note = mix_ambient(audio, sr, _bed_in, ambient_level)
        if _bed_note:
            notes.append(_built + _bed_note if _built else _bed_note)
        total = video.shape[0]
        # The finished chain is the largest thing this node holds, and it competes with
        # the MODELS for system RAM: ComfyUI offloads weights to RAM rather than
        # discarding them, so a shot boundary is a PCIe copy while that RAM is there
        # and a disk read once the frames have crowded the weights out.
        if cleanup_between_shots and total:
            # MEASURED off the tensor, not assumed. This said "* 4" for float32 while
            # the chain was float16 during the render and, since the join started
            # asking ComfyUI what dtype it wants, may be float16 when it is returned
            # too -- so a fixed width here is a number that is wrong on one install
            # or the other. element_size() is right on both.
            _bytes = video.element_size()
            _held = total * int(w) * int(h) * 3 * _bytes / GB
            _dt = "float16" if _bytes == 2 else "float32"
            if _held >= 2.0:
                notes.append(
                    f"the finished chain is {_held:.1f}GB in system RAM ({total} frames at "
                    f"{w}x{h}, {_dt}). It "
                    f"shares that RAM with the models, which ComfyUI offloads to it "
                    f"rather than discarding: while they fit, a shot boundary is a PCIe "
                    f"copy; once the frames crowd them out it becomes a disk read, once "
                    f"per model per shot. If the machine is thrashing, the levers are "
                    f"fewer frames per run (lower shot_seconds, or split a long script "
                    f"and join the parts outside the node), a lower megapixels, or a "
                    f"smaller diffusion quant -- every GB of weights is a GB not "
                    f"available to hold the render")
        if fresh:
            notes.append(
                f"shot(s) {', '.join(str(n) for n in fresh)} start fresh, because the shot "
                f"before each took something off -- continuing from a frame that may still "
                f"show the garment is how it comes back, and a picture outvotes the text. "
                f"That costs a cut there. Turn restart_after_removal off to keep the "
                f"continuity instead")
        if _carried:
            notes.append(
                "; ".join(
                    f"shot {s} carries the previous frame as a REFERENCE rather than "
                    f"as its first frame, so the room, the light and "
                    f"{' and '.join(w)} come with it while "
                    f"{' and '.join(j)} {'are' if len(j) > 1 else 'is'} already in "
                    f"place instead of walking in"
                    for s, w, j in _carried)
                + " -- a keyframe is frame one and a reference is not, which is what "
                  "lets a shot introduce somebody without re-imagining the room")
        wall = time.perf_counter() - t_start
        n = max(1, len(plan))
        other = max(0.0, wall - t_sample - t_decode)
        notes.append(
            f"rendered {total} frames (~{total / H3_FPS:.1f}s) in {wall:.0f}s -- "
            f"sampling {t_sample:.0f}s ({100 * t_sample / wall:.0f}%), "
            f"decode {t_decode:.0f}s ({100 * t_decode / wall:.0f}%), "
            f"other {other:.0f}s ({100 * other / wall:.0f}%); "
            f"per shot {t_sample / n:.1f}s + {t_decode / n:.1f}s")
        if av_fix:
            per_shot = abs(av_fix) / sr * 1000 / max(1, len(plan))
            notes.append(
                f"audio realigned to the picture by ~{abs(av_fix) / sr * 1000:.0f} ms "
                f"across {len(plan)} shot(s), {per_shot:.1f} ms each. H3's audio latent "
                f"runs at {AUDIO_LATENT_FPS}/s against {H3_FPS} fps video, so a shot's "
                f"sound lands exactly only when its frame count divides by 3 -- otherwise "
                f"it is up to 8.3 ms out, with the same sign every time when the shots "
                f"are the same length, which is how a chain drifts out of sync"
                + (". That is far more than the 8.3 ms the grid accounts for, so the "
                   "audio VAE is not returning the length its latent implies -- check "
                   "that the audio VAE is H3's own converted one"
                   if per_shot > 50 else ""))
        _detail = detail_report(shot_detail)
        if _detail:
            notes.append(_detail)
        _lvl = levels_report(_levels, len(plan.shots))
        if _lvl:
            notes.append(_lvl)
        if t_decode > t_sample:
            notes.append("decode is costing more than sampling here -- latent_upscale "
                         "trades cheaper sampling for a 4x more expensive decode, so it "
                         "is the wrong way round at this step count. megapixels is the "
                         "lever that lowers both")
        script = "\n---\n".join(f"[Shot {i}] {s}" for i, s in enumerate(plan.prompts, 1))
        # Whether the silence conditioning ACTUALLY went on. Reported from the
        # result, not from the flag: every failure inside _silent_audio_latent
        # returns None on purpose so a render never dies for a nicety, but that
        # meant a shot with a wide-open audio branch was described as "conditioned
        # on real silence" -- and a shot with no scripted line babbled with nothing
        # in the report saying why. This is the one note that has to come after the
        # loop, because before it there is no result to report.
        if _SILENCE_STATUS["asked"]:
            _missed = _SILENCE_STATUS["asked"] - _SILENCE_STATUS["applied"]
            if _missed > 0:
                notes.append(
                    f"SILENCE WAS ASKED FOR ON {_SILENCE_STATUS['asked']} shot(s) AND "
                    f"WENT ON {_SILENCE_STATUS['applied']}: {_missed} shot(s) have no "
                    f"working audio lock, because "
                    f"{_SILENCE_STATUS['why'] or 'the silent latent could not be built'}"
                    f". H3 is joint, so an unconditioned branch invents a voice and the "
                    f"picture lip-syncs to it -- a shot babbling with nothing scripted "
                    f"to say. The lips-closed sentence is still in the prompt and still "
                    f"loses to the stream")
            else:
                notes.append(
                    f"silence went on all {_SILENCE_STATUS['applied']} shot(s) that "
                    f"asked for it -- the requested full shot or dialogue lead-in is "
                    f"pinned to encoded silence, not merely told to be quiet")
        return (video, {"waveform": audio, "sample_rate": sr}, " | ".join(notes), script,
                plan.shots[0].frame_count, total, len(plan), round(total / H3_FPS, 2))


_NODE_IDS = ("H3LongVideos", "H3LongVideosFL2VA", "H3LongVideosV1",
             "H3LongVideosREF2VA")
NODE_CLASS_MAPPINGS = {name: H3LongVideos for name in _NODE_IDS}
NODE_DISPLAY_NAME_MAPPINGS = {name: "H3-LongVideos" for name in _NODE_IDS}
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
