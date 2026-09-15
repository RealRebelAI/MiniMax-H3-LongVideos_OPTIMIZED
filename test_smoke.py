"""Smoke test: run the whole render path with fake models.

Nothing here touches a GPU, loads a checkpoint or asks the real model for
anything -- the model, CLIP and both VAEs are stubs that return correctly shaped
tensors. What it exercises is the code the node runs on every render: the shot
loop, conditioning assembly, the keyframe handoff, decode, trim and concat.

This exists because a missing module-level name shipped once and only showed up
as a traceback on the user's first render. Parsing and unit-testing pure
functions did not catch it; running the path does.

Run: python test_smoke.py
"""

import importlib.util
import io
import os
import itertools
import re
import sys
import types

import torch

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# --- stub the ComfyUI surface the node imports ------------------------------
for _n in ("nodes", "comfy", "comfy.utils", "comfy.sample", "comfy.samplers",
           "comfy.nested_tensor", "comfy.model_management", "latent_preview", "node_helpers"):
    sys.modules.setdefault(_n, types.ModuleType(_n))
_c = sys.modules["comfy"]
for _sub in ("utils", "sample", "samplers", "nested_tensor", "model_management"):
    setattr(_c, _sub, sys.modules["comfy." + _sub])

sys.modules["comfy.samplers"].KSampler = type("K", (), {"SAMPLERS": ["res_multistep"],
                                                        "SCHEDULERS": ["simple"]})
sys.modules["comfy.samplers"].sampler_object = lambda name: object()


class FakeNested:
    """Stands in for comfy.nested_tensor.NestedTensor: a (video, audio) pair."""
    is_nested = True

    def __init__(self, parts):
        self.parts = tuple(parts)

    def unbind(self):
        return self.parts


sys.modules["comfy.nested_tensor"].NestedTensor = FakeNested

_mm = sys.modules["comfy.model_management"]
_mm.intermediate_device = lambda: torch.device("cpu")
_mm.get_torch_device = lambda: torch.device("cpu")
_mm.free_memory = lambda *a, **k: None
_mm.soft_empty_cache = lambda *a, **k: None
_mm.unload_all_models = lambda *a, **k: None
_mm.loaded_models = lambda *a, **k: []

sys.modules["comfy.utils"].PROGRESS_BAR_ENABLED = False
sys.modules["comfy.utils"].common_upscale = (
    lambda s, w, h, method, crop: torch.nn.functional.interpolate(s, size=(h, w)))
sys.modules["latent_preview"].prepare_callback = lambda *a, **k: None
sys.modules["node_helpers"].conditioning_set_values = (
    lambda cond, vals: [[c[0], {**c[1], **vals}] for c in cond])

_HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("h3smoke", os.path.join(_HERE, "sampler.py"))
S = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(S)

_fails = []


def check(label, ok, detail=""):
    print(("  PASS  " if ok else "  FAIL  ") + label + (f"  [{detail}]" if detail and not ok else ""))
    if not ok:
        _fails.append(label)


# --- fakes ------------------------------------------------------------------

W, H, FRAMES = 128, 96, 39            # small, on the 17k+5 grid
# A scene and one wordless beat. Kept as a constant because it is checked twice, with
# the mouth guard on and off, and the two spellings must not drift apart.
TWO_LINE_ROOM = "A room.\n\nHe walks in."


class FakeCLIP:
    def __init__(self):
        self.seen = []

    def tokenize(self, prompt, minimax_ref_items=None):
        self.seen.append((prompt, list(minimax_ref_items or [])))
        return {"prompt": prompt}

    def encode_from_tokens_scheduled(self, tokens):
        return [[torch.zeros(1, 8, 16), {}]]


def _vae_out_dtype():
    """What a real VAE.decode hands back: VAE.vae_output_dtype() IS
    model_management.intermediate_dtype() (comfy/sd.py). The stubs returned fp32
    unconditionally, so every assertion about what the node does with the VAE's
    dtype was passing on a fixture that could not produce the interesting case."""
    try:
        return _mm.intermediate_dtype()
    except Exception:
        return torch.float32


class FakeVAE:
    upscale_ratio = (4, 8, 8)
    latent_dim = 3

    def __init__(self):
        self.encodes = 0

    def encode(self, image):
        self.encodes += 1
        n = image.shape[0]
        return torch.zeros(1, 24, max(1, n), H // 16, W // 16)

    def decode(self, latent):
        # H3'S OWN GRID, not t*4. The real VAE's upscale_ratio is
        # (lambda a: max(1, (a - 2) // 5 * 17 + 5), 16, 16) -- comfy/sd.py -- so a
        # shot planned at 73 frames encodes to 22 latents and decodes back to 73.
        # t*4 gave 88, and nothing noticed because nothing asserted the round trip.
        #
        # It matters now: the shot loop is about to preallocate the finished chain
        # from sum(lens), which is only a valid bound if a shot decodes to the
        # length it was planned at. Against t*4 every shot overruns by 20%, the
        # whole suite takes the overflow path, and a disable-check on the streaming
        # fix would read "the fix does not work" when the fixture is what is wrong.
        t = latent.shape[2] if latent.ndim == 5 else 1
        return torch.rand(max(1, (t - 2) // 5 * 17 + 5), H, W, 3).to(_vae_out_dtype())

    def decode_tiled(self, latent, **kw):
        return self.decode(latent)


class FakeAudioVAE:
    upscale_ratio = 512
    latent_dim = 2
    audio_sample_rate = 32000
    audio_sample_rate_output = 44100

    class _M:
        latents_mean = torch.zeros(32)
        latents_std = torch.ones(32)

    first_stage_model = _M()

    def encode(self, wav):
        return torch.zeros(1, 32, 2, 16)

    def decode(self, latent):
        # Real layout is [B, L, C]; _decode_audio movedim's it to [B, C, L].
        n = latent.shape[-1] if latent.ndim >= 2 else 16
        return torch.rand(1, n * 800, 2).to(_vae_out_dtype())


class FakeModel:
    def clone(self):
        return self

    def get_model_object(self, name):
        class MS:
            def set_parameters(self, shift=1.0, **kw):
                self.shift = shift
        return MS()

    def add_object_patch(self, *a, **k):
        pass

    model = types.SimpleNamespace(model_config=types.SimpleNamespace(__class__=object))


def fake_ksampler(model, seed, steps, cfg, sn, sch, positive, negative, latent, denoise=1.0):
    """Return a latent shaped exactly as _empty_av_latent built it."""
    v, a = latent["samples"].unbind()
    out = dict(latent)
    out["samples"] = FakeNested((v.clone(), a.clone()))
    return (out,)


sys.modules["nodes"].common_ksampler = fake_ksampler


def run_node(prompt, **kw):
    node = S.H3LongVideos()
    args = dict(model=FakeModel(), clip=FakeCLIP(), vae=FakeVAE(), audio_vae=FakeAudioVAE(),
                prompt=prompt, resolution="4:3", megapixels=0.0,
                shot_seconds=FRAMES / S.H3_FPS, steps=2, cfg=1.0,
                sampler_name="res_multistep", scheduler="simple", seed=1,
                apply_model_sampling=False, tiled_decode=False)
    args.update(kw)
    # the preset is 1024x768; force the small canvas the fakes are built for
    S.NATIVE_RES["4:3"] = (W, H)
    return node.run(**args)


def test_independent_adult_arm_actions():
    print("\n=== separate adult arm actions stay separate across shots ===")
    memory = ("Maya: she, 38, green sweater, grey trousers.\n"
              "Owen: he, 42, blue shirt, brown trousers.")
    first = ("Maya raises her right hand to point at a painting. "
             "Owen keeps his hands at his sides.")
    second = ("Maya lowers her right hand to her side. "
              "Owen folds his arms across his chest.")
    clip = FakeCLIP()
    result = run_node("Maya and Owen stand in an art studio.\n\n"
                      + first + "\n\n" + second,
                      character_memory=memory, clip=clip)
    shots = result[3].split("\n---\n")
    check("both action beats reach rendering", len(shots) == 2)
    check("each character is described once in each shot",
          all(s.count("Maya:") == 1 and s.count("Owen:") == 1 for s in shots))
    check("raising and lowering stay in their own shots",
          first in shots[0] and second not in shots[0]
          and second in shots[1] and first not in shots[1])
    check("no restraint-derived arm pose is invented",
          all("Both arms are" not in s and "wrists together" not in s for s in shots))
    visual_inputs = [items for prompt, items in clip.seen if prompt.strip()]
    check("no first-shot picture and only one continuity picture on shot two",
          [len(items) for items in visual_inputs] == [0, 1])
    check("the reported script matches the text actually encoded",
          [s.split("] ", 1)[1] for s in shots]
          == [prompt for prompt, _ in clip.seen if prompt.strip()])


def test_plan():
    print("\n=== plan_only ===")
    imgs, audio, info, script, fps_shot, total, shots, secs = run_node(
        "A room.\n\nHe walks in.\n\nShe follows.", plan_only=True)
    check("it reports without rendering", "PLAN ONLY" in info)
    check("it counts the shots", shots == 2, f"{shots}")
    check("the script shows both shots", script.count("[Shot ") == 2)
    check("the scene is prepended to each", script.count("A room.") == 2)


def test_render():
    print("\n=== full render path ===")
    try:
        imgs, audio, info, script, per_shot, total, shots, secs = run_node(
            "A room.\n\nHe walks in.\n\nShe follows him and says: \"Wait.\"")
    except Exception as e:
        import traceback
        traceback.print_exc()
        check("the render path runs end to end", False, f"{type(e).__name__}: {e}")
        return
    check("the render path runs end to end", True)
    check("frames come back", imgs.ndim == 4 and imgs.shape[0] > 0, str(tuple(imgs.shape)))
    check("audio comes back", audio["waveform"].ndim == 3)
    check("two shots were rendered", shots == 2, str(shots))
    check("the seam frame is trimmed from shot 2",
          imgs.shape[0] == total and total > 0)
    check("info is populated", bool(info))


def test_a_cut_keeps_its_own_first_frame():
    print("\n=== a shot that opens on no keyframe keeps frame one ===")
    # Reported: the last frame and the first frame of the next beat not matching up.
    # trim_seam drops a shot's first frame because it is "the model's own reproduction
    # of the keyframe, so it is a duplicate" -- true only of a shot that OPENED on one.
    # restart_after_removal breaks the chain after a garment comes off, so the next
    # shot opens on nothing at all and its first frame is the genuine opening frame of
    # a deliberate cut. Trimming it threw away real footage and left frame TWO meeting
    # the shot before, which is what the mismatch looks like.
    mem = "Mara: she, 30, a grey coat, white top."
    P = ("A room.\n\nMara stands by the window.\n\n"
         "Mara pulls off her coat.\nremove: coat\n\n"
         "Mara waits.")
    on = run_node(P, character_memory=mem, restart_after_removal=True, trim_seam=True)
    off = run_node(P, character_memory=mem, restart_after_removal=False, trim_seam=True)
    check("the kept frame is reported, not a silent change in the count",
          "kept their FIRST frame" in on[2], on[2][:160])
    check("...and a fully chained run says nothing about it",
          "kept their FIRST frame" not in off[2], "")
    check("a cut keeps the frame a keyframed shot would have dropped",
          on[5] == off[5] + 1, f"{on[5]} vs {off[5]}")
    check("the frames returned match the reported total",
          int(on[0].shape[0]) == on[5], f"{int(on[0].shape[0])} vs {on[5]}")

    def _per_frame(r):
        return r[1]["waveform"].shape[-1] / max(1, int(r[0].shape[0]))
    check("picture and sound are trimmed together",
          abs(_per_frame(on) - _per_frame(off)) < 2.0,
          f"{_per_frame(on):.2f} vs {_per_frame(off):.2f}")
    plain = run_node("A room.\n\nHe walks in.\n\nShe follows.", trim_seam=True)
    check("a fully chained script still trims every seam",
          "kept their FIRST frame" not in plain[2]
          and int(plain[0].shape[0]) == plain[5], str(plain[5]))


def test_a_room_change_with_no_walk_is_a_cut():
    print("\n=== a shot that opens in another room starts fresh ===")
    # Reported: at the end beat the living room turned into a bathroom. Every shot is
    # anchored to the previous shot's last frame and a keyframe is a PICTURE, which
    # outvotes any sentence -- so a living-room shot opening on a frame of the kitchen
    # renders neither, it renders a blend, and a kitchen blended with the words "living
    # room" is a bathroom: tiles, a sink, cabinets. There was no room-change case in the
    # keyframe decision at all; only a removal or a newly placed character broke it.
    mem = "McKenna: she, 22, a grey t-shirt, black shorts."
    P = ("A small flat at night. Her bedroom has an unmade bed and a lamp.\n\n"
         "McKenna gets up and comes out of her bedroom.\n\n"
         "McKenna walks down the hallway to the living room.\n\n"
         "McKenna goes from the living room to the kitchen.\n\n"
         "McKenna sits on the sofa in the living room.")
    info = run_node(P, plan_only=True, character_memory=mem)[2]
    check("the shot that jumps rooms is cut, and only it",
          "shot(s) 4 START FRESH" in info, info[:160])
    # A WALK IS NOT THIS: a travel beat opens in the room it is leaving.
    walk = run_node("A flat.\n\nMcKenna walks from the bedroom to the hallway.\n\n"
                    "McKenna walks from the hallway to the kitchen.",
                    plan_only=True, character_memory=mem)[2]
    check("a journey keeps its keyframe", "START FRESH" not in walk, "")
    # ...but a journey starting somewhere the last shot did not end is still a cut.
    jump = run_node("A flat.\n\nMcKenna is in the kitchen.\n\n"
                    "McKenna walks from the bedroom to the bathroom.",
                    plan_only=True, character_memory=mem)[2]
    check("a walk whose origin is not where we were is a cut",
          "START FRESH" in jump, jump[:160])
    one = run_node("A kitchen with a white table.\n\nMcKenna fills the kettle.\n\n"
                   "McKenna sits down.", plan_only=True, character_memory=mem)[2]
    check("a one-room script is untouched",
          "START FRESH" not in one and "never describes" not in one, "")
    # A room the prompt never describes is a room the model invents -- say so.
    check("an undescribed room is reported",
          "never describes" in info and "living room" in info, "")
    desc = run_node("A flat. The living room has a green sofa. The kitchen is small and white."
                    "\n\nMcKenna is in the living room.\n\nMcKenna is in the kitchen.",
                    plan_only=True, character_memory=mem)[2]
    check("a described room draws no warning", "never describes" not in desc, desc[:150])
    # The render path still runs when a shot opens on no keyframe.
    r = run_node(P, character_memory=mem)
    check("the render path runs through a room cut",
          int(r[0].shape[0]) == r[5] and r[5] > 0, str(r[5]))


def test_extras_are_not_forbidden_by_the_body_count():
    print("\n=== a beat that stages extras keeps them ===")
    # Reported: character_guard turned off to get random girls into a scene, and the
    # camera stayed fixated on the primary character. Two separate causes.
    #
    # The body-count clause is what forbade the extras. With exactly two described
    # people, every shot was told "There are two people in the shot, with one body for
    # each person" -- right against a duplicated character, wrong against "two women
    # dance behind them", where it forbids as a positive fact the people the author
    # just asked for.
    check("a pair still gets the count",
          "one body for each person" in S.cast_hold(["Dan", "Crystal"]))
    check("...but not when the beat stages extras",
          S.cast_hold(["Dan", "Crystal"], "Two women dance behind them.") == "")
    check("...nor a crowd", S.cast_hold(["Dan", "Crystal"], "A crowd watches.") == "")
    # "both" and "they" are group cues about the NAMED cast -- group_beat owns those
    # and they must not stand the count down.
    check("a group cue about the cast is not extras",
          "one body for each person" in S.cast_hold(["Dan", "Crystal"], "They both sit down."))
    check("extras_in reads plural people, not body parts",
          S.extras_in("Other girls wait.") and not S.extras_in("He lowers his arms."))

    mem = "Dan: he, 40, a work coat.\nCrystal: she, 26, a red dress."
    sh = [" ".join(x.split()) for x in run_node(
        "A bar.\n\nDan and Crystal sit at the bar.\n\n"
        "Dan and Crystal talk while two women dance behind them.",
        plan_only=True, character_memory=mem)[3].split("---") if x.strip()]
    check("the pair-only shot keeps the count",
          "one body for each person" in sh[0], sh[0][-90:])
    check("...and the shot staging extras does not",
          "one body for each person" not in sh[1], sh[1][-120:])

    # THE CAMERA is the other half, and it is the guard itself: off, every sheet line
    # goes into every shot -- including the one carrying <Picture N> -- so a near-clean
    # reference rides all of them and the portrait's composition becomes the shot's.
    img = torch.rand(1, H, W, 3)
    P = "A bar.\n\nCrystal waits.\n\nCrystal turns."
    MEM1 = "Crystal: <Picture 1>, she, 26, a red dress."
    off = run_node(P, plan_only=True, character_memory=MEM1,
                   ref_image_1=img, character_guard=False)[2]
    check("guard off with a tagged reference is reported",
          "fixes the camera on one person" in off, "")
    check("...and it says the guard adds nobody", "ADDS NOBODY" in off, "")
    on = run_node(P, plan_only=True, character_memory=MEM1,
                  ref_image_1=img, character_guard=True)[2]
    check("...and stays quiet with the guard on",
          "fixes the camera on one person" not in on, "")


def test_a_walk_into_an_unlisted_room_is_still_walked():
    print("\n=== heading for the locker room is walked, not cut to ===")
    # Reported: girls playing volleyball in a gym head to the locker room, and the
    # scene shifted to a locker room instead of them walking into it. Four faults at
    # once: "gym" and "locker room" were in no place list, so the ORIGIN was never
    # established; travel_anchor returned nothing without an origin, so the shot was
    # never told to walk; _MOD swallowed "locker" so the destination read as bare
    # "room"; and the body-count clause re-forbade the other girls one beat after
    # allowing them.
    check("a gym is a place", S.first_place("A school gym, hard light.") == "gym")
    check("a locker room is one room, not 'room'",
          S.travel_legs("They head to the locker room.")[2] == "locker room")
    check("a journey with no origin still walks in",
          "enters the locker room" in S.travel_anchor("", "", "locker room"))
    # ...and the hardware words this file needs are still NOT places.
    check("bars are restraints, not a room",
          S.travel_legs("She is chained to the bars.")[2] == "")
    check("lifting is not travelling",
          S.travel_legs("He lifts her to the bed.")[2] == "")

    mem = "Mia: she, 19, a blue kit.\nTess: she, 19, a red kit."
    P = ("A school gym, hard overhead light.\n\n"
         "Mia and Tess play volleyball with other girls.\n\n"
         "They head to the locker room.")
    out = run_node(P, plan_only=True, character_memory=mem)
    sh = [" ".join(x.split()) for x in out[3].split("---") if x.strip()]
    check("the move is performed as a walk",
          "opens in the gym and arrives in the locker room" in sh[1]
          and "every step in frame" in sh[1], sh[1][-150:])
    check("the extras are not forbidden on the shot that stages them",
          "one body for each person" not in sh[0], sh[0][-110:])
    # THE LATCH HOLDS past a beat that simply stops mentioning them. Read per beat,
    # a shot whose beat said nothing about the extras got "There is one person in the
    # shot: one body, one face" while five women stood in it -- asserting four of them
    # out of existence. Background people do not leave because a sentence stopped
    # mentioning them.
    check("...and the count stays down while they are still there",
          "one body for each person" not in sh[1]
          and "one person in the shot" not in sh[1], sh[1][-110:])
    # ...and it comes back when the script says they have gone, which is the explicit
    # transition every other state in this file has.
    _SH = "Dan: he, 40, a work coat.\nCrystal: she, 26, a red dress."
    back = [" ".join(x.split()) for x in run_node(
        "A bar.\n\nDan and Crystal sit with two women dancing behind them.\n\n"
        "The two women have gone.\n\nDan and Crystal sit at the bar together.",
        plan_only=True, character_memory=_SH)[3].split("---") if x.strip()]
    check("the count returns once the extras are dismissed",
          "one body for each person" in back[-1], back[-1][-110:])
    check("the room is reported by its real name",
          "enters locker room" in out[2] or "locker room" in out[2], "")
    check("...and not as bare 'room'", "the film enters room," not in out[2], "")


def test_a_move_to_any_place_is_performed():
    print("\n=== a move to a place no list names is still walked ===")
    # Asked for: this has to work for ANY scene. Adding room words one report at a
    # time fixes one script each -- a dungeon, a cargo bay, a stable, a chapel would
    # each hit the same wall the locker room did. So the gate is inverted: an unlisted
    # destination is a PLACE unless it is furniture, a fitting, a body part, a vehicle
    # or a person, and there are far fewer common object destinations in English than
    # there are names for rooms.
    for where in ("dungeon", "cargo bay", "stable", "chapel", "morgue", "greenhouse",
                  "tent", "boiler room", "cockpit", "sauna", "vault", "crypt",
                  "laundromat", "observatory", "infirmary", "barracks"):
        b = f"They head to the {where}."
        check(f"a {where} is somewhere to go", S.moved_to(b) == where, S.moved_to(b))
        check(f"...and it costs the walk: {where}", S.travel_spaces(b) == 2)
    # What must stay inside the room it is in.
    for thing in ("door", "window", "bench", "bed", "lockers", "table", "bars",
                  "girl", "van"):
        check(f"a {thing} is not somewhere to go",
              S.moved_to(f"She walks to the {thing}.") == "", thing)
    for phrase in ("her hand", "his shoulder", "other girls"):
        check(f"{phrase!r} is not somewhere to go",
              S.moved_to(f"She walks to {phrase}.") == "", phrase)
    # A place named without MOVEMENT is not a journey, exactly as before.
    for b in ("She looks at the dungeon.", "The dungeon is cold.",
              "He waits in the chapel."):
        check(f"not a move: {b[:26]!r}", S.moved_to(b) == "")

    # End to end, across scenes nothing in this file was written for.
    mem = "Mia: she, 19, a blue kit."
    for label, P, dest in (
            ("dungeon", "A stone dungeon lit by torches.\n\nMia stands by the wall."
                        "\n\nMia is led to the armoury.", "armoury"),
            ("freighter", "The cargo bay of a freighter.\n\nMia checks a crate."
                          "\n\nMia heads to the cockpit.", "cockpit"),
            ("farmyard", "A muddy farmyard at dawn.\n\nMia feeds the chickens."
                         "\n\nMia walks to the stable.", "stable")):
        out = run_node(P, plan_only=True, character_memory=mem)
        last = " ".join([x for x in out[3].split("---") if x.strip()][-1].split())
        check(f"{label}: the arrival is performed",
              f"travels to the {dest}" in last and "first step to its last" in last,
              last[-120:])
        check(f"{label}: and it is reported",
              f"to the {dest}" in out[2] and "cannot name" in out[2], "")
    # An object destination adds nothing, so ordinary beats are not crowded.
    plain = run_node("A kitchen.\n\nMia fills the kettle.\n\nMia walks to the window.",
                     plan_only=True, character_memory=mem)[3]
    check("a move to an object inside the room adds no clause",
          "travels to the" not in plain, plain[-110:])


def test_an_unstated_frame_becomes_a_portrait():
    print("\n=== a shot that says nothing about the camera is told what the frame holds ===")
    # Reported: the camera fixated on one character, staring into the lens -- WITH THE
    # REFERENCE IMAGE REMOVED, which is what ruled the reference out and left the text.
    # Measured on a volleyball beat: 15 words of appearance from the sheet and 9 of
    # mouth guard against 8 words of action. The sheet describes a face in every shot
    # because clothing continuity needs it there; nothing said how much of the person
    # to show, and an unstated attribute is left to the model's PRIOR, which this file
    # already records as "a portrait: facing the lens, pleasantly".
    check("a whole-body action gets a frame",
          "whole body" in S.frame_hold("McKenna serves the ball hard across the net."))
    check("...and so does a move", "whole body" in S.frame_hold("McKenna walks to the bench."))
    check("a face acting does not", S.frame_hold("McKenna smiles.") == "")
    # The author's camera always wins -- a close-up included, since somebody asked for it.
    check("a close-up in the beat stands it down",
          S.frame_hold("Close-up on her face as she serves.") == "")
    # A SHOT SIZE stands it down; a lens, a stock or a camera move does not. This was
    # any camera word at all, and that silenced the clause on every shot of any real
    # script -- the README tells you to put the camera in the anchor, so "Shot on 35mm,
    # handheld" disabled it everywhere. Reported as the framing fix doing nothing, and
    # it was doing nothing: it never ran.
    check("a stated shot size stands it down",
          S.frame_hold("McKenna serves the ball.", "Shot on 35mm, wide shots throughout.") == ""
          and S.frame_hold("McKenna serves the ball.", "Medium shots, eye level.") == ""
          and S.frame_hold("McKenna serves the ball.", "Close-ups throughout.") == "")
    for _a in ("Shot on 35mm, handheld, natural light.", "Cinematic, anamorphic lens.",
               "35mm film, shallow depth of field.", "Handheld, available light, grainy."):
        check(f"a camera note with no size still fires: {_a[:30]!r}",
              S.frame_hold("McKenna serves the ball.", _a) != "", _a)
    check("...and a wide shot named in the beat too",
          S.frame_hold("A wide shot as McKenna serves.") == "")

    mem = "McKenna: she, 22, tall, long blonde hair, blue eyes, a blue kit."
    P = ("A school gym, hard overhead light.\n\n"
         "McKenna serves the ball hard across the net.\n\n"
         "McKenna walks to the bench and picks up a towel.")
    out = run_node(P, plan_only=True, character_memory=mem)
    sh = [" ".join(x.split()) for x in out[3].split("---") if x.strip()]
    for i, shot in enumerate(sh, 1):
        check(f"shot {i} is told what the frame holds",
              "whole body, head to feet" in shot, shot[-90:])
    check("the reason is reported", "frame HOLDS" in out[2], "")
    # REGRESSION: the destination capture must not run through a conjunction. It read
    # "the bench and picks", whose head is a verb, so the blocklist never saw the
    # bench -- and the shot was told to travel "to the bench and picks".
    check("a destination stops at a conjunction",
          S.moved_to("McKenna walks to the bench and picks up a towel.") == "")
    check("...and a compound room is still a room",
          S.moved_to("They head to the locker room.") == "locker room"
          and S.moved_to("He walks into the engine room.") == "engine room")
    check("no gibberish reaches the shot", "and picks on screen" not in sh[1], sh[1][-90:])


def test_an_interrupt_releases_the_frame_buffer():
    print("\n=== stopping a run drops the frame buffer before it unwinds ===")
    # Reported: stopping a run aborted the process with a CUDA illegal memory access,
    # thrown from cuMemFreeAsync inside a tensor destructor rather than from any line of
    # Python. An interrupt is a BaseException, so it skips every `del` in the render loop
    # and leaves a multi-gigabyte frame buffer for the collector to free in its own
    # order, after ComfyUI has begun unloading the models it was sized against.
    acc = S.FrameAccumulator(8, torch.float32, True)
    acc.add(torch.zeros((4, 1, 1, 3)))
    acc.add(torch.zeros((4, 1, 1, 3)))
    check("the buffer holds frames", acc.tensor is not None and acc.used == 8)
    acc.release()
    check("release drops the tensor", acc.tensor is None)
    check("...and the overflow list", acc.overflow == [] and acc.used == 0)

    # THE CAPACITY IS AN UPPER BOUND. It used to subtract one frame per seam, assuming
    # trim_seam drops one from every shot after the first -- and a shot that opens on no
    # keyframe keeps its first frame, so the buffer filled and the overflow list started
    # retaining whole shots of decoded frames, on the GPU, uncopied.
    acc2 = S.FrameAccumulator(6, torch.float32, False)
    acc2.add(torch.zeros((3, 1, 1, 3)))
    acc2.add(torch.zeros((3, 1, 1, 3)))
    check("a correctly sized buffer never reaches overflow", acc2.overflow == [])

    # An interrupt out of the render path releases the buffer and is re-raised intact.
    node = S.H3LongVideos()
    # run() clears _frames on entry, so the buffer has to be published the way _render
    # publishes it: assigned inside the render, then interrupted.
    held = S.FrameAccumulator(4, torch.float32, True)
    held.add(torch.zeros((4, 1, 1, 3)))
    import comfy.model_management as _mm
    _Interrupt = getattr(_mm, "InterruptProcessingException", None)
    if _Interrupt is None:
        class _Interrupt(BaseException):
            pass
    check("the interrupt is a BaseException, so no `except Exception` eats it",
          issubclass(_Interrupt, BaseException) and not issubclass(_Interrupt, Exception))

    def _boom(_self, _prepared):
        _self._frames = held
        raise _Interrupt()
    _orig = S.H3LongVideos._render
    S.H3LongVideos._render = _boom
    try:
        raised = None
        try:
            node.run(**_render_args())
        except BaseException as e:
            raised = e
        check("the interrupt is re-raised, never swallowed",
              isinstance(raised, _Interrupt), type(raised).__name__)
        check("...and the frame buffer was dropped on the way out",
              held.tensor is None and node._frames is None)
    finally:
        S.H3LongVideos._render = _orig


def _render_args():
    return dict(model=FakeModel(), clip=FakeCLIP(), vae=FakeVAE(), audio_vae=FakeAudioVAE(),
                prompt="A room.\n\nHe walks in.", resolution="4:3", megapixels=0.0,
                shot_seconds=FRAMES / S.H3_FPS, steps=2, cfg=1.0,
                sampler_name="res_multistep", scheduler="simple", seed=1,
                apply_model_sampling=False, tiled_decode=False)


def test_a_posture_denied_is_not_a_posture_taken():
    print("\n=== chained into a squat, she is not told she is standing ===")
    # Reported: chained by the ankles, forced into a squat she cannot stand out of --
    # and she stands. The chains were read as stretching.
    #
    # Every chain clause was correct: CHAIN_POSE_HOLD, with the metal "already drawn to
    # its full length, so the position it fixes is the position that keeps", was on
    # every shot after the squat. What sat beside it was the posture latch saying she
    # was STANDING -- because "She cannot stand." matched `stand` and nothing looked at
    # the `cannot`. The latch carried it forward, so every later shot asserted, flatly
    # and positively, the one thing the chains existed to prevent. At cfg 1 a positive
    # statement wins, so the chains stretched.
    P = "Dan forces McKenna down into a squat. She cannot stand."
    check("the squat is what she is in, not standing",
          S.posture_in(P, ["McKenna"]) == {"McKenna": "squatting"},
          str(S.posture_in(P, ["McKenna"])))
    # An ATTEMPT is not an arrival -- the same error in a friendlier disguise.
    for b in ("McKenna tries to stand.", "McKenna struggles to get up.",
              "McKenna is unable to stand.", "McKenna is no longer able to stand."):
        check(f"not a posture taken: {b[:34]!r}", S.posture_in(b, ["McKenna"]) == {})
    # ...and a posture plainly stated is still read.
    for b, w in (("McKenna stands up.", "standing"), ("McKenna sits down.", "sitting"),
                 ("McKenna squats down.", "squatting")):
        check(f"still read: {b[:22]!r}", S.posture_in(b, ["McKenna"]) == {"McKenna": w})
    # The cue belongs to its OWN clause: reaching past the comma would silence a
    # posture the beat plainly states.
    check("a cue does not cross a clause boundary",
          S.posture_in("McKenna cannot kneel, so she sits.", ["McKenna"]) == {"McKenna": "sitting"})
    check("...nor does a later strain undo a stated posture",
          S.posture_in("McKenna stands and strains against the chains.", ["McKenna"])
          == {"McKenna": "standing"})
    check("the engine reader agrees", S.engine.posture_in("She cannot stand.") == ""
          and S.engine.posture_in("She stands.") == "standing")

    mem = ("McKenna: she, 22, a grey vest, ankle chains, wrist cuffs.\n"
           "Dan: he, 40, a work coat.")
    SC = ("A bare cell, one bulb.\n\nDan chains her ankles to a ring in the floor.\n\n"
          "Dan forces McKenna down into a squat. She cannot stand.\n\n"
          "McKenna stays down, breathing hard.\n\nMcKenna tries to stand.")
    sh = [" ".join(x.split()) for x in
          run_node(SC, plan_only=True, character_memory=mem)[3].split("---") if x.strip()]
    # The squat is staged in shot 2 (index 1); every shot from there on.
    for i in (1, 2, 3):
        check(f"shot {i + 1} never says she is standing",
              "is still standing" not in sh[i], sh[i][-120:])
        check(f"...and shot {i + 1} keeps the chain at full length",
              "full length" in sh[i], sh[i][-120:])


def test_the_face_plays_the_feeling_the_author_named():
    print("\n=== a stated emotion is played, not clamped shut ===")
    # Reported: under duress she does not act like it, and the same where she is
    # supposed to be happy. The node was contradicting the beat in both directions.
    # "Mia hugs Tess, beaming." got "Mouths in the shot stay closed" beside it, and
    # "McKenna is terrified" got "The mood is grim; ... the mouth set" -- a generic
    # clenched face stamped over the specific word the author chose.
    check("an emotion is read in the author's word",
          S.emotion_in("Mia is delighted with the news.") == "delighted"
          and S.emotion_in("McKenna is terrified and shaking.") == "terrified")
    check("...and nothing is invented where none is named",
          S.emotion_in("McKenna walks to the door.") == "")
    check("the clause names the mouth, which the guard would have held shut",
          "mouth" in S.mood_face("delighted") and "delighted" in S.mood_face("delighted"))
    check("no feeling, no clause", S.mood_face("") == "")

    HAPPY = ("A sunny kitchen.\n\nMia and Tess laugh over breakfast.\n\n"
             "Mia is delighted with the news.\n\nMia hugs Tess, beaming.")
    HMEM = "Mia: she, 24, red hair.\nTess: she, 25, dark hair."
    sh = [" ".join(x.split()) for x in
          run_node(HAPPY, plan_only=True, character_memory=HMEM)[3].split("---") if x.strip()]
    # BOTH are the source, so there is nobody left to hold and voice_sources stays
    # quiet -- which is the guarantee. Before, only Tess was credited and Mia was told
    # her mouth stays closed in a beat that says she laughs.
    check("a compound subject makes both of them the source",
          [n for n, _ in S.vocal_sources_in("Mia and Tess laugh over breakfast.", HMEM)]
          == ["Mia", "Tess"])
    check("...so neither of them is told her mouth stays closed",
          "mouth in the shot stays closed" not in sh[0]
          and "Mouths in the shot stay closed" not in sh[0], sh[0][-130:])
    check("delight is played, not clamped",
          "expression is delighted" in sh[1] and "Mouths in the shot stay closed" not in sh[1],
          sh[1][-130:])
    check("...and so is beaming",
          "expression is beaming" in sh[2] and "Mouths in the shot stay closed" not in sh[2],
          sh[2][-130:])

    DUR = ("A bare cell, one bulb.\n\nMcKenna is cuffed to the bench.\n\n"
           "McKenna is terrified and shaking.\n\nMcKenna stares at the door, desperate.")
    ds = [" ".join(x.split()) for x in
          run_node(DUR, plan_only=True, character_memory="McKenna: she, 22, a grey vest, wrist cuffs.")[3].split("---") if x.strip()]
    check("the author's word replaces the generic mood",
          "expression is terrified" in ds[1] and "the mouth set" not in ds[1], ds[1][-130:])
    check("...and again where she is desperate",
          "expression is desperate" in ds[2] and "the mouth set" not in ds[2], ds[2][-130:])
    check("a duress shot that names no feeling keeps the film's mood",
          "The mood is grim" in ds[0], ds[0][-130:])
    # THE GUARD THAT EXISTS FOR A REASON still holds: `and` starting a new predicate
    # must not credit the wrong mouth.
    _sheet = "Dan: he, 40.\nMcKenna: she, 22."
    check("a new predicate after `and` credits only its own subject",
          [n for n, _ in S.vocal_sources_in("Dan holds the door and McKenna sobs.", _sheet)]
          == ["McKenna"])


def test_a_pronoun_pointing_away_keeps_the_person_it_means():
    print("\n=== 'Tess kneels beside her' keeps McKenna described ===")
    # Reported: scene cohesion lost -- restraints not matching between shots, hair
    # changing, a body not the same size twice. With two women on the sheet every
    # pronoun collapsed onto whoever the beat named: "Tess kneels beside her" credited
    # "her" to Tess, because Tess declares "she", so McKenna -- who is in the shot and
    # is what "her" refers to -- lost her sheet line entirely. A person in frame with no
    # description is a person the model dresses out of nothing, and with no reference
    # image in the run the sheet text is the only thing holding her together.
    same = "McKenna: she, 22, long blonde hair.\nTess: she, 21, short dark hair."
    mixed = "McKenna: she, 22, long blonde hair.\nDan: he, 40, a work coat."
    three = same + "\nMara: she, 30, grey hair."
    prev = ["McKenna", "Tess"]

    def kept(sheet, beat):
        return sorted(S.sheet_for_beat(sheet, beat, previous=prev)[1])

    check("a pronoun aimed away keeps the person it means",
          kept(same, "Tess kneels beside her.") == ["McKenna", "Tess"])
    check("...and so does a look", kept(same, "Tess looks at her.") == ["McKenna", "Tess"])
    # THE RECORDED REGRESSION ON THE OTHER SIDE. "behind" is deliberately not a
    # proximity preposition here: a door is shut behind ONESELF, and reading that as
    # another person present is the bug the resolver's own comment records fixing.
    check("a door shut behind him is still nobody else",
          kept(mixed, "Dan walks out and shuts the door behind him.") == ["Dan"])
    # A possessive is not an object pronoun, and an ordinary beat is untouched.
    check("a possessive keeps its own reading", kept(same, "Tess unlocks her cuffs.") == ["Tess"])
    check("a beat with no pronoun is untouched",
          kept(same, "Tess watches from the door.") == ["Tess"])
    check("two named already leaves the pronoun alone",
          kept(same, "Tess and McKenna look at her hands.") == ["McKenna", "Tess"])
    # A PRONOUN CAN ONLY POINT AT SOMEBODY IN THE SCENE, which is also what stops a
    # random: off the sheet alone, "Tess looks at her" dragged in whichever other
    # woman was written down, including one who had left or never appeared.
    check("somebody absent is not dragged in",
          S.sheet_for_beat(same, "Tess looks at her.", previous=["Tess"])[1] == ["Tess"])
    check("...nor when nothing is carried at all",
          S.sheet_for_beat(same, "Tess stares at her.", previous=[])[1] == ["Tess"])
    # Three on the sheet but only two in the scene RESOLVES, because there is only one
    # candidate present -- the sheet could not tell them apart and presence can.
    check("three on the sheet, two in the scene, resolved",
          sorted(S.sheet_for_beat(three, "Tess kneels beside her.",
                                  previous=["McKenna", "Tess"])[1]) == ["McKenna", "Tess"])
    # ...and NOTHING is guessed when two candidates are both present.
    check("two candidates present, no guess",
          S.sheet_for_beat(three, "Tess kneels beside her.",
                           previous=["McKenna", "Tess", "Mara"])[1] == ["Tess"])
    check("the reader itself is narrow",
          S.pronoun_points_away("Tess kneels beside her.")
          and not S.pronoun_points_away("shuts the door behind him.")
          and not S.pronoun_points_away("Tess looks at her hands."))

    mem = ("McKenna: she, 22, long blonde hair, large breasts, a grey vest, steel handcuffs.\n"
           "Tess: she, 21, short dark hair, a red top.")
    P = ("A bare cell, one bulb.\n\nDan cuffs McKenna's wrists behind her back.\n\n"
         "McKenna pulls at the cuffs.\n\nMcKenna sits down on the bench.\n\n"
         "Tess kneels beside her.")
    sh = [" ".join(x.split()) for x in
          run_node(P, plan_only=True, character_memory=mem)[3].split("---") if x.strip()]
    check("the shot that only names Tess still describes McKenna",
          "McKenna: she, 22, long blonde hair" in sh[-1], sh[-1][-150:])
    check("...and her restraints are held there too",
          "handcuffs" in sh[-1] or "cuffs" in sh[-1], sh[-1][-150:])


def test_a_feeling_belongs_to_the_face_the_beat_pins_it_on():
    print("\n=== one person's feeling is not worn by everybody in the shot ===")
    # Reported: actions performed by all the characters when that is not the case.
    # mood_face was impersonal -- "The face carries it: the expression is terrified" --
    # which in a two-hander is a sentence about whoever is on screen, so the captor
    # wore his victim's terror. gaze_hold already makes this call: name the owner once
    # a second person is in the shot. With one person there is nobody else it could be,
    # and naming them again is a second mention, which has its own cost.
    cast = ["McKenna", "Dan"]
    check("the owner is the person the beat puts in front of it",
          S.emotion_owner("McKenna is terrified as Dan steps closer.", cast, "terrified")
          == "McKenna")
    check("...and `and` opens a new predicate with its own subject",
          S.emotion_owner("Dan holds the door and McKenna is terrified.", cast, "terrified")
          == "McKenna")
    check("a feeling pinned on nobody has no owner",
          S.emotion_owner("The room is terrified.", cast, "terrified") == "")
    check("a named clause names them", "McKenna's face carries" in S.mood_face("terrified", "McKenna"))
    check("a solo clause stays impersonal", S.mood_face("terrified").startswith(" The face"))
    # TWO PEOPLE CAN FEEL DIFFERENT THINGS, and giving only the first left the other
    # face to the prior -- the same half-fix as naming one of two speakers.
    pairs = S.emotion_pairs("Dan is furious and McKenna is terrified.", cast)
    check("both feelings are pinned", pairs == [("Dan", "furious"), ("McKenna", "terrified")],
          str(pairs))
    check("...and said in one sentence",
          "Dan's face carries furious" in S.mood_faces(pairs)
          and "McKenna's carries terrified" in S.mood_faces(pairs))
    check("no pairs, nothing said", S.mood_faces([]) == "")

    mem = "McKenna: she, 22, long blonde hair, a grey vest.\nDan: he, 40, a work coat."
    P = ("A bare cell, one bulb.\n\nMcKenna is terrified as Dan steps closer.\n\n"
         "McKenna is delighted.\n\nDan is furious and McKenna is terrified.\n\n"
         "Dan holds the door and McKenna is terrified.")
    sh = [" ".join(x.split()) for x in
          run_node(P, plan_only=True, character_memory=mem)[3].split("---") if x.strip()]
    check("two in the shot: the terror is hers and is named",
          "McKenna's face carries" in sh[0] and "The face carries" not in sh[0], sh[0][-120:])
    check("one in the shot: nobody else it could be",
          "The face carries it: the expression is delighted" in sh[1], sh[1][-120:])
    check("two feelings, two faces", "Dan's face carries furious" in sh[2]
          and "McKenna's carries terrified" in sh[2], sh[2][-130:])
    check("a new predicate does not hand it to the other one",
          "McKenna's face carries" in sh[3] and "Dan's face" not in sh[3], sh[3][-120:])
    # The frame clause counts bodies too: "the whole body" of two people is one body.
    check("the frame clause is plural with two in the shot",
          "every body in it whole" in S.frame_hold("McKenna serves the ball.", "", 2))
    check("...and singular with one",
          "the whole body" in S.frame_hold("McKenna serves the ball.", "", 1))


def test_a_bare_region_says_whose_body_it_is():
    print("\n=== a bare region names the body, so the prior does not pick one ===")
    # Reported: the wrong anatomy applied to a female character. This file already
    # recorded the same failure for the chest -- "this said 'The arms and shoulders are
    # bare' and stopped there, so the one region a bra occupies was unspecified, and an
    # unspecified region is filled by the model's own prior". The hip-down clause had
    # the identical gap and it was never closed: "The legs are bare from the hip down"
    # names the region and says nothing about whose body it is, so the anatomy at the
    # hip came from the prior too -- and at cfg 1 nothing later takes that back.
    check("the declared pronoun is what names the body",
          S.body_of("she") == "a woman's body" and S.body_of("he") == "a man's body")
    # `they` is an UNDECLARED body, not a licence to guess one.
    check("an undeclared body is not guessed",
          S.body_of("they") == "" and S.body_of("") == "" and S.body_of(None) == "")
    check("the clause carries it",
          "on a woman's body" in S.bare_hold(["legs"], body="a woman's body"))
    check("...and says nothing extra without it",
          "body" not in S.bare_hold(["legs"]).replace("nothing else worn", ""))

    # THE AGE IS PART OF THE BODY NAMED, where the sheet states one. "A woman's body"
    # is true of a woman of 22 and a woman of 62, so it settles nothing between them
    # and the prior settles it instead -- always the twenties. These sheets declare an
    # age, so that is what comes out; the bare phrases above are what a sheet with no
    # age still gets. See body_of.
    for mem, beat, want in (
            ("McKenna: she, 22, a grey vest, denim shorts.",
             "McKenna takes off her shorts.\nremove: shorts", "the body of a woman of 22"),
            ("Dan: he, 40, a work coat, jeans.",
             "Dan takes off his jeans.\nremove: jeans", "the body of a man of 40")):
        sh = [" ".join(x.split()) for x in run_node(
            f"A bare room.\n\n{beat}\n\nThey stand still.",
            plan_only=True, character_memory=mem)[3].split("---") if x.strip()]
        check(f"the uncovering shot names {want}", want in sh[0], sh[0][-120:])
        # ...AND EVERY SHOT AFTER IT, from state: an unspecified region is refilled by
        # the prior on every shot that leaves it unspecified, not only the first.
        check(f"...and so does the shot after it ({want})", want in sh[1], sh[1][-120:])
    nb = [" ".join(x.split()) for x in run_node(
        "A bare room.\n\nAlex takes off their jeans.\nremove: jeans\n\nThey stand still.",
        plan_only=True, character_memory="Alex: they, 30, a coat, jeans.")[3].split("---") if x.strip()]
    # Scoped to the BARE clause: frame_hold legitimately says "the whole body" and a
    # bare "body" substring would pass or fail on the wrong sentence.
    check("an undeclared pronoun gets the region without a body",
          "bare from the hip down" in nb[0]
          and "on a woman" not in nb[0] and "on a man" not in nb[0]
          and "the body of a" not in nb[0],
          nb[0][-120:])


def test_a_plural_removal_still_comes_off():
    print("\n=== skirts come off and stay off ===")
    # Reported: a scene with the character and several unnamed women -- they take off
    # their skirts and panties, and the garments are back in the next beat.
    #
    # Nothing to do with the unnamed women. "take off their skirtS" yields the token
    # "skirts" while the sheet says "a denim skirt", and every reader downstream looks
    # that token up in the sheet: the scrub by pattern, infer_removals by entry head. A
    # plural garment matched nothing, so the removal did nothing at all, the sheet went
    # on listing the skirt, and it was re-stamped into every later shot. One woman
    # undressing wrote "her skirt" and worked; the moment the subject went plural so did
    # the garment.
    SC = "A changing room.\nMcKenna: she, 22, a grey vest, a denim skirt, a black thong."
    for b in ("McKenna takes off her skirt.", "McKenna takes off her skirts.",
              "McKenna and the other girls take off their skirts.",
              "The other girls take off their skirts.", "They take off their skirts."):
        check(f"read as coming off: {b[:40]!r}", S.infer_removals(b, SC) == ["skirt"],
              str(S.infer_removals(b, SC)))
    # INHERENTLY PLURAL GARMENTS KEEP THEIR "S". The vocabulary lists them only in the
    # plural, so the stem is not a garment word and nothing is stripped.
    SC2 = ("A room.\nDan: he, 40, brown boots, blue jeans, white socks, a grey top.")
    for b, want in (("Dan kicks off his boots.", ["boots"]),
                    ("Dan steps out of his jeans.", ["jeans"]),
                    ("Dan peels off his socks.", ["socks"]),
                    ("Dan and Mia take off their boots.", ["boots"]),
                    ("Dan takes off his tops.", ["top"])):
        check(f"kept as written: {b[:34]!r}", S.infer_removals(b, SC2) == want,
              str(S.infer_removals(b, SC2)))
    check("the normaliser is exact, not a guess",
          S.engine.singular_garment("skirts") == "skirt"
          and S.engine.singular_garment("boots") == "boots"
          and S.engine.singular_garment("jeans") == "jeans"
          and S.engine.singular_garment("panties") == "panties"
          and S.engine.singular_garment("tops") == "top")
    check("...and garment_words agrees with it, off the one owner",
          S.engine.garment_words("take off their skirts") == ["skirt"]
          and S.engine.garment_words("kicks off her boots") == ["boots"])

    # END TO END: scrubbed from the sheet on the next shot and every shot after it.
    mem = "McKenna: she, 22, a grey vest, a denim skirt, a black thong."
    for label, beat in (("one woman", "McKenna takes off her skirt."),
                        ("several", "McKenna and the other girls take off their skirts."),
                        ("unnamed only", "The other girls take off their skirts.")):
        sh = [" ".join(x.split()) for x in run_node(
            f"A changing room.\n\n{beat}\n\nThey stand still.\n\nThey turn around.",
            plan_only=True, character_memory=mem)[3].split("---") if x.strip()]
        def _has_skirt(shot):
            m = re.search(r"McKenna: [^.]*\.", shot)
            return "skirt" in (m.group(0) if m else "")
        check(f"{label}: the skirt is gone from the next shot", not _has_skirt(sh[1]), sh[1][:130])
        check(f"{label}: ...and stays gone", not _has_skirt(sh[2]), sh[2][:130])


def test_a_removal_undresses_only_its_wearer():
    print("\n=== one woman taking her shirt off leaves the other's on ===")
    # Reported: shirts looking half missing. Two women in white shirts, one takes hers
    # off, and the token "shirt" was scrubbed from the WHOLE text -- so the other one's
    # entry lost her shirt while the keyframe still showed her wearing it. That is the
    # contradiction this file already describes: "the shot then says it is not worn,
    # take it off, and the thing under it is already showing. The model renders that
    # contradiction as a garment half present -- open, or partly cut". The strip-bare
    # path has been scoped to "THEIR OWN entry" since it was written; the prose-removal
    # path was not.
    MEM = ("McKenna: she, 22, a white shirt, a black bra, steel handcuffs.\n"
           "Tess: she, 21, a white shirt, blue jeans.\nDan: he, 40, a work coat.")

    def last_shot(beat):
        sh = [" ".join(x.split()) for x in run_node(
            f"A room.\n\nMcKenna, Tess and Dan stand by the window.\n\n{beat}\n\nThey wait.",
            plan_only=True, character_memory=MEM, ref_noise_aug=0.999)[3].split("---")
            if x.strip()]
        return sh[-1]

    def entry(shot, who):
        m = re.search(who + r": [^.]*\.", shot)
        return m.group(0) if m else ""

    one = last_shot("McKenna takes off her shirt.")
    check("the remover loses her shirt", "shirt" not in entry(one, "McKenna"), entry(one, "McKenna"))
    check("...and the other woman keeps hers", "shirt" in entry(one, "Tess"), entry(one, "Tess"))
    check("...and nothing else of the remover's goes",
          "handcuff" in entry(one, "McKenna"), entry(one, "McKenna"))
    both = last_shot("McKenna and Tess take off their shirts.")
    check("a compound subject undresses both",
          "shirt" not in entry(both, "McKenna") and "shirt" not in entry(both, "Tess"),
          entry(both, "Tess"))
    other = last_shot("Tess takes off her shirt while McKenna watches.")
    check("the other way round too", "shirt" not in entry(other, "Tess")
          and "shirt" in entry(other, "McKenna"), entry(other, "McKenna"))

    # THE WEARER, NOT THE REMOVER. "Dan unlocks McKenna's handcuffs" is Dan removing
    # HER hardware, so scoping to whoever the beat names would strand it on her for
    # ever -- which is the regression the hardware tests caught when this was first
    # scoped by the remover.
    hw = last_shot("Dan unlocks McKenna's handcuffs.")
    check("somebody else's removal still reaches the wearer",
          "handcuff" not in entry(hw, "McKenna"), entry(hw, "McKenna"))
    check("...and takes nothing else of hers", "shirt" in entry(hw, "McKenna"), entry(hw, "McKenna"))
    check("the remover reader keeps a compound subject",
          sorted(S.strippers_in("McKenna and Tess take off their shirts.", MEM))
          == ["McKenna", "Tess"])
    check("...and a new predicate after `and` is not a remover",
          S.strippers_in("Dan holds the door and McKenna takes off her shirt.", MEM)
          == ["McKenna"])


def test_a_pairing_the_beat_wrote_is_said_back():
    print("\n=== who is with whom, where it could be read wrong ===")
    # Reported: girls kissing each other when they should be kissing the boys. The beat
    # said "Mia kisses Dan while Tess kisses Jon" and that was ALL the shot said about
    # it -- one sentence among four appearance descriptions, and at cfg 1 the model
    # reads the prompt as a bag of words and pairs by its own prior. This file names the
    # owner of a gaze, a vocal, a feeling, a posture, a restraint and a body count;
    # contact was the one relationship nothing restated.
    CAST = ["Mia", "Tess", "Dan", "Jon"]
    check("two pairs in one beat are both read",
          S.contact_pairs("Mia kisses Dan while Tess kisses Jon.", CAST)
          == [("Mia", "Dan"), ("Tess", "Jon")])
    check("one pair is read", S.contact_pairs("Mia kisses Dan.", CAST) == [("Mia", "Dan")])
    # THE OBJECT HAS TO BE A NAME, which is what makes the verb list safe: furniture is
    # never a partner, so hold/pull/take can be in it.
    check("furniture is not a partner",
          S.contact_pairs("Dan holds the door while Mia kisses Jon.", CAST) == [("Mia", "Jon")])
    check("...and a chain is not either", S.contact_pairs("Mia pulls the chain.", CAST) == [])
    # NOTHING IS GUESSED where the beat names nobody -- guessing which two is the bug.
    check("an unnamed pairing gets nothing", S.contact_pairs("They kiss.", CAST) == [])
    for b, w in (("Mia dances with Dan.", ("Mia", "Dan")),
                 ("Tess leans against Jon.", ("Tess", "Jon")),
                 ("Mia undresses Dan.", ("Mia", "Dan"))):
        check(f"read: {b[:26]!r}", S.contact_pairs(b, CAST) == [w])
    check("the clause names both sides",
          "Mia with Dan" in S.contact_hold([("Mia", "Dan")]))
    check("...and both pairs when there are two",
          "Mia with Dan" in S.contact_hold([("Mia", "Dan"), ("Tess", "Jon")])
          and "Tess with Jon" in S.contact_hold([("Mia", "Dan"), ("Tess", "Jon")]))
    check("no pair, no clause", S.contact_hold([]) == "")

    mem = ("Mia: she, 20, a red dress.\nTess: she, 21, a blue dress.\n"
           "Dan: he, 22, a work shirt.\nJon: he, 23, a hoodie.")
    sh = [" ".join(x.split()) for x in run_node(
        "A bar.\n\nMia, Tess, Dan and Jon stand together.\n\n"
        "Mia kisses Dan while Tess kisses Jon.\n\nThey all talk.",
        plan_only=True, character_memory=mem, ref_noise_aug=0.999)[3].split("---")
        if x.strip()]
    check("the contact shot names both pairs",
          "Mia with Dan" in sh[1] and "Tess with Jon" in sh[1], sh[1][-130:])
    check("...and a beat with no contact says nothing about it",
          "The contact is" not in sh[2], sh[2][-110:])
    # WITH TWO PEOPLE nothing is said: there is nobody else to pair with, and naming
    # them again costs a mention each.
    two = [" ".join(x.split()) for x in run_node(
        "A bar.\n\nMia and Dan stand together.\n\nMia kisses Dan.",
        plan_only=True, character_memory="Mia: she, 20, a red dress.\nDan: he, 22, a shirt.",
        ref_noise_aug=0.999)[3].split("---") if x.strip()]
    check("two in the shot need no pairing said", "The contact is" not in two[-1], two[-1][-110:])


def test_keyframe_handoff():
    print("\n=== the keyframe is encoded, once per boundary ===")
    vae = FakeVAE()
    run_node("A room.\n\nOne.\n\nTwo.\n\nThree.", vae=vae)
    # Shot 1 has no previous frame; shots 2-4 each encode the frame handed to them.
    # Passing the previous shot's own latent instead was tried and was wrong: a
    # keyframe is ONE pixel frame, which lands on H3's 5f grid point and encodes to
    # TWO latent frames, and a causal encoder's last latent is not a standalone
    # opening frame anyway. It degraded every shot after the first.
    # 3 shots = 2 boundaries; shot 1 has no previous frame and no first_frame here.
    check("one encode per boundary, none for shot 1", vae.encodes == 2,
          f"{vae.encodes} encodes")
    src = open(os.path.join(_HERE, "sampler.py"), encoding="utf-8").read()
    check("no latent is smuggled in as a keyframe", "handoff_latent" not in src)


def test_references_and_silence():
    print("\n=== references and silence ===")
    clip = FakeCLIP()
    run_node("A room.\n\nOne.\n\nTwo.", clip=clip,
             ref_image_1=torch.rand(1, 64, 64, 3))
    # seen[0] is the empty negative prompt, encoded once before the loop.
    shots_seen = clip.seen[1:]
    # No <Picture N> tag anywhere, so placing by tag would place the reference on no
    # shot at all -- a connected input that silently does nothing. It falls back to
    # every shot instead. Shot 1 carries the reference alone (no previous frame yet);
    # shot 2 carries the reference AND its keyframe.
    check("an untagged reference still reaches every shot",
          [len(i) for _, i in shots_seen] == [1, 2],
          str([len(i) for _, i in shots_seen]))
    clip2 = FakeCLIP()
    # auto_sound OFF: with it on, an ambient bed read from the scene opens every
    # branch, and this is the silence path.
    run_node("A room.\n\nHe walks in.\n\nShe says: \"Now.\"", clip=clip2,
             auto_sound=False)
    check("both shots reached the encoder", len(clip2.seen) == 3)   # + the negative
    check("the beat text is what was sent",
          "He walks in." in clip2.seen[1][0] and "Now." in clip2.seen[2][0])
    # auto_sound appends a sound sentence -- "He walks in." implies footsteps. Your
    # words are still never rewritten; the node only ever adds after them, and every
    # addition has a switch.
    # The beat has no line and no sound of its own, so it is pinned silent and gets no
    # sound sentence: a clause would describe an acoustic the conditioning removes,
    # and a free branch there is what put a voice in the mouth.
    # It carries the mouth clause and nothing else: no sound sentence, because a
    # clause would describe an acoustic the conditioning removes. The mouth clause is
    # the picture half of the same guarantee and has its own switch.
    # THE GUARANTEE, not the whole string -- the same lesson the clip4 check below
    # records: "asserting the whole string made this fail every time an unrelated
    # clause was added beside it". What matters here is that a silenced shot gets no
    # sound sentence, and that the mouth clause is the picture half of it.
    check("...with no sound sentence added to a silenced shot",
          "It sounds like" not in clip2.seen[1][0]
          and "the only sound" not in clip2.seen[1][0]
          and S.MOUTH_HOLD in clip2.seen[1][0]
          and "A room. He walks in." in clip2.seen[1][0], clip2.seen[1][0])
    clip2b = FakeCLIP()
    run_node(TWO_LINE_ROOM, clip=clip2b, mouths_shut_when_no_line=False,
             auto_sound=False)
    check("...and none at all with the mouth guard off",
          "It sounds like" not in clip2b.seen[1][0]
          and "the only sound" not in clip2b.seen[1][0]
          and S.MOUTH_HOLD not in clip2b.seen[1][0]
          and "A room. He walks in." in clip2b.seen[1][0], clip2b.seen[1][0])
    # The shot that DOES speak gets the open form: closing the list there would be
    # telling the model the line is not in it.
    clip4 = FakeCLIP()
    run_node("A room.\n\nShe walks in and says: \"Now.\"", clip=clip4)
    # The GUARANTEE, not the exact sentence: the sound clause is the open form,
    # and the beat and its line are there untouched. Asserting the whole string
    # made this fail every time an unrelated clause was added beside it.
    check("a shot with a line is not told that is all there is",
          "It sounds like" in clip4.seen[1][0]
          and "footsteps" in clip4.seen[1][0]
          and "the only sound" not in clip4.seen[1][0]
          and "She walks in and says:" in clip4.seen[1][0]
          and "Now." in clip4.seen[1][0], clip4.seen[1][0])
    clip3 = FakeCLIP()
    run_node("A room.\n\nHe walks in.", clip=clip3, auto_sound=False)
    check("...and with that off only the mouth clause remains",
          "It sounds like" not in clip3.seen[1][0]
          and "the only sound" not in clip3.seen[1][0]
          and S.MOUTH_HOLD in clip3.seen[1][0]
          and "A room. He walks in." in clip3.seen[1][0], clip3.seen[1][0])


def test_first_frame():
    print("\n=== first_frame anchors shot 1 ===")
    vae = FakeVAE()
    run_node("A room.\n\nOne.\n\nTwo.", vae=vae, first_frame=torch.rand(1, H, W, 3))
    # 2 shots: shot 1 encodes the supplied first_frame, shot 2 encodes the handoff.
    check("a supplied first frame is encoded as shot 1's keyframe", vae.encodes == 2,
          f"{vae.encodes}")
    bare = FakeVAE()
    run_node("A room.\n\nOne.\n\nTwo.", vae=bare)
    check("...and without one, shot 1 encodes nothing", bare.encodes == 1, f"{bare.encodes}")


def test_aug_protects_the_keyframe():
    print("\n=== ref_noise_aug must not corrupt the keyframe ===")
    # comfy/ldm/minimax/model.py applies visual_cond_noise_aug to BOTH the
    # reference rows and the keyframe rows: below 1.0 it noises the keyframe
    # latent and labels it max(t_v, aug) instead of 0.999. Softening references
    # would therefore corrupt the anchor of every shot after the first, and that
    # shows up during SAMPLING.
    P = "A room.\n\nOne.\n\nTwo."
    ref = torch.rand(1, 64, 64, 3)
    vae_hi = FakeVAE()
    info_hi = run_node(P, vae=vae_hi, ref_image_1=ref, ref_noise_aug=0.999)[2]
    check("at a safe aug the handoff is a real keyframe",
          "riding as an extra reference" not in info_hi)
    check("...and it is encoded", vae_hi.encodes >= 1, f"{vae_hi.encodes}")
    # One aug covers every visual condition row. Below KEYFRAME_SAFE_AUG it would
    # noise the keyframe too, so there the handoff stops anchoring and rides as an
    # extra reference: weaker continuity, but nothing pretending to anchor.
    info_lo = run_node(P, ref_image_1=ref, ref_noise_aug=0.90)[2]
    check("a soft aug demotes the keyframe rather than noising it",
          "riding as an extra reference" in info_lo)
    check("...and the note says what it costs", "weaker continuity" in info_lo)
    # With nothing connected there is no reference to soften, so the keyframe is safe
    # whatever the widget says.
    info_noref = run_node(P, ref_noise_aug=0.90)[2]
    check("no references means the keyframe is never demoted",
          "riding as an extra reference" not in info_noref)


def test_beat_reviving_a_garment():
    print("\n=== a later beat that names a removed garment ===")
    # Beats go to the model word for word. The scene can be perfectly scrubbed
    # and the removal honoured, and then beat 5 says "her coat" and it is
    # back. Ten beats in, that reads as the removal randomly failing.
    P = ("A basement. Kate is 20, blonde, grey wool coat, black jumper." + "\n\n"
         "Dan pulls off her coat.\nremove: coat" + "\n\n"
         "Kate lies still." + "\n\n"
         "Dan pulls at her coat again." + "\n\n"
         "Kate turns her head.")
    info = M_info = run_node(P)[2]
    check("the reviving beat is named", "shot 3 names coat" in info)
    check("...with the reason", "word for word" in info)
    check("...and the removing beat itself is not flagged",
          "shot 1 names coat" not in info)
    clean = run_node("A basement. Kate is 20." + "\n\n" +
                     "Dan pulls off her coat.\nremove: coat" + "\n\n" +
                     "Kate lies still.")[2]
    check("a clean script raises nothing", "in its own text" not in clean)


def test_restart_after_removal():
    print("\n=== a shot after a removal starts fresh ===")
    # Every shot is anchored to the previous shot's last frame. If the model
    # does not finish taking the garment off inside its own shot, that frame
    # still shows it -- and a keyframe is a PICTURE, which outvotes any
    # sentence. Inherit it once and every later shot inherits it too.
    P = ("A basement. Kate is 20, blonde, grey wool coat, black jumper." + "\n\n"
         "Dan pulls off her coat.\nremove: coat" + "\n\n"
         "Kate lies still." + "\n\n"
         "Kate breathes.")
    vae_on = FakeVAE()
    info_on = run_node(P, vae=vae_on, restart_after_removal=True)[2]
    check("the shot after the removal is named", "shot(s) 2 start fresh" in info_on)
    check("...with the reason", "picture outvotes the text" in info_on)
    check("...and the cost", "costs a cut" in info_on)
    vae_off = FakeVAE()
    info_off = run_node(P, vae=vae_off, restart_after_removal=False)[2]
    check("off, nothing restarts", "start fresh" not in info_off)
    # A dropped keyframe is one fewer frame to encode.
    check("the fresh shot encodes no keyframe", vae_on.encodes < vae_off.encodes,
          f"{vae_on.encodes} vs {vae_off.encodes}")
    check("a script with no removals is unaffected",
          "start fresh" not in run_node("A room.\n\nOne.\n\nTwo.")[2])


def test_auto_removal():
    print("\n=== removals read from the beat, with no directives ===")
    P = ("A basement. Kate is 20, blonde, grey jumper, wool scarf, black boots."
         + "\n\n" +
         "Dan pulls off her boots and throws them away, showing the wool scarf."
         + "\n\n" +
         "Dan pulls off her scarf and throws it away." + "\n\n" +
         "Kate lies still.")
    script = run_node(P)[3]
    sh = script.split("\n---\n")
    # Shot 1 takes the boots off and has NO keyframe, so the text is the only thing
    # saying they were on to start with and it keeps saying so. Gone from shot 2.
    check("the removing shot still says the boots are on", "black boots" in sh[0])
    check("...and shot 2 has lost them", "black boots" not in sh[1])
    # The scarf is under the boots -- read from "showing the wool scarf" -- so it is
    # not described as visible until they come off. Only the scene text counts here;
    # the beat names it either way, because beats go to the model word for word.
    _scene1 = sh[0].split("Dan pulls")[0]
    check("the covered scarf is not described yet", "wool scarf" not in _scene1)
    check("...and is once the boots are off", "wool scarf" in sh[1])
    check("shot 3 keeps neither", "boots" not in sh[2] and "scarf" not in sh[2])
    check("...and keeps what was never taken off", "grey jumper" in sh[2])
    info = run_node(P)[2]
    check("info says what it read", "read 'boots' as coming off" in info)
    # Off, nothing is inferred and the warning comes back instead.
    info_off = run_node(P, auto_remove=False)[2]
    check("auto_remove off infers nothing", "as coming off" not in info_off)
    check("...and warns instead", "no 'remove:' line" in info_off)


def test_fall_keeps_the_hardware():
    print("\n=== a fall does not open the cuffs ===")
    P = ("A bare cellar. Kate, 24, in a grey coat.\n\n"
         "Dan cuffs her wrists behind her back.\n\n"
         "Kate loses her balance and falls onto the floor.\n\n"
         "Kate lies still while Dan walks out.")
    blocks = [b for b in re.split(r"(?=\[Shot )", run_node(P, plan_only=True)[3])
              if b.strip()]
    check("three shots planned", len(blocks) == 3)
    fall = S.FALL_HOLD.strip()
    # The hold latches: once a restraint goes on it is held for the rest of the run.
    # Cuffs are rigid hardware, so the chain clause carries the guarantee here -- it
    # says "whole and closed" itself, and emitting both would say it twice.
    #
    # Shot 1 is the shot that PUTS them on, and it gets both ends instead of the
    # standing hold. The standing hold asserts the cuffs are fastened as they were put
    # on and still fastened at the last frame, which read at frame 1 says they are
    # already closed -- so they close first and the catching happens around them.
    check("the applying shot is told both ends",
          S.RESTRAINT_GOING_ON.strip() in blocks[0], blocks[0][-90:])
    check("...and keeps the metal rigid while it goes on",
          S.CHAIN_RIGID_TAIL.strip() in blocks[0], "")
    check("...and is not also told it is already fastened",
          "closed and fastened as" not in blocks[0], "")
    for i, b in enumerate(blocks[1:], 2):
        check(f"shot {i} holds the restraint",
              "closed and fastened as" in b)
    # The fall clause is per-beat -- it only earns its tokens where a body goes down.
    check("the fall beat says what takes the landing", fall in blocks[1])
    check("...the beat that puts them on does not", fall not in blocks[0])
    check("...and neither does lying still afterwards", fall not in blocks[2])
    # No restraint anywhere in the prompt: a fall is just a fall, nothing to protect.
    loose = run_node("A bare cellar. Kate, 24.\n\nKate trips and falls.",
                     plan_only=True)[3]
    check("an unbound fall adds nothing", fall not in loose)


def test_anchor_is_the_scene():
    print("\n=== an anchor makes every paragraph a beat ===")
    # Reported as "the character memory is not being passed" and "the first line is
    # ignored". One cause: with the framing moved into the anchor, the prompt starts
    # with an ACTION -- and paragraph 1 was still being taken as the scene. That beat
    # was then prepended to every shot, repeated to the end of the film, never given
    # a shot of its own, and any removal in it could never stick because the scene
    # restated the garment on every later shot.
    P = "Jon walks in and takes her scarf off.\n\nMaya lies still.\n\nJon leaves."
    sh = [x for x in re.split(r"(?=\[Shot )", run_node(
        P, plan_only=True, anchor="Wide lens, night.",
        character_memory="Maya: 27, grey scarf, black boots.")[3]) if x.strip()]
    check("every paragraph gets its own shot", len(sh) == 3)
    check("the first action is not stolen as the scene",
          "Jon walks in" in sh[0] and "Jon walks in" not in sh[1])
    check("the anchor leads every shot", all("Wide lens, night." in s for s in sh))
    check("...and the character sheet follows it",
          all(s.index("Wide lens") < s.index("Maya: 27") for s in sh))
    # Shot 1 removes the scarf and has no keyframe, so it still says the scarf is on
    # -- otherwise the shot reads "it is not worn, take it off". Gone after that.
    check("the removal sticks", all("grey scarf" not in s for s in sh[1:]))
    check("...and what was never removed is still described",
          all("black boots" in s for s in sh))
    # With no anchor, paragraph 1 is the scene exactly as before.
    sh2 = [x for x in re.split(r"(?=\[Shot )", run_node(
        "A basement.\n\nJon walks in.\n\nMaya lies still.", plan_only=True)[3])
        if x.strip()]
    check("no anchor, no change", len(sh2) == 2)
    check("...and the scene still leads", all("A basement." in s for s in sh2))


def test_guard_and_layers_end_to_end():
    print("\n=== the guard and the layers, through the whole path ===")
    P = ("Medium shadows. A basement workshop.\n\n"
         "Maya: 27, blonde hair, grey wool scarf, black quilted jacket, brown boots. "
         "Wrists cuffed behind back. She stays lying on her side on the floor.\n"
         "Jon: 34, navy overalls.\n\n"
         "Maya lies still on the floor, eyes closed.\n\n"
         "Jon walks in and takes her jacket off to expose the scarf.\n\n"
         "Maya lies still.\n\n"
         "Jon walks out and shuts the door.")
    imgs, audio, info, script = run_node(P, plan_only=True)[:4]
    sh = [x for x in re.split(r"(?=\[Shot )", script) if x.strip()]
    check("four beats, four shots", len(sh) == 4)
    check("Jon is absent from the shot he is not in", "Jon: 34" not in sh[0])
    check("...and present in the one he is", "Jon: 34" in sh[1])
    check("...and alone once he leaves her behind", "Maya: 27" not in sh[3])
    check("Maya is kept where a pronoun refers to her", "Maya: 27" in sh[1])
    # The scarf is under the jacket, read from the script's own wording.
    check("the covered layer is not described", "grey wool scarf" not in sh[0])
    check("...and appears once the jacket is off", "grey wool scarf" in sh[2])
    check("the jacket is described while it is on", "quilted jacket" in sh[0])
    check("...and gone after it comes off", "quilted jacket" not in sh[2])
    check("info names the layering", "scarf under jacket" in info)
    check("...and the opening-pose warning", "opening pose comes from the text" in info)
    # Off, every sheet line goes into every shot, as before.
    off = [x for x in re.split(r"(?=\[Shot )", run_node(P, plan_only=True,
           character_guard=False)[3]) if x.strip()]
    check("guard off puts everyone in every shot",
          all("Jon: 34" in s and "Maya: 27" in s for s in off))


def test_removing_shot_without_a_keyframe():
    print("\n=== a removal in a shot that has no keyframe ===")
    # Reported: the garment rendering half-off -- opened up, with the layer under it
    # showing. The scrub applies to the removing shot too, which is right when that
    # shot has a KEYFRAME: the picture already shows the garment on at the start, so
    # text saying it is worn would put it back at the end. Shot 1 has no keyframe.
    # Scrubbing there left the shot saying: it is not worn, take it off, and the
    # thing under it is already showing -- and the model rendered the contradiction.
    P = ("A basement.\n\n"
         "Maya: 27, grey wool scarf, black quilted jacket, brown boots.\n\n"
         "Jon cuts off her jacket to expose the scarf.\n\n"
         "Maya lies still.\n\n"
         "Jon pulls off her scarf.")
    sh = [x for x in re.split(r"(?=\[Shot )", run_node(P, plan_only=True)[3]) if x.strip()]
    check("the removing shot still says the garment is worn",
          "quilted jacket" in sh[0])
    check("...and the layer under it is not yet showing",
          "grey wool scarf" not in sh[0])
    # WHOSE HANDS, where the beat names them. An agentless "the jacket comes off"
    # describes a garment removing itself, which is what a belt dropping to the floor
    # on its own looked like.
    check("...and it is still told to come off",
          "jacket off during this shot" in sh[0] or "jacket comes off during this shot" in sh[0])
    # WHOSE HANDS -- and the limit on that. The agent can only be somebody the SHEET
    # describes, because that is the cast the guard resolves. Jon acts in this beat
    # but has no entry, so the only person in the shot is Maya and the clause names
    # her. Undressing herself is a defensible reading of a prompt that never says who
    # Jon is; naming a person the shot does not describe would be worse, since a
    # described person is a person the model draws.
    check("...by the hands of somebody the sheet describes",
          "Maya takes the black quilted jacket off during this shot" in sh[0],
          sh[0][-120:])
    check("the next shot has lost it", "quilted jacket" not in sh[1])
    check("...and shows what was under it", "grey wool scarf" in sh[1])
    check("info explains the exception", "no keyframe" in run_node(P, plan_only=True)[2])
    # A removing shot that DOES have a keyframe still scrubs in place -- the picture
    # carries the starting state there, so the text does not have to.
    check("a later removing shot still scrubs itself",
          "grey wool scarf" not in sh[2].split("Jon pulls")[0])
    # With a first_frame wired, shot 1 has a keyframe and behaves like the rest.
    sh_ff = [x for x in re.split(r"(?=\[Shot )", run_node(
        P, plan_only=True, first_frame=torch.rand(1, H, W, 3))[3]) if x.strip()]
    check("a wired first_frame restores the normal rule",
          "quilted jacket" not in sh_ff[0].split("Jon cuts")[0])


def test_hardware_anchor_end_to_end():
    print("\n=== a shown collar still belongs on a neck ===")
    P = ("A basement.\n\nMaya: 27, grey coat.\n\n"
         "Jon shows her a collar and leash.\n\n"
         "Jon buckles the collar around her neck.\n\n"
         "Maya walks to the window.")
    imgs, audio, info, script = run_node(P, plan_only=True)[:4]
    sh = [x for x in re.split(r"(?=\[Shot )", script) if x.strip()]
    check("the shot that shows it says where it goes",
          "a collar closes around the neck" in sh[0])
    check("...and where the leash goes", "leash clips to the collar" in sh[0])
    check("the shot that places it is left alone",
          "closes around the neck" not in sh[1])
    check("an ordinary beat gets nothing",
          "sits where it belongs" not in sh[2])
    check("info says it stepped in", "no body part beside it" in info)


def test_chain_hold_end_to_end():
    print("\n=== a chain holds its shape through the whole run ===")
    P = ("A basement.\n\n"
         "Maya: 27, grey coat. Wrists cuffed behind back.\n\n"
         "Jon locks a chain around her waist and padlocks it at the back.\n\n"
         "Maya pulls against the chain.\n\n"
         "Maya lies still.")
    sh = [x for x in re.split(r"(?=\[Shot )", run_node(P, plan_only=True)[3]) if x.strip()]
    chain = "links keeping their size"
    check("every shot with the hardware holds it rigid", all(chain in s for s in sh))
    # It REPLACES the restraint hold instead of joining it -- both say "whole and
    # closed", and two clauses for one guarantee is twice the stasis in the prompt.
    check("...instead of repeating the restraint hold",
          all("the run between them" in s for s in sh))
    check("...while still carrying its guarantee",
          all("closed and fastened as it was put on" in s
              or "closed and fastened as they were put on" in s for s in sh))
    # Rope flexes. Saying it holds a straight line would be wrong, so it does not.
    soft = run_node("A basement.\n\nMaya: 27, a rope around her wrists.\n\n"
                    "Maya lies still.", plan_only=True)[3]
    check("rope is not claimed to be rigid", chain not in soft)
    # ...and it is TIED, not closed and fastened. restraint_sentence has said so
    # since it was written -- "rope is not closed and it is not fastened, and
    # saying so of a cord describes a mechanism that is not there" -- but the
    # rope was declared in the SHEET and nothing read the sheet, so no hardware
    # was found and the shot fell through to the generic wording this assertion
    # was pinned to. The engine reads the sheet, so the soft branch now fires.
    # The check keeps its intent: whatever the mechanism is called, it is whole.
    check("...but it is still held whole",
          "tied and holding as" in soft or "closed and fastened as" in soft, soft[-120:])
    check("...and a cord is tied rather than fastened shut",
          "tied and holding as" in soft, soft[-120:])
    # Steel locked on in shot 1 is still steel in shot 5. Tested per shot rather than
    # latched, the shot naming the chain got the rigid clause and every shot after it
    # fell back to the soft one -- which is where the slack came back from.
    later = [x for x in re.split(r"(?=\[Shot )", run_node(
        "A basement.\n\nMaya: 27, grey coat.\n\nJon locks a chain around her waist.\n\n"
        "Maya walks to the window.\n\nMaya looks down.", plan_only=True)[3]) if x.strip()]
    # The shot that locks it on gets both ends rather than the standing clause, but
    # the METAL is rigid throughout -- steel is steel while it is being closed, and
    # letting it go soft on the shot that introduces the object is where the model's
    # idea of the object gets set.
    # The content, not the exact sentence: the applying shot carries it as its own
    # sentence and the standing clause carries it after a semicolon.
    _rigid = "the run between them"
    check("the metal is rigid from the shot that names it",
          all(_rigid in s for s in later))
    check("rigidity latches past the shot that names it",
          all(chain in s for s in later[1:]))
    check("...and that first shot is the one putting it on",
          S.RESTRAINT_GOING_ON.strip() in later[0], later[0][-90:])
    check("...and the soft clause is not used instead",
          all("the run between them" in s for s in later))
    # A position the hardware enforces latches too: the chain that put a body in a
    # squat is still that length three shots later, so the squat is still the position.
    posed = [x for x in re.split(r"(?=\[Shot )", run_node(
        "A basement.\n\nMaya: 27, grey coat. Wrists cuffed behind back.\n\n"
        "Jon locks a chain from her ankles to her collar, forcing her into a squat.\n\n"
        "Maya strains against the chain, trying to stand.\n\nMaya breathes hard.",
        plan_only=True)[3]) if x.strip()]
    check("a forced position keeps for the rest of the run",
          all("drawn to its full length" in s for s in posed))
    check("...replacing the plain chain clause rather than joining it",
          all(chain not in s for s in posed))
    # No position forced: the plain clause, so an unposed chain does not freeze anyone.
    check("a chain with no position forced stays plain",
          all("drawn to its full length" not in s for s in later))
    # Hardware with nothing restrained by it is scenery, not a restraint.
    loose = run_node("A yard with a chain-link fence.\n\nMaya walks past it.",
                     plan_only=True)[3]
    check("scenery does not arm it", chain not in loose)


def test_every_paragraph_accounted_for():
    print("\n=== no beat quietly goes missing ===")
    # Two ways a beat disappears: it reads as a character sheet and is folded into
    # the scene, or it was never a separate paragraph. Both look like beats being
    # absorbed into other beats, and both were silent.
    P = ("A basement.\n\n"
         "Maya: 27, grey coat\nJon: 34, navy overalls\n\n"
         "Maya walks in.\nMaya sits down.\nMaya stands up.\n\n"
         "Jon: pushes the door shut.\n\n"
         "Maya leaves.")
    imgs, audio, info, script = run_node(P, plan_only=True)[:4]
    sh = [x for x in re.split(r"(?=\[Shot )", script) if x.strip()]
    check("a labelled ACTION still gets its own shot", len(sh) == 3)
    check("...and is not folded into the scene",
          "pushes the door shut" in sh[1] and "pushes the door shut" not in sh[0])
    check("every paragraph is accounted for", "5 paragraph(s) in the prompt" in info)
    check("...naming what became shots", "3 rendered as shots" in info)
    check("...what was folded in", "2 folded in as character sheet(s)" in info)
    check("...and what became the scene", "1 kept as the scene" in info)
    # Lines joined by a single newline are ONE beat. That is not changed -- it is
    # reported, because three actions in one shot look like two were absorbed.
    check("a merged beat is flagged", "carry more than one line" in info)
    check("...naming the shot", "shot(s) 1 carry" in info)
    check("...and saying what to do", "put an empty line between them" in info)
    clean = run_node("A room.\n\nOne.\n\nTwo.", plan_only=True)[2]
    check("a clean script is not nagged", "carry more than one line" not in clean)


def test_person_described_once_end_to_end():
    print("\n=== two heads: one person described twice ===")
    P = "A basement.\n\nMaya: 27, silver hair, grey coat.\n\nMaya lies still on the floor."
    imgs, audio, info, script = run_node(
        P, plan_only=True, character_memory="Maya: 27, silver hair, grey coat.")[:4]
    shot = " ".join([x for x in re.split(r"(?=\[Shot )", script) if x.strip()][0].split())
    check("the person is described once", shot.count("Maya:") == 1)
    check("...and the duplicate is reported", "described more than once" in info)
    check("...naming who", "Maya described" in info)
    # No terminator on a sheet line welds it onto the beat: "grey coat Maya lies
    # still", which reads as one more item in the attribute list.
    s2 = run_node("A basement.\n\nMaya lies still on the floor.", plan_only=True,
                  character_memory="Maya: 27, silver hair, grey coat")[3]
    check("the sheet does not run into the beat", "grey coat Maya" not in s2)
    # THE GUARANTEE is the terminator, not what happens to sit after it: beat_leads
    # puts the beat ahead of the sheet, so the sheet no longer runs into it at all.
    check("...it is ended properly", "grey coat." in s2)
    # Using one channel only is unaffected.
    s3 = run_node(P, plan_only=True)[3]
    check("one channel alone still describes the person", "Maya: 27" in s3)
    check("...once", " ".join(s3.split()).count("Maya: 27") == 1)


def test_undressing_completely_end_to_end():
    print("\n=== undressing takes ALL of it off, and only theirs ===")
    mem = ("Nora: 34, she, tall, red hair, green canvas jacket, grey wool jumper, "
           "white t-shirt, black jeans, brown leather boots.\n"
           "Victor: he, 41, dark hair, navy overalls, tan work shoes")
    P = "\n\n".join(["Nora walks in and sets a toolbox on the bench.",
                     "Nora undresses completely and steps into the shower.",
                     "Nora reaches for a towel.",
                     "Victor walks in carrying a coil of cable."])
    info, script = run_node(P, plan_only=True, anchor="A room.",
                            character_memory=mem)[2:4]
    sh = [" ".join(x.split()) for x in re.split(r"(?=\[Shot )", script) if x.strip()]
    check("she is dressed to begin with", "green canvas jacket" in sh[0])
    for _g in ("jacket", "jumper", "t-shirt", "jeans", "boots"):
        check(f"{_g} is gone from the undressing shot", _g not in sh[1])
    # It has to STAY off: the scene is re-stamped into every later shot, so a garment
    # left in it is a garment back on.
    check("...and stays gone afterwards",
          not any(g in sh[2] for g in ("jacket", "jumper", "jeans", "boots")))
    # Scoped to the people the beat names. Undressing one person must not take the
    # other one's clothes off.
    check("the other character keeps his clothes", "navy overalls" in sh[3])
    # Said once, rather than reciting the wardrobe back at a shot whose point is that
    # there is none of it.
    check("one sentence, not five", sh[1].count("comes off during this shot") == 1)
    check("...and it is the bare one", "leaving bare skin" in sh[1])
    check("info names what it cleared", "read off the character sheet" in info)
    check("...and lists the garments", "jacket, jumper, t-shirt, jeans, boots" in info)


def test_a_tagged_object_comes_off_and_goes_back_on():
    print("\n=== an object's reference follows the object ===")
    # A <Picture N> can be attached to a THING, not only a person: "a silver locket
    # <Picture 2>" is a picture of the locket. It has to come off when the locket does
    # -- left behind, the reference keeps asserting what was just removed -- and come
    # back when an 'add:' puts it on again.
    mem = "Nora: <picture 1>, 34, she, red hair, a silver locket <picture 2>, green jacket"
    P = "\n\n".join([
        "Nora stands by the bench.",
        "Nora unclasps the silver locket and sets it down.\nremove: locket",
        "Nora looks out of the window.",
        "Nora picks it up again.\nadd: her silver locket <picture 2> is back around her neck",
        "Nora walks to the door."])
    script = run_node(P, plan_only=True, anchor="A workshop.", character_memory=mem,
                      ref_image_1=torch.rand(1, H, W, 3),
                      ref_image_2=torch.rand(1, H, W, 3))[3]
    tags = [re.findall(r"<Picture \d+>", " ".join(x.split()))
            for x in re.split(r"(?=\[Shot )", script) if x.strip()]
    check("both references start on", tags[0] == ["<Picture 1>", "<Picture 2>"],
          str(tags[0]))
    check("the object's reference goes with the object",
          tags[2] == ["<Picture 1>"], str(tags[2]))
    check("...and the person's stays throughout",
          all("<Picture 1>" in t for t in tags), str(tags))
    # Putting it back on. This could not work before: the token stayed in `gone` for
    # the rest of the film, so an 'add:' naming it was suppressed by the very removal
    # it was undoing.
    check("an add puts the object and its reference back",
          tags[3] == ["<Picture 1>", "<Picture 2>"], str(tags[3]))
    check("...and it stays on after that",
          tags[4] == ["<Picture 1>", "<Picture 2>"], str(tags[4]))


def test_hardware_stays_on_its_owner():
    print("\n=== hardware does not spread to the other character ===")
    # Reported: a belt locked onto one character turned up on the other, over their
    # clothes. The hold named nobody, so in a two-person shot it was an instruction
    # about whoever was on screen.
    mem = ("Nora: 34, she, red hair, bare skin, a locked steel waist belt.\n"
           "Victor: he, 41, dark hair, navy overalls, work boots")
    P = ("Nora stands by the bench, the steel belt locked on her hips.\n\n"
         "Victor walks in through the side door and looks at her.")
    sh = [" ".join(x.split()) for x in
          re.split(r"(?=\[Shot )", run_node(P, plan_only=True, anchor="A workshop.",
                                            character_memory=mem)[3]) if x.strip()]
    check("the shot with one person keeps the plain hold",
          "Every restraint stays closed" in sh[0])
    check("...and spends no words naming whose", "Every restraint on Nora" not in sh[0])
    check("the shot with two names the wearer", "Every restraint on Nora" in sh[1])
    # Once. The second sentence saying the same thing cost another naming of her, and
    # a described person is a person the model draws -- reported as a second girl
    # appearing at the moment of cuffing.
    check("...naming her once, not twice",
          len(re.findall(r"\bNora\b", sh[1].split("Nora:")[-1])) == 1, sh[1][-90:])
    check("...pinning the other to his own entry",
          "exactly what their own entry lists" in sh[1])
    # And his entry is still there to be pinned to.
    check("the other character keeps his clothes described", "navy overalls" in sh[1])


def test_a_state_in_the_scene_is_not_reasserted():
    print("\n=== a state changed in one beat is not re-asserted later ===")
    # The van usually stands in the SCENE paragraph, which is prepended to every
    # shot. So the text saying "doors closed" is in shot 3 as much as shot 1 -- and
    # by shot 3 the doors have been opened. Asserting the written state there would
    # shut them again, which is the reported bug wearing the other shoe.
    P = ("Daylight. A yard, and a van with its doors closed.\n\n"
         "Mara and Dom stand behind the van.\n\n"
         "Mara opens the van doors and climbs in.\n\n"
         "Dom looks back at the yard.")
    shots = [s for s in run_node(P, plan_only=True)[3].split("---") if s.strip()]
    check("three shots", len(shots) == 3, "")
    check("shot 1 is told the doors are already closed",
          "already closed at the first frame" in shots[0], "")
    # The beat that opens them is asking for that motion. Holding the state here
    # would be the node arguing with the script. It gets the two ENDS of the change
    # instead, which is a different sentence and the subject of its own test.
    check("the shot that opens them is not told the state holds",
          "already closed" not in shots[1], "")
    check("...and the shot after is not told the old state",
          "first frame" not in shots[2], "")
    # The author's own words still reach the model verbatim -- the node adds nothing
    # and takes nothing away. It is only the ADDED sentence that stops.
    check("the scene text itself is untouched",
          all("doors closed" in s for s in shots), "")
    info = run_node(P, plan_only=True)[2]
    check("info names the shot", "shot(s) 1 describe scenery in a state" in info, "")
    check("...and names the switch", "hold_scene_state" in info, "")
    off = run_node(P, plan_only=True, hold_scene_state=False)[3]
    check("the switch turns it off", "first frame" not in off, "")
    check("...and changes nothing else", off.count("doors closed") == 3, "")


def _prompts_sent(P, **kw):
    """Every prompt build_conditioning actually received, in shot order."""
    seen = []
    orig = S.build_conditioning
    def spy(clip, vae, audio_vae, prompt, *a, **k):
        seen.append((prompt, len(k.get("refs") or [])))
        return orig(clip, vae, audio_vae, prompt, *a, **k)
    S.build_conditioning = spy
    try:
        out = run_node(P, **kw)
    finally:
        S.build_conditioning = orig
    return seen, out[3]


def test_the_cuffs_stay_in_the_picture():
    print("\n=== the hardware is named on every shot it is on ===")
    # Reported after the sheet stopped listing the item: the cuffs disappeared while
    # she still looked restrained. The sheet had been the thing naming the object in
    # every shot. Taking it off the sheet is right -- it put the cuffs in the shots
    # before they went on -- and naming it here is what that costs.
    mem = "Mara: she, 22, grey dress.\nDan: he, 41."
    P = ("A bare room.\n\nMara backs away from Dan.\n\n"
         "Dan catches her and cuffs her wrists behind her back.\n\n"
         "Mara sits on the crate.\n\nMara looks at the door.")
    sh = [s for s in run_node(P, plan_only=True, character_memory=mem)[3].split("---")
          if s.strip()]
    check("before it goes on, nothing is claimed", "The cuffs stay" not in sh[0], "")
    check("the applying shot says it in the beat", "cuffs" in sh[1].lower(), "")
    check("...and is not told it a second time", "The cuffs stay" not in sh[1], "")
    for i in (2, 3):
        check(f"shot {i + 1} names the hardware", "cuffs" in sh[i].lower(), sh[i][-70:])
        check(f"...as the object, not just a category", "The cuffs stay" in sh[i], "")
    # A removal lets go of it, like every other latch here.
    P2 = ("A bare room.\n\nDan cuffs her wrists behind her back.\n\nMara sits.\n\n"
          "remove: cuffs\nDan takes the cuffs off.\n\nMara stands up.\n\nMara walks out.")
    sh2 = [s for s in run_node(P2, plan_only=True, character_memory=mem)[3].split("---")
           if s.strip()]
    check("held while they are on", "The cuffs stay" in sh2[1], "")
    check("let go after the removal",
          all("The cuffs stay" not in s for s in sh2[2:]), "")
    # Nothing restrained anywhere: this must not fire on an ordinary scene.
    plain = run_node("A room.\n\nMara waits.\n\nMara walks to the window.",
                     plan_only=True, character_memory=mem)[3]
    check("an unrestrained scene is untouched", "The cuffs stay" not in plain, "")


def test_a_sheet_that_claims_hardware_too_early():
    print("\n=== the sheet listing cuffs she has not been put in yet ===")
    # Why the applying fix does not reach a scene written this way: the sheet lists
    # the hardware, the sheet goes into EVERY shot, so she is restrained from shot 1
    # and the cuffing shot is told the restraint is already fastened. That renders as
    # restrained first and caught afterwards. The sheet is the author's standing
    # description and the beat is the author's action -- the node reports the clash
    # rather than picking a winner.
    mem = "Mara: she, 22, grey dress, handcuffs on her wrists.\nDan: he, 41."
    P = ("A bare room.\n\nMara backs away from Dan.\n\n"
         "Dan catches her and cuffs her wrists behind her back.")
    info, script = run_node(P, plan_only=True, character_memory=mem)[2:4]
    check("the clash is reported", "already lists it as worn" in info, "")
    check("...naming the shot that stages it", "shot(s) 2 stage hardware going ON" in info, "")
    # The sheet really is in the shot before it happens -- that is the point.
    sh = [s for s in script.split("---") if s.strip()]
    check("the cuffs are described before they go on", "handcuffs" in sh[0].lower(), "")
    # A clean sheet gets the applying clause and no complaint.
    clean = run_node(P, plan_only=True, character_memory="Mara: she, 22.\nDan: he, 41.")
    check("a clean sheet raises nothing", "already lists it as worn" not in clean[2], "")
    check("...and gets both ends on the applying shot",
          "hardware goes on during this shot" in clean[3], "")
    # Reported once. It is one authoring decision, not one per shot.
    many = run_node(P + "\n\nDan locks the cuffs tighter.\n\nDan checks them again.",
                    plan_only=True, character_memory=mem)[2]
    check("said once, not per shot", many.count("already lists it as worn") == 1, "")


def _shots(P, **kw):
    return [x for x in re.split(r"(?=\[Shot )", run_node(P, plan_only=True, **kw)[3])
            if x.strip()]


def test_what_comes_off_in_the_bathroom_stays_off():
    """REPORTED: she undresses in the bathroom, steps into the shower, and her thong
    is back in the next beat. It was supposed to stay off.

    The thong really was scrubbed out of the text -- that part worked. What brought it
    back was the SILENCE where it had been. The region table could not place any
    lower-body underwear, so nothing latched the hips as bare; and bare_hold, which
    says so on every shot after the removal, went quiet for any region the sheet named
    a layer under -- whether or not that layer had come off as well. A full strip is
    the case it matters most in, and it was the one case it could not speak in.

    So the shot said the chest was bare, the feet were bare, and nothing at all about
    the hips. An unspecified region is filled by the model's own prior, the prior for
    a hip is underwear, and the keyframe carried the invention into every shot after
    it. Nothing restored the thong. The prior drew one."""
    print("\n=== what comes off in the bathroom stays off ===")
    mem = "Kate: she, 28, a grey t-shirt, a denim skirt, a black thong, sandals."
    P = ("A tiled bathroom, warm light.\n\n"
         "Kate takes off her clothes and stands naked.\n\n"
         "Kate steps into the shower and the water runs over her.\n\n"
         "Kate turns under the water.\n\n"
         "Kate runs her hands through her wet hair.")
    sh = _shots(P, character_memory=mem)
    check("every beat reaches rendering", len(sh) == 4, str(len(sh)))
    # The text side, which already worked: the garment is out of the sheet.
    for _i in range(1, 4):
        check(f"shot {_i + 1} no longer describes the thong",
              "thong" not in sh[_i].lower(), sh[_i][:120])
    # ...and the half that did not: the space it left is SAID, on every shot after.
    for _i in range(1, 4):
        check(f"shot {_i + 1} says the hips and legs are bare",
              "legs are bare from the hip down" in sh[_i], sh[_i][-160:])
        check(f"...and the chest too", "chest, shoulders and arms are bare" in sh[_i],
              sh[_i][-160:])
    # Positively phrased, to the last clause: this sentence exists to keep the prior
    # out of a region, and at cfg 1 a negation in it hands the prior the word back.
    check("the bare clause names skin rather than what is absent",
          all("outermost surface there" in s for s in sh[1:]), sh[1][-90:])
    check("...and carries no negation",
          all(not re.search(r"\bnothing\b|\bno longer\b", s.split("] ", 1)[1])
              for s in sh[1:]), sh[1][-120:])
    # THE BRA HALF OF THE SAME REPORT. bare_hold was written for "a bra coming back on
    # somebody topless", and the suppression meant it only ever worked for a sheet
    # that did not layer the bra under anything -- which is the commonest way to
    # write one.
    bra = "Kate: she, 28, a grey t-shirt, a black bra, jeans."
    sh2 = _shots("A bedroom.\n\nKate pulls off the t-shirt and unhooks the bra.\n\n"
                 "Kate sits on the end of the bed.\n\nKate looks at the window.",
                 character_memory=bra)
    for _i in (1, 2):
        check(f"topless shot {_i + 1} says the chest is bare",
              "chest, shoulders and arms are bare" in sh2[_i], sh2[_i][-160:])
    # AND THE LAYER THAT IS STILL ON MUST NOT GET THIS. reveal_clause speaks when
    # something is underneath; this clause speaks when nothing is, and the two may
    # never both speak about one region.
    sh3 = _shots("A bedroom.\n\nKate unzips the denim skirt and lets it fall.\n\n"
                 "Kate sits down.\n\nKate looks up.",
                 character_memory="Kate: she, 28, a denim skirt, a black thong.")
    check("the uncovering shot says what shows there now",
          "thong underneath is what shows" in sh3[0], sh3[0][-160:])
    check("...and never also calls that region bare",
          all("legs are bare from the hip down" not in s for s in sh3),
          sh3[1][-160:])
    check("...and the thong stays described, because it is still on",
          all("thong" in s.lower() for s in sh3[1:]), sh3[1][:140])


def test_the_thong_comes_off_however_it_is_written():
    """The other half: eight of twenty ways people write underwear coming off did
    nothing at all, so the garment stayed in the sheet -- and the sheet goes into
    every shot, which is the garment back on and staying on.

    Worst of them, "lets the thong fall to the floor", matched the RESTORE
    vocabulary: the node read the author's removal and enacted its opposite."""
    print("\n=== the thong comes off however the beat is written ===")
    mem = "Kate: she, 28, a denim skirt, a black thong."
    for _beat in ("Kate hooks her thumbs in the thong and steps out of it.",
                  "Kate pushes the thong down her legs.",
                  "Kate lets the thong fall to the floor.",
                  "Kate shimmies out of the thong.",
                  "Kate pushes the thong off her hips."):
        # The skirt goes first, in a wording that always worked, so what is under
        # test is the second beat alone. It matters that the skirt IS off: while it
        # is on, the hips are covered by it and calling them bare would be a lie --
        # which is the one thing the suppression this work relaxed was right about.
        sh = _shots("A tiled bathroom with a wet floor.\n\n"
                    "Kate unzips the denim skirt and steps out of it.\n\n" + _beat
                    + "\n\nKate steps into the shower.\n\nKate turns under the water.",
                    character_memory=mem)
        check(f"off and stays off: {_beat[5:46]!r}",
              all("thong" not in s.lower() for s in sh[2:]), sh[2][:130])
        check(f"...and the hips are named instead: {_beat[5:34]!r}",
              all("legs are bare from the hip down" in s for s in sh[2:]),
              sh[2][-150:])
    # A FIXTURE IS NOT A GARMENT. A removal is read by POSITION, not by vocabulary,
    # so that an author can write a garment this file has never heard of -- and
    # position cannot tell a shower from a shirt. "Kate steps out of the shower" is
    # the same sentence shape as "Kate steps out of the thong", and the scene lists
    # the shower the way a sheet lists a skirt, so the shower was undressed off her
    # and scrubbed out of every later shot. In a bathroom scene.
    room = ("A tiled bathroom. A glass shower, a bath, a wooden stool.\n\n"
            "Kate steps out of the shower.\n\nKate reaches for the towel.\n\n"
            "Kate looks in the mirror.")
    sh4 = _shots(room, character_memory=mem)
    for _i, _s in enumerate(sh4):
        check(f"shot {_i + 1} still has its shower", "shower" in _s.lower(), _s[:130])


def test_a_breath_does_not_hold_the_branch_open():
    """REPORTED: micro-babble at the start of a scene, as somebody goes to talk.

    "Dana takes a breath." is the beat people write immediately before a line,
    and it read as the author asking for a sound -- so the audio branch stayed
    open for the whole shot with half a second of breath in it. An open branch on
    a joint model fills itself, and at 4-8 steps the last audio step clears
    30-50% of the denoising in one jump, so what it fills with is a voice."""
    print("\n=== a breath does not hold the branch open ===")
    for beat in ("Dana takes a breath.", "Dana draws breath to answer.",
                 "Dana takes a deep breath and turns to her.",
                 "Dana catches her breath."):
        check(f"closed: {beat!r}", not S.sound_described(beat))
    # SUSTAINED breathing is the sound of its shot and stays. So does everything
    # somebody asked for by name -- silencing those would be taking away what was
    # written, which is a different bug from the one being fixed.
    for beat in ("She breathes hard through the gag.", "Dana is breathing hard.",
                 "Dana sighs.", "Dana gasps.", "Dana moans.", "The door slams."):
        check(f"open: {beat!r}", S.sound_described(beat))
    # ...and not when something else in the beat lasts.
    check("a breath beside a chain still opens it",
          S.sound_described("Dana takes a breath as the chain rattles."))
    # The trade is REPORTED: a breath that will not be heard is a change to what
    # was written, and finding that out from the render is worse than reading it.
    info = run_node("A room.\n\nDana takes a breath.\n\nDana says: \"Listen.\"",
                    plan_only=True, character_memory="Dana: she, 35.")[2]
    check("the silenced breath is reported", "stage a breath" in info)
    check("...and says how to get it back", "in the same beat as the line" in info)
    quiet = run_node("A room.\n\nDana waits.", plan_only=True,
                     character_memory="Dana: she, 35.")[2]
    check("...and a beat with no breath says nothing", "stage a breath" not in quiet)


def test_appearing_is_not_arriving():
    """REPORTED: ghosting on a character introduction.

    A staged ARRIVAL keeps the previous frame as the keyframe, because somebody
    walking in through a door has a path into a frame that does not have them in
    it. "Appears", "shows up", "turns up" describe the result and not the
    movement -- there is no path, so the only way into that frame is to fade up
    inside it, which is what ghosting is."""
    print("\n=== appearing is not arriving ===")
    for beat, want in (("McKenna walks in through the door.", True),
                       ("McKenna enters the room.", True),
                       ("McKenna comes into the room.", True),
                       ("McKenna follows her in.", True),
                       ("McKenna appears in the doorway.", False),
                       ("McKenna shows up at the door.", False),
                       ("McKenna turns up beside her.", False),
                       ("McKenna appears.", False)):
        check(f"{beat!r} arrives={want}", S.arrives_in(beat) is want,
              str(S.arrives_in(beat)))
    # "appears" is also the ordinary word for SEEMS, and cancelling a real
    # entrance on it loses the arrival it was describing.
    check("a real entrance survives an opinion",
          S.arrives_in("McKenna walks in and appears calm.") is True)
    check("...and so does this one",
          S.arrives_in("McKenna enters and appears nervous.") is True)


def test_a_line_is_marked_however_it_is_punctuated():
    """REPORTED: dialogue duplication and words pronounced wrongly.

    '"Come here," Dana says.' is how half of written dialogue is punctuated, and
    only the text BEFORE a quote was consulted for a speech cue -- so that form
    was never wrapped in <d>...</d> at all. Quotation marks say nothing to the
    model; an unmarked line is one the audio branch was never told is spoken."""
    print("\n=== a line is marked however it is punctuated ===")
    for beat in ('"Come here," Dana says.', '"Come here," she says.',
                 '"Come here," Dana said quietly.'):
        check(f"marked: {beat!r}", "<d>" in S.mark_dialogue(beat),
              S.mark_dialogue(beat))
    check("the cue-first form still works",
          S.mark_dialogue('Dana says: "Come here."').count("<d>") == 1)
    # A scare quote is not a line, whichever side the words are on.
    for beat in ('She wore a "vintage" coat.', 'He called it a "problem" yesterday.',
                 'The sign said "exit" in red.'):
        check(f"not a line: {beat!r}", "<d>" not in S.mark_dialogue(beat),
              S.mark_dialogue(beat))
    # A SHORT LINE IN A LONG SHOT is where doubled dialogue comes from: the audio
    # branch runs the whole shot and fills what is left with the line again.
    # Reported, not clause -- "the line said once" as prompt text made it worse.
    mem = "Dana: she, 35."
    thin = run_node("A room.\n\nDana says: \"No.\"\n\nDana waits.",
                    plan_only=True, character_memory=mem)[2]
    check("a line that does not fill its shot is reported",
          "no line in it" in thin, thin[-200:])
    full = run_node("A room.\n\nDana says: \"I told you last night that this was "
                    "going to happen and you did not listen to a word of it.\"",
                    plan_only=True, character_memory=mem)[2]
    check("...and a line that does fill it is not", "no line in it" not in full)
    # Numbers and abbreviations have no single spoken form, and the model picks.
    hard = run_node('A room.\n\nDana says: "Dr. Vale gets here at 7:30."',
                    plan_only=True, character_memory=mem)[2]
    check("unsayable text in a line is reported", "no single way" in hard)
    check("...naming what it found", "7:30" in hard and "Dr." in hard)
    soft = run_node('A room.\n\nDana says: "Doctor Vale gets here at half seven."',
                    plan_only=True, character_memory=mem)[2]
    check("...and spelled-out dialogue is clean", "no single way" not in soft)
    # Only inside the quotes: narration is never spoken, so its digits are fine.
    narr = run_node('A room.\n\nDana checks the clock at 7:30.\n\n'
                    'Dana says: "You should have told me."',
                    plan_only=True, character_memory=mem)[2]
    check("...and narration is left alone", "no single way" not in narr)


def test_a_two_word_sheet_name_does_not_duplicate_her():
    """REPORTED: a duplicate Mistress, in the FIRST beat -- so nothing to do with
    keyframes or carried frames. sheet_lines took a one-word name, "Mistress
    Vale:" parsed as unlabelled, and an unlabelled line is global: her whole
    description went into every shot with no name attached."""
    print("\n=== a two-word sheet name does not duplicate her ===")
    mem = "Mistress Vale: she, 38, tall, dark hair, a black dress.\nAna: she, 24."
    got = _shots("A panelled study.\n\nThe Mistress stands at the window.\n\n"
                 "Ana kneels by the desk.", character_memory=mem)
    check("the sheet line is attributed, not global",
          "tall, dark hair" not in got[1], " ".join(got[1].split())[:170])
    check("...and she is described in her own shot",
          "tall, dark hair" in got[0], " ".join(got[0].split())[:170])
    # The old failure in one assertion: two descriptions of a woman in one shot.
    both = "Mistress: she, 38, a black dress.\nMistress Vale: she, 38, tall, dark hair."
    one = _shots("A panelled study.\n\nThe Mistress stands at the window.",
                 character_memory=both)
    check("one beat naming one woman describes one woman",
          one[0].count("she, 38") == 1, " ".join(one[0].split())[:200])


def test_dan_is_not_instantiated_twice():
    print("\n=== an exact pair gets one body each ===")
    mem = ("Dan: He, 35, black t-shirt, blue jeans, black shoes.\n\n"
           "Crystal: She, 35, white t-shirt, blue jeans, white shoes.")
    shot = _shots("In a home, Dan and Crystal are in the kitchen.",
                  character_memory=mem)[0]
    check("the exact cast is constrained",
          "There are two people in the shot" in shot, shot)
    check("the constraint asks for one body each",
          "one body for each person" in shot, shot)
    solo = _shots("In a home, Dan is in the kitchen.", character_memory=mem)[0]
    check("the constraint does not invent Crystal in a solo shot",
          "two people in the shot" not in solo, solo)


def test_a_carried_room_is_not_a_second_picture_of_somebody():
    """REPORTED: a duplicate Mistress.

    The room-carry sends the previous shot's last frame as a REFERENCE, so a shot
    introducing somebody in position keeps the set instead of re-imagining it. If
    the person in that frame already has a portrait on the sheet, the shot gets
    two pictures of her -- her <Picture 1> and the carried frame -- and two
    pictures of one person is how a second one gets drawn. The recovered-frame
    path skips tagged people for exactly this reason; this was written without
    that skip."""
    print("\n=== a carried room is not a second picture of somebody ===")
    sent = []
    _bc = S.build_conditioning

    def spy(clip, vae, avae, prompt, w, h, length, **kw):
        sent.append((len(kw.get("refs") or []), bool(kw.get("handoff_as_ref"))))
        return _bc(clip, vae, avae, prompt, w, h, length, **kw)

    P = ("A panelled study.\n\nThe Mistress stands at the window.\n\n"
         "Ana is already kneeling by the desk, watching the Mistress.\n\n"
         "Ana lowers her eyes.")
    S.build_conditioning = spy
    try:
        run_node(P, character_memory=("Mistress: she, 38, a black dress. <Picture 1>\n"
                                      "Ana: she, 24, a shirt."),
                 ref_image_1=torch.rand(1, H, W, 3))
    finally:
        S.build_conditioning = _bc
    check("the shot that introduces somebody carries no second picture of her",
          not any(as_ref for _n, as_ref in sent), str(sent))
    check("...and her own portrait still goes in", sent[1][0] == 1, str(sent))
    # Untagged: nothing carries her identity, so the room IS carried -- that is
    # the continuity fix and it has to survive this.
    sent2 = []
    S.build_conditioning = lambda c, v, a, p, w, h, l, **kw: (
        sent2.append(bool(kw.get("handoff_as_ref"))) or _bc(c, v, a, p, w, h, l, **kw))
    try:
        run_node(P, character_memory="Mistress: she, 38, a black dress.\nAna: she, 24.")
    finally:
        S.build_conditioning = _bc
    check("with no portrait in play the room is still carried", any(sent2), str(sent2))


def test_a_bare_region_is_said_on_every_shot():
    """REPORTED: a bra comes back on somebody topless, and the sheet never had one.

    Nothing restored it -- the model INVENTED it, because the clause saying the
    chest is uncovered was only ever on the beat that uncovered it. From the next
    shot that region was unspecified, and an unspecified region is filled from the
    model's own prior.

    End to end on purpose. The state half of this lives in the engine and is
    tested there; the clause is written here, and an engine test passes whether
    or not the shot ever says it."""
    print("\n=== a bare region is said on every shot ===")
    mem = "Kate: she, 24, a shirt, jeans.\nSam: he, 40."
    got = _shots("A bare room.\n\nKate is topless, sitting on the crate.\n\n"
                 "Kate looks at the door.\n\nKate listens.", character_memory=mem)
    for i, s in enumerate(got, 1):
        check(f"shot {i} says the chest is bare", "chest" in s and "bare" in s,
              " ".join(s.split())[-120:])
    check("...and it names the chest, not just the shoulders",
          all("chest, shoulders and arms are bare" in s for s in got))
    # Undressing reaches the same place, and the sheet must not re-dress her.
    off = _shots("A bare room.\n\nKate takes off her shirt.\n\n"
                 "Kate looks at the door.\n\nKate listens.", character_memory=mem)
    check("a removal is held the same way",
          all("chest" in s for s in off), " ".join(off[-1].split())[-120:])
    # THE BEAT THAT NAMES ONLY THE OTHER PERSON. Reported after the first fix
    # shipped: "bra still popped into ONE beat". That beat is this one -- she is
    # not in its cast, so the clause was skipped, and the keyframe still shows
    # her, so the region went unspecified for exactly one shot and the prior
    # filled it. Attributed, because the shot describes somebody else.
    other = _shots("A bare room with a crate.\n\n"
                   "Kate takes off her shirt and drops it on the crate.\n\n"
                   "Sam watches from the doorway.\n\n"
                   "Kate turns towards him.", character_memory=mem)
    check("a beat naming only the other person still says it",
          "chest" in other[1], " ".join(other[1].split())[-140:])
    check("...and says WHOSE chest it is", "Kate's chest" in other[1],
          " ".join(other[1].split())[-140:])
    check("...and names her when both are described",
          "Kate's chest" in other[2], " ".join(other[2].split())[-140:])
    # ...and dressing again stops it, or she is told her chest is bare over a shirt.
    back = _shots("A bare room.\n\nKate takes off her shirt.\n\n"
                  "Kate puts on her shirt.\n\nKate listens.", character_memory=mem)
    check("dressing again stops the clause", "chest" not in back[-1],
          " ".join(back[-1].split())[-120:])


def test_a_squat_survives_speech_and_undressing():
    """REPORTED: she does not stay squatting, she stands up on her own.

    posture_cleared read the whole beat INCLUDING quoted speech, so a line with a
    travel word in it cleared the pose -- "Someone is coming" is about somebody
    else. And "takes off her shirt" matched the travel list on "takes", so
    undressing cleared it too, which is why the two reports arrived together."""
    print("\n=== a squat survives speech and undressing ===")
    mem = "Kate: she, 24, a shirt, jeans.\nSam: he, 40."
    got = _shots("A bare room.\n\nKate squats down beside the crate.\n\n"
                 "Kate says: \"Someone is coming.\"\n\n"
                 "Kate takes off her shirt.\n\nKate listens.", character_memory=mem)
    # got[0] stages the squat and has the author's own words for it; the hold
    # starts on the shot after.
    check("the author's word is kept, not swapped for a crouch",
          "is squatting" in got[1], " ".join(got[1].split())[-120:])
    check("a spoken travel word does not stand her up",
          "is squatting" in got[2], " ".join(got[2].split())[-120:])
    check("...nor does taking a garment off",
          "is squatting" in got[3], " ".join(got[3].split())[-120:])
    # Real travel still clears it: a hold that survives walking away is a hold
    # arguing with its own shot.
    walk = _shots("A bare room.\n\nKate squats down beside the crate.\n\n"
                  "Kate walks to the door.\n\nKate listens.", character_memory=mem)
    check("walking away does clear it", "is squatting" not in walk[2],
          " ".join(walk[2].split())[-120:])


def test_the_audit_findings_stay_fixed():
    print("\n=== nine defects found by audit, each held down ===")
    mem3 = "Mara: she, 22.\nKate: she, 20.\nDan: he, 41."
    def sh(P, **kw):
        return [s for s in run_node(P, plan_only=True, **kw)[3].split("---") if s.strip()]

    # 1. restrained_who latched only at the FIRST transition, so a second person
    #    cuffed later was never recorded and their hardware went undescribed forever.
    got = sh("A room.\n\nDan cuffs Mara's wrists.\n\nDan cuffs Kate's wrists too.\n\n"
             "Kate sits alone.\n\nMara looks up.", character_memory=mem3)
    check("a second person cuffed later is held too",
          all("closed and fastened" in s or "hardware goes on" in s for s in got),
          str([("closed and fastened" in s) for s in got]))

    # 2. worn_item was one string, so a second item overwrote the first and the cuffs
    #    stopped being named from the gag onward.
    # Two people, not three: with two women on the sheet "her" is ambiguous, the guard
    # resolves nobody, and that is a separate problem the node already reports.
    last = sh("A room.\n\nDan cuffs her wrists.\n\nDan gags her with duct tape.\n\n"
              "Mara sits still.",
              character_memory="Mara: she, 22.\nDan: he, 41.")[-1].lower()
    check("both items stay named", "cuff" in last and "tape" in last, last[-90:])
    # Mixed hardware and soft goods is described as hardware: steel that is only
    # "tied and holding" is steel nobody has said is closed.
    check("...and cuffs with tape are still closed and fastened",
          "closed and fastened" in last, last[-90:])

    # 3. A displaced outer garment never uncovered what was under it: the beat showed
    #    the thong in its own words and the layering hid it again the next shot.
    m = "Mara: she, 22, denim shorts, a black thong."
    got = sh("A room.\n\nMara stands.\n\nMara pulls her shorts down to show the thong.\n\n"
             "Mara waits.\n\nMara pulls them back up.\n\nMara stands.", character_memory=m)
    check("displacement uncovers what is under it",
          "thong" in got[2].lower(), got[2][-80:])
    # UNDERWEAR IS PLACED, NOT DELETED, as of the report that said so for
    # the third time. It stays in the text and the node says where it is;
    # the fact protected is the same one, that it is not rendered on top of
    # what covers it. A locket or a scarf under a coat still waits.
    check("...and putting it back covers it again",
          "whole, opaque and unbroken" in got[4].lower(), got[4][-130:])

    # 4. `gone` only ever grew, so an add: could not re-cover.
    got = sh("A room.\n\nMara pulls off her shorts.\n\nMara waits.\n\n"
             "add: her denim shorts are back on\nMara pulls her shorts on.\n\nMara stands.",
             character_memory=m)
    check("an add: re-covers the layer under it",
          "whole, opaque and unbroken" in got[3].lower(), got[3][-130:])

    # 5. Stockings stop at the thigh and cover no waistband.
    check("stockings do not cover a belt",
          not S.implied_layers("Mara: she, stockings, a chastity belt."))
    check("...while tights do",
          S.implied_layers("Mara: she, tights, a chastity belt.").get("chastity belt"))

    # 6. One of two speaking made the shot a speaking shot, so the LISTENER's mouth
    #    was left free -- which is where invented lip-sync lands.
    one = sh('A room.\n\nMara says: "Wait." Dan listens.',
             character_memory="Mara: she, 22.\nDan: he, 41.")[0]
    check("the listener's mouth is closed", "Only Mara speaks" in one, one[-80:])
    both = sh('A room.\n\nMara says: "Wait." Dan says: "No."',
              character_memory="Mara: she, 22.\nDan: he, 41.")[0]
    check("...and nobody is closed when both speak", "Only" not in both, "")

    # 7. Rope is tied, not closed and fastened.
    rope = sh("A room.\n\nHer wrists are tied above her head with rope.\n\nMara pulls at it.",
              character_memory="Mara: she, 22.")[1]
    check("rope is tied, not fastened shut",
          "tied and holding" in rope and "closed and fastened" not in rope, rope[-90:])

    # 8. Whether the chain actually restarted was invisible.
    on = run_node("A room.\n\nMara pulls off her coat.\n\nMara waits.", plan_only=True,
                  restart_after_removal=True,
                  character_memory="Mara: she, 22, a grey coat, white top.")[2]
    check("a restart is reported", "start FRESH" in on, "")

    # 9. The gaze target was stated once and dropped, while every other state latches.
    got = sh("A room.\n\nMara watches the TV.\n\nMara sits still.\n\nMara walks out.",
             character_memory="Mara: she, 22.")
    check("the look is held while she stays put",
          "turned to the TV" in got[1], got[1][-70:])
    check("...and let go when she leaves", "eyes and the head" not in got[2], "")


def test_no_cuffs_described_without_their_wearer():
    print("\n=== a shot without the restrained person is not told about cuffs ===")
    # Reported as nasty duplicates. The hold latched for the whole film, so a shot
    # describing only the other character still said cuffs were closed on wrists --
    # and with nobody in the text to own those wrists, the model draws one.
    mem = "Mara: she, 22, grey dress.\nDan: he, 41, dark coat."
    P = ("A bare room.\n\nMara waits by the window.\n\n"
         "Dan walks in and cuffs her wrists behind her back.\n\n"
         "Mara sits on the crate.\n\nDan checks the cuffs.\n\nMara looks up.")
    info, script = run_node(P, plan_only=True, character_memory=mem)[2:4]
    sh = [s for s in script.split("---") if s.strip()]
    hw = ["yes" if ("closed and fastened" in s or "hardware goes on" in s) else "no"
          for s in sh]
    check("before it goes on, nothing", hw[0] == "no", str(hw))
    check("the applying shot has it", hw[1] == "yes", str(hw))
    check("the wearer's own shots have it", hw[2] == "yes" and hw[4] == "yes", str(hw))
    check("the shot with only the other man does NOT",
          hw[3] == "no", sh[3][-90:])
    check("info names that shot", "shot(s) 4 describe nobody who is wearing" in info, "")
    # It LATCHES: leaving it out of his shot must not lose it for hers.
    check("...and it is back when she is", "closed and fastened" in sh[4], "")


def test_caught_first_then_restrained():
    print("\n=== the shot that puts the cuffs on gets both ends ===")
    mem = "Mara: she, 30.\nDan: he, 41."
    def kinds(P):
        script = run_node(P, plan_only=True, character_memory=mem)[3]
        return [("APPLY" if "hardware goes on during this shot" in s
                 else "HOLD" if "closed and fastened as" in s else "-")
                for s in script.split("---") if s.strip()]
    got = kinds("A living room.\n\nMara runs for the door. Dan catches her and cuffs "
                "her wrists.\n\nMara stands by the wall.\n\nMara pulls against the cuffs.")
    check("the applying shot is told both ends", got[0] == "APPLY", str(got))
    check("...and every shot after gets the standing hold",
          got[1:] == ["HOLD", "HOLD"], str(got))
    # Already wearing it: the standing hold is right and "off at the first frame"
    # would be a lie about somebody who has been in cuffs since the scene began.
    worn = kinds("A room. Mara is handcuffed to the rail.\n\n"
                 "Mara pulls against the cuffs.\n\nMara looks at the door.")
    check("hardware already worn is never called new", "APPLY" not in worn, str(worn))
    check("...and still gets the standing hold", worn == ["HOLD", "HOLD"], str(worn))
    plain = kinds("A room.\n\nMara walks to the window.\n\nMara sits down.")
    check("no hardware, no clause either way", plain == ["-", "-"], str(plain))
    info = run_node("A living room.\n\nDan catches her and cuffs her wrists.",
                    plan_only=True, character_memory=mem)[2]
    check("info names the shot", "shot(s) 1 put the hardware ON" in info, "")


def test_a_television_keeps_its_own_voice():
    print("\n=== the line on the TV does not come out of her mouth ===")
    mem = "Mara: she, 30."
    P = ('A living room.\n\nMara sits on the sofa. The TV says: "Storms tonight."\n\n'
         'Mara says: "Again?"\n\nThe TV plays in the empty room.')
    info, script = run_node(P, plan_only=True, character_memory=mem)[2:4]
    sh = [s for s in script.split("---") if s.strip()]
    check("the voice is given back to the set", "the TV's" in sh[0], sh[0][-90:])
    check("...and the mouths are held closed", "Mouths in the shot" in sh[0], "")
    # The branch must STAY OPEN. The set is supposed to be heard -- silencing it
    # would trade one wrong thing for another. Checked on its own, because the empty
    # room later in this prompt IS silenced and should be.
    solo = run_node('A living room.\n\nMara sits on the sofa. '
                    'The TV says: "Storms tonight."',
                    plan_only=True, character_memory=mem)[2]
    check("the shot is not silenced", "conditioned on real silence" not in solo, "")
    # The WORDS reach the model unchanged. The quotation marks do not: they are
    # exchanged for H3's own <d>...</d>, which is the token pair that says a span
    # is spoken. Quote marks say nothing to the model, and a quoted imperative was
    # reaching it as an imperative sentence and being performed.
    check("...and the line reaches the model as written",
          "Storms tonight." in sh[0] and "<d>Storms tonight.</d>" in sh[0], "")
    # Her own line is untouched: she is speaking, and her mouth must move.
    check("her own line is left alone", "the TV's" not in sh[1], "")
    check("...and her mouth is not held shut", "Mouths in the shot" not in sh[1], "")
    # Nobody in the beat, nothing about mouths -- ca75672 again.
    check("an empty room is told nothing about mouths", "Mouths in the shot" not in sh[2], "")
    check("info names the shot", "shot(s) 1 have a spoken line that belongs" in info, "")
    off = run_node(P, plan_only=True, character_memory=mem,
                   mouths_shut_when_no_line=False)[3]
    check("the switch turns it off", "the TV's" not in off, "")


def test_a_shifted_workflow_stops_before_rendering():
    print("\n=== a slid workflow fails with an explanation, not a bad render ===")
    # The exact shape from the report: values slid up one slot after a widget was
    # converted to an input. Rendering anyway would use a scheduler as a sampler and
    # settings nobody picked, and the output would look like a broken model.
    try:
        run_node("A room.\n\nMara waits.", plan_only=True,
                 resolution=0.7, sampler_name="beta", scheduler=48,
                 shot_length=True, cfg=float("nan"))
        check("it refuses to render", False, "no error raised")
    except RuntimeError as e:
        msg = str(e)
        check("it refuses to render", True, "")
        check("...naming the widgets that are wrong",
              "sampler_name" in msg and "scheduler" in msg, "")
        check("...the cause", "restored by POSITION" in msg, "")
        check("...and the fix", "Fix node (recreate)" in msg, "")
    # It must not fire on a healthy graph, or nobody can render at all.
    ok = run_node("A room.\n\nMara waits.", plan_only=True)
    check("a healthy workflow is untouched", "PLAN ONLY" in ok[2], "")
    # A NaN in a NUMBER is still repaired rather than refused -- that one is
    # recoverable, and sane_widgets says so in info.
    num = run_node("A room.\n\nMara waits.", plan_only=True, pace=float("nan"))
    check("a NaN number is still repaired, not refused",
          "not usable numbers" in num[2], "")


def test_the_removal_shot_says_what_is_under():
    print("\n=== taking the shorts off shows the panties, not skin ===")
    # Reported: the shorts come off and the render goes straight to bare, past the
    # underwear the sheet named. The removal clause is emphatic and specific -- off
    # the body, dropped out of frame -- while the layer beneath is one entry in an
    # attribute list, and against a prior that says trousers coming off means bare
    # skin, a list entry does not compete.
    mem = "Mara: she, 22, blue denim shorts, white top, black panties."
    P = "A room.\n\nMara stands.\n\nMara pulls off her shorts.\n\nMara turns."
    info, script = run_node(P, plan_only=True, character_memory=mem)[2:4]
    sh = [s for s in script.split("---") if s.strip()]
    check("the removing shot is told what shows there",
          "what shows there now" in sh[1], sh[1][-90:])
    check("...naming the layer", "panties underneath" in sh[1].lower(), "")
    check("...and saying it stays on", "on and unchanged" in sh[1], "")
    check("said only on the shot that uncovers it",
          "what shows there now" not in sh[0] and "what shows there now" not in sh[2], "")
    check("info names the shot", "take off a garment that was covering" in info, "")
    # Nothing underneath: there is nothing to promise, and promising anyway would
    # dress her in something the sheet never gave her.
    bare = run_node("A room.\n\nMara pulls off her shorts.\n\nMara turns.",
                    plan_only=True,
                    character_memory="Mara: she, 22, blue denim shorts, white top.")[3]
    check("nothing underneath, nothing claimed", "what shows there now" not in bare, "")
    # A full strip takes the cover AND what was under it. Saying the panties show
    # would put back the garment the beat was most explicit about removing.
    strip = run_node("A room.\n\nMara undresses completely.\n\nMara turns.",
                     plan_only=True, character_memory=mem)[3]
    check("a full strip promises nothing", "what shows there now" not in strip, "")


def test_a_beat_can_name_what_the_layering_hid():
    print("\n=== a beat naming a covered garment puts it back, and is reported ===")
    # Three fixes into "the belt is on top of the jeans", with the sheet correct, the
    # tag correct and the picture correctly withheld. This is the route none of them
    # could touch: beats are passed through word for word and are never scrubbed, so
    # the layering takes the belt out of the sheet and the beat puts it straight back.
    # A described thing is a drawn thing, drawn over whatever is on top of it -- and
    # the next shot starts from this one's last frame, so it is carried forward from
    # there and looks permanent rather than like one bad shot.
    mem = "Mara: <Picture 1>, she, 22, blue jeans, a chastity belt <Picture 2>."
    P = ("A room.\n\nMara stands by the window.\n\n"
         "Mara runs a hand over the chastity belt under her jeans.\n\nMara waits.")
    info, script = run_node(P, plan_only=True, character_memory=mem)[2:4]
    sh = [s for s in script.split("---") if s.strip()]
    # The sheet KEEPS it and the node says where it is. Deleting it was what
    # put the author's own words out of the prompt, three reports running.
    # The WORDS wait too. One mention at cfg 1 is enough to draw it, and the
    # sheet's was the last one left after the picture and the clause stopped
    # naming it. It returns on the shot the cover moves.
    check("the covered shot does not name it",
          "chastity belt" not in sh[0].lower(), sh[0][:200])
    # The clause names the COVER. The belt is named once, by the sheet.
    check("...and the jeans are described as covering it",
          "jeans cover the hips and waist completely" in sh[0].lower(),
          sh[0][:240])
    check("...and nothing else in her entry is touched",
          "blue jeans" in sh[0].lower(), sh[0][:240])
    check("...and the jeans are described as covering it",
          "whole, opaque and unbroken" in sh[0].lower(), sh[0][:260])
    # The PICTURE waits even though the words do not: a reference reproduces
    # an image and outweighs any sentence about what is on top of what.
    check("...and its picture waits for the cover to come off",
          "<Picture 2>" not in sh[0], sh[0][:220])
    check("the beat still says it", "chastity belt" in sh[1].lower(), "")
    check("...word for word, unedited",
          "runs a hand over the chastity belt under her jeans" in sh[1], "")
    check("the clash is reported", "says is covered" in info, "")
    check("...naming the shot and the garment",
          "shot 2: chastity belt" in info, "")
    # And it must not fire when the beat leaves it alone.
    quiet = run_node("A room.\n\nMara stands.\n\nMara waits.", plan_only=True,
                     character_memory=mem)[2]
    check("no mention, no warning", "says is covered" not in quiet, "")


def test_a_removal_says_whose_hands():
    print("\n=== a garment does not take itself off ===")
    # Reported: she asks to have the chastity belt taken off, and it drops off on its
    # own. The removal clause named no agent -- "the belt comes off during this shot
    # and is away by the last frame, dropped out of frame" is true of a belt falling
    # to the floor by itself, and that is what it rendered. The beat named the person;
    # the clause was simply not carrying it.
    mem = ("McKenna: she, 22, Shiny white crop top, chastity belt, blue jeans shorts.\n"
           "Dan: he, 41, dark coat.")
    P = ("A room.\n\nMcKenna stands.\n\n"
         "remove: shorts\nMcKenna takes off her shorts.\n\n"
         'McKenna asks Dan to take the chastity belt off. "Please take it off."\n'
         "remove: belt\n\nMcKenna waits.")
    sh = [s for s in run_node(P, plan_only=True, character_memory=mem)[3].split("---")
          if s.strip()]
    check("she undresses herself with her own hands",
          "McKenna takes off her shorts" in sh[1]
          and "away by the last frame" in sh[1], sh[1][-110:])
    # ASKING gives the hands to the other person. "She asks Dan to take it off" is
    # Dan's doing -- reading the asker as the agent is what put them back on her.
    check("asking hands it to the other person",
          "Dan takes the chastity belt off during this shot" in sh[2], sh[2][-110:])
    check("...and not to the one who asked",
          "McKenna takes the chastity belt off" not in sh[2], "")
    # One person in the shot is that person, with no ambiguity to resolve.
    solo = run_node("A room.\n\nremove: coat\nMara takes off her coat.", plan_only=True,
                    character_memory="Mara: she, 22, a grey coat, white top.")[3]
    check("a solo shot names her",
          "Mara takes off her coat" in solo
          and "away by the last frame" in solo, "")
    # The unit rule, directly.
    beat = 'McKenna asks Dan to take the chastity belt off.'
    check("the agent is read from the beat",
          S.removal_agent(beat, ["McKenna", "Dan"], "McKenna") == "Dan")
    check("...and without a wearer the first named acts",
          S.removal_agent(beat, ["McKenna", "Dan"], None) == "McKenna")
    check("no cast, no agent claimed", S.removal_agent(beat, []) == "")


def test_the_only_tag_being_on_a_covered_thing():
    print("\n=== the belt is the only tagged thing, and it is under the shorts ===")
    # The reported sheet, verbatim. Two faults met in it, and each hid the other.
    sheet = ("McKenna: she, 22, Shiny white crop top, chastity belt <picture 2>, "
             "blue jeans shorts.")
    # 1. "blue jeans shorts" is a pair of SHORTS. Taking the first outer match made the
    #    cover "jeans" while a removal names it "shorts", so the two never lined up and
    #    the belt was hidden correctly and then never uncovered.
    check("the cover is the head noun, not the first word",
          S.implied_layers(sheet) == {"chastity belt": "shorts"},
          str(S.implied_layers(sheet)))
    check("...so taking the shorts off uncovers it",
          S.hidden_layers(S.implied_layers(sheet), ["shorts"]) == [])
    # 2. The belt's tag is the ONLY tag. Layering scrubs the belt on every covered
    #    shot and the tag goes with it -- correctly -- and the node then saw no tags
    #    anywhere and fell back to "untagged references ride EVERY shot", sending the
    #    belt's picture into exactly the shots that had just hidden it.
    BELT = torch.full((1, H, W, 3), 0.80)
    rows = []
    ob = S.build_conditioning
    def spy(clip, vae, audio_vae, p, *a, **k):
        rows.append((p, len(k.get("refs") or [])))
        return ob(clip, vae, audio_vae, p, *a, **k)
    S.build_conditioning = spy
    try:
        run_node("A room.\n\nMcKenna stands.\n\nMcKenna waits.\n\n"
                 "remove: shorts\nMcKenna takes off her shorts.\n\nMcKenna turns.",
                 character_memory=sheet + "\nDan: he, 41.", ref_image_2=BELT)
    finally:
        S.build_conditioning = ob
    # The picture travels with the words. The tag is how ref_image_N reaches
    # the shot at all, so withholding it lost the reference -- and a picture
    # the text never claims is read as an extra subject.
    check("the picture waits while it is covered",
          rows[0][1] == 0 and rows[1][1] == 0, str([n for _, n in rows]))
    check("...and the words wait with it",
          not any("chastity" in p.lower() for p, _ in rows[:2]),
          rows[0][0][-160:])
    # The occlusion is carried by the COVER, since a reference cannot be
    # weakened for one image. If the belt still shows through, ref_noise_aug
    # is the only lever left and it moves every reference at once.
    check("...with the cover described as opaque over it",
          all("whole, opaque and unbroken" in p.lower() for p, _ in rows[:2]),
          rows[0][0][-200:])
    check("...with the cover described as opaque over it",
          all("whole, opaque and unbroken" in p.lower() for p, _ in rows[:2]),
          rows[0][0][-200:])
    check("the picture arrives when the shorts come off", rows[2][1] == 1, "")
    check("...and stays after", rows[3][1] == 1, "")
    check("...described too", all("chastity" in p.lower() for p, _ in rows[2:]), "")


def test_an_object_tag_works_without_a_face_picture():
    print("\n=== a tagged object is placed even when nobody is tagged ===")
    # A picture is not always somebody's face. With the person untagged, the belt's
    # own tag was the only one in the entry -- and once the layering scrubbed the
    # belt's WORDS, ownership was decided on the leftover comma fragment, which has
    # nothing in front of it. So the tag read as the person's, survived, and went on
    # fetching the belt's picture into every shot where the belt was covered.
    FACE = torch.full((1, H, W, 3), 0.10)
    BELT = torch.full((1, H, W, 3), 0.80)
    which = lambda t: "FACE" if abs(float(t.mean()) - 0.10) < 0.01 else "BELT"
    def imgs(mem):
        rows = []
        ob = S.build_conditioning
        def spy(clip, vae, audio_vae, p, *a, **k):
            rows.append([which(r) for r in (k.get("refs") or [])])
            return ob(clip, vae, audio_vae, p, *a, **k)
        S.build_conditioning = spy
        try:
            run_node("A room.\n\nMara stands.\n\nMara waits.\n\nMara pulls off her jeans.",
                     character_memory=mem, ref_image_1=FACE, ref_image_2=BELT)
        finally:
            S.build_conditioning = ob
        return rows
    for label, mem in (
            ("nobody tagged, belt tag first",
             "Mara: she, 22, blue jeans, <Picture 2> a chastity belt."),
            ("nobody tagged, belt tag last",
             "Mara: she, 22, blue jeans, a chastity belt <Picture 2>.")):
        got = imgs(mem)
        # The tag is how ref_image_N reaches the shot, so withholding it lost
        # the reference altogether -- and a picture the text never claims is
        # read as an extra subject. It travels with the words now.
        check(f"{label}: picture waits while covered",
              "BELT" not in got[0] and "BELT" not in got[1], str(got))
        check(f"{label}: and arrives when uncovered", "BELT" in got[2], str(got))
    # A face picture alongside it still behaves, and still travels every shot.
    got = imgs("Mara: <Picture 1>, she, blue jeans, a chastity belt <Picture 2>.")
    check("with a face too, the face is always there",
          all("FACE" in g for g in got), str(got))
    check("...and the belt from the shot that uncovers it",
          "BELT" not in got[0] and "BELT" in got[2], str(got))


def test_an_untagged_picture_defeats_the_layering():
    print("\n=== an untagged reference is sent even where its garment is covered ===")
    # Reported: the belt drawn on top of the jeans, with "chastity belt" spelled the
    # way the layering matches. The words were right -- the belt had stopped being
    # described -- and the PICTURE was still going in. An untagged reference rides
    # every shot, so an image of something currently underneath something else is
    # sent on the shots where it is covered, and the model draws what it is shown.
    img = lambda v: torch.full((1, H, W, 3), v)
    P = "A room.\n\nMara stands.\n\nMara pulls off her jeans."
    untagged = run_node(P, plan_only=True, ref_image_1=img(0.1), ref_image_2=img(0.8),
                        character_memory="Mara: she, 22, blue jeans, a chastity belt.")[2]
    check("the clash is reported",
          "not one <Picture N> tag anywhere" in untagged, "")
    check("...naming the layer it will override",
          "chastity belt under jeans" in untagged, "")
    check("...and saying what fixes it", "<Picture 2>" in untagged, "")
    # Tagged, the layering controls the image and there is nothing to warn about.
    tagged = run_node(P, plan_only=True, ref_image_1=img(0.1), ref_image_2=img(0.8),
                      character_memory="Mara: <Picture 1>, she, 22, blue jeans, "
                                       "a chastity belt <Picture 2>.")[2]
    check("a tagged wardrobe raises nothing",
          "not one <Picture N> tag anywhere" not in tagged, "")
    # No layers to defeat: an untagged reference is then just an untagged reference,
    # which the node already warns about separately.
    flat = run_node(P, plan_only=True, ref_image_1=img(0.1),
                    character_memory="Mara: she, 22, a white top and boots.")[2]
    check("no layers, no layer warning",
          "not one <Picture N> tag anywhere" not in flat, "")


def test_underwear_is_hidden_until_it_is_not():
    print("\n=== underwear stays out of the text while something is over it ===")
    mem = "Mara: she, 22, blue denim shorts, white top, panties, a chastity belt."
    P = ("A room.\n\nMara stands by the window.\n\n"
         "Mara pulls off her shorts.\n\nMara turns around.")
    info, script = run_node(P, plan_only=True, character_memory=mem)[2:4]
    sh = [s.lower() for s in script.split("---") if s.strip()]
    # UNDERWEAR IS PLACED, NOT DELETED, as of the report that said so for
    # the third time. It stays in the text and the node says where it is;
    # the fact protected is the same one, that it is not rendered on top of
    # what covers it. A locket or a scarf under a coat still waits.
    check("while the shorts are on, the shorts are the outermost layer",
          "whole, opaque and unbroken" in sh[0], sh[0][-150:])
    check("...and the panties are not named",
          "panties" not in sh[0], sh[0][-150:])
    # The belt goes under as well: worn beneath jeans it is COVERED, not forgotten,
    # and it comes back by the same route as any other layer -- the cover comes off,
    # the reveal clause names it, and it stays in the scene from then on.
    check("...nor the belt", "chastity belt" not in sh[0], sh[0][-150:])
    check("...while the shorts themselves are", "shorts" in sh[0], "")
    # The shot that takes them off is where both become visible, and it has to say so
    # or the reveal happens against a body the text says is bare.
    check("the reveal shot describes them", "panties" in sh[1] and "chastity belt" in sh[1], "")
    check("...and every shot after", "panties" in sh[2] and "chastity belt" in sh[2], "")
    check("info explains the layering", "read as layers" in info, "")
    # Nothing over it: underwear is on show and must not be described away.
    solo = run_node("A room.\n\nMara stands by the window.", plan_only=True,
                    character_memory="Mara: she, 22, panties and a bra.")[3]
    check("underwear alone is still described", "panties" in solo.lower(), "")


def test_a_garment_moved_is_not_a_garment_gone():
    print("\n=== shorts pulled down stay on, and stay described ===")
    # Reported: the shorts changed appearance in the next beat. "Pulls down her
    # shorts" was read as a full removal, so the shot was told they come off and are
    # "dropped out of frame", and the entry was scrubbed from the scene -- leaving
    # every later shot describing nothing where something still was. An undescribed
    # garment is one the model re-invents, which is the same pair coming back a
    # different pair.
    mem = "Mara: she, 22, blue denim shorts, white top."
    P = ("A room.\n\nMara stands and pulls down her shorts.\n\n"
         "Mara steps to the window.\n\nMara looks back.")
    info, script = run_node(P, plan_only=True, character_memory=mem)[2:4]
    sh = [s for s in script.split("---") if s.strip()]
    check("the shorts are not taken off",
          "come off during this shot" not in sh[0]
          and "comes off during this shot" not in sh[0], sh[0][-80:])
    check("...and are not scrubbed from the scene",
          all("shorts" in s.lower() for s in sh), "")
    check("the later shots say where they now sit",
          all("On the body and" in s for s in sh[1:]), "")
    check("...and the staging shot is not told it twice",
          "On the body and" not in sh[0], "")
    check("info names the shots", "MOVED rather than taken off" in info, "")
    # Put back up: the latch lets go, by name or by pronoun.
    for _put in ("Mara pulls her shorts back up.", "Mara pulls them back up."):
        back = run_node("A room.\n\nMara pulls down her shorts.\n\nMara waits.\n\n"
                        + _put + "\n\nMara walks out.",
                        plan_only=True, character_memory=mem)[3]
        got = ["yes" if "On the body and" in s else "no"
               for s in back.split("---") if s.strip()]
        check(f"restored by {_put[:28]!r}", got == ["no", "yes", "no", "no"], str(got))
    # A real removal still empties the wardrobe and says so.
    off = run_node("A room.\n\nMara pulls off her shorts.\n\nMara waits.",
                   plan_only=True, character_memory=mem)[3]
    # The guarantee is that the removal FINISHES in the shot -- the last frame is
    # the next shot's keyframe. Not the exact sentence: where the beat already
    # stages the act, the clause says only the part the beat leaves out, because
    # restating who and what was the same removal written twice.
    check("a real removal still removes",
          "away by the last frame" in off and "fully removed" in off, "")
    check("...and stops describing them", "shorts" not in off.split("---")[1].lower(), "")


def test_undressing_does_not_drop_her():
    print("\n=== taking a garment off does not put a body on the floor ===")
    # Whole-path, because the unit test cannot show the clause reaching the shot. The
    # fall cue read "pulls down her shorts" as a body being put on the floor, and the
    # shot was then told what takes the landing and how the legs fold under it.
    mem = "Mara: she, 22, denim shorts.\nDan: he, 41."
    P = ("A room.\n\nMara stands and pulls down her shorts.\n\n"
         "Dan pushes her down onto the floor.")
    sh = [s for s in run_node(P, plan_only=True, character_memory=mem)[3].split("---")
          if s.strip()]
    check("the undressing shot is not told to fall",
          "falls as one piece" not in sh[0], sh[0][-80:])
    check("...and the real fall still is", "falls as one piece" in sh[1], "")
    # The removal itself must still work -- this only changes what reads as a FALL.
    check("the garment still comes off", "shorts" in sh[0].lower(), "")
    info = run_node(P, plan_only=True, character_memory=mem)[2]
    check("only the real fall is reported", "shot(s) 2 put a body down" in info, "")


def test_an_unbound_fall_is_told_what_catches_it():
    print("\n=== a fall with no hardware in it still names the landing ===")
    # FALL_HOLD only ever fired on a RESTRAINED fall -- the concern there was the
    # hold giving way when the hands came up to break it. A free body falling had
    # nothing said about it at all, and that is the shot the spare leg turned up on.
    mem = "Mara: she, 30."
    free = run_node("A room.\n\nMara trips and falls to the floor.",
                    plan_only=True, character_memory=mem)[3]
    check("a free fall is told what takes the landing",
          "The body falls as one piece" in free, free[-90:])
    check("...and it is not the bound wording",
          "A bound body falls" not in free, "")
    bound = run_node("A room.\n\nMara is handcuffed behind her back.\n\n"
                     "Mara falls to the floor.", plan_only=True, character_memory=mem)[3]
    last = [s for s in bound.split("---") if s.strip()][-1]
    check("a bound fall keeps the bound wording", "A bound body falls" in last, "")
    check("...and not both at once", "The body falls as one piece" not in last, "")
    # It must not fire on shots that are not a fall -- every clause costs the beat
    # some of the shot.
    still = run_node("A room.\n\nMara walks to the window.\n\nMara drops the keys.",
                     plan_only=True, character_memory=mem)[3]
    check("no fall, no clause",
          "falls as one piece" not in still and "bound body falls" not in still, "")
    info = run_node("A room.\n\nMara trips and falls to the floor.",
                    plan_only=True, character_memory=mem)[2]
    check("info names the shot", "shot(s) 1 put a body down" in info, "")


def test_a_named_look_target_is_restated():
    print("\n=== a beat that names something to look at gets it said twice ===")
    P = ("A living room.\n\nMara sits on the sofa looking at the TV.\n\n"
         "Mara looks at her.\n\nMara walks to the window.")
    info, script = run_node(P, plan_only=True, character_memory="Mara: she, 30.")[2:4]
    sh = [s for s in script.split("---") if s.strip()]
    check("the named target is restated",
          "The eyes and the head are turned to the TV" in sh[0], sh[0][-70:])
    # It follows the beat rather than leading it -- ca75672: anatomy in the opening
    # tokens is what a distilled LoRA settles composition on.
    check("...after the beat, not before it",
          sh[0].index("looking at the TV") < sh[0].index("The eyes and the head"), "")
    check("a pronoun target adds nothing", "The eyes and the head" not in sh[1], "")
    check("a beat with no look adds nothing", "The eyes and the head" not in sh[2], "")
    check("info names the shot", "shot(s) 1 name something to look at" in info, "")
    off = run_node(P, plan_only=True, character_memory="Mara: she, 30.",
                   hold_gaze=False)[3]
    check("the switch turns it off", "The eyes and the head" not in off, "")
    check("...and leaves the beat exactly as written", "looking at the TV" in off, "")


def test_a_line_with_no_look_turns_the_faces_to_each_other():
    print("\n=== a dialogue shot that names no look faces the speakers to each other ===")
    # Reported: "it looks like they are talking to a camera and not to each other."
    # gaze_hold only restates a look the beat NAMED; a line with none left both
    # faces to the portrait prior. Two people and a line get one impersonal sentence.
    mem = "Dan: he, 40, a grey jacket.\nMara: she, 30, a red coat."
    P = ("A kitchen.\n\nDan and Mara stand at the counter. Dan asks: \"Is that the last one?\"\n\n"
         "Mara looks at the window and says: \"Nearly.\"\n\n"
         "Dan waits alone.")
    info, script = run_node(P, plan_only=True, character_memory=mem)[2:4]
    sh = [s for s in script.split("---") if s.strip()]
    check("two people and a line, no look staged: they face each other",
          "face each other" in sh[0], sh[0][-90:])
    check("...said positively, naming no camera", "camera" not in sh[0].lower(), "")
    check("a look the beat stages wins, and this stands down",
          "turned to the window" in sh[1] and "face each other" not in sh[1], sh[1][-90:])
    check("no line, nothing to face", "face each other" not in sh[2], "")
    check("info names the shot", "shot(s) 1 carry a line and two or more people" in info, "")
    off = run_node(P, plan_only=True, character_memory=mem, hold_gaze=False)[3]
    check("the switch turns it off", "face each other" not in off, "")


def test_a_walk_along_a_place_arrives_in_it():
    print("\n=== a beat that travels along a place ends in it ===")
    # Reported: she leaves the bedroom, walks down the hallway, comes across a BED,
    # then reaches the kitchen. "down the hallway" is a via with no destination;
    # travel_anchor said nothing, `here` stayed "bedroom", and the kitchen beat was
    # told it opened in the bedroom -- two shots after she had left it.
    mem = "McKenna: she, 22, a grey t-shirt, black shorts."
    P = ("A small flat at night. Her bedroom has an unmade bed and a lamp.\n\n"
         "McKenna gets up and comes out of her bedroom.\n\n"
         "McKenna walks down the hallway.\n\n"
         "McKenna goes into the kitchen.")
    script = run_node(P, plan_only=True, character_memory=mem)[3]
    sh = [" ".join(x.split()) for x in script.split("---") if x.strip()]
    check("the hallway shot is told it opens where she left off and arrives in the hallway",
          "opens in the bedroom and arrives in the hallway" in sh[1], sh[1][-160:])
    check("...with the walk played out in frame", "every step in frame" in sh[1], "")
    check("the kitchen shot opens in the HALLWAY, not the bedroom",
          "opens in the hallway and arrives in the kitchen" in sh[2], sh[2][-160:])
    check("...and is never sent back to the bedroom", "opens in the bedroom" not in sh[2], "")
    # A via WITH a destination is untouched: the destination still wins.
    both = run_node("A flat.\n\nMcKenna walks down the hallway to the kitchen.",
                    plan_only=True, character_memory=mem)[3]
    check("a via with a destination keeps both",
          "carries along the hallway" in both and "arrives in the kitchen" in both, both[-160:])


def test_a_rooms_description_waits_outside_that_room():
    print("\n=== the scene's description of a room waits outside that room ===")
    # Reported: a scene paragraph reading "Her bedroom has an unmade bed and a lamp",
    # a walk from the bedroom down the hallway TO THE LIVING ROOM (which worked -- the
    # room label was right), then a beat from the living room to the kitchen -- and a
    # BED IN THE LIVING ROOM. The paragraph is stamped into every shot, so the bed was
    # described in the living-room shot, and at cfg 1 nothing takes a named thing back.
    SCENE = "A small flat at night. Her bedroom has an unmade bed and a lamp."
    out, held, blocked, waited = S.scene_for_here(SCENE, "kitchen")
    check("another room's description is held", "unmade bed" not in out, out)
    check("...the film's own framing is kept", "small flat at night" in out, out)
    check("...the room is named for the report", held == ["bedroom"] and not blocked, str(held))
    check("...and the sentence that waited is quoted back",
          waited == ["Her bedroom has an unmade bed and a lamp."], str(waited))
    # A ROOM DESCRIBED WITH A COLON is the author describing a room, not a person.
    colon = S.scene_for_here("A small flat at night. Her bedroom: an unmade bed and a lamp.",
                             "kitchen", "", ["McKenna"])
    check("a room written with a colon is still held",
          "unmade bed" not in colon[0] and "at night" in colon[0], colon[0])
    # ...while a declared name with a colon is protected even naming another room.
    keepn = S.scene_for_here("McKenna: she, 22, asleep in the bedroom.", "kitchen", "", ["McKenna"])
    check("a declared name's line is protected", "McKenna: she, 22" in keepn[0], keepn[0])
    # A ROOM THE BEAT NAMES IS NEVER HELD -- `here` goes stale on an unlisted verb.
    stale = S.scene_for_here("Her bedroom has an unmade bed. The kitchen is small and white.",
                             "bedroom", "", ["McKenna"], "McKenna pads into the kitchen.")
    check("a room the beat names survives a stale tracked room",
          "small and white" in stale[0], stale[0])
    same, held2, _, _ = S.scene_for_here(SCENE, "bedroom")
    check("the room we are IN keeps its description", "unmade bed" in same, same)
    check("...and reports nothing held", held2 == [], str(held2))
    # HOLDING NOTHING CHANGES NOTHING, byte for byte -- a script that never leaves one
    # room must not be touched, and every other suite depends on that.
    for txt, room in (("A kitchen. She is at the sink.", "kitchen"),
                      ("Shot on 35mm, shallow depth of field.", "bedroom"),
                      ("Two people, late evening.", "kitchen")):
        check(f"unchanged: {txt[:28]!r}", S.scene_for_here(txt, room)[0] == txt)
    check("no room known, nothing held", S.scene_for_here(SCENE, "")[1] == [])
    # A SHEET LINE IS NEVER TOUCHED, whatever it names.
    sheet = S.scene_for_here("A kitchen.\nMcKenna: she, 22, a grey t-shirt.", "bedroom")[0]
    check("a sheet entry survives a room it does not match",
          "McKenna: she, 22, a grey t-shirt." in sheet, sheet)
    # Two rooms named: each is kept only in its own.
    two = S.scene_for_here("Her bedroom has an unmade bed. The kitchen is small and white.",
                           "kitchen")[0]
    check("the other room goes and this one stays",
          "unmade bed" not in two and "small and white" in two, two)
    # A sentence WELDING the film's framing to one room's furniture is kept, and the
    # caller is told, because holding it would take the night with the bedroom.
    weld, weld_held, weld_blocked, _ = S.scene_for_here(
        "A small flat at night, her bedroom with an unmade bed.", "kitchen")
    check("a welded sentence is kept rather than losing the film's framing",
          "at night" in weld and weld_blocked and weld_held == ["bedroom"], weld)
    # THE ANCHOR IS NEVER HELD. build_scene fuses the anchor and the scene paragraph
    # before either reaches a shot, and an anchor is what belongs to the WHOLE film --
    # "look, camera, lighting, location". An anchor naming a room was being held on
    # every shot outside it, so the film lost its stock and its lens to a rule about
    # furniture.
    ANCH = "Shot on 35mm in a cramped kitchen"
    a_in = S.scene_for_here(ANCH + ".\nHer bedroom has an unmade bed.", "bedroom", ANCH)
    check("an anchor naming another room is kept", "35mm" in a_in[0] and a_in[1] == [], a_in[0])
    a_out = S.scene_for_here(ANCH + ".\nHer bedroom has an unmade bed.", "kitchen", ANCH)
    check("...and the bedroom still goes while it stays",
          "35mm" in a_out[0] and "unmade bed" not in a_out[0], a_out[0])

    # END TO END, the reported script, including the return.
    mem = "McKenna: she, 22, a grey t-shirt, black shorts."
    P = ("A small flat at night. Her bedroom has an unmade bed and a lamp.\n\n"
         "McKenna gets up and comes out of her bedroom.\n\n"
         "McKenna walks down the hallway to the living room.\n\n"
         "McKenna goes from the living room to the kitchen.\n\n"
         "McKenna goes back into her bedroom and lies down.")
    info, script = run_node(P, plan_only=True, character_memory=mem)[2:4]
    sh = [" ".join(x.split()) for x in script.split("---") if x.strip()]
    check("shot 1 is in the bedroom and keeps the bed", "unmade bed" in sh[0], sh[0][:140])
    check("the shot ARRIVING in the living room has no bed", "unmade bed" not in sh[1], sh[1][:200])
    check("...and the living-room-to-kitchen shot has no bed", "unmade bed" not in sh[2], sh[2][:200])
    check("...and the bed comes back when she walks back in", "unmade bed" in sh[3], sh[3][:140])
    check("the film's framing is on every shot",
          all("small flat at night" in x for x in sh), "")
    check("nobody loses their sheet line", all("McKenna: she, 22" in x for x in sh), "")
    check("info reports which shots it waited on",
          "WAITS OUTSIDE it, on shot(s) 2, 3" in info, "")
    # A one-room script is untouched and says nothing about waiting.
    info1 = run_node("A kitchen.\n\nMcKenna fills the kettle.\n\nMcKenna opens a cupboard.",
                     plan_only=True, character_memory=mem)[2]
    check("a script that never leaves one room reports nothing",
          "WAITS OUTSIDE" not in info1, "")


def test_a_covered_object_does_not_send_its_picture():
    print("\n=== an object out of view does not carry its reference ===")
    # Reported: the object looked different when it came back into view. While it was
    # covered, the scrubber removed its WORDS and left its <Picture N> -- so the shot
    # still sent the reference, unclaimed, and whatever the model made of a picture
    # nothing accounted for became the keyframe the next shot was built on.
    FACE = torch.full((1, H, W, 3), 0.10)
    OBJ = torch.full((1, H, W, 3), 0.80)
    which = lambda t: "FACE" if abs(float(t.mean()) - 0.10) < 0.01 else "OBJ"
    mem = ("Mara: <Picture 1>, she, 30, wearing a silver locket <Picture 2>, "
           "and a long coat.")
    P = ("A room.\n\nMara stands by the window in her coat.\n\n"
         "Mara takes off the coat, showing the locket.\n\nMara walks to the door.")
    rows = []
    ob = S.build_conditioning
    def spy(clip, vae, audio_vae, prompt, *a, **k):
        rows.append((prompt, [which(r) for r in (k.get("refs") or [])]))
        return ob(clip, vae, audio_vae, prompt, *a, **k)
    S.build_conditioning = spy
    try:
        run_node(P, character_memory=mem, ref_image_1=FACE, ref_image_2=OBJ)
    finally:
        S.build_conditioning = ob
    check("the covered shot sends only the face", rows[0][1] == ["FACE"], str(rows[0][1]))
    check("...and names no picture it does not carry",
          sorted(set(re.findall(r"<Picture (\d+)>", rows[0][0]))) == ["1"], rows[0][0][:80])
    check("...and does not describe the covered object",
          "locket" not in rows[0][0].lower(), "")
    # Back in view: the words and the picture return together.
    check("the reveal brings the picture back", rows[1][1] == ["FACE", "OBJ"], str(rows[1][1]))
    check("...claimed by the text",
          sorted(set(re.findall(r"<Picture (\d+)>", rows[1][0]))) == ["1", "2"], "")
    check("...and it stays for the shot after", rows[2][1] == ["FACE", "OBJ"], str(rows[2][1]))
    # Every shot: as many references as the text names. The invariant this broke.
    bad = [i + 1 for i, (p, imgs) in enumerate(rows)
           if len(imgs) != len(set(re.findall(r"<Picture (\d+)>", p)))]
    check("no shot carries a picture its text never names", not bad, str(bad))


def test_the_anchor_survives_a_close_shot():
    print("\n=== cuffs above the head are still above the head next shot ===")
    P = ("A room.\n\nMara is handcuffed above her head to the bed frame.\n\n"
         "A close shot of her face.\n\nMara turns her head.\n\n"
         "remove: handcuffs\nMara sits up and rubs her wrists.")
    info, script = run_node(P, plan_only=True, character_memory="Mara: she, 30.")[2:4]
    sh = [s for s in script.split("---") if s.strip()]
    # The staging shot has the author's own words and gets no second sentence about it.
    check("the staging shot is not argued with",
          "the wrists" not in sh[0], "")
    # Both facts still reach the shot; they are two sentences now. The POSITION is
    # a pose about the body, because buried in the hardware sentence it was the
    # least prominent thing in it and the wrists rendered in front. The POINT
    # stays with the hardware, which is what is fastened to the frame.
    check("the next shot is told where the arms are",
          "wrists together above the head" in sh[1], sh[1][-120:])
    check("...and what they are fastened to",
          "fast at the bed frame" in sh[1], sh[1][-120:])
    # The one that matters: a close shot crops the anchor out of the picture, so the
    # text is the only thing still carrying it.
    check("...including the close shot", "the wrists" in sh[1], "")
    check("...and the shot after that", "the wrists" in sh[2], "")
    # It latches like the hardware and is released by the same `remove:`.
    check("a removal lets go of it", "the wrists" not in sh[3], sh[3][-70:])
    check("info names the held shots", "fastened limbs held in place on shot(s) 2, 3" in info, "")
    check("...and names the tight framing", "frame tight enough to crop" in info, "")
    # Nothing to anchor, nothing said -- this must not fire on ordinary shots.
    plain = run_node("A room.\n\nMara waits.\n\nMara walks to the window.",
                     plan_only=True, character_memory="Mara: she, 30.")[3]
    check("an unrestrained scene is untouched", "the wrists" not in plain, "")


def test_a_grin_is_not_a_closed_mouth():
    print("\n=== the guard stops countermanding the beat ===")
    # Reported as bad acting. The mouth guard lands on every shot with a person and
    # no line, and for a face doing nothing that is right. But it was landing on the
    # beats that ARE the performance:
    #
    #     Dana grins, wide and mean.   -> Mouths in the shot stay closed, jaws still.
    #     Dana's mouth falls open.     -> Mouths in the shot stay closed, jaws still.
    #     Dana yawns.                  -> Mouths in the shot stay closed, jaws still.
    #
    # The second one is contradicted word for word. Only the VOCAL reactions stood
    # down -- laughs, sobs -- because exertion_in covers those; every silent facial
    # performance was answered with an instruction to freeze it.
    MEM = "Dana: she, 34, red coat."

    def held(beat):
        sc = run_node("A kitchen.\n\n" + beat, plan_only=True,
                      character_memory=MEM)[3]
        return "Mouths in the shot stay closed" in sc

    for _b in ("Dana smiles at him.",
               "Dana grins, wide and mean.",
               "Dana bites her lip.",
               "Dana yawns.",
               "Dana's mouth falls open.",
               "Dana sneers at the badge.",
               "Dana licks her lips.",
               "Dana mouths the words."):
        check(f"the beat keeps its mouth: {_b!r}", not held(_b), "")
    # ONLY the mouth. A stare or a wince is a face acting with its mouth shut, and
    # the guard costs it nothing -- standing down for those would free a mouth for
    # no gain, and a free mouth on an open branch is where lip-sync lands.
    for _b in ("Dana stares at the door.",
               "Dana walks to the window.",
               "Dana finds her sister's body."):
        check(f"a shut mouth is still held: {_b!r}", held(_b), "")

    # ...and ordinary English that happens to use these verbs is left alone. Every
    # one of these was in the first draft of the pattern: a dog bites, a mill
    # grinds, a tongue licks an envelope, an engine spits.
    for _b in ("The dog bites the postman.", "She grinds the coffee.",
               "He licks the envelope.", "The engine spits and dies.",
               "She mouths off at the guard.", "The jaw of the vice opens."):
        check(f"ordinary English is quiet: {_b!r}", not S.mouth_performs(_b), "")
    check("...and a person spitting is not", S.mouth_performs("He spits on the floor."))

    # THE BRANCH MUST NOT MOVE. exertion_in stands the guard down AND unpins the
    # audio; a smile is silent, so this frees the picture only. If a silent
    # expression started opening branches it would be the babble hole all over
    # again, on the most common beat in any script.
    info = run_node("A kitchen.\n\nDana grins, wide and mean.", plan_only=True,
                    character_memory=MEM, auto_sound=False)[2]
    check("...and a silent expression opens no audio branch",
          re.search(r"shot\(s\) [^|]*\b1\b[^|]*conditioned on real silence", info)
          is not None, info[:300])
    check("info says which shots kept their mouths",
          "the beat itself puts the mouth to work" in info, info[:300])

    # The guard no longer orders the lower face to hold still. "Mouths stay closed"
    # is the whole of the lip-sync guarantee; "jaws still" was a stillness
    # instruction riding along on it, on every quiet shot in the film.
    for _c in (S.MOUTH_HOLD, S.MOUTH_HOLD_OTHERS):
        check(f"no stillness ordered: {_c[:34]!r}",
              not re.search(r"\bstill\b|\bmotionless\b|\bfrozen\b", _c, re.I), _c)
        check("...and the mouth is still shut", "clos" in _c)
    check("the face is given something to do", "expressions moving" in S.MOUTH_HOLD)


def test_nothing_tells_the_cast_to_hold_still():
    print("\n=== the guards stopped ordering stillness ===")
    # Reported: characters look lifeless. This file already knows why, in
    # RESTRAINT_HOLD's own comment -- an earlier wording let the body "reach only as
    # far as the metal allows before it stops", and "read plainly, that is an
    # instruction to hold still ... The performance died under its own continuity
    # guards." It was fixed there and left standing everywhere else. Eight clauses
    # were still carrying it, three of them freezing PEOPLE by name:
    #
    #     Mara listens, still, wearing what the sheet already lists.
    #     ...and the people listening hold still and let it play.
    #     ...every other jaw in the shot stays still.
    #     McKenna is still lying down.
    #     Still on the body and the shorts pulled down...
    #     ...beginning at the first frame and still finishing on the last.
    #     ...untouched and still fastened.
    #     The thong underneath is what shows there now, still on and unchanged.
    #
    # Five of those meant "as before" and one meant "nevertheless". The model does
    # not get the distinction -- `still` is one token and it damps motion wherever
    # it is pointed. The first two are the worst of them, because they land on the
    # LISTENER: the reaction shot, which is where acting actually happens.
    STILL = re.compile(r"\bstill\b|\bmotionless\b|\bfrozen\b|\bunmoving\b|"
                       r"\brigid\b|\bstatic\b", re.I)
    _tv = 'The TV says: "x"'
    built = [
        ("told_hold", S.told_hold(["Mara"])),
        ("told_hold, two", S.told_hold(["Mara", "Dan"])),
        ("posture_hold", S.posture_hold({"Mara": "lying down"}, ["Mara"])),
        ("posture_hold, two", S.posture_hold(
            {"Mara": "lying down", "Dan": "kneeling"}, ["Mara", "Dan"])),
        ("device_voice_clause", S.device_voice_clause(_tv)),
        ("displaced_hold", S.displaced_hold([("shorts", "pulled down")])),
        ("pace_clause", S.pace_clause(4.0, 12.0)),
        ("reveal_clause", S.reveal_clause(["thong"])),
        ("off_by_last_frame", S.off_by_last_frame(["shorts"], "McKenna", "A van.", "")),
    ]
    for _n, _c in built:
        check(f"no stillness ordered by {_n}", not STILL.search(_c), _c)
    # ...and every clause CONSTANT in the file, so the next one added cannot quietly
    # bring it back. The filter is the clause idiom: an upper-case name holding real
    # prose rather than a regex source.
    for _n in sorted(dir(S)):
        _v = getattr(S, _n, None)
        if (isinstance(_v, str) and _n.isupper() and len(_v.split()) >= 4
                and not re.search(r"\(\?|\\b|\|", _v)):
            check(f"no stillness ordered by {_n}", not STILL.search(_v), _v)

    # EVERY GUARANTEE SURVIVES. This is a vocabulary change, not a retreat: each of
    # these clauses exists because of a report, and none of them stops saying what
    # it was added to say.
    check("the listener is still given something to do",
          "listens" in S.told_hold(["Mara"]))
    check("...and is still held to the sheet's wardrobe",
          "sheet already lists" in S.told_hold(["Mara"]))
    check("the latched pose is still named",
          "lying down" in S.posture_hold({"Mara": "lying down"}, ["Mara"]))
    check("the machine still owns the voice", "TV's" in S.device_voice_clause(_tv))
    check("...and the room's own mouths are still closed",
          "mouths closed" in S.device_voice_clause(_tv))
    check("...and it is still positively phrased",
          not re.search(r"\bno\b|\bnot\b|\bnever\b", S.device_voice_clause(_tv), re.I))
    check("one voice is still asserted",
          "Only the person speaking" in S.ONE_VOICE and "closed" in S.ONE_VOICE)
    check("the moved garment is still on the body",
          "On the body" in S.displaced_hold([("shorts", "pulled down")]))
    check("...and still where the beat left it",
          "where the beat put them" in S.displaced_hold([("shorts", "pulled down")]))
    check("the layer underneath is still unchanged",
          "unchanged" in S.reveal_clause(["thong"]))
    check("the rest of the wardrobe is still bound",
          "untouched" in S.off_by_last_frame(["shorts"], "McKenna", "A van.", ""))
    check("the beat still runs to the last frame",
          "last" in S.pace_clause(4.0, 12.0))


def test_a_face_under_duress_is_not_a_portrait():
    print("\n=== the face stops being left to the model's prior ===")
    # Reported: she smiles at the camera in a situation of duress. Dumped, a
    # four-shot scene of a woman handcuffed in the back of a van -- pulling at the
    # cuffs, struggling, going limp -- contained NOT ONE WORD about anybody's face.
    # Every clause in it was hardware, limbs, or mouths-closed.
    #
    # SILENCE IS NOT NEUTRAL. An attribute the prompt does not state is filled from
    # the model's prior, and the prior for a named, described person is a PORTRAIT:
    # facing the lens, pleasantly. This file already knows the half of that about
    # the eyes -- it is the whole reason gaze_hold exists -- but gaze_hold only
    # fires when the beat NAMES something to look at, which most beats do not, and
    # nothing has ever spoken for the expression at all.
    MEM = "McKenna: she, 26, dark hair, a red jacket, handcuffs behind her back."
    PLAIN = "Kate: she, 30, a grey coat."

    def face(beat, mem=MEM, **kw):
        return run_node("A van interior, night.\n\n" + beat, plan_only=True,
                        character_memory=mem, **kw)[3]

    check("a restrained shot says what the face is doing",
          "shows the strain" in face("McKenna lies against the wheel arch."), "")
    # A beat with UNAMBIGUOUS duress in it does too, with no sheet hardware at all.
    check("...and so does a beat staging unambiguous duress",
          "shows the strain" in face("Kate is bound and gagged on the floor.",
                                     mem=PLAIN), "")
    # ...but an ambiguous one alone does NOT, and that is deliberate. See
    # _DURESS_STRONG: "struggles" is a maths homework as often as a restraint, and
    # one such beat used to stamp a grim mood on a whole comedy.
    check("...while one ambiguous beat alone does not",
          "shows the strain" not in face("Kate struggles and twists away.",
                                         mem=PLAIN), "")
    check("...unless the anchor says what the film is",
          "shows the strain" in face("Kate struggles and twists away.", mem=PLAIN,
                                     anchor="Handheld. A tense, grim abduction."), "")
    # NOT INVENTED. Only what the scene already establishes -- hardware the sheet
    # lists, or the author's own distress verbs. A shot staging neither gets
    # nothing: a node that decides how everybody feels is a node writing the film.
    check("an ordinary scene is left alone",
          "shows the strain" not in face("Kate makes the coffee.", mem=PLAIN), "")
    check("...and so is an ordinary beat about a calm person",
          "shows the strain" not in face("Kate reads the letter.", mem=PLAIN), "")
    # A COLLAR ALONE IS NOT DURESS. restraint_present is right for the continuity
    # holds -- a collar must stay fastened and stay the object it was -- but it is
    # worn in scenes that are not distressing, and stamping strain on one of those
    # faces is the mouth guard's bug in a new place.
    check("a collar alone does not stage duress",
          "shows the strain" not in face("McKenna sits on the bed.",
                                         mem="McKenna: she, 26, a leather collar."), "")
    check("...but cuffs do",
          "shows the strain" in face("McKenna sits on the bed.",
                                     mem="McKenna: she, 26, cuffs on her wrists."), "")
    check("...and so does rope",
          "shows the strain" in face("McKenna sits on the bed.",
                                     mem="McKenna: she, 26, rope around her wrists."), "")
    # The author's own face beat wins, exactly as it does against the mouth guard:
    # where the beat says what the face does, the node has nothing to add over it.
    check("a written smile is not argued with",
          "shows the strain" not in face("McKenna smiles at him."), "")

    # IMPERSONAL, even with two people in the shot. Naming her here is a SECOND
    # naming -- the hardware hold has already said "Every restraint on McKenna" --
    # and a named person is a person the model draws: that is what put a second girl
    # in frame at the moment of cuffing. gaze_hold made the same call.
    two = run_node("A van interior, night.\n\nMcKenna pulls at the cuffs while Dan "
                   "watches.", plan_only=True,
                   character_memory=MEM + "\nDan: he, 40, a work coat.")[3]
    check("two in the shot, and the clause still names nobody",
          "the face shows the strain" in two and "McKenna's face" not in two,
          two[-200:])

    # PICTURE ONLY. The audio branch must not move: a face is not a sound, and a
    # wordless duress shot is still pinned to silence.
    info = run_node("A van interior, night.\n\nMcKenna lies against the wheel arch.",
                    plan_only=True, character_memory=MEM, auto_sound=False)[2]
    check("...and it opens no audio branch",
          "conditioned on real silence" in info, info[:200])
    check("info names the shots it spoke for",
          "left to the model's prior" in info, info[:200])

    # Positively phrased: at cfg 1 there is no negative prompt, so "not smiling"
    # would name the smile. A set mouth is a thing that can be drawn.
    cl = S.DURESS_FACE
    check("the clause is positively phrased",
          not re.search(r"\bno\b|\bnot\b|\bnever\b|\bun\w+ing\b", cl, re.I), cl)
    check("...and orders no stillness",
          not re.search(r"\bstill\b|\bmotionless\b|\bfrozen\b", cl, re.I), cl)
    check("...and names no camera", not re.search(r"camera|lens", cl, re.I), cl)
    check("...in one sentence", cl.count(".") == 1, cl)

    # THE SWITCH IS hold_gaze, not one of its own. Every inference in this file has
    # an off switch, but the widget count is capped by a test whose comment says the
    # old node had 38 and nobody could find anything -- so this rides the
    # switch that already exists for the same prior. hold_gaze is the portrait's
    # POSE; this is the portrait's EXPRESSION.
    off = face("McKenna lies against the wheel arch.", hold_gaze=False)
    check("off with hold_gaze, the face is left alone again",
          "shows the strain" not in off, "")


def test_her_whimper_does_not_free_his_mouth():
    print("\n=== a vocal belongs to somebody ===")
    # Reported: "her whimpering is opening up his ability to babble. Dialogue is
    # not being localized to the characters."
    #
    # _voiced is a SHOT-LEVEL flag with no owner. exertion_in(body) says this beat
    # stages a vocal, and both mouth guards then stand down -- for EVERYBODY in the
    # shot. The comment says why they stand down: "straining is vocal and that mouth
    # should be open." That mouth. Not every mouth.
    #
    # So "McKenna sobs while Dan watches" left Dan's mouth as free as hers on a shot
    # whose audio branch her sob had just opened, which is the exact machinery the
    # speech guard exists to stop -- and it fires for a spoken line while a vocal
    # switched it off entirely. Measured on a six-shot scene: not one shot had any
    # mouth guard at all.
    MEM = ("McKenna: she, 26, dark hair, handcuffs behind her back.\n"
           "Dan: he, 40, a work coat.")

    def shot(beat):
        return " ".join(run_node("A van interior, night.\n\n" + beat, plan_only=True,
                                 character_memory=MEM)[3].split())

    a = shot("McKenna sobs while Dan watches.")
    check("the vocal is given an owner", "The sobbing is McKenna's" in a, a[-200:])
    check("...and the other mouth is closed",
          "every other mouth in the shot stays closed" in a, a[-200:])

    # LOCALIZATION. Two sound sources in one shot and nothing saying which is which:
    # the model is free to give her the words and him the whimper.
    b = shot('McKenna whimpers, and Dan says: "Nearly there."')
    check("the line is localized to the speaker", "Only Dan speaks" in b, b[-220:])
    check("...and the vocal to the other character",
          "the whimpering is McKenna's" in b, b[-220:])
    check("...and neither of their mouths is held shut",
          "Mouths in the shot stay closed" not in b, b[-220:])

    # The person MAKING the sound is never muted -- that was the whole reason the
    # guard stood down, and it is still right for her.
    c = shot("McKenna whimpers behind the gag.")
    check("a lone vocaliser is not told to close her mouth",
          "Mouths in the shot stay closed" not in c, c[-200:])
    check("...and with nobody else in the beat, nothing is claimed",
          "every other mouth" not in c, c[-200:])

    # UNATTRIBUTABLE STAYS UNGUARDED. If the beat does not say whose the sound is,
    # closing mouths could close the mouth of whoever is making it -- and muting a
    # real sound is worse than a mouth moving, which is this file's standing rule.
    d = shot("Somebody sobs in the dark while Dan waits.")
    check("an unattributed vocal holds nobody",
          "every other mouth in the shot stays closed" not in d, d[-200:])

    # THE BRANCH IS UNTOUCHED. Her sob is still heard: this changes which faces move,
    # never what the audio is conditioned on.
    info = run_node("A van interior, night.\n\nMcKenna sobs while Dan watches.",
                    plan_only=True, character_memory=MEM, auto_sound=False)[2]
    check("the vocal still keeps its audio",
          re.search(r"shot\(s\) [^|]*\b1\b[^|]*stage EFFORT", info) is not None,
          info[:300])
    check("info names the shots whose vocal was attributed",
          "belongs to somebody" in info, info[:300])

    cl = S.voice_sources(["Dan"], "whimpering", ["McKenna"], ["Sam"])
    check("the clause is positively phrased",
          not re.search(r"\bno\b|\bnot\b|\bnever\b", cl, re.I), cl)
    check("...and orders no stillness",
          not re.search(r"\bstill\b|\bmotionless\b|\bfrozen\b", cl, re.I), cl)


def test_her_look_does_not_land_on_him():
    print("\n=== a held look belongs to whoever is doing the looking ===")
    # Reported: "Girl is stuck gazing at a camera while the other character does his
    # part." The gaze latch has no owner. `looking_at` is ONE string held across
    # shots, and gaze_hold is impersonal by design, so a look staged by one person
    # goes on being said in shots that person is not even in -- landing on whoever
    # IS there.
    #
    # Measured: McKenna looks at the lane in shot 3; shot 4 describes only Dan and
    # was still told "The eyes and the head are turned to the lane behind them."
    # That is his head being turned to her sightline. Same class of defect as the
    # vocal flag: a per-person fact kept in a shot-level variable.
    MEM = ("McKenna: she, 26, dark hair, handcuffs behind her back.\n"
           "Dan: he, 40, a work coat.")
    P = ("A white van parked facing away down the lane, rear doors open, night.\n\n"
         "Dan lifts McKenna into the back of the van.\n\n"
         "McKenna looks at the lane behind them.\n\n"
         "Dan closes one of the rear doors.")
    sh = [" ".join(x.split()) for x in
          re.split(r"(?=\[Shot )", run_node(P, plan_only=True,
                                            character_memory=MEM)[3]) if x.strip()]
    check("the beat that stages the look gets the clause",
          "turned to the lane" in sh[1], sh[1][-160:])
    check("...and the shot she is not in never claims it impersonally",
          "The eyes and the head are turned to the lane" not in sh[2], sh[2][-160:])

    # ...and where she IS still in the shot, the look is hers by name, because a
    # second person in frame is a second pair of eyes the impersonal wording could
    # land on. One person in the shot keeps the impersonal wording: naming somebody
    # is a second mention of them, and that has its own cost.
    P2 = ("A lane, night.\n\n"
          "McKenna looks at the treeline.\n\n"
          "Dan checks the wheel while McKenna waits.")
    sh2 = [" ".join(x.split()) for x in
           re.split(r"(?=\[Shot )", run_node(P2, plan_only=True,
                                             character_memory=MEM)[3]) if x.strip()]
    # Attributed -- by name, or by her pronoun where another clause in the same
    # shot has already spent the naming. Either way it is HERS and cannot be read
    # as his, which is the guarantee. A bare "The eyes and the head" cannot.
    check("the held look is attributed once two people are in the shot",
          ("McKenna's eyes and head are turned to the treeline" in sh2[1]
           or "Her eyes and head are turned to the treeline" in sh2[1]), sh2[1][-200:])

    # The clause itself, both ways.
    check("impersonal with nobody named",
          S.gaze_hold("TV") == " The eyes and the head are turned to the TV.")
    check("...and named when it has to be",
          S.gaze_hold("TV", "Mara's") == " Mara's eyes and head are turned to the TV.")
    check("...or by pronoun where the name is already spent in this shot",
          S.gaze_hold("TV", "her") == " Her eyes and head are turned to the TV.")
    check("...and a person target takes no article",
          S.gaze_hold("Dan", "", True) == " The eyes and the head are turned to Dan.")
    check("nothing named, nothing said", S.gaze_hold("", "Mara's") == "")
    check("info says the look was attributed",
          "a look belongs to whoever" in run_node(P, plan_only=True,
                                                  character_memory=MEM)[2])


def test_an_anchor_says_when_it_has_taken_the_scenes_place():
    print("\n=== the scene paragraph that quietly became a shot ===")
    # Reported: "van was not in the same direction as the beat prior. Van direction
    # moved."
    #
    # Filling in `anchor` makes EVERY paragraph a beat -- the anchor is then the
    # scene. That is deliberate and the tooltip says so. What it means in practice
    # is that a prompt whose first paragraph is scene text loses that text after
    # shot 1: the van's heading, the location, the time of night are stated once,
    # rendered as their own shot, and never carried again.
    #
    #     no anchor    4 shots   van direction carried: yes yes yes yes
    #     with anchor  5 shots   van direction carried: yes NO  NO  NO  NO
    #
    # Nothing said so at runtime. The tooltip is in the widget; the person hits this
    # after a render, and what they see is the van turning round between shots.
    P = ("A white van parked facing away down the lane, rear doors open, night.\n\n"
         "Dan lifts her into the back.\n\n"
         "Dan closes one of the rear doors.")
    MEM = "Dan: he, 40, a work coat."
    info = run_node(P, plan_only=True, character_memory=MEM,
                    anchor="Handheld, night exterior.")[2]
    check("the node says the scene paragraph became a shot",
          "is being spent as shot 1" in info, info[:300])
    check("...and quotes it back so it can be recognised",
          "A white van parked facing away down the lane" in info, info[:300])
    check("...and says what that costs",
          "not carried into any other shot" in info, info[:300])

    # NOT when the first paragraph stages an action -- that IS a beat, and telling
    # somebody to move it into the anchor would be wrong.
    P2 = ("Dan opens the rear doors.\n\nDan lifts her into the back.")
    q = run_node(P2, plan_only=True, character_memory=MEM,
                 anchor="Handheld, night exterior.")[2]
    check("an opening ACTION is not reported as lost scene text",
          "is being spent as shot 1" not in q, q[:300])
    # ...and never without an anchor, where the first paragraph IS the scene.
    r = run_node(P, plan_only=True, character_memory=MEM)[2]
    check("with no anchor there is nothing to report",
          "is being spent as shot 1" not in r, r[:300])
    check("...because the scene is carried instead",
          all("facing away down the lane" in b for b in
              [x for x in re.split(r"(?=\[Shot )",
                                   run_node(P, plan_only=True,
                                            character_memory=MEM)[3]) if x.strip()]))


def test_a_grim_film_is_grim_in_every_shot():
    print("\n=== the scene stops reading as a happy one ===")
    # Reported: "the last run had them smiling and thinking this was a happy scene,
    # when indeed it was not." THEM -- plural. 8418805 gave the restrained person a
    # face and stopped there, and measured on a five-shot kidnapping it shows:
    #
    #     shot 1  cast=          -- NOTHING --
    #     shot 2  cast=McKenna   strain
    #     shot 3  cast=McKenna Dan   strain      <- impersonal; Dan gets nothing
    #     shot 4  cast=Dan       -- NOTHING --
    #     shot 5  cast=Dan       -- NOTHING --
    #
    # Every shot the captor is alone in had no tone at all, so it was filled from
    # the prior, and the prior for a man in a work coat is a pleasant one. A film
    # has a MOOD and it does not belong to one character's face.
    MEM = ("McKenna: she, 26, dark hair, handcuffs behind her back.\n"
           "Dan: he, 40, a work coat.")
    P = ("A van interior, night.\n\nMcKenna pulls at the cuffs.\n\n"
         "Dan watches her.\n\nDan closes the hatch.\n\nDan starts the engine.")
    sh = [" ".join(x.split()) for x in
          re.split(r"(?=\[Shot )", run_node(P, plan_only=True, character_memory=MEM,
                                            anchor="Handheld, tight interior.")[3])
          if x.strip()]
    for _i in (1, 2, 3, 4):
        check(f"shot {_i + 1} carries the film's mood",
              "The mood is grim" in sh[_i], sh[_i][-140:])
    check("the shot with the restrained person still gets her face",
          "shows the strain" in sh[1], sh[1][-140:])
    check("...and the shot with only the captor gets the mood alone",
          "The mood is grim" in sh[3] and "shows the strain" not in sh[3], sh[3][-140:])

    # NOT INVENTED, and not everywhere. A film that stages no duress is left alone
    # in every shot -- the node does not get to decide that your film is bleak.
    PLAIN = "Kate: she, 30, a grey coat.\nSam: he, 33."
    Q = ("A kitchen, morning.\n\nKate makes the coffee.\n\n"
         "Sam reads the paper.\n\nKate sits down.")
    qs = [" ".join(x.split()) for x in
          re.split(r"(?=\[Shot )", run_node(Q, plan_only=True, character_memory=PLAIN,
                                            anchor="Wide, morning light.")[3])
          if x.strip()]
    check("an ordinary film gets no mood anywhere",
          not any("mood is grim" in x for x in qs), qs[0][-140:])

    # It reads the FILM, not just this beat: a shot of the captor alone is grim
    # because of what the sheet says is on her wrists, three beats ago.
    check("the mood is read from the sheet's hardware",
          S.film_stages_duress(["Dan closes the hatch."],
                               "McKenna: she, 26, cuffs on her wrists."))
    check("...or from unambiguous duress in any beat",
          S.film_stages_duress(["Kate makes the coffee.",
                                "Kate is held captive in the cellar."], PLAIN))
    # AND NOT from an ambiguous one. Measured over 512 beats of six scenario
    # families, inferring from ambiguous verbs called an ORDINARY film grim as often
    # as a duress one -- 87.8% against 73.5% at two weak beats -- so it does not.
    check("...but never from an ambiguous verb alone",
          not S.film_stages_duress(["Kate makes the coffee.", "Kate sobs."], PLAIN))
    # THE ANCHOR SETTLES IT, both ways, and outranks everything.
    check("a declared tone turns it on",
          S.film_stages_duress(["Kate makes the coffee."], PLAIN,
                               "Wide. A grim, claustrophobic film."))
    check("...and a declared light tone turns it off outright",
          not S.film_stages_duress(["Kate is bound and gagged in the cellar."], PLAIN,
                                   "Wide. A warm, comic short."))
    check("mood_declared reads both directions",
          S.mood_declared("A tense kidnapping, handheld") == "grim"
          and S.mood_declared("Warm, romantic, golden hour") == "light"
          and S.mood_declared("Handheld, 35mm, night") == "")
    check("...and a collar alone is still not duress",
          not S.film_stages_duress(["Kate makes the coffee."],
                                   "Kate: she, 30, a leather collar."))
    check("...and an ordinary film is not", not S.film_stages_duress(
        ["Kate makes the coffee.", "Sam reads the paper."], PLAIN))

    check("the mood clause is positively phrased",
          not re.search(r"\bno\b|\bnot\b|\bnever\b", S.DURESS_MOOD, re.I), S.DURESS_MOOD)
    check("...and orders no stillness",
          not re.search(r"\bstill\b|\bmotionless\b|\bfrozen\b", S.DURESS_MOOD, re.I))
    check("off with hold_gaze, like the face it belongs to",
          "mood is grim" not in run_node(P, plan_only=True, character_memory=MEM,
                                         hold_gaze=False)[3])


def test_a_look_survives_the_next_beat():
    print("\n=== the look she was given does not evaporate ===")
    # Reported: "when she looks at the van in one beat, she's gazing at the camera in
    # the next beat and not looking at the van."
    #
    # ed0d2a6 gave the latch an owner. It did not make the latch SURVIVE. Measured
    # over seven plausible next-beats, the look held in two:
    #
    #     McKenna waits.                    -> held
    #     McKenna pulls at the cuffs.       -> held
    #     McKenna watches him.              -> LOST
    #     Dan opens the driver's door.      -> LOST
    #     Dan walks to the driver's door.   -> LOST
    #     Dan lifts the case into the back. -> LOST
    #
    # Two different causes, neither of them the latch:
    #
    #   1. "watches him" clears the look -- she IS looking somewhere else -- but
    #      `him` is refused as a target, so nothing replaces it and the shot says
    #      nothing about her eyes at all. Nothing is the lens.
    #   2. A beat that names only Dan drops her from the shot text completely, so
    #      she has no sheet line, no pose and no look, while still being in the
    #      van in the picture the next shot starts from.
    MEM = ("McKenna: she, 26, dark hair, handcuffs behind her back.\n"
           "Dan: he, 40, a work coat.")

    def two(second):
        sh = [" ".join(x.split()) for x in
              re.split(r"(?=\[Shot )",
                       run_node("A lane at night.\n\nMcKenna looks at the van.\n\n" + second,
                                plan_only=True, character_memory=MEM,
                                anchor="Handheld, night exterior.")[3]) if x.strip()]
        return sh[1], sh[2]

    # 1. A LOOK AT A PERSON IS A LOOK. The old rule -- "restating a pronoun says
    # nothing the beat did not" -- is true against a neutral model and false against
    # one whose prior is a portrait: the choice is not between the beat's word and a
    # restatement, it is between the beat's word and the lens.
    _, b = two("McKenna watches him.")
    check("a look at a person is said, not dropped",
          "turned to Dan" in b, b[-200:])
    _, b2 = two("McKenna looks at Dan.")
    check("...and a look at a NAME is a look too", "turned to Dan" in b2, b2[-200:])
    # Only where it is unambiguous. Three people and a bare "him" resolves to
    # nobody, and guessing which is worse than saying nothing.
    THREE = MEM + "\nSam: he, 35, a hood."
    amb = " ".join(run_node("A lane at night.\n\nMcKenna watches him while Dan and "
                            "Sam wait.", plan_only=True, character_memory=THREE,
                            anchor="Handheld, night exterior.")[3].split())
    check("an ambiguous pronoun resolves to nobody",
          "turned to Dan" not in amb and "turned to Sam" not in amb, amb[-200:])

    # 2. SHE IS STILL IN THE VAN. A beat about Dan does not take her out of the
    # frame -- the next shot starts from a picture with her in it -- so a look she
    # was given goes on being hers. Named, because the shot's own text describes
    # only Dan and an impersonal sentence would land on him: that was ed0d2a6.
    for _second in ("Dan opens the driver's door.",
                    "Dan lifts the case into the back."):
        _, c = two(_second)
        check(f"her look is carried past {_second[:22]!r}",
              "McKenna's eyes and head are turned to the van" in c, c[-200:])
    # ...for ONE shot, not forever. Two beats she is absent from and the claim
    # stops: the node knows she was in the previous picture, not where she is now.
    sh = [" ".join(x.split()) for x in
          re.split(r"(?=\[Shot )",
                   run_node("A lane at night.\n\nMcKenna looks at the van.\n\n"
                            "Dan opens the driver's door.\n\nDan starts the engine.",
                            plan_only=True, character_memory=MEM,
                            anchor="Handheld, night exterior.")[3]) if x.strip()]
    check("...and is not still being claimed two beats later",
          "McKenna's eyes" not in sh[3], sh[3][-200:])
    # And a beat that walks her off ends it, as before.
    _, d = two("McKenna walks away down the lane.")
    check("walking her off still ends the look",
          "turned to the van" not in d, d[-200:])


def test_the_bed_survives_an_anchor():
    print("\n=== the room tone that was supposed to fill the lead-in ===")
    # Reported: "there is still babble at the opening stages of the beat."
    #
    # This file already knows the mechanism and wrote it down: "a line is a second
    # or two; the shot is five to ten, and the audio branch is open for all of it
    # ... the only thing it knows is happening in this shot is somebody talking, so
    # it invents more talking to occupy the lead-in." Its answer was that "what
    # occupies the time around it is ROOM TONE. A branch with a bed to lay down does
    # not need to invent a voice to fill the space."
    #
    # The bed was not there. scene_ambient reads the anchor and the SCENE -- and
    # with `anchor` filled in there is no scene paragraph, because every paragraph
    # is a beat. Measured on the same prompt:
    #
    #     anchor set   3 shots   ambient bed per shot: NO  NO  NO
    #     no anchor    2 shots   ambient bed per shot: yes yes
    #
    # room_tone already has the fallback for exactly this and says so at its call
    # site: "The opening beat is the fallback: with `anchor` set there is no scene
    # paragraph, and an anchor describes the camera rather than the room."
    # scene_ambient never got one, so the film's bed died whenever anyone filled in
    # the widget the tooltips tell them to fill in.
    MEM = "McKenna: she, 26, dark hair.\nDan: he, 40, a work coat."
    P = ("A van interior, night. The engine is running.\n\n"
         "Dan looks back and says: \"Sit still.\"\n\n"
         "McKenna says: \"Where are we going?\"")

    def beds(**kw):
        sh = [" ".join(x.split()) for x in
              re.split(r"(?=\[Shot )", run_node(P, plan_only=True,
                                                character_memory=MEM, **kw)[3])
              if x.strip()]
        return [("sounds like" in x or "only sound" in x) for x in sh]

    withq = beds(anchor="Handheld, tight interior.")
    check("the speaking shots keep their bed with an anchor set",
          all(withq[1:]), str(withq))
    plain = beds()
    check("...exactly as they do without one", all(plain), str(plain))
    check("the bed itself is read from the opening beat",
          S.scene_ambient("Handheld, tight interior.") == ""
          and S.scene_ambient("Handheld, tight interior.",
                              "A van interior, night. The engine is running.")
          == "an engine idling")
    # It still cannot OPEN a branch -- that is the one thing an inference may never
    # do here, and it is why every wordless shot does not get a voice with its bed.
    Q = ("A van interior, night. The engine is running.\n\nMcKenna waits.")
    q = [" ".join(x.split()) for x in
         re.split(r"(?=\[Shot )", run_node(Q, plan_only=True, character_memory=MEM,
                                           anchor="Handheld, tight interior.")[3])
         if x.strip()]
    check("a wordless shot still gets no bed and stays silent",
          not any("sounds like" in x for x in q), q[-1][-150:])
    check("info says where the bed was read from",
          "read from the opening beat" in run_node(P, plan_only=True,
                                                   character_memory=MEM,
                                                   anchor="Handheld, tight interior.")[2])


def test_the_audio_branch_gets_a_soft_landing():
    print("\n=== the audio branch stops landing from a great height ===")
    # Reported repeatedly: babble at the opening of the beat, and prose was not
    # fixing it. It cannot. Every clause in this file changes what the branch is
    # TOLD; none of them changes how much noise it still has to clear when it
    # stops. Computed from ComfyUI's own scheduler code at shift 12/3:
    #
    #     scheduler     5 steps   8 steps      <- last AUDIO sigma before zero
    #     simple         0.4286    0.3000
    #     beta           0.2981    0.1559
    #     normal         0.0348    0.0348
    #     kl_optimal     0.0030    0.0030
    #     exponential    0.0030    0.0030
    #
    # On `simple` at 5 steps the audio branch resolves 43% of its denoising in ONE
    # final jump -- 140x what kl_optimal leaves. A branch resolving that much at
    # once invents whatever is easiest to invent, and for a branch conditioned on
    # "somebody speaks" that is a voice. It surfaces at the OPENING because that is
    # where the branch has the least conditioning to anchor it: the line has not
    # started yet.
    #
    # The audio branch has no schedule of its own -- comfy/ldm/minimax/model.py
    # derives it per step, sigma_a = time_shift_sigma(sigma_v, shift_v, shift_a) --
    # so the audio tail is decided by the VIDEO schedule, and picking a scheduler
    # for the audio means giving up the one chosen for the picture. Inserting one
    # step does not: it splits the final jump and leaves every earlier sigma alone.
    #
    # Pure arithmetic, deliberately. comfy is not importable in this suite, which
    # is why last_audio_sigma silently falls back here -- so the schedule build
    # lives apart from the maths, and the maths is what gets tested.
    f = S.audio_sigma_of
    g = S.video_sigma_for_audio
    for want in (0.20, 0.10, 0.05, 0.03, 0.01):
        got = f(g(want, 12.0, 3.0), 12.0, 3.0)
        check(f"audio {want} round-trips through the video grid",
              abs(got - want) < 1e-9, f"{got}")
    check("the formula matches comfy's own for the known tail",
          abs(f(0.75, 12.0, 3.0) - 0.42857142857) < 1e-9, str(f(0.75, 12.0, 3.0)))

    # simple @ 5 steps, shift 12 -- the schedule this actually happens on.
    SIMPLE = [1.0, 0.9796, 0.9474, 0.8889, 0.75, 0.0]
    out = S.insert_audio_landing(SIMPLE, 12.0, 3.0)
    check("a coarse tail gets exactly one step inserted",
          len(out) == len(SIMPLE) + 1, str(out))
    check("...inserted before the zero, not after",
          out[-1] == 0.0 and out[-2] > 0.0, str(out))
    check("...and the schedule stays strictly decreasing",
          all(a > b for a, b in zip(out, out[1:])), str(out))
    check("...leaving every earlier sigma untouched",
          out[:len(SIMPLE) - 1] == SIMPLE[:-1], str(out))
    check("the audio's last jump drops from 43% to about 3%",
          abs(f(out[-2], 12.0, 3.0) - 0.03) < 1e-6, str(f(out[-2], 12.0, 3.0)))

    # A schedule that already lands softly is left completely alone -- picking
    # kl_optimal or exponential is the better fix and must not be taxed a step.
    FINE = [1.0, 0.5, 0.154, 0.062, 0.0109, 0.0]
    check("a soft tail is not touched", S.insert_audio_landing(FINE, 12.0, 3.0) == FINE)
    # ...and neither is anything malformed. This runs inside the render path, so it
    # returns the input unchanged rather than raising on anything unexpected.
    for _bad in ([], [0.0], [1.0], [1.0, 0.5], None):
        check(f"malformed input is returned as-is: {_bad!r}",
              S.insert_audio_landing(_bad, 12.0, 3.0) == (_bad if _bad else _bad))
    check("a schedule with no trailing zero is left alone",
          S.insert_audio_landing([1.0, 0.75, 0.5], 12.0, 3.0) == [1.0, 0.75, 0.5])
    # Never twice, however coarse.
    once = S.insert_audio_landing(S.insert_audio_landing(SIMPLE, 12.0, 3.0), 12.0, 3.0)
    check("it is never inserted twice", len(once) == len(SIMPLE) + 1, str(once))


def test_a_kidnapping_reads_as_one_without_being_declared():
    print("\n=== the beats are allowed to say what the film is ===")
    # Reported: "she is still smiling in every beat, despite this being a kidnapping
    # situation that was not defined in the anchor."
    #
    # That last clause is the whole bug. film_stages_duress read binding hardware
    # from the CHARACTER SHEET and distress verbs from the beats -- and a kidnapping
    # is neither. Nobody writes "McKenna: she, 26, handcuffs" for an abduction; the
    # hardware goes on during the film, in the beats, and the beats use coercion
    # verbs that were nowhere in the distress list. Ten beats of an explicit
    # abduction returned False:
    #
    #     Dan grabs McKenna from behind and covers her mouth.
    #     Dan drags McKenna towards the van.
    #     Dan ties McKenna's wrists behind her back.
    #     Dan puts a strip of tape over McKenna's mouth.
    #     McKenna is bound and gagged in the back.
    #     Dan pulls a hood over McKenna's head.
    #
    # Not one of them registered, so the film had no mood, so every face came from
    # the portrait prior, which is pleasant. She smiled through her own abduction.
    MEM = "McKenna: she, 26, dark hair, a red jacket.\nDan: he, 40, a work coat."
    SCRIPT = ["Dan grabs McKenna from behind and covers her mouth.",
              "Dan drags McKenna towards the van.",
              "Dan forces McKenna into the back of the van.",
              "Dan ties McKenna's wrists behind her back.",
              "Dan puts a strip of tape over McKenna's mouth.",
              "Dan holds McKenna down.",
              "Dan shoves McKenna against the wheel arch.",
              "McKenna is bound and gagged in the back.",
              "Dan pulls a hood over McKenna's head.",
              "McKenna tries to get away."]
    check("the whole script reads as an abduction",
          S.film_stages_duress(SCRIPT, MEM), "")
    # Graded, because the words differ in how much they prove. Hardware on a body
    # says duress on its own; grabbing and dragging are equally a garden centre.
    for _b in ("Dan ties McKenna's wrists behind her back.",
               "Dan puts a strip of tape over McKenna's mouth.",
               "McKenna is bound and gagged in the back.",
               "Dan pulls a hood over McKenna's head."):
        check(f"unambiguous: {_b[:40]!r}",
              S.beat_duress_strength(_b) == "strong", S.beat_duress_strength(_b))
    for _b in ("Dan grabs McKenna from behind and covers her mouth.",
               "Dan drags McKenna towards the van.",
               "Dan forces McKenna into the back of the van.",
               "Dan holds McKenna down.",
               "Dan shoves McKenna against the wheel arch.",
               "McKenna tries to get away."):
        check(f"seen, but ambiguous: {_b[:40]!r}",
              S.beat_duress_strength(_b) == "weak", S.beat_duress_strength(_b))
    # PASSIVE VOICE, which is how half an abduction gets written and which every
    # active pattern missed: the victim is the subject, so nothing follows the verb.
    for _b in ("She was grabbed from behind on the towpath.",
               "She is bundled into the back of the van.",
               "They were hauled out of the church one at a time.",
               "He was walked out of the building between two men.",
               "She is loaded into the back like freight.",
               "He is being held captive somewhere in the north of the city."):
        check(f"passive voice is seen: {_b[:40]!r}",
              S.beat_duress_strength(_b) != "", "")

    # AND IT IS NOT JUST THE FILM'S MOOD -- the shot he drags her in is a shot her
    # face is in, so that shot gets the face clause too, not only the tone.
    sh = [" ".join(x.split()) for x in
          re.split(r"(?=\[Shot )",
                   run_node("A lane at night.\n\nDan drags McKenna towards the van."
                            "\n\nDan opens the rear doors.", plan_only=True,
                            character_memory=MEM,
                            anchor="Handheld, night. A grim abduction.")[3])
          if x.strip()]
    check("the coercion beat gets the face, not just the tone",
          "shows the strain" in sh[1], sh[1][-170:])
    check("...and the beat without her still carries the film's mood",
          "The mood is grim" in sh[2], sh[2][-170:])

    # STILL NOT INVENTED. The verbs need a PERSON: an object being grabbed, dragged
    # or forced is a prop, not a victim, and a film of those is not a grim film.
    for _b in ("Kate grabs a coffee and leaves.",
               "Kate drags the case to the door.",
               "Kate forces the window open.",
               "Kate ties her hair back.",
               "Kate tapes the box shut.",
               "Kate pulls the hood of the car open."):
        check(f"a prop is not a victim: {_b[:34]!r}",
              not S.film_stages_duress([_b], "Kate: she, 30, a grey coat."), "")
    check("an ordinary film is still left alone",
          not S.film_stages_duress(["Kate makes the coffee.", "Sam reads the paper."],
                                   "Kate: she, 30.\nSam: he, 33."))


def test_a_line_is_locked_to_the_person_who_says_it():
    print("\n=== the line belongs to one mouth and the others are shut ===")
    # Asked for: "an algorithm that keeps dialogue locked to who says it and nobody
    # else talks." Attribution was not the problem -- speakers_in reads 19 of 20
    # phrasings, including inverted ('"Wait," says Dan'), interrupted and
    # addressee forms. What was missing is that the LOCK was only emitted when
    # somebody else was described in the same beat:
    #
    #     Dan says: "Wait here."                        -> no lock at all
    #     Dan says: "Wait." Mara says: "No."            -> no lock at all
    #     Dan says: "Wait here." Mara says nothing.     -> no lock at all
    #
    # The first is the commonest beat in any script. Mara is standing there from
    # the previous shot -- the picture this shot starts from has her in it -- and
    # nothing told the shot she is not the one talking.
    MEM = "Dan: he, 40, a work coat.\nMara: she, 33, a green scarf."

    def shot(second, first="Dan and Mara stand by the door."):
        sc = re.split(r"(?=\[Shot )",
                      run_node("A hallway.\n\n" + first + "\n\n" + second,
                               plan_only=True, character_memory=MEM,
                               anchor="Wide, day.")[3])
        return " ".join([x for x in sc if x.strip()][-1].split())

    # 1. THE CARRIED CAST IS STILL IN THE ROOM. A beat naming only the speaker does
    # not empty the frame, and the person it does not name is exactly the mouth an
    # invented voice lands on.
    a = shot('Dan says: "Wait here."')
    check("a lone speaking beat still locks the line",
          "Only Dan speaks" in a, a[-190:])
    check("...and closes the mouth of whoever else is in the room",
          "every other mouth in the shot stays closed" in a, a[-190:])

    # 2. TWO SPEAKERS, TWO LINES. Nothing said which was whose, so they could be
    # swapped between the two faces.
    b = shot('Dan says: "Wait." Mara says: "No."')
    check("two speakers are put in order",
          "Dan speaks first, then Mara" in b, b[-190:])

    # 3. "SAYS NOTHING" IS NOT SPEAKING. It read as a second speaker, which left
    # nobody silent and so cancelled the lock entirely -- a negation turning the
    # guard off is the worst possible reading of it.
    check("saying nothing is not saying something",
          S.speakers_in('Dan says: "Wait here." Mara says nothing.', MEM) == ["Dan"],
          str(S.speakers_in('Dan says: "Wait here." Mara says nothing.', MEM)))
    for _b in ('Mara says nothing.', 'Mara said nothing at all.',
               'Mara does not say a word.', 'Mara never says a word.'):
        check(f"...{_b!r}", S.speakers_in(_b, MEM) == [], str(S.speakers_in(_b, MEM)))
    c = shot('Dan says: "Wait here." Mara says nothing.')
    check("...so the lock is not cancelled by it", "Only Dan speaks" in c, c[-190:])

    # NOT INVENTED. With nobody else anywhere in the film there is no second mouth
    # to close, and the shot is left alone rather than told about absent people.
    solo = " ".join([x for x in re.split(r"(?=\[Shot )",
                     run_node('A hallway.\n\nDan says: "Wait here."', plan_only=True,
                              character_memory="Dan: he, 40, a work coat.",
                              anchor="Wide, day.")[3]) if x.strip()][-1].split())
    check("one person in the whole film gets no lock clause",
          "every other mouth" not in solo, solo[-160:])
    # ...and an unattributable line still falls back to counting the voices.
    amb = shot('"Wait here."')
    check("an unattributed line still says how many voices there are",
          "Only the person speaking" in amb, amb[-190:])


def test_mouths_stay_shut_with_no_line():
    print("\n=== a shot with nobody speaking keeps its mouth closed ===")
    # H3 is joint: the face follows the audio branch. A shot with no line but a sound
    # the AUTHOR wrote -- "a low hum off the strip light" -- kept its branch open, and
    # an open branch invents a voice the picture lip-syncs to. Nobody is speaking and
    # the mouth moves anyway. That was the hole; a written sound was enough to open it.
    P = ('A workshop.\n\n'
         'Kate walks to the window.\n\n'
         'Kate says: "Wait there."\n\n'
         'A low hum comes off the strip light.\n\n'
         'Kate strains against the cuffs.')
    # With a sheet, because that is how the node knows Kate is a person. It does not
    # scan prose for capitals on purpose -- a sheet LABELS people and guessing from
    # capitalisation picks up place names. A beat with a bare pronoun still works
    # without one; a beat with only an unlisted name does not, and that is the
    # conservative direction: no clause rather than a clause about nobody.
    MEM = "Kate: she, 30, red coat."
    # auto_sound OFF: the trade this reports -- a written sound given up so the
    # mouth can be held -- only arises when the branch would otherwise be pinned.
    info, script = run_node(P, plan_only=True, character_memory=MEM,
                            auto_sound=False)[2:4]
    sh = [s for s in script.split("---") if s.strip()]
    mouth = [i + 1 for i, s in enumerate(sh) if "Mouths in the shot stay closed" in s]
    check("the wordless shot with a person is told to close", mouth == [1], str(mouth))
    check("the speaking shot is not", "Mouths in the shot stay closed" not in sh[1], "")
    # ca75672, which this must not undo: a mouth sentence on a beat with nobody in it
    # describes a person who is not there, and the only way to satisfy it is to draw a
    # face into an empty frame. The AUDIO half has no such limit -- an empty room still
    # babbles -- so shot 3 is silenced without being told anything about mouths.
    check("the scenery beat is told nothing about mouths",
          "Mouths in the shot stay closed" not in sh[2], "")
    check("...but is still silenced", "sound is" not in sh[2], "")
    # Effort is vocal and its mouth SHOULD be open. Silencing a straining body was a
    # bug once already -- it renders as a flat, unreacting face.
    check("a straining body is left alone", "Mouths in the shot stay closed" not in sh[3], "")
    # ...and its branch stays OPEN. With auto_sound off there is no sound sentence
    # to look for, so the guarantee is read from the report: shot 4 is one of the
    # shots whose audio is left free, which is what "keeps its audio" means.
    check("...and keeps its audio",
          "either describe a sound IN THE BEAT or stage EFFORT" in info
          and re.search(r"shot\(s\) [^|]*\b4\b[^|]*stage EFFORT", info) is not None,
          "")
    # The written sound on a wordless shot is given up, because conditioning the
    # branch is the only thing that actually settles the mouth.
    check("the wordless sound shot is silenced", "sound is" not in sh[2], "")
    check("info names the shots held closed", "mouths held closed on shot(s) 1" in info, "")
    check("...and names what it cost", "gave up the sound you wrote" in info, "")
    # The switch puts it all back.
    off = run_node(P, plan_only=True, character_memory=MEM,
                   mouths_shut_when_no_line=False)[3]
    check("off, nothing is told to close", "Mouths in the shot stay closed" not in off, "")
    check("off, the written sound comes back",
          "sound" in [s for s in off.split("---") if s.strip()][2].lower(), "")
    # Positively phrased: at cfg 1 no negative is evaluated, so an absence cannot be
    # asked for -- only a closed mouth can be drawn.
    check("the clause is positively phrased",
          not re.search(r"\bno\b|\bnot\b|\bnever\b|\bnobody\b", S.MOUTH_HOLD, re.I), "")
    check("...and is one short sentence",
          S.MOUTH_HOLD.count(".") == 1 and len(S.MOUTH_HOLD.split()) <= 14, "")


def test_script_is_what_was_sent():
    print("\n=== script reports the text the model was actually given ===")
    # script is documented as the exact per-shot text, and it is what the reader is
    # told to check when a shot renders somebody they did not ask for. It was built
    # before the render loop and never touched again, while the recovered-face claim
    # was written onto the loop's own copy -- so on exactly the shot most likely to
    # be under investigation, the model got "Dom: <Picture 1>, he, 41" and script
    # said "Dom: he, 41". Diagnosing a duplicate from that leads to the wrong fix.
    mem = "Mara: <Picture 1>, she, 30, red coat.\nDom: he, 41, grey jacket."
    P = ("Daylight. A yard.\n\nDom stands by the gate.\n\n"
         "Mara walks along the fence.\n\nDom looks at the sky.")
    sent, script = _prompts_sent(P, character_memory=mem,
                                 ref_image_1=torch.rand(1, H, W, 3))
    rep = [b.split("] ", 1)[1].strip() for b in script.split("\n---\n")]
    check("every shot matches, claim included",
          all(s.strip() == r for (s, _), r in zip(sent, rep)), "")
    check("the recovery shot really does carry a claim",
          "<Picture 1>" in sent[2][0] and "<Picture 1>" in rep[2], "")


def test_every_reference_is_claimed():
    print("\n=== no shot carries a picture its text never names ===")
    # The node's own rule, and the cause of every duplicate reported so far: a
    # picture the prompt refers to is that subject, and one it never mentions is
    # ANOTHER subject -- a second person with the same face and clothes. So the
    # count of references sent must equal the count of tags in the prompt, and the
    # numbers must run 1..n with no gaps, or a tag points at the wrong image.
    img = lambda: torch.rand(1, H, W, 3)
    cases = [
        ("one tagged, one untagged returning",
         "A yard.\n\nDom stands by the gate.\n\nMara walks along the fence.\n\n"
         "Dom looks at the sky.",
         dict(character_memory="Mara: <Picture 1>, she, 30, red coat.\nDom: he, 41.",
              ref_image_1=img())),
        ("two people, two tagged references",
         "A yard.\n\nMara waits.\n\nDom arrives.\n\nMara and Dom talk.",
         dict(character_memory="Mara: <Picture 1>, she, 30.\nDom: <Picture 2>, he, 41.",
              ref_image_1=img(), ref_image_2=img())),
        ("a reference nobody tags",
         "A yard.\n\nMara waits.\n\nDom arrives.",
         dict(character_memory="Mara: <Picture 1>, she, 30.\nDom: he, 41.",
              ref_image_1=img(), ref_image_2=img())),
        ("an object tag beside a person tag",
         "A yard.\n\nMara waits.\n\nMara holds the locket.\n\nDom arrives.",
         dict(character_memory="Mara: <Picture 1>, she, a silver locket <Picture 2>.\n"
                               "Dom: he, 41.",
              ref_image_1=img(), ref_image_2=img())),
    ]
    for name, P, kw in cases:
        sent, _ = _prompts_sent(P, **kw)
        bad = []
        for i, (p, n) in enumerate(sent, 1):
            tags = sorted({int(x) for x in re.findall(r"<Picture (\d+)>", p)})
            if n != len(tags) or tags != list(range(1, n + 1)):
                bad.append(f"shot {i}: {n} sent, tags {tags}")
        check(f"{name}: every picture is claimed", not bad, "; ".join(bad))


def _encoded_refs(P, **kw):
    """(prompt, number of images actually encoded as references) per shot.

    The demotion happens inside build_conditioning, so counting the refs handed TO
    it misses the handoff being added. This counts what reaches the encoder."""
    rows, box = [], {}
    ob, orr = S.build_conditioning, S._cond_module._build_ref_images
    def spy_r(vae, imgs, w, h, size):
        box["n"] = len(imgs); return orr(vae, imgs, w, h, size)
    def spy_b(clip, vae, audio_vae, prompt, *a, **k):
        box["n"] = 0
        out = ob(clip, vae, audio_vae, prompt, *a, **k)
        rows.append((prompt, box["n"]))
        return out
    S._cond_module._build_ref_images, S.build_conditioning = spy_r, spy_b
    try:
        run_node(P, **kw)
    finally:
        S._cond_module._build_ref_images, S.build_conditioning = orr, ob
    return rows


def test_the_demoted_handoff_is_claimed():
    print("\n=== below the safe aug, the handoff is named too ===")
    # Reported as a duplicate Dan at the END of the video. Below KEYFRAME_SAFE_AUG
    # the handoff stops being a keyframe and is encoded as an extra reference -- and
    # it went in unclaimed, on the reasoning that a first frame is not a subject. In
    # the reference rows it is not a first frame any more: it is picture N of N, and
    # a picture the prompt never names is read as ANOTHER subject. The picture is the
    # previous shot, so the other subject wears that shot's face and clothes.
    #
    # Later shots only, because a shot needs BOTH a handoff and a reference to demote
    # anything -- which is exactly "at the end of the video".
    mem = "Dan: <Picture 1>, he, 41, grey jacket.\nMara: she, 30, red coat."
    P = "A yard.\n\nDan waits.\n\nMara arrives.\n\nDan and Mara talk.\n\nDan looks up."
    img = torch.rand(1, H, W, 3)
    for aug in (0.999, 0.98, 0.95, 0.90):
        rows = _encoded_refs(P, character_memory=mem, ref_image_1=img, ref_noise_aug=aug)
        bad = []
        for i, (p, enc) in enumerate(rows, 1):
            tags = {int(x) for x in re.findall(r"<Picture (\d+)>", p)}
            if enc != len(tags) or sorted(tags) != list(range(1, enc + 1)):
                bad.append(f"shot {i}: {enc} encoded, tags {sorted(tags)}")
        check(f"every picture claimed at aug {aug}", not bad, "; ".join(bad))
    # The claim must say it is the SAME people, or naming it invites a new one.
    rows = _encoded_refs(P, character_memory=mem, ref_image_1=img, ref_noise_aug=0.95)
    late = rows[2][0]
    check("the handoff is named as the opening frame",
          "is the frame this shot opens on" in late, "")
    check("...and as the same people, not new ones",
          "the same people" in late and "anybody new" in late, "")
    # At a safe aug nothing is demoted, so nothing extra is said.
    early = _encoded_refs(P, character_memory=mem, ref_image_1=img, ref_noise_aug=0.999)
    check("nothing added when the handoff stays a keyframe",
          all("opens on" not in p for p, _ in early), "")
    info = run_node(P, character_memory=mem, ref_image_1=img, ref_noise_aug=0.95)[2]
    check("info explains it", "encoded as a reference rather than a keyframe" in info, "")


def test_a_gapped_socket_still_sends_its_image():
    print("\n=== wiring ref_image_1 and ref_image_3 sends both ===")
    # Whole-path, because the unit test cannot show the image being dropped. Wire a
    # gap and the second picture used to vanish: its tag matched nothing in a roster
    # of two, so it was stripped, and a reference nothing claims is not sent.
    img = lambda: torch.rand(1, H, W, 3)
    P = "A yard.\n\nMara waits.\n\nMara walks."
    sent, _ = _prompts_sent(P, character_memory="Mara: <Picture 1>, she, a locket <Picture 3>.",
                            ref_image_1=img(), ref_image_3=img())
    p, n = sent[0]
    check("both images are sent", n == 2, f"{n}")
    check("...and both are claimed in the text",
          sorted(re.findall(r"<Picture (\d+)>", p)) == ["1", "2"], p[:90])
    # A single image on a socket that is not the first.
    sent, _ = _prompts_sent(P, character_memory="Mara: <Picture 2>, she, 30.",
                            ref_image_2=img())
    p, n = sent[0]
    check("a lone image on socket 2 is sent", n == 1 and "<Picture 1>" in p, f"{n} {p[:60]}")
    # Sockets filled from the top are untouched -- this must not move anybody's
    # working setup.
    sent, _ = _prompts_sent(P, character_memory="Mara: <Picture 1>, she, a locket <Picture 2>.",
                            ref_image_1=img(), ref_image_2=img())
    p, n = sent[0]
    check("no gap, nothing changes",
          n == 2 and sorted(re.findall(r"<Picture (\d+)>", p)) == ["1", "2"], f"{n}")
    # The renumbered tag must still fetch the RIGHT image. Two pictures that can be
    # told apart by value, so a mix-up shows up as the wrong face rather than as a
    # number that merely looks tidy. This is the check that matters: renumbering is
    # only safe if the image follows the number.
    A = torch.full((1, H, W, 3), 0.10)
    B = torch.full((1, H, W, 3), 0.70)
    val = lambda t: round(float(t.mean()), 2)
    seen = []
    orig = S.build_conditioning
    def spy(clip, vae, audio_vae, prompt, *a, **k):
        seen.append((prompt, [val(r) for r in (k.get("refs") or [])]))
        return orig(clip, vae, audio_vae, prompt, *a, **k)
    S.build_conditioning = spy
    try:
        run_node("A yard.\n\nMara waits.\n\nDom arrives.\n\nMara and Dom talk.",
                 character_memory="Mara: <Picture 1>, she, 30.\nDom: <Picture 3>, he, 41.",
                 ref_image_1=A, ref_image_3=B)
    finally:
        S.build_conditioning = orig
    check("the lone-reference shot renumbers to 1",
          re.findall(r"<Picture (\d+)>", seen[1][0]) == ["1"], str(seen[1][0][-60:]))
    check("...and still sends that person's own image",
          seen[1][1] == [0.7], str(seen[1][1]))
    check("the shared shot keeps both, in order",
          sorted(re.findall(r"<Picture (\d+)>", seen[2][0])) == ["1", "2"]
          and seen[2][1] == [0.1, 0.7], str(seen[2][1]))
    # And a tag on an empty socket is reported rather than silently dropped.
    info = run_node(P, plan_only=True, character_memory="Mara: <Picture 3>, she, 30.",
                    ref_image_1=img())[2]
    check("a tag with no image behind it is named",
          "names a socket with no image on it" in info, "")


def test_a_modified_state_is_not_read_as_an_act():
    print("\n=== 'closed rear doors' is not somebody closing them ===")
    # Reported after the state hold shipped: the doors were STILL opening and being
    # closed on camera. The node was doing it. A word between the state and its noun
    # made "closed" parse as the verb, and the shot was handed "the doors are open at
    # the first frame and shut by the last" -- the inverted guard, asking for exactly
    # the render it exists to prevent. Whole-path, because that is where it bit.
    for _beat in ("Mara and Dom walk out from behind a van with closed rear doors.",
                  "Mara and Dom step out from behind the van, its back doors closed.",
                  "Mara and Dom stand behind a van with shut cargo doors."):
        s = run_node("Daylight. A yard.\n\n" + _beat, plan_only=True)[3]
        check(f"held, not anchored: {_beat[36:60]!r}",
              "already" in s and "open at the first frame" not in s, "")
    # The one that really does stage it still gets its two ends.
    s = run_node("Daylight. A yard.\n\nMara closed the van's doors.", plan_only=True)[3]
    check("a possessive act is still an act",
          "The doors are open at the first frame and shut by the last." in s, "")
    # And the state is bounded: "stays closed" with no end on it is a state something
    # can happen to before the shot is out.
    s = run_node("Daylight. A yard.\n\nThey stand by a van with its doors closed.",
                 plan_only=True)[3]
    check("the held state is bounded", "for the whole shot" in s, "")
    # Whole-path: the auto sound must not ask for the swing the hold forbids. Needs a
    # line in the beat, because a silent shot gets no sound clause at all.
    P = ('Daylight. A yard.\n\nMara stands behind a van with closed rear doors. '
         'Mara says: "We wait here."')
    s = run_node(P, plan_only=True)[3]
    check("the held doors are not also heard swinging",
          "already closed" in s and "a door on its hinges" not in s, "")
    check("...while the shot still has its other sound", "an engine outside" in s, "")
    # The shot that stages the opening keeps the sound of one.
    s = run_node('Daylight. A yard.\n\nMara opens the van doors. Mara says: "Here."',
                 plan_only=True)[3]
    check("a staged opening still sounds like one", "a door on its hinges" in s, "")


def test_a_staged_change_gets_both_ends():
    print("\n=== a staged change is anchored at both ends ===")
    # The shot that WORKS the thing gets no held state -- it is asking for that
    # motion -- but it is exactly the shot a reversing LoRA renders backwards. It
    # gets the two ends instead.
    P = ("Daylight. A yard, and a van with its doors closed.\n\n"
         "Mara and Dom stand behind the van.\n\n"
         "Mara opens the van doors and climbs in.\n\n"
         "Dom slams the tailgate shut.")
    shots = [s for s in run_node(P, plan_only=True)[3].split("---") if s.strip()]
    check("shot 1 holds the standing state",
          "already closed at the first frame" in shots[0], "")
    check("shot 2 gets both ends of the opening",
          "The doors are shut at the first frame and open by the last." in shots[1], "")
    check("shot 3 gets both ends of the shutting",
          "The tailgate is open at the first frame and shut by the last." in shots[2], "")
    check("the working shot is not also told the state holds",
          "already" not in shots[1].split("climbs in.")[1], "")
    info = run_node(P, plan_only=True)[2]
    check("info names the anchored shots", "shot(s) 2, 3 stage a change" in info, "")
    check("...and says where reversal is likeliest", "shot 1, which has no previous" in info, "")
    off = run_node(P, plan_only=True, hold_scene_state=False)[3]
    check("the switch turns it off too", "by the last" not in off, "")
    # Continuity and beat share one shot. Two anchors plus two held states would be
    # four continuity sentences on a beat that is one line long.
    B = ("A yard. The gate is open and the blinds are drawn.\n\n"
         "Mara opens the van doors and Dom lifts the lid of the crate.")
    one = run_node(B, plan_only=True)[3]
    check("at most two frame sentences in a shot", one.count("first frame") == 2, "")
    check("...and the beat's own action wins the budget",
          "The doors are shut" in one and "The lid is shut" in one, "")


def test_dialogue_headroom():
    print("\n=== how much of a speaking shot the line does not fill ===")
    # The audio branch is open for the WHOLE shot once there is a line in it, so the
    # seconds the line does not cover are unconditioned audio in a shot the model
    # already knows has a voice. That is where speech carries on past the line and
    # turns into babble.
    P = "\n\n".join([
        'Dan says: "Wait."',
        'Dan says: "Wait there a moment, I need to check the cable before you '
        'start it up."',
        "Nora picks up the spanner."])
    fixed = run_node(P, plan_only=True, anchor="A workshop.",
                     shot_length="fixed", shot_seconds=15.0)[2]
    check("a one-word line in a fixed shot is flagged", "dialogue headroom" in fixed)
    check("...with the words and the seconds", "shot 1: 1 word(s), about 0.4s" in fixed)
    check("...and a long line in the same shot too",
          "shot 2: 15 word(s), about 6.0s" in fixed)
    check("...naming the two ways out", "longer line, or a shorter shot" in fixed)
    # Sized from the beat, the shot follows the line and there is no tail to report.
    beat = run_node(P, plan_only=True, anchor="A workshop.",
                    shot_length="from the beat", shot_seconds=15.0)[2]
    check("sizing from the beat leaves no headroom", "dialogue headroom" not in beat)
    # A shot with no line has no dialogue to outlast; silence covers it instead.
    quiet = run_node("Nora walks to the window.\n\nNora sits down.", plan_only=True,
                     anchor="A workshop.", shot_length="fixed", shot_seconds=15.0)[2]
    check("a shot with no line is not flagged", "dialogue headroom" not in quiet)


def test_introducing_somebody_already_in_position():
    print("\n=== a character introduced in position starts fresh ===")
    # Reported: a character is thrown into the shot and then moved to where they
    # belong. Every shot continues from the previous shot's LAST FRAME, and somebody
    # appearing for the first time is not in it -- so the beat says where they are and
    # the picture says they are nowhere. The picture wins, and they have to arrive out
    # of nothing and travel to the spot.
    mem = "Nora: 34, she, red hair.\nDan: 41, he, dark hair, navy overalls"
    tail = "\n\nNora picks up the spanner."

    def encodes(beat2):
        vae = FakeVAE()
        info = run_node("Nora sets a toolbox on the bench.\n\n" + beat2 + tail,
                        anchor="A workshop.", character_memory=mem, vae=vae)[2]
        return vae.encodes, info

    # In position: the shot cannot inherit a frame that has him in it, so it starts
    # fresh. Three shots, and only shot 3 encodes a keyframe.
    # The frame stops being frame ONE -- and is still carried, as a reference, so the
    # room comes with it. Dropping it outright is what re-imagined the scenery
    # between shots, which build_conditioning's own note calls "same place, new room".
    n_placed, info = encodes("Dan is already sitting on the crate, watching her.")
    check("an in-position introduction still carries the frame",
          "carries the previous frame as a REFERENCE" in info)
    check("...and the run says so", "introduces Dan in position" in info)
    check("...naming what the room brings", "the room, the light and Nora come" in info)
    check("...and how to keep it as the anchor", "Write the entrance" in info)
    check("intentional room carry does not report a low-strength reference",
          "anchor would be noised" not in info)
    # CLAIMED, and claimed differently. The standing handoff claim says the shot is
    # "carried forward rather than joined by anybody new" -- exactly wrong in the one
    # case where somebody new is the reason the frame was demoted. Unclaimed, a
    # picture of Nora is a second Nora.
    # The full path on purpose: the carry is decided in the render loop, because it
    # needs a frame to carry. plan_only has none and reports none.
    _s2 = re.split(r"\[Shot ", run_node(
        "Nora sets a toolbox on the bench.\n\nDan is already sitting on the crate, "
        "watching her." + tail, anchor="A workshop.", character_memory=mem)[3])[2]
    check("the carried frame is claimed as the room",
          "is this room a moment earlier" in _s2, _s2[-260:])
    check("...naming who was in it", "Nora is the person there" in _s2)
    check("...and who is already in place", "Dan is in this room too" in _s2)
    check("...without the claim that nobody new joins",
          "joined by anybody new" not in _s2)
    # NOT claimable: somebody in that frame is absent from this shot, so the picture
    # would carry a person the prompt cannot account for. The old fresh start stands.
    _info3 = run_node(
        "Nora and Ada set a toolbox on the bench.\n\nDan is already sitting on the "
        "crate, watching Ada.\n\nAda picks up the spanner.", anchor="A workshop.",
        character_memory=mem + "\nAda: 29, she, short hair")[2]
    check("an unaccountable person in the frame keeps the fresh start",
          "carries the previous frame as a REFERENCE" not in _info3)
    # Arriving is what the chain is FOR: he walks in from the frame before.
    n_arrive, info2 = encodes("Dan walks in through the side door and looks at her.")
    check("an arriving introduction keeps the chain", n_arrive == 2, str(n_arrive))
    check("...and claims nothing", "introduces Dan in position" not in info2)
    for _b, _want in (("Dan enters the workshop.", True),
                      ("Dan comes back in.", True),
                      ("Nora follows him in.", True),
                      ("Dan steps into the room.", True),
                      ("Dan is at the far wall, watching her.", False),
                      ("Dan stands at the bench.", False),
                      ("Dan waits by the roller door.", False)):
        check(f"arrival={_want}: {_b[:34]!r}", S.arrives_in(_b) == _want)


def test_back_after_a_shot_away():
    print("\n=== somebody back after a shot away ===")
    # Reported: a character's appearance is lost when they walk out of frame and
    # return. Every shot starts from the PREVIOUS shot's last frame, so somebody who
    # was not in that shot is not in the picture this one begins from -- their
    # appearance comes from the sheet text and nothing else, and text drifts where a
    # picture does not.
    P = "\n\n".join(["Nora sets a toolbox on the bench.",
                     "Nora walks out through the side entrance.",
                     "Victor walks in and kneels by the cable, alone in the workshop.",
                     "Nora comes back in and picks up the spanner."])
    mem = "Nora: 34, she, tall, red hair.\nVictor: he, 41, dark hair"
    info = run_node(P, plan_only=True, anchor="A room.", character_memory=mem)[2]
    check("the return is detected", "back after a shot away" in info)
    check("...naming the shot and who", "shot 4: Nora" in info)
    check("...and why the keyframe cannot carry them",
          "not in the picture this one begins from" in info)
    # A reference tag IS the picture that pins them, so the advice differs.
    check("with no tag, it says to add one", "no <Picture N> tag" in info)
    tagged = run_node(P, plan_only=True, anchor="A room.",
                      character_memory="Nora: <picture 1>, 34, she, red hair.\n"
                                       "Victor: he, 41, dark hair",
                      ref_image_1=torch.rand(1, H, W, 3))[2]
    check("with a tag, it says they are pinned", "All of them carry a reference" in tagged)
    check("...and does not ask for one", "no <Picture N> tag" not in tagged)
    # Nobody leaves, nobody returns.
    straight = run_node("Nora walks in.\n\nNora sits down.", plan_only=True,
                        anchor="A room.", character_memory=mem)[2]
    check("a chain nobody leaves reports nothing",
          "back after a shot away" not in straight)

    # And the fix: a frame from the middle of the last shot they WERE in, sent as a
    # reference on the shot they come back to. The middle because somebody walking
    # out is gone by the last frame and somebody walking in is missing from the first.
    seen = []
    orig = FakeCLIP.tokenize

    def spy(self, text, minimax_ref_items=None, **kw):
        seen.append(sum(1 for it in (minimax_ref_items or []) if it["type"] == "image"))
        return orig(self, text, minimax_ref_items=minimax_ref_items, **kw)

    FakeCLIP.tokenize = spy
    try:
        base = ["Nora sets a toolbox on the bench.",
                "Nora walks out through the side entrance.",
                "Victor walks in and kneels by the cable, alone in the workshop."]
        seen.clear()
        info = run_node("\n\n".join(base + ["Nora comes back in and picks up the spanner."]),
                        anchor="A room.", character_memory=mem)[2]
        # shot 1 no keyframe and no refs; 2 and 3 their keyframe; 4 keyframe + recovered.
        check("the return shot gets a second picture", seen[1:] == [0, 1, 1, 2],
              str(seen[1:]))
        check("...and the run says whose face and from where",
              "recovered a face for Nora on shot 4, from shot 2" in info)
        # NARROW: the recovered frame carries whoever else was on screen, so a return
        # into company is reported and left alone rather than risking a second person.
        seen.clear()
        multi = run_node("\n\n".join(
            base + ["Nora comes back in and Victor hands her the spanner."]),
            anchor="A room.", character_memory=mem)[2]
        check("a return into company is left alone", seen[1:] == [0, 1, 1, 1],
              str(seen[1:]))
        check("...and nothing is claimed for it", "recovered a face" not in multi)
        # A tagged character already has their own reference travelling with them.
        seen.clear()
        tagged = run_node("\n\n".join(base + ["Nora comes back in and picks up the spanner."]),
                          anchor="A room.",
                          character_memory="Nora: <picture 1>, 34, she, red hair.\n"
                                           "Victor: he, 41, dark hair",
                          ref_image_1=torch.rand(1, H, W, 3))[2]
        check("a tagged character is not given a second picture",
              "recovered a face" not in tagged)
        # THE RECOVERED FRAME HAS TO BE CLAIMED IN THE PROSE. A picture the prompt
        # refers to is that subject; one it never mentions is ANOTHER subject. Sent
        # unclaimed, a frame of somebody is read as a second person who looks exactly
        # like them -- same face, same clothes -- beside the one the beat asked for.
        tagseen = []
        _o = FakeCLIP.tokenize

        def _spy(self, text, minimax_ref_items=None, **kw):
            tagseen.append((sum(1 for it in (minimax_ref_items or [])
                                if it["type"] == "image"),
                            re.findall(r"<Picture \d+>", text), text))
            return _o(self, text, minimax_ref_items=minimax_ref_items, **kw)

        FakeCLIP.tokenize = _spy
        try:
            run_node("\n\n".join(["Dan walks into the workshop alone.",
                                  "Dan walks out through the side door.",
                                  "Nora kneels by the cable, alone.",
                                  "Dan comes back in and picks up the spanner."]),
                     anchor="A workshop.",
                     character_memory="Nora: 34, she, red hair.\n"
                                      "Dan: he, 41, dark hair, navy overalls")
        finally:
            FakeCLIP.tokenize = _o
        _pics, _tags, _txt = tagseen[4]          # [0] is the negative; shot 4
        check("the return shot carries the recovered frame and its keyframe",
              _pics == 2, str(_pics))
        check("...and the prose claims the recovered one", _tags == ["<Picture 1>"],
              str(_tags))
        check("...on the entry of the person it depicts",
              "Dan: <Picture 1>," in _txt, _txt[:80])
        # THE SOURCE has to be solo too, not just the destination. A frame is a
        # picture of everyone in it, so one captured from a shared shot brings the
        # other person back into a shot that does not call for them -- which is the
        # second character turning up uninvited.
        shared = ["Nora walks into the workshop.",
                  "Victor comes in and Nora hands him the spanner.",
                  "Victor walks in and kneels by the cable, alone.",
                  "Nora comes back in and picks up the toolbox."]
        info2 = run_node("\n\n".join(shared), anchor="A room.", character_memory=mem)[2]
        src = re.search(r"recovered a face for Nora on shot 4, from shot (\d+)", info2)
        check("the frame comes from a shot that was hers alone",
              bool(src) and src.group(1) == "1", src.group(1) if src else "none")
        # Never on screen alone: no clean frame exists, so nothing is sent. Leaving
        # the sheet text to carry her beats importing somebody who is not in the shot.
        only_shared = run_node("\n\n".join(shared[1:]), anchor="A room.",
                               character_memory=mem)[2]
        check("no solo frame anywhere means nothing is recovered",
              "recovered a face" not in only_shared)
    finally:
        FakeCLIP.tokenize = orig


def test_a_name_with_no_entry_end_to_end():
    print("\n=== a person the sheet never describes ===")
    P = "\n\n".join(["Maya walks in.", "Alex says hello to Maya.",
                     "Alex walks to the window.",
                     "Maya stands up and Alex takes her hand."])
    mem = "Maya: she, 27, grey coat.\nJon: he, 35, jeans."
    info, script = run_node(P, plan_only=True, anchor="A room.",
                            character_memory=mem)[2:4]
    check("the run reports the undescribed person", "Alex, who has no entry" in info)
    check("...naming the shots", "shot(s) 2, 3, 4 name Alex" in info)
    check("...and what to do about it", "use one name throughout" in info)
    # Nothing is acted on: whether Alex is Jon under another name or a third person
    # is not answerable from the text, and guessing would rewrite the script.
    shots = [x for x in re.split(r"(?=\[Shot )", script) if x.strip()]
    check("the script is not rewritten", "Alex says hello to Maya." in shots[1])
    check("...and no entry is invented for them", "Alex:" not in script)
    # A sheet that covers everyone says nothing.
    clean = run_node("Maya walks in.\n\nJon greets Maya.", plan_only=True,
                     anchor="A room.", character_memory=mem)[2]
    check("a complete sheet is not reported", "has no entry" not in clean)


def test_references_ride_with_the_keyframe():
    print("\n=== a reference and the keyframe ride together ===")
    # I broke this, then removed it, on the conclusion that fl2va could not carry an
    # identity reference at all. Wrong, and the old node's own comment said so:
    # ComfyUI packs both channels in AGREEING orders -- model_base.py:2183-2191 builds
    # cond_video_latents as keyframe latents then ref latents, and PackedLayout emits
    # keyframe "cond" segments then ref "ref_img" ones -- so a shot takes references
    # AND a real keyframe. The keyframe anchors the first frame; a reference only
    # says who somebody is. They were never alternatives.
    #
    # Which image is the first frame is decided by resolved_frame_index in
    # minimax_keyframes, NOT by a label's number. The labels only have to line up with
    # the <Picture N> tags in the prompt, so references keep slots 1..N and the
    # handoff is appended after them where it disturbs no numbering.
    seen = []
    orig = FakeCLIP.tokenize

    def spy(self, text, minimax_ref_items=None, **kw):
        n = sum(1 for it in (minimax_ref_items or []) if it["type"] == "image")
        seen.append((n, len(re.findall(r"<Picture \d+>", text))))
        return orig(self, text, minimax_ref_items=minimax_ref_items, **kw)

    FakeCLIP.tokenize = spy
    try:
        mem = "Kate: <picture 1>, 22, she, blonde hair.\nMike: he, 35, jeans"
        seen.clear()
        # Mike ARRIVES, so the chain is kept and this stays a test of the picture
        # roster rather than of the fresh-start rule for an in-position introduction.
        run_node("Kate walks in.\n\nMike walks in and Kate sits beside him.\n\n"
                 "Mike stands up.",
                 anchor="A room.", character_memory=mem,
                 ref_image_1=torch.rand(1, H, W, 3))
        shots = seen[1:]                     # seen[0] is the negative
        # shot 1: the reference only -- no previous frame to hand over yet.
        # shot 2: the reference AND the keyframe. This is the pairing I forbade.
        # shot 3: the guard drops Kate, so no tag, so no reference; keyframe only.
        check("the roster is ref + keyframe, not one or the other",
              [p for p, _ in shots] == [1, 2, 1], str([p for p, _ in shots]))
        check("the shot carrying both still names exactly one picture",
              shots[1] == (2, 1), str(shots[1]))
        check("a shot the guard trimmed carries no reference", shots[2][1] == 0)
        # The tag stays IN the prompt: it is the binding between the picture and the
        # person, which comfy_extras/nodes_minimax_h3.py tells you to write there.
        info, script = run_node("Kate walks in.\n\nKate sits down.", plan_only=True,
                                anchor="A room.", character_memory=mem,
                                ref_image_1=torch.rand(1, H, W, 3))[2:4]
        check("the binding survives into the script", "<Picture 1>" in script)
        check("...on the person it depicts", "Kate: <Picture 1>, 22" in script)
        check("the run explains the pairing", "ride alongside the keyframe" in info)
        # No reference connected: the handoff is the only picture, which is H3's own
        # first-frame shape.
        seen.clear()
        run_node("A room.\n\nOne.\n\nTwo.\n\nThree.")
        check("a chain with no reference keeps the keyframe picture",
              any(pics == 1 for pics, _ in seen))
    finally:
        FakeCLIP.tokenize = orig


def test_audio_sigma_reads_the_scheduler():
    print("\n=== the last audio sigma follows the SCHEDULER, not just the formula ===")
    # The closed form a/(steps + a - 1) is exact for `simple` and WRONG for every
    # other scheduler. Measured on a real ComfyUI at 5 steps, shift_audio 3.0:
    #     simple      0.4286   (the formula agrees)
    #     beta        0.2981
    #     kl_optimal  0.0030   <- 143x smaller than the formula claims
    # The note that fires above 0.40 therefore warned about babble that a scheduler
    # had already fixed, and advised lowering shift_audio -- the wrong lever, when
    # the fix was one dropdown away. comfy/ldm/minimax/model.py:569 derives the
    # audio sigma from the VIDEO sigma (time_shift_sigma), so the ladder the
    # scheduler produces is what decides this, not a closed form.
    import comfy.samplers as _cs
    _ms_mod = sys.modules.setdefault("comfy.model_sampling",
                                     types.ModuleType("comfy.model_sampling"))
    setattr(sys.modules["comfy"], "model_sampling", _ms_mod)
    seen = []
    def _fake(model_sampling, scheduler, steps):
        seen.append((scheduler, steps))
        return [1.0, 0.669, 0.416, 0.201, 0.003, 0.0]   # kl_optimal-shaped tail
    class _FakeMS:
        def set_parameters(self, **kw): self.kw = kw
    _had = getattr(_cs, "calculate_sigmas", None)
    _had_ms = getattr(_ms_mod, "ModelSamplingDiscreteFlow", None)
    _cs.calculate_sigmas = _fake
    _ms_mod.ModelSamplingDiscreteFlow = _FakeMS
    try:
        got = S.last_audio_sigma(5, 3.0, scheduler="kl_optimal")
    finally:
        if _had is None:
            del _cs.calculate_sigmas
        else:
            _cs.calculate_sigmas = _had
        if _had_ms is None:
            del _ms_mod.ModelSamplingDiscreteFlow
        else:
            _ms_mod.ModelSamplingDiscreteFlow = _had_ms
    check("it asks the scheduler for the real ladder", seen == [("kl_optimal", 5)], str(seen))
    check("...and reports that, not the formula's 0.43", got < 0.05, f"{got:.4f}")


def test_audio_sigma_falls_back_to_the_closed_form():
    print("\n=== ...and without a real ComfyUI it still gives the `simple` answer ===")
    # The stub has no calculate_sigmas. The closed form is exact for `simple`, which
    # is the shipped default, so nothing regresses where the formula was right.
    got = S.last_audio_sigma(5, 3.0)
    check("closed form still exact for simple", abs(got - 3.0 / 7.0) < 1e-6, f"{got:.4f}")
    check("...and 8 steps still gives the documented 0.30",
          abs(S.last_audio_sigma(8, 3.0) - 0.30) < 1e-6, f"{S.last_audio_sigma(8, 3.0):.4f}")

def test_a_hidden_garments_lettering_goes_with_it():
    print("\n=== the print on a covered garment is covered too ===")
    # Reported from a real render: a thong under shorts, with lettering on the
    # thong, and the lettering came out ON THE SHORTS.
    #
    # hide_item took the item plus up to three words IN FRONT of it and nothing
    # behind, so
    #     "denim shorts, a black thong with \"PRINCESS\" across the front."
    # became
    #     "denim shorts, with \"PRINCESS\" across the front."
    # -- the garment deleted out from under its own modifier, which then sits in
    # the list directly after the shorts. A described print is a drawn print, and
    # it is drawn on whatever garment is still there to carry it.
    #
    # The fragment survived because the emptiness test only strips articles: what
    # was left read as '"PRINCESS"acrossthefront', which is not empty.
    T = 'McKenna: she, 26, denim shorts, a black thong with "PRINCESS" across the front.'
    got = S.hide_item(T, ["a black thong"])
    check("the lettering leaves with the garment", "princess" not in got.lower(), got)
    check("...and the outer garment stays", "denim shorts" in got.lower(), got)
    check("...and the person keeps her line", "McKenna" in got, got)

    # Lettering written BEFORE the noun fails the same way, because the adjective
    # window is \\w-only and a quoted word breaks it: it left 'a "PRINCESS" .'
    T2 = 'McKenna: she, 26, denim shorts, a "PRINCESS" lettered thong.'
    got2 = S.hide_item(T2, ["thong"])
    check("...also when the print is written first",
          "princess" not in got2.lower(), got2)

    # THE NARROW HALF. A fragment naming a SECOND garment must not be dropped
    # whole -- that is the bug hide_item exists to avoid ("green dress, steel
    # collar" going to nothing), so only a fragment with no garment left in it goes.
    T3 = "McKenna: she, 26, a thong and denim shorts."
    got3 = S.hide_item(T3, ["thong"])
    check("a fragment naming another garment keeps it",
          "denim shorts" in got3.lower(), got3)

    # THE PRINT IN ITS OWN FRAGMENT. Above, the lettering sits inside the
    # garment's fragment, so the removal empties it and it goes whole. Written
    # after the comma -- which is how a sheet lists attributes -- nothing was
    # removed from it, so it stayed, right before the shorts with no garment to
    # carry it. Reported as the thong's lettering on the shorts, again. A
    # sentence of its own failed worse: the fragment kept the shorts, so the
    # whole of it stayed, shipping "She . BRAT is printed across the back."
    for label, T in (
            ("its own fragment",
             'McKenna: she, 26, a black thong, "BRAT" printed across the back, denim shorts.'),
            ("pointing back with a pronoun",
             "McKenna: she, 26, a black thong, the word BRAT across its back, denim shorts."),
            ("a bare capitalised word",
             "McKenna: she, 26, a black thong, BRAT across the back, denim shorts."),
            ("a trailing modifier that is not a print",
             "McKenna: she, 26, a black thong, with a bow at the hip, denim shorts."),
            ("a sentence of its own",
             "McKenna: she, 26, denim shorts. She wears a black thong. BRAT is printed across the back.")):
        got = S.hide_item(T, ["thong"])
        check(f"the print goes with the garment: {label}",
              "brat" not in got.lower() and "bow" not in got.lower()
              and "denim shorts" in got.lower() and "McKenna" in got
              and "She ." not in got, got)
    # ...and only what CONTINUES the garment. An item that merely follows it in
    # the list stands on its own, or a covered thong would take the lipstick with it.
    T5 = "McKenna: she, 26, a black thong, red lipstick, a tattoo across the lower back, denim shorts."
    got5 = S.hide_item(T5, ["thong"])
    check("an unrelated item after the garment stays",
          "red lipstick" in got5 and "tattoo across the lower back" in got5
          and "thong" not in got5, got5)

def test_camera_framing_is_read_from_the_anchor():
    print("\n=== a close frame written in the ANCHOR is still a close frame ===")
    # The anchor is the documented home for camera -- "Framing that belongs to the
    # whole film -- look, camera, lighting, location" -- and it is carried at the
    # front of every shot. But tight_framing() was only ever handed the BEAT, so a
    # film shot entirely in close-up, declared once in the anchor the way the
    # tooltip asks for, registered as no close framing anywhere.
    #
    # What that cost: the note warning that a tight frame crops the anchor point
    # out of the frame the NEXT shot inherits -- the one thing that still knows
    # where the limbs are fastened -- never fired for anybody who put their camera
    # where they were told to put it. It fired only for people who wrote "close-up"
    # into a beat, which the tooltip does not ask them to do.
    check("the anchor's framing is seen", S.tight_framing("Close-up on her face."))
    check("...and a beat's still is", S.tight_framing("She leans in, tight on her eyes."))
    check("...and a wide anchor is not", not S.tight_framing("Wide shot, night."))

    mem = "McKenna: she, 26, denim shorts."
    P = ("McKenna lies on the bed, wrists cuffed behind her back.\n\n"
         "She turns her head.")
    info = run_node(P, plan_only=True, character_memory=mem,
                    anchor="Close-up on her face. 85mm lens.")[2]
    check("the tight-frame warning fires from the anchor alone",
          "frame tight enough to crop the anchor point out" in info,
          info[:200])
    wide = run_node(P, plan_only=True, character_memory=mem,
                    anchor="Wide shot, night.")[2]
    check("...and does not fire on a wide film",
          "frame tight enough to crop the anchor point out" not in wide)

def test_a_tight_frame_drops_wardrobe_it_cannot_contain():
    print("\n=== a close-up stops describing what is outside it ===")
    # Reported: camera types written in the anchor did not take. The anchor text
    # was reaching the model verbatim -- it is 11-13% of a shot -- but the other
    # 87% asserted denim shorts, wrists at the small of the back and the weight on
    # shoulder and hip, none of which a close-up on a face can contain. The camera
    # was not ignored, it was outvoted.
    #
    # So a frame that names its SUBJECT drops the wardrobe that subject's frame
    # cannot hold. Only wardrobe: the limb and restraint prose STAYS, because on a
    # tight shot it is the only thing that still knows where the limbs are
    # fastened -- the node's own warning says so, and the picture the next shot
    # inherits has already lost it.
    mem = "McKenna: she, 26, dark hair, a red jacket, denim shorts, black boots."
    P = "McKenna lies on the bed, wrists cuffed behind her back.\n\nShe turns her head."
    tight = run_node(P, plan_only=True, character_memory=mem,
                     anchor="Close-up on her face. 85mm lens.")[3]
    check("legs are out of a face frame", "denim shorts" not in tight.lower(), tight[:200])
    check("...and so are the feet", "black boots" not in tight.lower(), tight[:200])
    check("...but head-and-shoulders keeps the jacket",
          "red jacket" in tight.lower(), tight[:200])
    check("...and she is still herself", "McKenna" in tight)
    check("...and the restraint is STILL said",
          "cuff" in tight.lower(), tight[:260])
    check("...and where the limbs are held is still said",
          "small of the back" in tight.lower(), tight[:260])

    # A frame naming the HANDS keeps the gloves and drops the rest.
    hands = run_node(P, plan_only=True,
                     character_memory="McKenna: she, 26, gloves, denim shorts.",
                     anchor="Macro lens on her hands.")[3]
    check("a hand frame keeps the gloves", "gloves" in hands.lower(), hands[:200])
    check("...and drops the shorts", "denim shorts" not in hands.lower(), hands[:200])

    # THE NARROW HALVES. A wide film changes nothing, and a close-up that names no
    # subject changes nothing either -- there is no way to know what it is close ON.
    wide = run_node(P, plan_only=True, character_memory=mem, anchor="Wide shot, night.")[3]
    check("a wide film keeps everything", "denim shorts" in wide.lower(), wide[:200])
    bare = run_node(P, plan_only=True, character_memory=mem, anchor="Close-up. 85mm.")[3]
    check("a subjectless close-up changes nothing",
          "denim shorts" in bare.lower(), bare[:200])

def test_the_soundtrack_is_the_models_own():
    """REPORTED: "Just get rid of the ambient sounds all together. They sound horrid.
    Go back to the model's natural audio."

    The node used to build the non-vocal half of the soundtrack itself -- a room tone
    shaped from the scene's own wording, and 21 foley recipes laid into the shots whose
    audio branch is pinned to silence. The reasoning was sound: a pinned shot cannot
    get audio from the model at all, because prompt text never opens a branch, so
    auto_sound was writing sounds into prompts that could not make them.

    It still did not pass. Reported first as footsteps sounding like heartbeats and a
    bathroom that tapped; then, once both of those measured clean, as horrid anyway.
    Synthesis that measures right and sounds wrong is the end of that road.

    So the test is the absence: what comes out is what the model made, and neither
    level widget may move a single sample of it."""
    print("\n=== the soundtrack is the model's own ===")
    P = ("A tiled bathroom. The shower runs and water moves in the pipes.\n\n"
         "Kate walks across the tiles to the shower.\n\n"
         "Kate stands under the water.")

    def _run(**kw):
        # The fake decoder draws unseeded noise, so two runs differ unless the global
        # generator is pinned. Without this the comparison below cannot mean anything
        # -- and it quietly did not, the first time it was written.
        torch.manual_seed(12345)
        return run_node(P, character_memory="Kate: she, 28, a towel.", **kw)[1]["waveform"]

    base = _run(ambient_level=0.0, foley_level=0.0)
    check("the run is reproducible with the seed pinned",
          torch.equal(base, _run(ambient_level=0.0, foley_level=0.0)))
    for _al, _fl in ((0.9, 0.9), (0.25, 0.35), (1.0, 0.0), (0.0, 1.0)):
        check(f"ambient_level={_al} foley_level={_fl} changes not one sample",
              torch.equal(base, _run(ambient_level=_al, foley_level=_fl)))
    # THE BUILDERS ARE GONE, not merely unreachable. A dead synthesiser behind a
    # level of 0 is a thing that comes back.
    for _n in ("foley_for", "synth_ambient", "plain_bed", "_FOLEY", "phrase_seed",
               "bed_recipe", "_BED_RECIPE", "_BED_EVENTFUL", "_BED_RMS", "_MODES",
               "_band", "_hits", "_even", "_contact", "_flow", "_creak",
               "_gait", "_walk", "_step_period", "_MOTION_TIMED", "motion_envelope"):
        check(f"no builder named {_n!r} survives", not hasattr(S, _n))
    check("ShotAudio no longer offers to accept built foley",
          not hasattr(S.ShotAudio(False, True, False, True, 0.0, S.AUDIO_LATENT_FPS),
                      "accepts_built_foley"))
    # ...and no call site is left behind either. The same check test_verbatim makes on
    # the guard text it banned: the source is the evidence.
    _src = open(os.path.join(_HERE, "sampler.py"), encoding="utf-8").read()
    for _gone in ("foley_for(", "synth_ambient(", "plain_bed(", "motion_envelope(",
                  "_foley_on", "accepts_built_foley"):
        check(f"no call site remains: {_gone!r}", _gone not in _src)
    # WIDGET POSITIONS ARE UNTOUCHED. Both levels stay where they are even though one
    # of them now does nothing: widgets are restored BY POSITION out of a saved
    # workflow, so deleting one in the middle loads the wrong number into every widget
    # after it -- here that would be four, ending at hold_levels.
    _opt = list(S.H3LongVideos.INPUT_TYPES()["optional"].keys())
    check("ambient_audio, ambient_level and foley_level keep their positions",
          _opt[34:37] == ["ambient_audio", "ambient_level", "foley_level"])
    check("...and the four after them have not shifted",
          _opt[37:41] == ["speech_lead_seconds", "speech_tail_seconds", "beat_leads",
                          "hold_levels"])
    # A WIRED FILE STILL PLAYS. That is the user's own recording, which is a different
    # thing from ambience the node invents: not synthesised, conditions nothing, and it
    # cannot put a voice in a wordless shot. It is the path the info note points at.
    _bed = {"waveform": torch.full((1, 2, 8000), 0.5), "sample_rate": 44100}
    _mixed, _note = S.mix_ambient(torch.zeros((1, 2, 16000)), 44100, _bed, 0.5)
    check("a wired bed reaches the soundtrack", float(_mixed.abs().max()) > 0.1)
    check("a wired bed at level 0 is a no-op",
          torch.equal(S.mix_ambient(torch.zeros((1, 2, 16000)), 44100, _bed, 0.0)[0],
                      torch.zeros((1, 2, 16000))))
    check("...as is no bed at all",
          torch.equal(S.mix_ambient(torch.zeros((1, 2, 16000)), 44100, None, 0.9)[0],
                      torch.zeros((1, 2, 16000))))
    # THE CONSEQUENCE IS REPORTED rather than left to be discovered: a wordless shot is
    # pinned to silence and is now silent, and the pin is deliberately untouched -- it
    # is what stops a free branch filling itself with a voice.
    _info = run_node(P, character_memory="Kate: she, 28.", foley_level=0.35)[2]
    check("info says the level does nothing now", "does nothing any more" in _info)
    check("...and says what that costs", "is SILENT" in _info)
    check("...and how to get sound back",
          "write the sound into that beat" in _info and "ambient_audio" in _info)


def test_shot_one_is_the_only_unpinned_shot():
    """REPORTED: "The girl doesn't look the same from the first to last beat... The
    remaining beats are fine. I even have image reference strength set to 0.999", with
    a hardware artefact in beat 1 alone -- her hair caught in a collar.

    Both halves are one fact, and it is structural rather than a bug: shot 1 is the
    only shot whose opening frame is pinned by nothing. Every later shot opens on the
    previous shot's last frame, which fixes pose, framing and the arrangement of
    everything on the body, so the chain agrees with itself and shot 1 is the one that
    can disagree. An arrangement no picture settles -- how hair sits against a collar
    -- is settled by the model, once, in the only shot with no picture.

    And ref_noise_aug cannot reach it, which is the part the report turned on.
    build_conditioning's own comment is plain: "the keyframe ANCHORS the first frame,
    which is what continuity needs, while a reference only supplies identity. They are
    not alternatives." A cleaner reference sharpens identity; there is no frame on
    shot 1 for it to sharpen."""
    print("\n=== shot 1 is the only unpinned shot ===")
    mem = "Nora: <Picture 1>, she, 24, long dark hair, a steel collar, a grey dress."
    P = ("A bare room, cold light.\n\nNora stands by the window.\n\n"
         "Nora turns to look at the door.\n\nNora sits on the crate.")
    ref = torch.rand((1, 64, 64, 3))

    def _pins(**kw):
        """What pins each shot's opening frame, as seen by build_conditioning."""
        got, orig = [], S.build_conditioning
        def spy(clip, vae, audio_vae, prompt, width, height, length,
                handoff=None, refs=None, ref_noise_aug=0.999, **k):
            got.append(("keyframe" if (handoff is not None and not k.get("handoff_as_ref"))
                        else "reference" if handoff is not None else "nothing",
                        len(refs or [])))
            return orig(clip, vae, audio_vae, prompt, width, height, length,
                        handoff=handoff, refs=refs, ref_noise_aug=ref_noise_aug, **k)
        S.build_conditioning = spy
        _p = kw.pop("prompt_override", P)
        try:
            run_node(_p, character_memory=mem, **kw)
        finally:
            S.build_conditioning = orig
        return got

    bare = _pins(ref_image_1=ref, ref_noise_aug=0.999)
    check(f"three shots measured: {[p for p, _ in bare]}", len(bare) == 3)
    check("shot 1's opening frame is pinned by NOTHING", bare[0][0] == "nothing")
    check("...while every later shot has a keyframe",
          all(p == "keyframe" for p, _ in bare[1:]))
    # The reference IS there on shot 1 -- it is not a missing input, it is the wrong
    # kind of input for this job.
    check("...and shot 1 does carry the reference anyway",
          all(n == 1 for _p, n in bare))
    # RAISING THE AUG CHANGES NOTHING ABOUT THE PINNING. This is the dial the report
    # reached for, at the top of its range, and it cannot move this.
    for _aug in (0.90, 0.95, 0.999, 1.0):
        _p = _pins(ref_image_1=ref, ref_noise_aug=_aug)
        check(f"at ref_noise_aug {_aug} shot 1 is still unpinned", _p[0][0] == "nothing")
    # WIRING first_frame IS WHAT REACHES SHOT 1 -- but what it then does depends on
    # whether the script reads it as an opening frame or as the SET. This beat places
    # her and she has a portrait, so it is the set: see
    # test_a_first_frame_can_be_the_set_instead_of_frame_one. An entrance in beat 1
    # leaves it as the hard keyframe it always was.
    _ff = _pins(ref_image_1=ref, first_frame=ref, ref_noise_aug=0.999)
    check("first_frame reaches shot 1 either way", _ff[0][0] != "nothing")
    _ent = _pins(ref_image_1=ref, first_frame=ref, ref_noise_aug=0.999,
                 prompt_override=("A bare room, cold light.\n\n"
                                  "Nora walks in through the side door.\n\n"
                                  "Nora turns to look at the door.\n\n"
                                  "Nora sits on the crate."))
    check("...and pins frame one when beat 1 stages an entrance",
          _ent[0][0] == "keyframe")
    # ...and info says all of that, including which dial is NOT this one.
    _info = run_node(P, plan_only=True, character_memory=mem, ref_image_1=ref,
                     ref_noise_aug=0.999)[2]
    _n = [x for x in _info.split(" | ") if "NO first_frame" in x]
    check("info reports the asymmetry", len(_n) == 1)
    _n = _n[0] if _n else ""
    check("...saying shot 1 alone is pinned by nothing",
          "only shot in this film whose opening frame is pinned by NOTHING" in _n)
    check("...that every other shot opens on the previous last frame",
          "previous shot's last frame" in _n)
    check("...that ref_noise_aug cannot fix it",
          "ref_noise_aug IS NOT THE DIAL FOR THIS" in _n)
    check("...naming the one-shot-only symptom that was reported",
          "hair sitting differently against a collar" in _n)
    check("...and first_frame as the fix", "Wire first_frame to fix it" in _n)
    # It must go quiet once the fix is applied, or it is noise.
    check("...and says nothing once first_frame is wired",
          "NO first_frame" not in run_node(P, plan_only=True, character_memory=mem,
                                          ref_image_1=ref, first_frame=ref)[2])
    # NOT SAID TWICE. With a reference wired, the ref_noise_aug note already explains
    # at length what to put in first_frame and why a portrait there is a trap; this one
    # defers to it rather than repeating it. Two notes making the same point is the
    # same vice as two clauses making it -- and info is already long.
    check("with a reference wired it defers rather than repeating",
          "see the ref_noise_aug note above" in _n
          and "subject, pose, framing, background" not in _n)
    _no_ref = [x for x in run_node(P, plan_only=True, character_memory=mem)[2].split(" | ")
               if "NO first_frame" in x][0]
    check("...and carries the advice itself when there is no reference note",
          "subject, pose, framing, background" in _no_ref)


def test_a_first_frame_can_be_the_set_instead_of_frame_one():
    """REPORTED, after being told to wire first_frame: "No, this should be fixed using
    reference images from the start. I only use first_frame for the scene."

    Which is a workflow the node could not serve. first_frame was always taken as
    shot 1's opening frame, pinned whole -- and a picture of a SET has nobody in it,
    so the cast the beat places there has to appear out of nothing during shot 1. That
    is the reported artefact: beat 1 wrong, every later beat fine, because beat 2
    inherits the settled version from its handoff.

    The node already refuses exactly this on every OTHER shot, with its own reasoning
    written down -- "that frame does not have them in it, and a keyframe is a picture,
    so they would have to appear out of nothing and travel to the spot the beat
    describes". Shot 1 could not reach it: the guard read `and plan`, which is empty on
    the first shot. True before first_frame could supply a keyframe; wrong after.

    The set is told from an opening frame BY THE SCRIPT, never by looking at pixels:
    the beat has to place the cast rather than stage an entrance, and every one of them
    has to already have a portrait of their own. Then this picture has nothing left to
    say about who they are, only about where they are."""
    print("\n=== a first_frame can be the set instead of frame one ===")
    plate = torch.rand((1, 256, 256, 3))
    ref = torch.rand((1, 256, 256, 3))
    TAG = "Nora: <Picture 1>, she, 24, long dark hair, a steel collar."
    BARE = "Nora: she, 24, long dark hair, a steel collar."
    PLACED = "A bare room.\n\nNora stands by the window.\n\nNora turns to the door."
    ARRIVES = ("A bare room.\n\nNora walks in through the side door.\n\n"
               "Nora turns to the door.")

    def _first(mem, P, **kw):
        """(how shot 1's first_frame is used, refs passed, shot 1's prompt)."""
        got, orig = [], S.build_conditioning
        def spy(clip, vae, audio_vae, prompt, width, height, length,
                handoff=None, refs=None, ref_noise_aug=0.999, **k):
            got.append(("set" if (handoff is not None and k.get("handoff_as_ref"))
                        else "keyframe" if handoff is not None else "nothing",
                        len(refs or []), prompt))
            return orig(clip, vae, audio_vae, prompt, width, height, length,
                        handoff=handoff, refs=refs, ref_noise_aug=ref_noise_aug, **k)
        S.build_conditioning = spy
        try:
            run_node(P, character_memory=mem, **kw)
        finally:
            S.build_conditioning = orig
        return got[0]

    # THE REPORTED CASE. A plate, a beat that places her, a portrait of her.
    _how, _n, _pr = _first(TAG, PLACED, first_frame=plate, ref_image_1=ref)
    check("a plate with the cast placed and portrayed rides as the SET", _how == "set")
    # BOTH pictures are there and both are claimed. Her portrait keeps slot 1 -- which
    # is what her sheet tag points at -- and the plate is appended after it, so the
    # numbering the author wrote still lands on the author's image.
    check("...her portrait keeps <Picture 1>", "<Picture 1>, she, 24" in _pr)
    check("...the plate is claimed as the set, after her",
          "<Picture 2> is the set this shot takes place in" in _pr)
    check("...and claimed as having nobody in it",
          "a picture of the place only, with nobody in it" in _pr)
    # An unclaimed picture is another subject, which is this file's oldest bug. A plate
    # claimed with room_claim's wording would be worse than unclaimed: it says "this
    # room a moment earlier" and names who was standing in it, and on shot 1 there is
    # no earlier and nobody was.
    check("...and not as a room a moment earlier", "a moment earlier" not in _pr)

    # THE THREE CASES THAT MUST NOT CHANGE.
    # An entrance WANTS a frame she is absent from -- walking in is the one case where
    # starting from a picture without her is right.
    check("an entrance in beat 1 still pins frame one",
          _first(TAG, ARRIVES, first_frame=plate, ref_image_1=ref)[0] == "keyframe")
    check("no portrait means the frame is her only picture, so it stays frame one",
          _first(BARE, PLACED, first_frame=plate)[0] == "keyframe")
    check("...and a tag pointing at an empty socket portrays nobody",
          _first(TAG, PLACED, first_frame=plate)[0] == "keyframe")
    check("no first_frame at all is unchanged",
          _first(TAG, PLACED, ref_image_1=ref)[0] == "nothing")

    # TWO PEOPLE: every one of them has to be covered, not just the first.
    _two = ("Nora: <Picture 1>, she, 24, long dark hair.\n"
            "Dan: <Picture 2>, he, 41, a grey coat.")
    _one = "Nora: <Picture 1>, she, 24, long dark hair.\nDan: he, 41, a grey coat."
    _P2 = ("A bare room.\n\nNora stands by the window and Dan sits on the crate.\n\n"
           "Nora turns to the door.")
    check("both portrayed -> the plate is the set",
          _first(_two, _P2, first_frame=plate, ref_image_1=ref,
                 ref_image_2=ref)[0] == "set")
    check("one of them unportrayed -> it stays frame one",
          _first(_one, _P2, first_frame=plate, ref_image_1=ref)[0] == "keyframe")

    # ...and it is REPORTED, with what to do if the reading is wrong.
    _info = run_node(PLACED, plan_only=True, character_memory=TAG,
                     first_frame=plate, ref_image_1=ref)[2]
    check("info says the frame was read as the set",
          "first_frame is being read as the SET" in _info)
    check("...and that it is not discarded", "NOT discarded" in _info)
    check("...and how to pin frame one instead",
          "put the cast IN that frame" in _info and "write the entrance" in _info)
    check("...and says nothing when the frame is taken as frame one",
          "read as the SET" not in run_node(ARRIVES, plan_only=True, character_memory=TAG,
                                           first_frame=plate, ref_image_1=ref)[2])


def test_the_age_reaches_the_shot_and_the_floor_holds():
    """End to end for "Breast development should also be correct, given the age of a
    person" -- and for the floor that request makes necessary.

    An age driving anatomy has to answer for the ages it was not meant for, so the same
    reader that names a woman of 45 names nothing at all below 18."""
    print("\n=== the age reaches the shot, and the floor holds ===")
    P = "A bare room.\n\nKate takes off her vest.\nremove: vest\n\nKate stands still."

    def _chest(mem):
        sh = [" ".join(x.split()) for x in
              run_node(P, plan_only=True, character_memory=mem)[3].split("---")
              if x.strip()]
        m = re.search(r"(The chest[^.]*\.)", sh[1])
        return m.group(1) if m else ""

    _22 = _chest("Kate: she, 22, a grey vest, denim shorts.")
    _58 = _chest("Kate: she, 58, a grey vest, denim shorts.")
    _none = _chest("Kate: she, a grey vest, denim shorts.")
    check("the age reaches the shot", "the body of a woman of 22" in _22, _22[-90:])
    check("...and so does the chest at that age", "sitting high on the chest" in _22)
    check("a different age says something different",
          "the body of a woman of 58" in _58 and "hanging low" in _58, _58[-90:])
    check("...so two ages are not one description", _22 != _58)
    # A sheet with no age keeps exactly what it had: there is nothing to be consistent
    # with, and the old silence beats a guess.
    check("no age keeps the plain phrasing",
          "on a woman's body" in _none and "the breasts" not in _none, _none[-90:])
    # THE FLOOR, end to end. The scene renders -- children are in films -- and not one
    # clause names a body for the entry that declares one.
    mem = "Kate: she, 38, a grey vest, denim shorts.\nSam: he, 9, a school jumper."
    info, script = run_node(
        "A kitchen.\n\nSam eats breakfast.\n\nKate reads the paper.",
        plan_only=True, character_memory=mem)[2:4]
    check("a scene with a child in it still renders", "[Shot 1]" in script)
    check("...and no body is named for them anywhere",
          "the body of a man of 9" not in script and "a man's body" not in script)
    check("...and info says so rather than leaving it to be noticed",
          "declared under 18 on the sheet" in info and "NO body is described" in info)
    check("...and says the scene itself renders", "The scene renders" in info)
    check("...and that it would not have with sex staged in it",
          "nothing would have rendered at all" in info)
    check("a film with no minor says none of that",
          "declared under 18" not in run_node(
              P, plan_only=True,
              character_memory="Kate: she, 38, a grey vest, denim shorts.")[2])
    # ...AND THE REFUSAL IS A REFUSAL: it raises before anything is sampled, so there is
    # no half-rendered output and no quietly-softened one either.
    _raised = ""
    try:
        run_node("A bedroom.\n\nKate undresses and lies down.\n\nShe moans.",
                 character_memory="Kate: she, 28.\nSam: she, 15.")
    except RuntimeError as _e:
        _raised = str(_e)
    check("a declared minor plus sexual staging renders nothing", _raised != "")
    check("...saying so plainly", "REFUSED" in _raised and "nothing was rendered" in _raised)
    check("...and naming the entry that tripped it", "Sam" in _raised)
    # Both halves of that really are required, end to end.
    _ok = run_node("A bedroom.\n\nKate undresses and lies down.\n\nShe moans.",
                   plan_only=True, character_memory="Kate: she, 28.")
    check("the same script with no minor declared renders", "[Shot 1]" in _ok[3])
    _ok2 = run_node("A kitchen.\n\nSam eats breakfast.\n\nKate reads.",
                    plan_only=True, character_memory="Kate: she, 38.\nSam: she, 15.")
    check("a declared minor in an ordinary scene renders", "[Shot 1]" in _ok2[3])


def test_sound_survives_silencing():
    print("\n=== a described sound is not silenced away ===")
    # No space named, so no room tone -- this test is about the SILENCE path, and a
    # bed under every shot would mean nothing is silenced and there is nothing to see.
    P = ("Two people, late evening.\n\n"
         "Maya: 27, grey coat.\n\n"
         "Maya lies still.\n\n"
         "The chain drags and rattles beside her.\n\n"
         'Jon says: "Get up."')
    vae = FakeAudioVAE()
    info = run_node(P, audio_vae=vae, auto_sound=False)[2]
    check("the beat that describes a sound keeps its audio",
          "describe a sound IN THE BEAT" in info)
    # NAMED, not counted. "2 shot(s) have an open branch" tells a reader that two
    # of eleven can babble and gives no way to find out which two -- and the note's
    # own advice is that the beat's own sound wording opened it, which cannot be
    # acted on without knowing which beat.
    check("...and is named", "shot(s) 2 have no line but either describe a sound" in info)
    check("the beat with none is silenced",
          "shot(s) 1 have no quoted line and no sound" in info)
    check("...and the guidance says what silence actually is",
          "not 'no speech', it is 'no sound at all'" in info)
    check("...and how to score a scene", "DESCRIBE it in the prose" in info)
    check("...and warns off a label", "read as text to draw" in info)
    # With silencing off, nothing is silenced and nothing is claimed about it.
    off = run_node(P, silence_nonspeech=False)[2]
    check("silencing off silences nothing", "conditioned on real silence" not in off)


def test_auto_sound_end_to_end():
    print("\n=== sound generated from the prompt ===")
    # No space named: this test isolates the sound a beat's own ACTION implies, and
    # room tone would put a bed on every shot including the ones meant to be bare.
    #
    # Derived sound is TEXT ONLY and can never unsilence a shot, so it lands only on
    # shots whose audio branch is already open: ones with a line, or with a sound the
    # author wrote. A shot with neither stays pinned to silence -- the mouth follows
    # the audio, and an inference is not reason enough to let it move.
    P = ("Two people, late evening.\n\n"
         "Maya: 27, grey coat. Wrists cuffed behind back.\n\n"
         "Jon walks in holding a pair of scissors and says: \"Hold still.\"\n\n"
         "Jon walks to the bench and looks at the box.\n\n"
         "Maya lies still.\n\n"
         "The chain drags and rattles beside her.")
    imgs, audio, info, script = run_node(P, plan_only=True)[:4]
    sh = [" ".join(x.split()) for x in re.split(r"(?=\[Shot )", script) if x.strip()]
    # Shot 1 speaks, so its branch is open anyway and the action's sound is added.
    check("walking is heard on the shot that speaks", "footsteps" in sh[0])
    check("...and the scissors", "blades through fabric" in sh[0])
    check("...in the open form, because it has a line", "It sounds like" in sh[0])
    # Shots 2 and 3 have no line and no sound of their own. With auto_sound OFF
    # they are pinned silent and told nothing about sound, since the clause would
    # describe an acoustic that is not there. This is the one that was babbling,
    # and it is still the behaviour with the switch off -- an ambient bed read from
    # the anchor is what opens them now, chosen deliberately.
    q_info, q_script = run_node(P, plan_only=True, auto_sound=False)[2:4]
    q_sh = [" ".join(x.split()) for x in re.split(r"(?=\[Shot )", q_script)
            if x.strip()]
    check("a shot with no line gets no derived sound (auto_sound off)",
          "footsteps" not in q_sh[1])
    check("...and no sound sentence at all",
          "sounds like" not in q_sh[1] and "only sound" not in q_sh[1])
    check("a beat staging nothing audible gets nothing", "sounds like" not in sh[2])
    # What you wrote wins: a beat describing its own sound is left alone AND stays open.
    check("a beat with its own sound is not overwritten", "It sounds like" not in sh[3])
    check("...and it still counts as asking for audio",
          "either describe a sound IN THE BEAT or" in info)
    check("info lists the shots it scored", "were given the sound" in info)
    check("...saying it can never unsilence one", "never unsilence a shot" in info)
    # Sound is counted apart from the continuity guards, which ask for the opposite
    # thing -- for something to stay as it is rather than to happen.
    check("the balance separates sound from guards", "sound " in info.split("balance")[1][:120])
    # Off, nothing is added and those shots go back to being silenced.
    off = run_node(P, plan_only=True, auto_sound=False)[3]
    check("auto_sound off adds nothing", "It sounds like" not in off)


def test_room_tone_under_every_shot():
    print("\n=== the room is heard even when nothing happens ===")
    # Real footage has a bed under the events. Digital silence between them is what
    # makes a scene sound staged rather than recorded.
    P = ("A cold concrete basement with bare walls.\n\n"
         "Maya: 27, grey coat.\n\n"
         "Jon walks in and says: \"Get up.\"\n\n"
         "Maya lies still.\n\n"
         "Maya breathes hard, the sound of it loud in the room.")
    imgs, audio, info, script = run_node(P, plan_only=True)[:4]
    sh = [" ".join(x.split()) for x in re.split(r"(?=\[Shot )", script) if x.strip()]
    check("a shot that speaks carries the room too",
          "hard walls giving the sound back" in sh[0])
    check("the acting shot still gets its events", "footsteps" in sh[0])
    check("info names the acoustic", "room tone read from the scene" in info)
    # The ambient bed goes under shots whose audio branch is ALREADY open -- ones
    # with a line, or with a sound the author wrote. It is NOT put under a wordless
    # shot, and that reversal is the point.
    #
    # "Ambience everywhere" was tried and does not work. The bed was allowed to open
    # a branch, which is the one thing nothing inferred here may do: an open branch
    # on a joint model fills itself, and at 4-8 steps the last audio step clears
    # 50%-30% of the denoising in a single jump, so what it fills with is a voice.
    # Ambience everywhere and silence are mutually exclusive by construction -- the
    # silence latent IS the audio, and there is no room in it for a room tone.
    check("the speaking shot carries the bed", "low hum off the strip light" in sh[0])
    check("the wordless shot is left silent",
          "low hum off the strip light" not in sh[1])
    check("...and info says so", "ambient bed read from the anchor" in info)
    # THE CONDITIONING, not just the text. `sounded` decides whether the silence
    # latent is applied at render time, and the bed used to be counted into it --
    # so a wordless shot was reported as one that "describes a sound IN THE BEAT",
    # which it does not, and its branch was left open. plan_only shows the text
    # either way, so this reads the node's own account of which shots are pinned.
    _pin = re.search(r"shot\(s\) ([\d, ]+) have no quoted line and no sound described",
                     info)
    _open = re.search(r"shot\(s\) ([\d, ]+) have no line but either", info)
    check("the wordless shot is pinned to silence",
          _pin is not None and "2" in _pin.group(1), _pin.group(1) if _pin else "none")
    check("...and is not counted as describing its own sound",
          _open is None or "2" not in _open.group(1),
          _open.group(1) if _open else "none")

    # THE SILENCE PATH, which is still the path with auto_sound OFF.
    #
    # Reported twice: auto_sound was moving the mouth and babbling. H3 is joint, so
    # a free audio branch fills itself with a VOICE and the face lip-syncs to it. No
    # wording suppresses that -- only the silent keyframe does, and it pins the
    # whole shot. Scoring the film from the anchor re-opens that branch on purpose,
    # chosen on 2026-09-06 with the trade stated; auto_sound is the way back.
    q_info, q_script = run_node(P, plan_only=True, auto_sound=False)[2:4]
    q_sh = [" ".join(x.split()) for x in re.split(r"(?=\[Shot )", q_script)
            if x.strip()]
    check("auto_sound off: a wordless shot is silenced",
          "conditioned on real silence" in q_info)
    check("...and the room does not go under it",
          "hard walls" not in q_sh[1], q_sh[1][-70:])
    check("...and it is told nothing about sound",
          "sounds like" not in q_sh[1] and "only sound" not in q_sh[1])
    # A sound the AUTHOR wrote used to keep that shot open even with no line -- and
    # an open branch is exactly where the invented voice and the lip-sync came from.
    # mouths_shut_when_no_line, on by default, silences it instead. The old rule is
    # still the rule with the switch off, and that is the trade, stated both ways.
    # auto_sound stays ON here: the room tone this asserts comes from it, and the
    # point is the MOUTH guard, not the bed.
    off = run_node(P, plan_only=True, mouths_shut_when_no_line=False)[3]
    off_sh = [b for b in off.split("---") if b.strip()]
    check("a sound you wrote keeps the branch open, guard off",
          "hard walls giving the sound back" in off_sh[2])
    check("...in the closed form, because it has no line",
          "The only sound" in off_sh[2])
    check("...while on, that shot is silenced so the mouth cannot move",
          "The only sound" not in q_sh[2]
          and "Mouths in the shot stay closed" in q_sh[2])
    check("...and info explains the mouth",
          "so the mouths could be held shut" in q_info)
    # A scene naming no space gets no bed, and the silence guard still applies.
    plain = run_node("Two people talking.\n\nHe waits.\n\nShe waits.",
                     plan_only=True)[2]
    check("no space named, no room tone", "room tone read" not in plain)
    check("...and no ambient bed either", "ambient bed read" not in plain)


def test_the_decode_keeps_the_vae_it_is_about_to_use():
    print("\n=== eviction does not throw away the VAE it needs ===")
    # Reported: forced eviction costing 37% of wall-clock on a 3090. The half that is
    # wrong on EVERY card is this one -- free_memory ran with keep_loaded=[], which
    # unloads every resident model including the video VAE, and ComfyUI then reloads
    # it three lines later to run the decode. Peak VRAM is identical either way (the
    # VAE has to be resident to decode), so the round trip was pure cost.
    class _LM:                      # stands in for ComfyUI's LoadedModel
        def __init__(self, m):
            self.model = m

    calls = []
    _orig_free = _mm.free_memory
    model, vae, avae = FakeModel(), FakeVAE(), FakeAudioVAE()
    _mm.current_loaded_models = [_LM(model), _LM(vae), _LM(avae)]

    def spy(memory_required, device, keep_loaded=(), **kw):
        calls.append((memory_required, [lm.model for lm in keep_loaded]))

    _mm.free_memory = spy
    try:
        run_node("A room.\n\nOne.\n\nTwo.", model=model, vae=vae, audio_vae=avae)
    finally:
        _mm.free_memory = _orig_free
        del _mm.current_loaded_models

    check("eviction still runs", bool(calls))
    # Two call sites per shot: _evict_all_but before sampling, free_first before decode.
    pre_sample = [c for c in calls if model in c[1]]
    pre_decode = [c for c in calls if vae in c[1]]
    check("before sampling, the DiT is what is kept", bool(pre_sample))
    check("...and the VAEs are not", all(vae not in c[1] for c in pre_sample))
    check("before decode, the video VAE is kept", bool(pre_decode))
    check("...and the audio VAE with it, used on the next line",
          all(avae in c[1] for c in pre_decode))
    check("...while the DiT goes, which is what makes the decode fit",
          all(model not in c[1] for c in pre_decode))
    # Not changed, and deliberately: sizing the request needs real hardware.
    check("the request is still unsized", all(c[0] >= 1e29 for c in calls))
    # A model ComfyUI does not hold cannot be kept, and must not raise.
    _mm.current_loaded_models = []
    check("nothing resident, nothing kept", S._resident([model, vae]) == [])
    del _mm.current_loaded_models
    check("no current_loaded_models at all is survivable", S._resident([model]) == [])


def test_a_vocal_shot_says_what_fills_the_gaps():
    print("\n=== between the moans ===")
    # Reported as babble between the moans. A named vocal opens the audio branch on
    # purpose -- it is meant to be heard -- and the clause then named nothing but the
    # vocal. A moan is INTERMITTENT and the branch is open for the whole shot, so the
    # list described the peaks and left the troughs blank; a blank trough on a joint
    # model, next to a face, fills itself with speech.
    #
    # There is no way to ask for the absence of speech at cfg 1, so the fix is to say
    # what IS in the troughs. Room tone does not answer it -- the ambient bed already
    # appended two continuous phrases and the babble was reported anyway, because the
    # gap is a PERSON's audio presence and a soft room is not a person. Breath is.
    def _clause(_beat, _scene="Inside a van at night."):
        _sh = run_node(f"{_scene}\n\n{_beat}", plan_only=True)[3]
        _m = re.search(r"(The only sound[^.]*\.|It sounds like[^.]*\.)", _sh)
        return _m.group(1) if _m else ""

    for _beat, _word in (("She moans.", "moaning"),
                         ("He thrusts into her, and she moans.", "moaning"),
                         ("She whimpers and thrashes in her restraints.", "whimpering"),
                         ("She sobs quietly.", "sobbing"),
                         ("She screams.", "screaming"),
                         ("She groans.", "groaning"),
                         ("She whines.", "whining")):
        _c = _clause(_beat)
        check(f"the vocal is still named: {_word}", _word in _c, f"{_beat!r} -> {_c!r}")
        check(f"...and the gaps between them are too: {_word}",
              S._VOCAL_BETWEEN in _c, f"{_beat!r} -> {_c!r}")
        check(f"...with the author's word first: {_word}",
              _c.index(_word) < _c.index(S._VOCAL_BETWEEN), _c)
    # The list stays CLOSED on a shot with no line -- that exclusivity is the only
    # lever against speech this model has, and it is what the breath now backs up.
    check("the clause is still exclusive with no line spoken",
          _clause("She moans.").startswith("The only sound"), _clause("She moans."))
    # A beat naming NO vocal must gain nothing. The node does not start inventing
    # breath on shots that never opened a branch for a voice.
    check("a beat with no vocal gains no breath", _clause("He walks in.") == "")
    check("...nor does a written NON-vocal sound, which takes the muted path",
          _clause("The chain rattles against the frame.") == "")
    # A shot that also carries a LINE is not closed, and the breath rides along
    # harmlessly -- the line is what occupies the branch there.
    _both = _clause('McKenna moans, and Dan says: "Nearly there."')
    check("a vocal beside a line keeps an open list",
          _both.startswith("It sounds like"), _both)
    check("...and still names both the vocal and the breath",
          "moaning" in _both and S._VOCAL_BETWEEN in _both, _both)
    # The breath is the node's own existing vocabulary, not a new word invented here.
    check("the between-vocal sound is a phrase the node already used elsewhere",
          S._VOCAL_BETWEEN in S._VOCAL_RETIRES, S._VOCAL_BETWEEN)
    # And a pinned-silent shot must never gain a clause -- the conditioning says
    # there is no sound at all, so describing one would contradict it.
    _sil = run_node("A quiet room.\n\nShe stands by the window.", plan_only=True)[3]
    check("a silenced shot gains no breath", S._VOCAL_BETWEEN not in _sil, _sil[-200:])


def test_an_exclusive_sound_clause_never_denies_the_beat():
    print("\n=== the closed list names the sound the author wrote ===")
    # REPRODUCED, then fixed. sound_described("She screams.") is true, so _own zeroes
    # the inferred list; exertion_in is ALSO true, so _voiced keeps the branch open
    # instead of letting the shot be muted; then the ambient bed is appended and
    # only=not _speaks CLOSES the list. The shot was conditioned on
    #
    #     "The only sound is an engine idling."
    #
    # against a beat that says she screams -- an exclusive claim that the scream is
    # not happening, made on exactly the shots whose branch is open and which
    # therefore have to fill themselves with something.
    def _clause(_beat):
        _sh = run_node(f"Inside a van at night.\n\n{_beat}", plan_only=True)[3]
        _m = re.search(r"(The only sound[^.]*\.|It sounds like[^.]*\.)", _sh)
        return _m.group(1) if _m else ""

    for _beat, _word in (("She screams.", "screaming"),
                         ("She sobs quietly.", "sobbing"),
                         ("She starts whimpering and thrashes in her restraints.",
                          "whimpering"),
                         ("She moans and pulls against the cuffs.", "moaning")):
        _c = _clause(_beat)
        check(f"the clause names it: {_word}", _word in _c, f"{_beat!r} -> {_c!r}")
        check(f"...and no longer claims the bed is the only sound: {_word}",
              not re.match(r"The only sound is an? [a-z ]+\.$", _c), _c)
    # A NON-VOCAL written sound is a different path and must stay on it: it is muted
    # outright by _mute_written, reaches no clause, and the info reports the trade.
    _chain = "The chain rattles against the frame."
    check("a written non-vocal sound still takes the muted path", _clause(_chain) == "",
          _clause(_chain))
    _info = run_node(f"Inside a van at night.\n\n{_chain}", plan_only=True)[2]
    check("...and the trade is still reported",
          "gave up the sound you wrote" in _info)
    # And a beat that names NO sound is untouched -- the node must not start
    # inventing vocals here.
    check("a silent beat gains no vocal", _clause("He walks in.") == "")
    # named_vocals_in is the author's own word, matched literally -- not an inference.
    check("named_vocals_in reads the beat", S.named_vocals_in("She screams.") == ["screaming"])
    check("...and finds nothing where nothing is named",
          S.named_vocals_in("He walks in.") == [])
    check("...and it is the same table sounds_for uses",
          all(v in S._NAMED_VOCALS for _p, v in S._VOCAL_FROM)
          and len(S._VOCAL_FROM) == len(S._NAMED_VOCALS))


def test_the_chain_is_never_held_twice():
    print("\n=== the chain is one allocation from first shot to return ===")
    # The planned capacity accounts for the frame removed at each seam.
    # The returned tensor should not retain storage for those discarded frames.
    for _n in (2, 4, 8):
        _P = "A room.\n\n" + "\n\n".join(f"Beat {_i}." for _i in range(_n))
        _out = run_node(_P)
        _v = _out[0]
        _slack = _v.untyped_storage().nbytes() - _v.numel() * _v.element_size()
        check(f"{_n} beats: no storage retained for trimmed seam frames",
              _slack == 0, f"unused storage: {_slack} bytes")
    # ...and it is still the right pixels, in range, in the output dtype.
    _out = run_node("A room.\n\nOne.\n\nTwo.\n\nThree.")
    _v = _out[0]
    check("...and still float32 by default", _v.dtype == torch.float32, str(_v.dtype))
    check("...and still in range", float(_v.min()) >= 0.0 and float(_v.max()) <= 1.0)
    check("...and the frame count is unchanged", _v.shape[0] == _out[5])

    # THE OVERFLOW PATH. Not reachable on the real VAE, which decodes a shot to
    # exactly the length it was planned at -- but "not reachable" is a claim about
    # somebody else's code, so a VAE that over-decodes must still return a correct
    # chain, just by the slower route.
    _real_decode = FakeVAE.decode
    try:
        FakeVAE.decode = lambda self, latent: torch.rand(
            max(1, (latent.shape[2] - 2) // 5 * 17 + 5) + 3, H, W, 3).to(_vae_out_dtype())
        _o = run_node("A room.\n\nOne.\n\nTwo.\n\nThree.")
        _vo = _o[0]
        check("a VAE that over-decodes still returns a chain", _vo.shape[0] > 0)
        check("...of the length it actually decoded", _vo.shape[0] == _o[5])
        check("...in range", float(_vo.min()) >= 0.0 and float(_vo.max()) <= 1.0)
        check("...and in the output dtype", _vo.dtype == torch.float32)
    finally:
        FakeVAE.decode = _real_decode


def test_the_position_may_only_be_written_once_in_the_scene():
    print("\n=== the wrists are placed even when only the scene says where ===")
    # restrained is set by the beat OR the scene; the anchor was read from the beat
    # alone. So the ordinary way of writing this -- said once in the scene or on the
    # sheet, never repeated in a beat -- marked her restrained with no position, and
    # pose_clause looks its argument up in a dict, so empty is NO sentence rather
    # than a shorter one. Never told the wrists are together, never told the arms are
    # behind her, and never told what carries her weight when she lies down. Reported
    # twice as her arm propping her up.
    _p = ("Inside a van at night. McKenna lies in the back, wrists cuffed behind her back.\n"
          "McKenna: she, 26, dark hair.\n\n"
          "Dan gets into the van and looks at her.\n\n"
          "She lays down on her side.")
    _shots = run_node(_p)[3].split("\n---\n")
    check("the position reaches every shot from the scene alone",
          all("wrists together at the small of the back" in _s for _s in _shots))
    check("...and the lying shot is told what carries her weight",
          "take the weight of the body" in _shots[-1])
    # A BEAT THAT MOVES THEM STILL WINS. The scene is only the fallback.
    _moved = run_node("Inside a van. McKenna sits, wrists cuffed behind her back.\n\n"
                      "Dan cuffs her wrists above her head.")[3]
    check("a beat that moves the wrists overrides the scene",
          "above the head" in _moved and "small of the back" not in _moved.split("Both arms")[-1])
    # THE POSTURE HAD THE SAME ASYMMETRY. A scene reading "McKenna lies in the back"
    # put nobody in a posture, so a restrained body the script never lays down ON
    # SCREEN was never known to be off its feet and the weight clause could not fire.
    def _weight(_p):
        return ["take the weight" in _s for _s in run_node(_p)[3].split("\n---\n")]
    check("the scene alone can say she is lying",
          _weight("Inside a van at night. McKenna lies in the back, wrists cuffed "
                  "behind her back.\n\nDan gets into the van and looks at her.\n\n"
                  "He watches her.") == [True, True])
    check("...with a sheet too",
          _weight("Inside a van. McKenna lies in the back, wrists cuffed behind her "
                  "back.\nMcKenna: she, 26, dark hair.\n\nDan gets into the van."
                  "\n\nHe watches her.") == [True, True])
    # AND A BEAT OUTRANKS IT. The scene paragraph is stamped into every shot, so
    # believing it over a beat that stands her up would hold her down for the rest of
    # the film. Note the latch cannot carry this when the script names no cast --
    # posture_in needs people -- so the guard is "no beat has stated a posture yet".
    check("a beat that stands her up ends it, for good",
          _weight("Inside a van. McKenna lies in the back, wrists cuffed behind her "
                  "back.\n\nShe gets to her feet.\n\nDan looks at her.") == [False, False])
    check("nobody in hardware, nothing said",
          _weight("A bedroom. Nora lies on the bed.\n\nShe looks at the ceiling.") == [False])
    check("restrained but nobody said lying, nothing said",
          _weight("A barn. McKenna stands, wrists cuffed behind her back.\n\n"
                  "Dan looks at her.") == [False])
    check("and the beat that lays her down still fires it",
          _weight("Inside a van. McKenna sits in the back, wrists cuffed behind her "
                  "back.\nMcKenna: she, 26, dark hair.\n\nShe lays down on her side.")
          == [True])


def test_finished_shots_are_held_in_half_precision():
    print("\n=== the chain does not crowd the weights out of RAM ===")
    # ComfyUI offloads models to system RAM rather than discarding them, so a shot
    # boundary is a PCIe copy while that RAM is there and a disk read once it is not.
    # The finished chain is the largest thing this node holds and the one thing it can
    # shrink: a 107s chain at 1056x608 is 18.5GB as float32 and 9.3GB as float16,
    # against ~39GB of weights on a 64GB machine.
    imgs = run_node("A room.\n\nOne.\n\nTwo.\n\nThree.")[0]
    check("what comes out is still float32", imgs.dtype == torch.float32, str(imgs.dtype))
    check("...and still in range",
          float(imgs.min()) >= 0.0 and float(imgs.max()) <= 1.0)
    # Free, not a trade: fp16 resolves far finer than the 8 bits the output has.
    x = torch.rand(100000)
    err = (x - x.half().float()).abs().max().item()
    check(f"fp16 error {err:.1e} is inside one 8-bit step {1 / 255:.1e}", err < 1 / 255)
    # With cleanup off the frames stay where they were; nothing is converted, and the
    # concat must not trip over a dtype it did not expect.
    off = run_node("A room.\n\nOne.\n\nTwo.", cleanup_between_shots=False)[0]
    check("cleanup off still returns float32", off.dtype == torch.float32, str(off.dtype))
    # ...and float32 because THAT is what this ComfyUI asks for between nodes, not
    # because the number is written into the node. An install run with
    # --fp16-intermediates has a VAE whose own decode returns fp16
    # (VAE.vae_output_dtype is model_management.intermediate_dtype), so widening the
    # chain to fp32 there is 9.3GB of system RAM spent against the staged weights to
    # hand fp16-native nodes something they never asked for. That machine was killed
    # by the OOM killer at 96% twice.
    check("no flag at all -> float32, as before", S._image_out_dtype() == torch.float32)
    _mm.intermediate_dtype = lambda: torch.float16
    try:
        check("--fp16-intermediates -> the chain follows it",
              S._image_out_dtype() == torch.float16)
        _h = run_node("A room.\n\nOne.\n\nTwo.\n\nThree.")[0]
        check("...and what comes out really is float16", _h.dtype == torch.float16,
              str(_h.dtype))
        check("...still in range", float(_h.min()) >= 0.0 and float(_h.max()) <= 1.0)
        _o = run_node("A room.\n\nOne.\n\nTwo.", cleanup_between_shots=False)[0]
        check("...cleanup off too", _o.dtype == torch.float16, str(_o.dtype))
        # THE AUDIO DOES NOT FOLLOW IT. fp16 is 16x finer than the 8 bits an image
        # goes out at and 8x COARSER than the 16 a waveform does -- ~12 effective
        # bits -- and the whole soundtrack is 0.018GB against 9.3GB of frames, so
        # following the flag here would trade an audible loss for nothing. The bed is
        # mixed on after the join, so a narrow accumulator adds up coarse as well.
        _pair = run_node("A room.\n\nOne.\n\nTwo.")
        _w = _pair[1]["waveform"] if isinstance(_pair[1], dict) else _pair[1]
        check("images follow the flag, audio does not",
              _pair[0].dtype == torch.float16 and _w.dtype == torch.float32,
              f"{_pair[0].dtype}/{_w.dtype}")
        _mm.intermediate_dtype = lambda: torch.float32
        check("flag off -> float32 again", S._image_out_dtype() == torch.float32)
        check("...byte for byte what it returned before",
              run_node("A room.\n\nOne.\n\nTwo.")[0].dtype == torch.float32)
    finally:
        del _mm.intermediate_dtype


class DriftVAE(FakeVAE):
    """A VAE whose decode carries the grade of the last frame it was handed.

    THE STOCK FAKE CANNOT SHOW THIS BUG AND THAT IS WHY IT REACHED A USER. FakeVAE.decode
    returns fresh torch.rand, independent of anything encoded, so every shot's output is
    statistically identical by construction and a compounding grade defect is invisible to
    every test in this file. Closing the loop is the whole fixture: what the keyframe looked
    like decides what comes back, and each pass expands contrast about mid grey the way a
    4-step distill at cfg 1 does.

    Keeps FakeVAE's frame-count formula, because the chain preallocates from sum(lens) and
    a shot that decodes to the wrong length sends the whole suite down the overflow path."""

    GAIN = 1.06

    def __init__(self):
        super().__init__()
        self.seen = []           # (mean, std, clipped fraction) of every frame encoded
        self._last = None

    def encode(self, image):
        x = image.float()
        if x.dim() == 4:
            x = x[0]
        if x.dim() == 3 and int(x.shape[-1]) >= 3:
            self.seen.append((float(x.mean()), float(x[..., :3].std()),
                              float(((x <= 0.0) | (x >= 1.0)).float().mean())))
            self._last = x[..., :3].clone()
        return super().encode(image)

    def decode(self, latent):
        t = latent.shape[2] if latent.ndim == 5 else 1
        n = max(1, (t - 2) // 5 * 17 + 5)
        base = torch.rand(H, W, 3) if self._last is None else self._last
        burned = ((base - 0.5) * self.GAIN + 0.5).clamp(0.0, 1.0)
        return burned.unsqueeze(0).repeat(n, 1, 1, 1).to(_vae_out_dtype())


def test_the_chain_does_not_burn_in():
    print("\n=== the chain does not cook itself ===")
    P = "\n\n".join(["A kitchen at night."]
                     + [f"She takes a step to the left. Beat {i}." for i in range(1, 8)])

    off = DriftVAE()
    info_off = run_node(P, vae=off, hold_levels=0.0)[2]
    on = DriftVAE()
    info_on = run_node(P, vae=on, hold_levels=1.0)[2]

    s_off = [s for _, s, _ in off.seen]
    s_on = [s for _, s, _ in on.seen]
    # THE DISABLE CHECK FIRST. Without it the rest could pass on a fixture incapable of
    # producing the defect, which is the mistake FakeVAE.decode's own comment records.
    check("the fixture really does cook the chain with the correction off",
          len(s_off) >= 3 and s_off[-1] > s_off[0] * 1.10, f"{[round(v, 4) for v in s_off]}")
    check("...and hold_levels holds it down",
          s_on[-1] < s_off[-1], f"on={[round(v, 4) for v in s_on]} off={[round(v, 4) for v in s_off]}")
    grow_off = s_off[-1] / s_off[0] if s_off[0] else 0.0
    grow_on = s_on[-1] / s_on[0] if s_on[0] else 0.0
    check("...by most of the growth, not a sliver of it",
          grow_on < 1.0 + (grow_off - 1.0) * 0.6, f"off x{grow_off:.3f} on x{grow_on:.3f}")
    c_off = [c for _, _, c in off.seen]
    c_on = [c for _, _, c in on.seen]
    check("clipping does not keep widening either",
          max(c_on) <= max(c_off) + 1e-6,
          f"on={max(c_on):.4f} off={max(c_off):.4f}")

    check("the contrast trend is reported at all", "contrast per shot" in info_off,
          info_off[-400:])
    check("...and a climb is named as cooking", "COOKING" in info_off, info_off[-400:])
    check("hold_levels says what it measured", "hold_levels:" in info_on, info_on[-400:])
    check("...and off it never claims to have corrected anything",
          "took it back out" not in info_off, info_off[-300:])

    # pure functions, no render
    lv = S.HandoffLevels()
    flat = torch.full((H, W, 3), 0.5)
    check("a flat frame is refused rather than divided by",
          lv.observe(flat, flat) is False)
    check("nothing measured means nothing applied",
          S.HandoffLevels().gains(1.0) == (None, None))
    g = S.HandoffLevels()
    lo = torch.rand(H, W, 3) * 0.4 + 0.3
    hi = ((lo - 0.5) * 1.08 + 0.5).clamp(0.0, 1.0)
    for _ in range(4):
        g.observe(lo, hi, hi, hi)
    gain, off_v = g.gains(1.0)
    check("a measured expansion comes back as a gain below 1",
          gain is not None and float(gain.max()) < 1.0, f"{gain}")
    check("...and strength 0 is off even with a measurement in hand",
          g.gains(0.0) == (None, None))
    rep = S.detail_report([(0.10, 0.08), (0.12, 0.10), (0.14, 0.13)])
    check("a rising chain is no longer called 'not softening'",
          "not softening" not in rep, rep)
    check("...the rise is named as cooking instead", "COOKING" in rep, rep)


def test_detail_trend():
    print("\n=== the chain is measured for softening ===")
    # Every boundary decodes a shot, takes its LAST frame and re-encodes it as the
    # next shot's keyframe. That round trip is lossy and it runs on the model's own
    # output, so shot 11 is sampled from a picture that has been through ten
    # decode/encode cycles. The softening is invisible shot to shot and obvious end
    # to end, which is exactly the kind of thing to measure rather than argue about.
    sharp = torch.rand(48, 48, 3)
    soft = sharp.clone()
    for _ in range(4):
        soft[1:-1, 1:-1] = (soft[:-2, 1:-1] + soft[2:, 1:-1]
                            + soft[1:-1, :-2] + soft[1:-1, 2:]) / 4
    check("a blurred frame measures less detail",
          S.frame_detail(sharp)[0] > S.frame_detail(soft)[0])
    check("a flat frame measures no detail", S.frame_detail(torch.zeros(8, 8, 3))[0] == 0)
    check("a 1px frame does not divide by zero", S.frame_detail(torch.rand(1, 1, 3)) == (0.0, 0.0))
    # The report only claims a trend when there is one.
    falling = S.detail_report([(0.09, .2), (0.08, .2), (0.07, .2), (0.06, .2)])
    check("a falling chain is called out", "DOWN 33%" in falling)
    check("...with the cause named", "re-encodes it as the next" in falling)
    check("...and a way out", "restart_after_removal" in falling)
    check("a flat chain is not alarming",
          "flat within" in S.detail_report([(0.09, .2), (0.089, .2), (0.091, .2)]))
    check("one shot claims no trend", S.detail_report([(0.09, .2)]) == "")
    check("no shots, no line", S.detail_report([]) == "")
    check("the run reports it", "detail per shot" in run_node("A room.\n\nOne.\n\nTwo.")[2])
    check("...and plan_only does not",
          "detail per shot" not in run_node("A room.\n\nOne.\n\nTwo.", plan_only=True)[2])


def test_av_stays_in_sync():
    print("\n=== the sound is as long as the picture ===")
    # Reported: video and audio out of sync. Each shot's audio latent is
    # round(frames / 24 * 40), exact only when the frame count divides by 3, so most
    # lengths leave the sound up to 8.3 ms off its own picture -- and concatenated
    # with equal shot lengths that error has the same sign every time and adds up.
    for n_beats in (2, 3, 5):
        P = "A room.\n\n" + "\n\n".join(f"Beat {i}." for i in range(n_beats))
        imgs, audio, info, script, fps_shot, total, shots, secs = run_node(P)
        sr = audio["sample_rate"]
        v = total / S.H3_FPS
        a = audio["waveform"].shape[-1] / sr
        drift_ms = abs(a - v) * 1000
        check(f"{n_beats} beats: sound matches picture within a sample "
              f"({drift_ms:.3f} ms)", drift_ms < 1.0)
        check(f"...and the frame count is what the video reports",
              imgs.shape[0] == total)
    # It is corrected per shot, so every interior cut lands too -- not just the
    # total duration at the end.
    check("the correction is reported", "realigned to the picture" in
          run_node("A room.\n\nOne.\n\nTwo.")[2])


def test_nothing_wearable_is_ever_added():
    """No clothing is invented, WHATEVER the character memory says.

    The hand-written cases below it check specific wardrobes. This one is the
    general claim, and it is made two ways because either alone has a hole:

      1. A SWEEP over awkward wardrobes x beat orderings, asserting the script's
         garment vocabulary against the author's own. Catches a known garment word
         appearing where the prompt never asked for it.

      2. Every WORD the node adds that the prompt did not contain, checked for
         anything wearable. A garment word that is in no hand-written list -- the
         hole in (1) -- shows up here, because this asserts on what the node
         actually emitted rather than on a list somebody remembered to write.

    Reported repeatedly, and each earlier fix looked complete because the wardrobe
    that exposed the next one was not in the tests."""
    print("\n=== nothing wearable is ever added ===")
    garment = re.compile(
        r"\b(leggings|stockings|tights|pantyhose|hold-?ups|nylons|hosiery|socks|"
        r"panties|knickers|thong|g-?string|briefs|boxers|underwear|undies|"
        r"jockstrap|bra|bralette|brassiere|corset|bustier|camisole|undershirt|"
        r"shorts|trousers|jeans|slacks|chinos|skirt|kilt|joggers|jeggings|"
        r"culottes|dungarees|overalls|dress|gown|robe|jumper|sweater|sweatshirt|"
        r"hoodie|cardigan|jacket|coat|shirt|blouse|t-?shirt|tee|top|tunic|boots|"
        r"shoes|trainers|sneakers|sandals|heels|gloves|mittens|scarf|hat|belt|"
        r"apron|cape|poncho|shawl|swimsuit|bikini|leotard)\b", re.I)
    wardrobes = (
        # A tagged entry: the <Picture N> made the head noun "2" and broke the
        # name lookup for exactly the garment a reference was pinning.
        "McKenna: <Picture 1>, she, 22, Shiny white crop top, "
        "chastity belt <Picture 2>, blue jeans shorts.",
        "Kate: she, 30, blouse, panties, skirt.",
        "Mara: she, 25, red dress.",
        "Jon: he, 50.",
        "Bea: she, 28, corset, stockings, garter belt, heels.",
        "Ana: she, 19, a t-shirt, tights, ankle socks, trainers.",
        "Dee: she, 40, sundress.\nEve: she, 41, sundress.",
    )
    beats = (
        "{N} stands by the chair.",
        "{N} takes off her clothes.",
        "{N} pulls the skirt down.",
        "{N} falls to the floor.",
        "{N} sits, wrists bound above her head.",
        "{N} walks out of frame.",
        "{N} comes back in.",
        "{N} is stripped bare.",
        # Plain removals that leave a region with nothing named under it -- the
        # bare-region clause fires here, and it is the one guard that talks about
        # an uncovered part of the body, so it is the likeliest to name a garment.
        "{N} takes off the skirt.",
        "{N} takes off her tights.",
        "{N} takes off the dress.",
        "{N} takes off her boots.",
    )
    invented, added, runs = {}, {}, 0
    for w in wardrobes:
        n = w.split(":", 1)[0].strip()
        for combo in itertools.islice(itertools.permutations(beats, 3), 0, 12):
            P = w + "\n\n" + "\n\n".join(b.format(N=n) for b in combo)
            try:
                script = run_node(P, plan_only=True)[3]
            except Exception as e:
                invented.setdefault("RAISED " + repr(e)[:50], P[:60])
                continue
            runs += 1
            allowed = {x.lower().replace("-", "") for x in garment.findall(P)}
            for x in garment.findall(script):
                x = x.lower().replace("-", "")
                if x not in allowed:
                    invented.setdefault(x, P[:60])
            # ...and every word the node ADDS, so a garment word nobody listed
            # above still surfaces.
            src = {t for t in re.findall(r"[a-z][a-z-]{2,}", P.lower())}
            for word in re.findall(r"[a-z][a-z-]{2,}", script.lower()):
                if word not in src:
                    added.setdefault(word, P[:60])
    check(f"no garment invented across {runs} prompts", not invented,
          f"invented {sorted(invented)[:6]}")
    wearable = re.compile(
        r"(leggings|stocking|tight|pantyhose|nylon|hosiery|sock|panti|knicker|"
        r"thong|brief|boxer|underwear|undies|bra|corset|camisole|short|trouser|"
        r"jean|skirt|kilt|jogger|dress|gown|robe|jumper|sweater|hoodie|cardigan|"
        r"jacket|coat|shirt|blouse|tunic|boot|shoe|trainer|sneaker|sandal|heel|"
        r"glove|mitten|scarf|apron|cape|poncho|shawl|bikini|leotard)", re.I)
    hits = sorted(w for w in added if wearable.search(w))
    check("no wearable word is ever ADDED by the node", not hits, f"added {hits[:6]}")


def test_no_garment_is_ever_invented():
    """END TO END: no garment word reaches a prompt unless the author wrote it.

    Reported repeatedly -- legwear appearing in shots that never asked for it. The
    unit tests cover each builder alone; this one reads the finished script and
    asserts against the AUTHOR'S OWN vocabulary, which is the only check that
    catches a word introduced by a path nobody thought to test."""
    print("\n=== nothing is invented ===")
    garment = re.compile(
        r"\b(leggings|stockings|tights|pantyhose|hold-?ups|nylons|hosiery|socks|"
        r"panties|knickers|thong|briefs|boxers|underwear|undies|bra|bralette|"
        r"corset|camisole|shorts|trousers|jeans|slacks|chinos|skirt|kilt|joggers|"
        r"dress|gown|robe|jumper|sweater|sweatshirt|hoodie|cardigan|jacket|coat|"
        r"shirt|blouse|t-shirt|tee|top|tunic|boots|shoes|trainers|sneakers|"
        r"sandals|heels|gloves|mittens|scarf|hat|belt)\b", re.I)
    cases = (
        ("a request, nothing removed",
         "McKenna: she, 22, crop top, chastity belt, blue jeans shorts.\n"
         "Dan: he, 40, t-shirt.\n\n"
         "McKenna stands by the chair.\n\n"
         'McKenna approaches Dan. "Will you take the chastity belt off?"\n\n'
         "McKenna sits in the chair."),
        ("a strip with nothing underneath",
         "McKenna: she, 22, crop top, blue jeans shorts.\n\n"
         "McKenna stands.\n\n"
         "Dan pulls off her jeans shorts.\n\n"
         "McKenna sits."),
        ("a strip with a layer underneath",
         "Kate: she, 30, blouse, panties, skirt.\n\n"
         "Kate stands.\n\n"
         "Kate takes off her skirt.\n\n"
         "Kate walks away."),
        ("one garment only",
         "Mara: she, 25, red dress.\n\n"
         "Mara stands in the room.\n\n"
         "Mara takes off her dress.\n\n"
         "Mara sits down."),
        ("no garment at all",
         "Jon: he, 50.\n\n"
         "Jon walks in.\n\n"
         "Jon sits down."),
    )
    for name, P in cases:
        allowed = {w.lower() for w in garment.findall(P)}
        script = run_node(P, plan_only=True)[3]
        invented = {w.lower() for w in garment.findall(script)} - allowed
        check("nothing invented: %s" % name, not invented,
              "invented %s" % sorted(invented))


def test_a_garment_keeps_its_description():
    """END TO END: a displaced garment keeps the sheet's words, and a garment put
    back stops being described as displaced.

    Reported as shorts coming back "a different kind". They were not re-invented by
    the model out of nothing: the node itself named them twice, once fully from the
    sheet and once bare from the beat."""
    print("\n=== a garment keeps its description ===")
    sheet = ("McKenna: she, 22, Shiny white crop top, blue jeans shorts, "
             "black leather boots.\n\n")
    # The beat uses the SHORT name; the guard must still use the sheet's.
    s = run_node(sheet + "McKenna stands.\n\n"
                 "McKenna pulls the shorts down.\n\n"
                 "McKenna looks at the window.", plan_only=True)[3]
    check("the displacement carries the sheet's full name",
          "blue jeans shorts pulled down" in s)
    check("...and never a bare one beside it", "the shorts pulled" not in s)
    # Put back up, under a different name than it went down under.
    s2 = run_node(sheet + "McKenna stands.\n\n"
                  "McKenna pulls her blue jeans shorts down.\n\n"
                  "McKenna pulls the shorts back up.\n\n"
                  "McKenna sits.", plan_only=True)[3]
    last = s2.split("[Shot 4]")[-1]
    check("a garment put back is no longer displaced", "pulled down" not in last)
    check("...and is not both up and down at once",
          not ("pulled down" in last and "pulled up" in last))


def test_a_removal_always_names_hands():
    """END TO END: the removal clause names WHOSE hands, in a real prompt.

    Reported as the action happening twice -- she takes the shorts off herself and
    he takes them off as well. The beat named her; the clause named nobody, so the
    shot carried two removals and the model gave the unattributed one to the other
    person in the frame.

    The cause was upstream of the agent logic, which was right all along: a sheet
    paragraph folded into the scene never reaches pull_character_sheets -- it only
    ever sees the beat -- so `sheet` was "" for the whole run, and the wearer and
    the cast read off it came back empty for EVERY shot."""
    print("\n=== a removal names whose hands ===")
    two = ("McKenna: she, 22, Shiny white crop top, blue jean shorts.\n"
           "Dan: he, 40, t-shirt, jeans.\n\n")
    s1 = run_node(two + "McKenna takes off her jean shorts.", plan_only=True)[3]
    check("she undresses herself",
          "McKenna takes off her jean shorts" in s1
          and "away by the last frame" in s1)
    s2 = run_node(two + "Dan pulls off her jean shorts.", plan_only=True)[3]
    check("...and he does it when the beat says so",
          "Dan pulls off her jean shorts" in s2
          and "away by the last frame" in s2)
    # WHOSE HANDS has to be somewhere in the shot -- that is the guarantee, and
    # the beat supplies it here. The completion clause may then be agentless:
    # restating who and what is the same removal written twice, which rendered
    # it twice. What must never happen is a shot where NOBODY has the hands, so
    # a beat that does not stage the removal still gets the named clause.
    for _s, _who in ((s1, "McKenna"), (s2, "Dan")):
        check("the hands are named in the shot (%s)" % _who, _who in _s)
    _unstaged = run_node(two + "remove: shorts\nMcKenna stands by the door.",
                         plan_only=True)[3]
    check("a removal the beat does not stage still names the hands",
          "McKenna takes the blue jean shorts off during this shot" in _unstaged)
    # THE ACTION, not just the end state. This clause exists because scrubbing the
    # scene stops a garment being DESCRIBED and does not tell the model to complete
    # the removal -- the last frame is the next shot's keyframe, and a cut still in
    # progress hands on a garment half worn. A version that said only "the shorts
    # are away by the last frame" asserted the outcome and never the act, and
    # garments stopped coming off.
    for _s, _lbl in ((s1, "hers"), (s2, "his"), (_unstaged, "unstaged")):
        check("the clause says the removal HAPPENS (%s)" % _lbl,
              "off during this shot" in _s)
    # One person in the shot is still that person.
    solo = run_node("Mara: she, 25, red dress.\n\nMara takes off her dress.",
                    plan_only=True)[3]
    check("alone, it is still her hands",
          "Mara takes off her dress" in solo
          and "away by the last frame" in solo)


def test_hands_and_holds_follow_the_beat():
    """Two defects seen in one real script, on neutral text.

    1. ONE agent for the whole beat. A beat that takes a coat off and then asks
       about a scarf gave BOTH removals to the other person -- her own coat came
       off by his hands. Attribution is per garment now, on the garment's own
       clause, and the first-named-acts fallback reads that clause too.

    2. The restraint hold could only be cleared by a `remove:` line. auto_remove
       filters hardware out of infer_removals on purpose, so a script that unlocks
       the cuffs IN ITS PROSE never cleared it: every later shot went on saying they
       stay closed and fastened, over hardware the beat had put on the floor. And
       clearing the latch alone was not enough -- the sheet still listed the cuffs,
       so the next shot read them back out and latched again."""
    print("\n=== hands and holds follow the beat ===")
    mem = "Kate: she, 30, blue coat, wool scarf, grey jumper.\nSam: he, 34, shirt."
    s1 = run_node("A hallway.\n\n"
                  "Kate takes off her coat and hangs it up. She finds Sam and asks "
                  'him: "Can you get this scarf off?"\n\n'
                  "Sam unties the scarf. Kate takes off her jumper.",
                  plan_only=True, character_memory=mem)[3]
    sh = [x for x in s1.split("---") if x.strip()]
    check("her own coat is by her hands",
          "Kate takes off her coat" in sh[0]
          and "away by the last frame" in sh[0])
    check("...not the person she asks about something else",
          "Sam takes the blue coat off" not in sh[0])
    check("the garment she asked about is by his",
          "Sam takes the wool scarf off during this shot" in sh[1]
          or "Sam unties the scarf" in sh[1])
    check("...and the one she removes herself is hers",
          "Kate takes off her jumper" in sh[1]
          and "away by the last frame" in sh[1])

    hw = "Kate: she, 30, coat, handcuffs.\nSam: he, 34, shirt."
    for _beats, _held, _lbl in (
            ("Kate sits with the handcuffs on.\n\nSam looks at the handcuffs.\n\n"
             "Kate waits.", True, "a mention does not unlock them"),
            ("Kate sits with the handcuffs on.\n\nKate tugs at the handcuffs.\n\n"
             "Kate waits.", True, "nor does pulling at them"),
            ("Kate sits with the handcuffs on.\n\nSam checks the handcuffs are "
             "tight.\n\nKate waits.", True, "nor does checking them"),
            ("Kate sits with the handcuffs on.\n\nSam unlocks the handcuffs.\n\n"
             "Kate waits.", False, "unlocking them in the prose does"),
            ("Kate sits with the handcuffs on.\n\nSam unlocks them. They drop to "
             "the floor.\n\nKate waits.", False, "...by pronoun too"),
            ("Kate sits with the handcuffs on.\n\nSam unfastens the handcuffs.\n\n"
             "Kate waits.", False, "...and unfastening them"),
            ("Kate sits with the handcuffs on.\n\nremove: handcuffs\nSam takes them "
             "off.\n\nKate waits.", False, "a remove: line still works")):
        _s = run_node("A room.\n\n" + _beats, plan_only=True, character_memory=hw)[3]
        _last = [x for x in _s.split("---") if x.strip()][-1]
        _on = bool(re.search(r"Every restraint[^.]*\.|The handcuffs stay[^.]*\.", _last))
        check(_lbl, _on == _held, "held=%s want=%s" % (_on, _held))
    # ...and the hardware leaves the SHEET, or the next shot reads it back out.
    _s = run_node("A room.\n\nKate sits with the handcuffs on.\n\n"
                  "Sam unlocks the handcuffs. They drop to the floor.\n\n"
                  "Kate rubs her wrists.", plan_only=True, character_memory=hw)[3]
    check("the hardware leaves the sheet",
          "handcuffs" not in [x for x in _s.split("---") if x.strip()][-1])


def test_one_line_is_one_voice():
    """END TO END: when one person has the line, the other does not babble.

    The guard said only that the other MOUTHS stay closed -- the picture half. On a
    joint model the face follows the audio, so a second voice in the stream moves a
    second mouth whatever the prose says about jaws. The shot now hears how many
    voices there are, not just how many mouths, and the prose is what conditions
    the audio branch.

    And a line with NO name on it got no guard at all: whose mouth to hold was
    unknowable, so both were free. How many voices there are does not require
    knowing whose."""
    print("\n=== one line is one voice ===")
    mem = "Kate: she, 30, blue coat.\nSam: he, 34, black shirt."
    named = run_node('A room.\n\nKate and Sam stand together. Kate says: "Come here."',
                     plan_only=True, character_memory=mem)[3]
    check("the speaker is named", "Only Kate speaks" in named)
    check("...and the other jaws are held",
          "every other mouth in the shot stays closed" in named)
    # SHORT on purpose. Every word in this clause is speech vocabulary, and on a
    # joint model the prose conditions the audio branch: a longer version ("one
    # voice in the shot, the line said once, with room tone either side of it")
    # was added to stop a listener babbling and was reported as causing it.
    check("...and says no more than that",
          "room tone either side" not in named and "said once" not in named)
    # A line is a second or two in a shot of five to ten, and the branch is open for
    # all of it: told only that there is ONE voice, the model still had seconds to
    # fill on either side and invented more talking to fill them. Reported as babble
    # before the dialogue starts.
    # <d>...</d> is the same case.
    tag = run_node("A room.\n\nKate turns to Sam. <d>Come here.</d>",
                   plan_only=True, character_memory=mem)[3]
    check("a <d> line is attributed too", "Only Kate speaks" in tag)
    # No name on the line, two people who could be saying it.
    for _b in ('The two of them wait. "Now?"', "They face each other. <d>Now?</d>"):
        _out = run_node("A room.\n\n" + _b, plan_only=True, character_memory=mem)
        _info, _s = _out[2], _out[3]
        check(f"unattributed still gets one voice: {_b[:26]!r}",
              "Only the person speaking" in _s)
        check("...and says so in info", "names no speaker" in _info)
    # One person alone needs no guard: there is nobody else to babble, and a
    # sentence about other mouths implies other people.
    solo = run_node('A room.\n\nKate waits alone. "Now?"', plan_only=True,
                    character_memory=mem)[3]
    check("one person alone gets no mouth guard",
          "One voice in the shot" not in solo and "Only Kate speaks" not in solo)
    # The speaker is named ONCE. Naming a person twice in one shot is what put a
    # second copy of them in frame.
    check("the speaker is named once in the guard",
          named.split("Only Kate speaks")[1].split(".")[0].count("Kate") == 0)


def test_the_plan_says_whether_silence_can_be_applied():
    """A wrong VAE on audio_vae costs nothing at load time and silently unpins
    every line-free shot -- and the first anybody knows of it is a shot with no
    dialogue that babbles. Probed in the PLAN, so finding out does not cost a full
    render."""
    print("\n=== the plan says whether silence can be applied ===")
    P = "A room.\n\nKate looks around.\n\nKate sits down."
    mem = "Kate: she, 30, coat."
    ok = run_node(P, plan_only=True, character_memory=mem,
                  audio_vae=FakeAudioVAE())[2]
    check("a working audio VAE reports it can", "silence can be applied" in ok)
    check("...and does not cry wolf", "SILENCE CANNOT BE APPLIED" not in ok)
    none = run_node(P, plan_only=True, character_memory=mem, audio_vae=None)[2]
    check("a missing audio VAE is called out", "SILENCE CANNOT BE APPLIED" in none)
    check("...naming the input", "audio_vae input" in none)
    # The likeliest wiring mistake: the VIDEO vae on the audio input. It loads
    # without complaint and reports no sample rate.
    S._SILENT_UNIT["lat"] = None

    class NotAnAudioVae:
        def encode(self, x):
            raise RuntimeError("not an audio vae")

    wrong = run_node(P, plan_only=True, character_memory=mem,
                     audio_vae=NotAnAudioVae())[2]
    check("a VAE that cannot encode silence is called out",
          "SILENCE CANNOT BE APPLIED" in wrong)
    check("...and says which file the input wants",
          "minimax_h3_audio_vae" in wrong)
    S._SILENT_UNIT["lat"] = None


def test_the_silent_latent_looks_like_silence():
    """The conditioning has to look like what the encoder produces, not just decode
    quietly. The old build kept ONE interior frame and repeated it, on the argument
    that silence is homogeneous -- it is not, in latent space. Encoded silence has
    frame-to-frame variation (mean 0.002-0.004, max 0.021); a repeated frame has a
    delta of exactly zero, which is a signal no encoder makes, and a model handed
    conditioning outside its own distribution has every reason to disregard it.

    Checked with a stand-in whose encode has the same shape and the same edge
    artifacts as the real VAE, so the shape of the fix is tested without loading a
    checkpoint. The numbers in the docstring were measured against the real one.

    This lives in the smoke suite because test_node stubs torch, and a stand-in
    built from stub tensors cannot exercise a function that slices and flips."""
    print("\n=== the silent latent looks like silence ===")

    class EdgyVae:
        """Encodes to [B, 32, 2, T] with big deltas at the ends, like the real one."""
        audio_sample_rate = 32000

        def encode(self, x):
            n = x.shape[1] // 800
            t = torch.arange(n, dtype=torch.float32)
            base = 0.46 + 0.002 * torch.sin(t * 0.7)      # interior variation
            base[:4] += torch.tensor([0.22, 0.10, 0.05, 0.03])   # padded head
            base[-4:] += torch.tensor([0.03, 0.05, 0.10, 0.17])  # padded tail
            return base.reshape(1, 1, 1, n).repeat(1, 32, 2, 1)

    vae = EdgyVae()
    S._SILENT_UNIT["lat"] = None
    lat = S._silent_audio_latent(vae, 226, S.H3_FPS)
    check("a latent is produced", lat is not None)
    _, _, want_t = S.temporal_shape(226, S.H3_FPS)
    check("...of exactly the length the layout wants", lat.shape[-1] == want_t)
    check("...with the layout's channel count", lat.shape[1] == 32)
    d = (lat[..., 1:] - lat[..., :-1]).abs()
    check("it is NOT a flat signal", float(d.mean()) > 1e-5)
    # The edges the encoder pads are thrown away, so no join carries their spike.
    check("no seam spike from the padded ends", float(d.max()) < 0.05)
    # Ping-pong: every join repeats a frame, so plain tiling's seam is gone.
    S._SILENT_UNIT["lat"] = None
    long = S._silent_audio_latent(vae, 462, S.H3_FPS)
    dl = (long[..., 1:] - long[..., :-1]).abs()
    check("a longer shot has no seam either", float(dl.max()) < 0.05)
    check("...and is still the right length",
          long.shape[-1] == S.temporal_shape(462, S.H3_FPS)[2])
    class OtherVae(EdgyVae):
        def encode(self, x):
            return super().encode(x) + 1.0

    other = S._silent_audio_latent(OtherVae(), 226, S.H3_FPS)
    check("changing audio VAE rebuilds the cached silence",
          float(other.mean()) > float(lat.mean()) + 0.5)
    # Still defensive: anything unexpected returns None rather than raising.
    S._SILENT_UNIT["lat"] = None

    class Broken:
        audio_sample_rate = 32000

        def encode(self, x):
            raise RuntimeError("nope")

    check("a failing encode returns None",
          S._silent_audio_latent(Broken(), 226, S.H3_FPS) is None)
    check("no sample rate returns None",
          S._silent_audio_latent(object(), 226, S.H3_FPS) is None)

    class TooShort:
        audio_sample_rate = 32000

        def encode(self, x):
            return torch.zeros((1, 32, 2, 3))

    S._SILENT_UNIT["lat"] = None
    check("an encode too short to trim returns None",
          S._silent_audio_latent(TooShort(), 226, S.H3_FPS) is None)
    S._SILENT_UNIT["lat"] = None


def test_silence_pins_the_generated_audio():
    print("\n=== silence pins the generated audio target ===")
    video = torch.randn((1, 24, 4, 3, 5))
    audio = torch.randn((1, 32, 2, 40))
    silence = torch.zeros_like(audio) + 0.125

    latent = {"samples": FakeNested((video, audio))}
    check("a full-shot silence mask is installed",
          S._pin_audio_silence(latent, silence, None))
    got_video, got_audio = latent["samples"].unbind()
    video_mask, audio_mask = latent["noise_mask"].unbind()
    check("the target audio is encoded silence", torch.equal(got_audio, silence))
    check("video remains free to denoise", bool(torch.all(video_mask == 1)))
    check("the whole silent shot is locked", bool(torch.all(audio_mask == 0)))
    check("the video target is untouched", torch.equal(got_video, video))

    latent = {"samples": FakeNested((video, audio))}
    check("a dialogue lead-in mask is installed",
          S._pin_audio_silence(latent, silence, 20))
    _, audio_mask = latent["noise_mask"].unbind()
    check("the first half-second is locked", bool(torch.all(audio_mask[..., :20] == 0)))
    check("speech can denoise after the lead-in", bool(torch.all(audio_mask[..., 20:] == 1)))

    latent = {"samples": FakeNested((video, audio))}
    check("a lead-in and a tail install together", S._pin_audio_silence(latent, silence, 20, 8))
    _, audio_mask = latent["noise_mask"].unbind()
    check("the opening is locked", bool(torch.all(audio_mask[..., :20] == 0)))
    check("the close is locked", bool(torch.all(audio_mask[..., -8:] == 0)))
    check("the line's span between them is free", bool(torch.all(audio_mask[..., 20:-8] == 1)))

    latent = {"samples": FakeNested((video, audio))}
    check("a tail alone is enough to install", S._pin_audio_silence(latent, silence, 0, 8))
    _, audio_mask = latent["noise_mask"].unbind()
    check("...and it locks only the close",
          bool(torch.all(audio_mask[..., -8:] == 0)) and bool(torch.all(audio_mask[..., :-8] == 1)))

    latent = {"samples": FakeNested((video, audio))}
    check("a tail that would overlap the lead is clipped to what is left",
          S._pin_audio_silence(latent, silence, 30, 20))
    _, audio_mask = latent["noise_mask"].unbind()
    check("...so the lead keeps its 30 and the tail takes the other 10", bool(torch.all(audio_mask == 0)))
    check("nothing to pin is reported, not counted as applied",
          not S._pin_audio_silence({"samples": FakeNested((video, audio))}, silence, 0, 0))

    bad = {"samples": FakeNested((video, audio))}
    check("a mismatched silent latent is rejected",
          not S._pin_audio_silence(bad, silence[..., :-1], None))
    check("a rejected mask does not partially mutate the latent", "noise_mask" not in bad)


def test_a_softened_handoff_is_not_a_keyframe():
    """The removing shot keeps describing the garment when nothing anchors it.

    The scrub applies to the removing shot too, and the only reason that is safe is
    that the shot's KEYFRAME already shows the garment on at the start. Below
    KEYFRAME_SAFE_AUG the handoff stops being a keyframe and rides as an extra
    reference -- it says who somebody is, not what the opening frame holds -- so
    the premise is false and scrubbing took the belt out of the text of the very
    shot that removes it. It was gone a beat early with nothing holding it on."""
    print("\n=== a softened handoff is not a keyframe ===")
    # No outer garment over the belt: implied_layers would keep it out of the sheet
    # until that garment came off, which is a different rule and would mask what
    # this test is about.
    P = ("A room.\n\nMcKenna: she, 22, crop top, chastity belt.\n\n"
         "Dan: he, 30, shirt.\n\nMcKenna walks in.\n\nDan looks at her.\n\n"
         "Dan unlocks the chastity belt.\n\nMcKenna sits down.")
    soft = [x for x in re.split(r"(?=\[Shot )",
                                run_node(P, plan_only=True, ref_noise_aug=0.95)[3])
            if x.strip()]
    # Wherever the carried sheet sits in the shot -- beat_leads moves it after the
    # beat -- what matters is that the removing shot still carries the belt and the
    # shot after it does not.
    check("softened: the removing shot still says it is worn",
          "chastity belt" in soft[2])
    check("...and the next shot does not", "chastity belt" not in soft[3])
    # At an anchoring aug the keyframe does hold the first frame, and scrubbing the
    # removing shot is right: text saying it is worn would put it back at the end.
    hard = [x for x in re.split(r"(?=\[Shot )",
                                run_node(P, plan_only=True, ref_noise_aug=0.999)[3])
            if x.strip()]
    check("anchored: the removing shot is scrubbed",
          "chastity belt" not in hard[2].split("Dan unlocks")[0])
    check("...and it still comes off there", "comes off during this shot" in hard[2])
    # The report has to name WHICH reason, or it cannot be acted on.
    info = run_node(P, plan_only=True, ref_noise_aug=0.95)[2]
    check("info names the aug", "ref_noise_aug 0.95 is below" in info)


def test_a_journey_reaches_the_shot():
    """END TO END: the travel clause lands in the shot, with the room the previous
    beat established as its origin.

    The unit tests exercise travel_anchor directly and passed with the clause
    disconnected from the render loop, which is the failure that matters: a
    journey given only its destination is one the model can satisfy by starting
    there."""
    print("\n=== a journey reaches the shot ===")
    mem = "McKenna: she, 22, top.\nDan: he, 30, shirt."
    P = ("A scene in a home.\n\n"
         "McKenna finds Dan in the living room.\n\n"
         "She takes his hand and walks him down the hallway to the bedroom.\n\n"
         "They sit on the bed.")
    out = run_node(P, plan_only=True, character_memory=mem)
    info, script = out[2], out[3]
    sh = [x for x in re.split(r"(?=\[Shot )", script) if x.strip()]
    check("the travelling shot is told where it begins",
          "opens in the living room" in sh[1])
    check("...and where it ends", "arrives in the bedroom" in sh[1])
    check("...and what it passes through", "along the hallway" in sh[1])
    check("...and that the walk happens on screen",
          "played out on screen" in sh[1] and "every step in frame" in sh[1])
    # The origin came from the PREVIOUS beat: this one never names it.
    check("the origin is latched from the earlier beat",
          "living room" not in P.split("\n\n")[2])
    # A shot that goes nowhere gets nothing.
    check("a stationary shot carries no travel clause",
          "The shot begins in" not in sh[0] and "The shot begins in" not in sh[2])
    check("info names the travelling shots", "move between places" in info)


def test_a_line_is_spoken_in_one_language():
    """H3 is joint and multilingual. The prose conditions the audio branch, and a
    branch told a line is spoken but never told in WHAT will pick a language --
    fluent delivery in one nobody asked for sounds like babble to anybody expecting
    English."""
    print("\n=== a line is spoken in one language ===")
    mem = "Kate: she, 30, coat.\nSam: he, 34, shirt."
    P = ('A room.\n\nKate waits.\n\nKate says to Sam: "Come here."\n\n'
         "They sit down.")
    out = run_node(P, plan_only=True, character_memory=mem)
    info, script = out[2], out[3]
    sh = [x for x in re.split(r"(?=\[Shot )", script) if x.strip()]
    check("the speaking shot is told its language",
          "The language is English" in sh[1])
    check("...and the silent ones are not",
          "The language is" not in sh[0] and "The language is" not in sh[2])
    check("info names the shots", "told which language it is spoken in" in info)
    # Positively phrased: naming the unwanted language would ask for it.
    check("no language is named but the wanted one",
          not any(w in sh[1] for w in ("not in", "Spanish", "French", "Chinese")))
    # ...AND IT IS NOT ALWAYS ENGLISH. The clause named English and only English,
    # so a script written in anything else was told its own line is spoken in a
    # language it is not, and the delivery fought the words. Read from the line.
    def said_in(script_text, which=1):
        s = [x for x in re.split(r"(?=\[Shot )",
                                 run_node(script_text, plan_only=True)[3]) if x.strip()]
        m = re.search(r"The language is ([^.]*)\.", " ".join(s[which].split()))
        return m.group(1) if m else ""

    for _lang, _line in (
            ("Spanish", 'Kate dice: "No sé lo que estás haciendo con eso."'),
            ("French", 'Kate dit: "Je ne sais pas ce que vous faites."'),
            ("German", 'Kate sagt: "Ich weiss nicht was du da machst."'),
            ("Russian", 'Kate says: "Не знаю '
                        'что ты дела'
                        'ешь."'),
            ("Japanese", 'Kate says: "何をしているの？"'),
    ):
        got = said_in("A room.\n\nKate waits.\n\n" + _line)
        check(f"a line in {_lang} is called {_lang}", got == _lang, got)
    # A line too short to carry evidence takes the language of the script it is in,
    # not English on a technicality.
    # Shot 3 is the short one -- "Si." carries no evidence of its own, and read
    # alone it would be called English inside a Spanish script.
    short = said_in('Una habitación.\n\nKate espera.\n\n'
                    'Kate dice: "No sé lo que estás haciendo con eso."\n\n'
                    'Kate dice: "Sí."', which=2)
    check("...and the short line follows the script", short == "Spanish", short)
    # ...AND ORDINARY SPEECH CARRIES ONE FUNCTION WORD, NOT TWO. The vote needs two
    # before it will name a language, and these are unmistakably German to a reader:
    #     "Oh, du siehst heute toll aus, Schatz!"   -- du
    #     "OK, das reicht mir jetzt wirklich!"      -- das
    # Both scored 1 and abstained. Inside a script whose other lines are English the
    # fallback is English, so the German line was actively TOLD it is English and the
    # delivery then fights its own words. The author had already said which language
    # it is, in the stage direction -- and spoken_text() strips everything outside
    # the quotes before the vote sees it, so that statement was the one thing thrown
    # away. Reported by a user reading the code.
    for _line in ('Klaus approaches her and says in German '
                  '"Oh, du siehst heute toll aus, Schatz!"',
                  'Klaus angrily stares at her and says in German '
                  '"OK, das reicht mir jetzt wirklich!"'):
        _got = said_in("A kitchen.\n\nKlaus waits.\n\n" + _line)
        check(f"one function word, but the author said German: {_line[42:62]!r}",
              _got == "German", _got)
    # The worst case is a mixed script, where the fallback is not merely uninformed
    # but wrong: this beat's own stage direction has to outrank it.
    _mixed = ('A kitchen.\n\nMara says: "Where have you been all evening, exactly?"\n\n'
              'Klaus says in German "OK, das reicht mir jetzt wirklich!"\n\n'
              'Mara says: "That is not what I asked you and you know it."')
    # s[0] is Shot 1: the scene paragraph is not a shot, so the German line is s[1].
    check("the German line in an English script is German",
          said_in(_mixed, 1) == "German", said_in(_mixed, 1))
    check("...and the English ones stay English",
          said_in(_mixed, 0) == "English" and said_in(_mixed, 2) == "English",
          f"{said_in(_mixed, 0)}/{said_in(_mixed, 2)}")
    # A NATIONALITY IS NOT A LANGUAGE. A speech frame is required, so these say
    # nothing and the line is read the way it always was.
    for _t in ("The German soldier looks up.", "A German car pulls in.",
               "She pets the German shepherd.", "Klaus is German.",
               "The Spanish tiles are cold.", "A French window stands open."):
        check(f"not a language: {_t[:34]!r}", not S.engine.language_named(_t))
    for _t, _w in (("She speaks German to him.", "German"),
                   ("He replies in broken German.", "German"),
                   ("She switches to Spanish.", "Spanish"),
                   ("He asked in Russian where the key was.", "Russian")):
        check(f"named: {_t[:34]!r}", S.engine.language_named(_t) == _w)
    # A CONFIDENT VOTE STILL WINS. The statement resolves an abstention; it does not
    # overrule the words, because naming the wrong language is the failure here.
    check("the vote is not overruled by a stray mention",
          S.engine.language_of("Je ne sais pas ce que vous faites",
                               named="German") == "French")
    # Non-Latin characters are a strong language signal and easy to miss by eye.
    check("plain English is clean", S.non_latin_in('Kate says: "Hello."') == [])
    check("...and accented Latin is not flagged",
          S.non_latin_in("caf\u00e9 na\u00efve") == []
          and S.non_latin_in("cafe\u0301") == [])
    check("...nor curly quotes and dashes",
          S.non_latin_in("\u201cHello\u201d \u2014 she said\u2026") == [])
    check("a non-Latin script IS flagged",
          S.non_latin_in("\u041f\u0440\u0438\u0432\u0435\u0442") != [])
    odd = run_node('A room.\n\nKate says: "\u041f\u0440\u0438\u0432\u0435\u0442."',
                   plan_only=True, character_memory="Kate: she, 30, coat.")[2]
    check("...and reported", "not Latin text" in odd)
    check("...without being removed from the prompt", "NOT removed" in odd)


def test_undressing_does_not_spread():
    """END TO END: one character undressing leaves the other dressed."""
    print("\n=== undressing does not spread ===")
    mem = "McKenna: she, 22, crop top, shorts.\nDan: he, 30, shirt, jeans."
    P = ("A room.\n\nMcKenna and Dan stand.\n\n"
         "McKenna and Dan sit down. McKenna takes off her clothes.\n\nThey wait.")
    sh = [x for x in re.split(r"(?=\[Shot )",
                              run_node(P, plan_only=True, character_memory=mem)[3])
          if x.strip()]
    check("the other character keeps his wardrobe", "shirt, jeans" in sh[1])
    check("...and hers is gone", "crop top" not in sh[1].split("Dan:")[0])
    check("the clause says whose body it is",
          "Everything McKenna is wearing" in sh[1])
    check("...and pins everyone else to their own entry",
          "own entry lists" in sh[1])
    check("...once, not twice", sh[1].count("own entry lists") == 1)
    # It stays gone for him on later shots too.
    check("he is still dressed afterwards", "shirt, jeans" in sh[2])


def test_a_working_character_is_not_still_lying_down():
    """END TO END: the hold lets go when the beat contradicts it."""
    print("\n=== a working character is not still lying down ===")
    mem = "McKenna: she, 2, blonde.\nDana: she, 35, brunette, shirt, jeans."
    P = ("A nursery.\n\nDana lies down on the sofa with McKenna.\n\n"
         "Dana takes out a new blanket and places it on the change table.\n\n"
         "Dana waits.")
    sh = [x for x in re.split(r"(?=\[Shot )",
                              run_node(P, plan_only=True, character_memory=mem)[3])
          if x.strip()]
    check("the working shot does not say she is lying down",
          "Dana is lying down" not in sh[1])
    check("...nor the shot after it", "Dana is lying down" not in sh[2])
    # And a pose that is NOT contradicted is still held.
    P2 = ("A nursery.\n\nDana and McKenna sit down on the sofa.\n\n"
          "Dana and McKenna look at the window.\n\nDana and McKenna wait.")
    sh2 = [x for x in re.split(r"(?=\[Shot )",
                               run_node(P2, plan_only=True, character_memory=mem)[3])
           if x.strip()]
    check("a pose nothing contradicts is still held",
          "is sitting" in sh2[1])


def test_pacing_reaches_the_thin_shots():
    """END TO END: the pacing clause lands on shots that outlast their beat, and
    on no others."""
    print("\n=== pacing reaches the thin shots ===")
    mem = "Kate: she, 30, coat."
    P = ("A room.\n\nKate waits.\n\n"
         "Kate walks in, drops her bag, takes off her coat, hangs it up and "
         "crosses the room.\n\nKate looks at the window.")
    # 'fixed' gives every shot the same length regardless of content, which is
    # where a three second action lands in a ten second shot.
    out = run_node(P, plan_only=True, character_memory=mem,
                   shot_length="fixed", shot_seconds=10.0)
    info, script = out[2], out[3]
    paced = [i for i, b in enumerate(script.split("---"), 1)
             if "even pace across the whole shot" in b]
    check("the thin shots are paced", paced == [1, 3], str(paced))
    check("...and the full one is not", 2 not in paced)
    check("info names them", "stage less than their length" in info)
    # Sizing from the beat already matches content, so nothing is thin and the
    # clause stays out of the way.
    fitted = run_node(P, plan_only=True, character_memory=mem)[3]
    check("sizing from the beat needs no pacing",
          "even pace across the whole shot" not in fitted)


def test_an_instruction_is_not_the_action():
    """END TO END: a beat that TELLS somebody to undress and lie down does neither
    until the beat that does it."""
    print("\n=== an instruction is not the action ===")
    mem = "McKenna: she, 22, shorts.\nDana: she, 35, jeans."
    P = ("A public bathroom.\n\n"
         "McKenna and Dana walk in. Dana says to McKenna: "
         '"Take off your shorts and lie down on the change table." '
         "Dana puts the bag on the change table.\n\n"
         "McKenna takes off her shorts and lies down on the change table.\n\n"
         # She has to be IN the last shot for her pose to be stated: a pose for
         # somebody the text does not describe is a pose for nobody.
         "Dana opens the bag and looks at McKenna.")
    sh = [x for x in re.split(r"(?=\[Shot )",
                              run_node(P, plan_only=True, character_memory=mem)[3])
          if x.strip()]
    # The sheet entry itself, not the bare word: the quoted ORDER says "shorts" too,
    # so a plain substring would pass even if the wardrobe had been scrubbed. Asserted
    # on the entry because beat_leads moves the sheet after the beat.
    check("the shorts are still on while she is only told",
          re.search(r"McKenna: she, 22,[^.]*shorts", sh[0]) is not None)
    check("...and nothing is taken off yet",
          "off during this shot" not in sh[0])
    check("...and nobody is lying down yet", "still lying down" not in sh[0])
    # The beat that actually does it.
    check("the next beat takes them off", "off during this shot" in sh[1])
    # ...and the pose it sets is held afterwards, for her only.
    check("the pose is held after that", "McKenna is lying down" in sh[2])
    check("...and the speaker is not lying down", "Dana is still lying" not in sh[2])
    # ...and with NOTHING happening after the instruction, so a wrongly latched
    # pose has nothing to clear it. Without the guard both of them are laid down
    # by a sentence Dana only spoke.
    idle = ("A public bathroom.\n\nMcKenna and Dana walk in. Dana says to McKenna: "
            '"Take off your shorts and lie down on the change table."\n\n'
            "McKenna and Dana wait.")
    idle_sh = [x for x in re.split(r"(?=\[Shot )",
                                   run_node(idle, plan_only=True,
                                            character_memory=mem)[3]) if x.strip()]
    # The words of the order are still in the shot -- they have to be, beats go to
    # the model verbatim -- so the LISTENER is given something to be doing, or the
    # instruction is the only thing in the frame about their body.
    check("the listener is given something to do",
          "McKenna listens" in sh[0])
    # Split defensively: with the clause absent this used to raise IndexError
    # rather than report a failure, which hides the result behind a traceback.
    _tail = sh[0].split("McKenna listens")
    check("...and is named once, not twice",
          len(_tail) > 1 and _tail[1].split(".")[0].count("McKenna") == 0)
    check("...and a line with no order in it adds nothing",
          "listens, still" not in run_node(
              'A room.\n\nDana says: "Hello there."', plan_only=True,
              character_memory=mem)[3])
    check("a spoken instruction latches nobody",
          "still lying down" not in idle_sh[1], idle_sh[1][-90:])


def test_the_scene_does_not_reset():
    """END TO END: after a move, later shots are in the new room, and the journey
    itself has both its ends."""
    print("\n=== the scene does not reset ===")
    mem = "Kate: she, 30, coat.\nSam: he, 34, shirt."
    P = ("A living room.\n\nKate and Sam sit on the sofa.\n\n"
         "Kate walks him down the hallway to the bedroom.\n\n"
         "They sit on the bed.\n\nKate looks at him.")
    out = run_node(P, plan_only=True, character_memory=mem)
    info, script = out[2], out[3]
    sh = [x for x in re.split(r"(?=\[Shot )", script) if x.strip()]
    # The journey has an origin, which it only has because `here` is seeded from
    # the scene: stated as a destination alone it renders as a cut.
    check("the move begins where the scene says", "opens in the living room" in sh[1])
    check("...and ends in the new room", "arrives in the bedroom" in sh[1])
    # ...and the shots after it stay there.
    check("the next shot is in the new room", "takes place in the bedroom" in sh[2])
    check("...and so is the one after", "takes place in the bedroom" in sh[3])
    check("info names them", "room the scene text does not name" in info)
    # A script that never moves is untouched.
    still = run_node("A living room.\n\nKate waits.\n\nKate looks up.",
                     plan_only=True, character_memory=mem)[3]
    check("a script that stays put says nothing", "This shot is in the" not in still)


def test_a_described_room_still_holds():
    """END TO END: the same journey as above, into a room the author DESCRIBED. One
    adjective used to switch the whole room tracker off, silently, and the acoustic
    stayed behind in the room they left."""
    print("\n=== a described room still holds ===")
    mem = "Kate: she, 30, coat.\nSam: he, 34, shirt."
    # The shots that check the acoustic carry a LINE, because the bed and the room
    # tone go under shots whose audio branch is already open. A wordless shot is
    # pinned to silence and is given no acoustic at all -- see
    # test_room_tone_under_every_shot for why ambience everywhere does not work.
    P = ("A carpeted living room, lamps low.\n\n"
         "Kate and Sam sit on the sofa and she says: \"Come with me.\"\n\n"
         "Kate walks him down the hallway to the tiled bathroom.\n\n"
         "They stand by the sink and she says: \"Wait here.\"\n\n"
         "Kate turns the tap on.")
    out = run_node(P, plan_only=True, character_memory=mem)
    info, script = out[2], out[3]
    sh = [x for x in re.split(r"(?=\[Shot )", script) if x.strip()]
    check("the journey has both ends", "opens in the living room" in sh[1]
          and "arrives in the bathroom" in sh[1])
    check("the next shot is in the new room", "takes place in the bathroom" in sh[2])
    check("...and so is the one after", "takes place in the bathroom" in sh[3])
    # The acoustic goes with them.
    check("the room tone follows", "tiled walls ringing" in sh[2])
    check("...and the ambient bed", "water moving in the pipes" in sh[2])
    check("the room they left keeps its own", "a soft room with little echo" in sh[0])
    check("...and is not given the new one", "tiled walls ringing" not in sh[0])
    check("info names the shots", "sound followed them into the new room" in info)
    # auto_sound off leaves the picture fix alone and takes only the sound.
    quiet = run_node(P, plan_only=True, character_memory=mem, auto_sound=False)[3]
    check("auto_sound off keeps the room hold", "takes place in the bathroom" in quiet)
    check("...and drops the acoustic", "tiled walls ringing" not in quiet)


def test_the_sound_clause_is_inside_the_budget():
    """END TO END, and testing the WIRING rather than the function.

    The unit test for this passed with the fix reverted, because it called
    fit_guards directly with clauses of its own -- which says nothing about whether
    the sound clause is actually in the list the node passes. It was appended after
    the call for a long time, so it was the one piece of node-written text no cap
    could reach.

    The shipped budget almost never binds, so this squeezes it deliberately and
    checks the node's own report names what went."""
    print("\n=== the sound clause is inside the budget ===")
    mem = "Kate: she, 30, coat, scarf.\nSam: he, 34, shirt."
    # Beats carry LINES, because a sound clause only exists on a shot whose audio
    # branch is open -- a wordless shot is pinned to silence and gets none, so there
    # would be nothing for the budget to drop.
    P = ("A living room.\n\nKate and Sam sit on the sofa and she says: \"Sit down.\"\n\n"
         "Kate takes off her scarf and says: \"It is warm in here.\"\n\n"
         "Kate walks him down the hallway to the tiled bathroom.\n\n"
         "They stand by the sink and she says: \"Wait here.\"\n\n"
         "Sam waits and says: \"All right.\"")
    floor, ratio = S.GUARD_FLOOR_WORDS, S.GUARD_WORDS_PER_BEAT_WORD
    try:
        S.GUARD_FLOOR_WORDS, S.GUARD_WORDS_PER_BEAT_WORD = 20, 1
        info = run_node(P, plan_only=True, character_memory=mem)[2]
    finally:
        S.GUARD_FLOOR_WORDS, S.GUARD_WORDS_PER_BEAT_WORD = floor, ratio
    check("a squeezed budget drops something", "guard clauses dropped for room" in info)
    # ...and the thing it drops is the sound clause, because it ranks last. If it
    # were still appended after the cap it could not be dropped at all.
    check("the sound clause is what gives way", re.search(r"dropped for room[^|]*sound",
                                                          info) is not None)
    # At the SHIPPED budget nothing is dropped: the cap catches runaways and does
    # not trim routinely, so no current render changes because of this.
    clean = run_node(P, plan_only=True, character_memory=mem)[2]
    check("the shipped budget drops nothing", "guard clauses dropped for room" not in clean)


def test_the_anchor_is_not_read_as_a_room():
    """END TO END: an anchor is the CAMERA. room_tone has always said so and reads the
    opening beat instead, but the room reader was handed the whole assembled string --
    anchor, scene and sheet -- so a lens line was being asked where everybody is.

    With "shallow depth of field" in the anchor, every shot was told it was in a hall,
    the first journey took the hall as its origin, and the acoustic became a large
    room with a long tail. None of it was in the script."""
    print("\n=== the anchor is not read as a room ===")
    anchor = ("Shot on 35mm, shallow depth of field, warm practical lighting, "
              "handheld, muted colour grade, film grain.")
    mem = "Kate: she, 30, coat.\nSam: he, 34, shirt."
    P = ("Kate and Sam sit on the sofa.\n\n"
         "Kate walks him down the hallway to the bedroom.\n\n"
         "They sit on the bed.\n\nSam waits.")
    out = run_node(P, plan_only=True, character_memory=mem, anchor=anchor)
    script = out[3]
    sh = [x for x in re.split(r"(?=\[Shot )", script) if x.strip()]
    check("no shot is put in a phantom hall", "in the hall," not in script)
    check("...and no journey starts in one", "opens in the hall" not in script)
    check("...and the lens gets no cathedral", "a large room with a long tail" not in script)
    # The anchor itself is still carried on every shot -- that is what it is for.
    check("the anchor is on every shot",
          all("Shot on 35mm" in x for x in sh), f"{len(sh)} shots")
    check("...once per shot", all(x.count("Shot on 35mm") == 1 for x in sh))
    # The journey itself still works; it simply has no invented origin.
    check("the move still reaches the bedroom", "arrives in the bedroom" in sh[1])
    # ...and a scene paragraph that DOES name a room still seeds it.
    named = run_node("A living room.\n\nKate waits.\n\n"
                     "Kate walks him down the hallway to the bedroom.",
                     plan_only=True, character_memory=mem)[3]
    check("a real scene room still seeds the origin", "opens in the living room" in named)
    # THE HALF THAT CAUGHT A WRONG FIX. Excluding the anchor from the room reader
    # looks right -- a lens line is not a location -- but the anchor is DOCUMENTED
    # to carry the location, and with one set there is no scene paragraph for the
    # room to live in instead. Excluded, this loses the origin of its first journey,
    # which is the destination-with-no-origin case that renders as a cut. The word
    # boundary was the whole fix; the anchor was never at fault.
    located = run_node(P, plan_only=True, character_memory=mem,
                       anchor="A carpeted living room. Shot on 35mm, "
                              "shallow depth of field, handheld.")[3]
    check("a location IN the anchor still seeds the origin",
          "opens in the living room" in located)
    check("...and still no phantom hall", "in the hall" not in located)


def test_a_garment_does_not_appear_at_the_first_frame():
    """END TO END: an outer garment comes off, several beats pass, and it goes back
    on. The shot that puts it on must describe the CHANGE, not the end state, and
    must not also list it as already worn -- one shot holding the garment in two
    states is the disagreement that made it appear at frame one."""
    print("\n=== a garment does not appear at the first frame ===")
    mem = "Kate: she, 30, brown hair, grey leggings, blue shorts."
    P = ("A changing room.\n\nKate stands by the bench.\n\n"
         "Kate takes off her shorts.\n\nKate sits down on the bench.\n\n"
         "Kate reaches for the bench.\n\n"
         "Kate puts her shorts back on.\nadd: her blue shorts\n\n"
         "Kate stands up and walks out.")
    out = run_node(P, plan_only=True, character_memory=mem)
    info, script = out[2], out[3]
    sh = [x for x in re.split(r"(?=\[Shot )", script) if x.strip()]
    # Off in the shots between.
    check("the shorts are gone after the removal", "shorts" not in sh[2])
    check("...and stay gone", "shorts" not in sh[3])
    # The staging shot describes the CHANGE...
    check("the staging shot gives both ends",
          "off the body as the shot opens" in sh[4] and "by the last frame" in sh[4])
    # ...and does NOT also list them as already worn. The add phrase is held back.
    check("...and does not also list them as worn", "Her blue shorts." not in sh[4])
    # From the next shot they are simply worn, the way a removal scrubs from its own.
    check("the next shot wears them", "Her blue shorts." in sh[5])
    check("info names the shot", "put a garment back ON" in info)
    # The OTHER use of add: -- revealing a layer already underneath -- is untouched.
    rev = run_node("A yard.\n\nNora stands by the gate.\n\n"
                   "Dan cuts off her jacket and throws it away.\n"
                   "remove: jacket\nadd: her white shirt underneath\n\n"
                   "Nora turns to the gate.",
                   plan_only=True,
                   character_memory="Nora: she, 34, red hair, green jacket, white shirt.")[3]
    check("a reveal stages no dressing", "off the body as the shot opens" not in rev)
    check("...and is described in its own shot", "Her white shirt underneath." in
          [x for x in re.split(r"(?=\[Shot )", rev) if x.strip()][1])
    # ...and the case the beat_stages_wearing guard actually protects: the SAME
    # garment coming back, on a beat that does not stage anybody putting it on. It
    # went back on between shots, so there is no dressing to give both ends to, and
    # the shot should simply describe it as worn.
    off = run_node("A changing room.\n\nKate stands by the bench.\n\n"
                   "Kate takes off her shorts.\n\nKate walks to the door.\n\n"
                   "Kate is at the door.\nadd: her blue shorts\n\n"
                   "Kate opens the door.",
                   plan_only=True, character_memory=mem)[3]
    osh = [x for x in re.split(r"(?=\[Shot )", off) if x.strip()]
    check("a garment back with no dressing staged gets no clause",
          "off the body as the shot opens" not in off)
    check("...and is described as worn in that shot", "Her blue shorts." in osh[3])


def test_one_picture_two_people_is_reported():
    """Reported: a scene written for two people rendered the same woman twice, and
    the second character was not recognised as a separate person.

    A shot carrying a reference for one person and describing another who has none
    gives the model a photographed face and two faces to draw. A reference is the
    strongest identity signal in the prompt -- far stronger than "38, dark hair" --
    so the face that exists gets used twice.

    The node cannot fix it: nothing in the text outranks a photograph, and it is the
    model resolving a shot with more subjects than pictures. What it must do is SAY
    so, in terms of the fix, instead of leaving it to be discovered in a render."""
    print("\n=== one picture, two people ===")
    img = torch.rand(1, H, W, 3)
    P = ("A hotel room.\n\nKristy and Dan sit on the bed.\n\n"
         "Dan takes her hand.")
    one = run_node(P, plan_only=True, ref_image_1=img,
                   character_memory="Kristy: <Picture 1>, she, 26, blonde.\n"
                                    "Dan: he, 38, dark hair.")[2]
    check("the mismatch is reported",
          "carry a reference picture for one person" in one)
    check("...naming who has no picture", "describe Dan" in one)
    check("...and naming the shots", "shot(s) 1, 2" in one)
    check("...and saying what to do about it", "<Picture 2>" in one)
    # It must not cry wolf. Both tagged is the state it is asking for.
    both = run_node(P, plan_only=True, ref_image_1=img, ref_image_2=img,
                    character_memory="Kristy: <Picture 1>, she, 26, blonde.\n"
                                     "Dan: <Picture 2>, he, 38, dark hair.")[2]
    check("both tagged says nothing", "carry a reference picture for one" not in both)
    # ...nor when there is no picture in play at all, which is a different note.
    none = run_node(P, plan_only=True,
                    character_memory="Kristy: <Picture 1>, she, 26, blonde.\n"
                                     "Dan: he, 38, dark hair.")[2]
    check("no reference wired says nothing", "carry a reference picture for one" not in none)
    # ...nor on a scene with one person in it.
    solo = run_node("A room.\n\nKristy sits down.\n\nKristy stands up.",
                    plan_only=True, ref_image_1=img,
                    character_memory="Kristy: <Picture 1>, she, 26, blonde.")[2]
    check("a solo scene says nothing", "carry a reference picture for one" not in solo)


def test_a_collar_chained_to_a_wall_stays_on():
    """Reported: her collar was chained to a wall and she took it off.

    END TO END, because the unit halves both passed while the film was still wrong.
    limb_anchor could read a wall and held_part could say "neck", and neither was
    reached: the wall was missing from _ANCHOR_POINT so nothing anchored, and the
    part was computed from an item name that is dropped whenever the beat already
    says it. Testing the pieces is what let this ship -- so this asserts the SHOTS."""
    print("\n=== a collar chained to a wall stays on ===")
    P = ("A basement.\n\n"
         "The guard locks a steel collar around Ana's neck and chains it to the "
         "wall.\n\n"
         "Ana sits on the floor.\n\n"
         "Ana pulls at the chain.\n\n"
         "Ana stands and walks to the far side of the room.\n\n"
         "Ana looks up at the door.")
    script = run_node(P, plan_only=True,
                      character_memory="Ana: she, 28, grey shirt.\n"
                                       "Guard: he, 40, uniform.")[3]
    shots = [b.split("]", 1)[1] for b in script.split("[Shot ")[1:]]
    check("every shot after the first is tethered",
          all("at the wall" in s for s in shots[1:]),
          f"{[('at the wall' in s) for s in shots]}")
    check("...and it is the NECK being held, not the wrists",
          all("holding the neck fast at the wall" in s for s in shots[1:]),
          f"{[('holding the neck' in s) for s in shots]}")
    check("no shot claims the wrists", not any("the wrists" in s for s in shots))
    check("the hardware stays fastened in every later shot",
          all("closed and fastened" in s for s in shots[1:]))
    # The shot that STAGES the chaining says it in the author's own words and must
    # not get the clause back a second time.
    check("the staging shot is not told twice", "at the wall" not in shots[0])
    # A scene with no anchor in it must not sprout one.
    plain = run_node("A room.\n\nThe guard cuffs Ana's wrists.\n\n"
                     "Ana sits down.", plan_only=True,
                     character_memory="Ana: she, 28.\nGuard: he, 40.")[3]
    check("cuffs with no anchor named stay unanchored",
          "at the wall" not in plain and "holding the neck" not in plain)


COLLAR_PHRASINGS = [
    "The guard locks a steel collar around Ana's neck and chains it to the wall.",
    "The guard chains her collar to the wall.",
    "Her collar is chained to the wall.",
    "She is chained to the wall by her collar.",
    "A chain runs from her collar to the wall.",
    "The collar at her throat is padlocked to a ring in the wall.",
    "The guard puts a collar on her and clips the chain to a ring in the wall.",
    "Ana is collared and chained to the wall.",
    "The chain from her collar is bolted to the wall.",
    "Her collar is attached to a chain fixed to the wall.",
    "The guard fastens the collar and secures the chain to the wall.",
    "She wears a steel collar chained to the wall.",
    "The collar around her neck is locked to a wall ring.",
    "A short chain holds her collar to the wall.",
]


def test_every_way_of_chaining_a_collar_to_a_wall():
    """Reported twice. The first fix -- adding the wall to _ANCHOR_POINT -- was
    shipped without a render and did not fix it, because the anchor reader is only
    consulted once a restraint is LATCHED, and six of these fourteen latched nothing.

    restraint_present needs an ambiguous noun (chain, collar) plus a binding verb or
    a body part in the same sentence. _BINDING_VERB holds participles almost
    exclusively -- "chains", "clips" and "holds" are absent and "bolted" is not in it
    at all -- and "collared" matches no noun pattern, so the whole latch stayed down
    and every later shot was free of hardware entirely.

    Hardware fastened to something that does not move IS a restraint, whatever verb
    form said so. Tested on the SHOTS, across every phrasing, because the unit halves
    passed while the film was wrong the first time."""
    print("\n=== every way of chaining a collar to a wall ===")
    for p in COLLAR_PHRASINGS:
        script = run_node("A basement.\n\n" + p + "\n\nAna sits on the floor.\n\n"
                          "Ana stands and walks across the room.",
                          plan_only=True,
                          character_memory="Ana: she, 28.\nGuard: he, 40.")[3]
        last = script.split("[Shot ")[-1]
        check(f"tethered: {p[:44]!r}",
              "at the wall" in last or "at the ring" in last)
        check(f"...and fastened: {p[:40]!r}",
              "closed and fastened" in last or "tied and holding" in last)


def test_moving_towards_something_is_not_being_chained_to_it():
    """The other half of the same rule, and the reason it cannot simply be loosened.

    restraint_present now accepts hardware plus an anchor point, and limb_anchor
    accepts a chain that "runs" or "holds" to a fixed thing. Both would read ordinary
    movement as a fastening if the verb were not constrained -- "she runs to the
    wall" is the exact shape of "a chain runs to the wall"."""
    print("\n=== movement is not fastening ===")
    for t_ in ("she sinks to the floor", "he walks to the table",
               "she is dragged to the bed", "she falls to the floor",
               "he crosses to the window", "she runs to the wall",
               "they move to the bed", "she looks to the door",
               "he runs to the fence", "a chain-link fence runs along the wall",
               "the rope runs to the mast", "she leans back against the wall",
               "the dog runs to the gate",
               # HARDWARE BEING MOVED, not fastened. These are the ones the first
               # attempt let through: _FASTEN_TO was written as bare stems with an
               # optional suffix, and "chain", "rope", "clip" and "lock" are nouns
               # as well as verbs, so the pattern matched its own hardware and called
               # it a fastening. With the restraint gate leaning on the anchor, each
               # of these then latched a hold over hardware lying on the ground.
               "She drops the rope to the floor.", "He throws the chain to the floor.",
               "She kicks the cuffs to the wall.", "The rope lies next to the bed.",
               "He carries the chain to the table.", "The clip fell to the floor.",
               "The chains hang next to the door."):
        check(f"no anchor: {t_!r}", not S.limb_anchor(t_))
        check(f"no restraint: {t_!r}", not S.restraint_present(t_))


def test_two_restraints_put_on_together_both_survive():
    """END TO END, and the unit half would have passed while the film stayed wrong.

    Cuffs and a collar applied in ONE beat recorded only the collar, because
    hardware_named returns a single item and the worn list appended that string. So
    from shot 2 the handcuffs were absent from the prompt entirely -- not held, not
    even named -- and the model drew free hands. Reported as her breaking out of the
    handcuffs, which is the model rendering exactly what it was given."""
    print("\n=== two restraints, one beat ===")
    P = ("A basement cell, bare concrete.\n\n"
         "The guard handcuffs Ana's wrists behind her back and locks a steel "
         "collar around her neck, chained to the wall.\n\n"
         "Ana sits against the wall.\n\n"
         "Ana pulls at the chain.\n\n"
         # This beat calls them "the cuffs" where shot 1 said "handcuffs". Same
         # pair, and the worn list has to know that.
         "Ana twists her wrists in the cuffs.\n\n"
         "Ana looks at the door.")
    script = run_node(P, plan_only=True,
                      character_memory="Ana: she, 28.\nGuard: he, 40.")[3]
    shots = [b.split("]", 1)[1] for b in script.split("[Shot ")[1:]]
    check("the cuffs are named in shot 2", "cuffs" in shots[1], shots[1][:220])
    check("...and the collar with them", "collar" in shots[1])
    check("both stay fastened in every later shot",
          all("closed and fastened" in x for x in shots[1:]))
    # The merged anchor used to read "holding the neck behind the back, at the
    # wall" -- a neck behind a back.
    check("no neck is behind a back",
          not any("neck behind the back" in x for x in shots))
    # A limb position is a POSE sentence of its own now.
    check("the wrists take the limb position",
          all("wrists together at the small of the back" in x
              for x in shots[1:]))
    check("the collar takes the fixed point",
          all("fast at the wall" in x for x in shots[1:]))
    # "Ana looks at the door" must not relocate the camera into a door.
    check("a door does not move the shot",
          not any("in the door," in x for x in shots))
    # ONE pair of handcuffs, however the beats word it. An exact-match dedupe on the
    # worn list produced "The handcuffs, steel collar, chain and cuffs stay closed",
    # which reads as four things and invites the model to draw a spare set.
    held = [x.split(" stay closed")[0].split(" stays closed")[0] for x in shots[1:]]
    check("the cuffs are listed once", all(h.count("cuffs") <= 1 for h in held),
          f"{[h[-90:] for h in held]}")


def test_a_name_that_is_only_spoken_is_not_in_the_shot():
    """Reported: a character turned up in a scene they were not supposed to be in.

        Dana opens the door and calls out: "McKenna where are you?"

    put McKenna's whole sheet line into that shot -- "McKenna: she, 27, green
    dress" -- so the model was handed a full description of her and drew her
    standing there. She is the one person the beat says is NOT in the room.

    Calling for somebody is the commonest way to write their absence, and it was
    reading as their presence. Presence is decided on the beat with its spoken
    spans removed; a name staged OUTSIDE the quote still counts."""
    print("\n=== a spoken name is not a staged one ===")
    mem = "Dana: she, 31, blue jacket.\nMcKenna: she, 27, green dress."

    def described(beat):
        s = run_node("A hallway.\n\n" + beat + "\n\nDana walks on.",
                     plan_only=True, character_memory=mem)[3]
        first = s.split("[Shot ")[1].split("]", 1)[1]
        return {n for n in ("Dana", "McKenna") if n + ":" in first}

    check("called for, and absent",
          described('Dana opens the door and calls out: "McKenna where are you? '
                    'I am looking for you."') == {"Dana"})
    check("spoken about, and absent",
          described('Dana says: "McKenna took the keys."') == {"Dana"})
    check("...even named twice in the line",
          described('Dana shouts: "McKenna! McKenna, answer me."') == {"Dana"})
    # Staged AND spoken still counts: the staging half is what puts her there.
    check("staged as well as spoken",
          described('Dana turns to McKenna and says: "McKenna, wait."')
          == {"Dana", "McKenna"})
    check("staged with no speech at all",
          described("Dana and McKenna walk together.") == {"Dana", "McKenna"})
    # And the same rule where hardware is concerned. Asserted on the SHOT, not on
    # restrained_by_beat: that one deliberately returns everybody when it cannot
    # tell who is applying what -- a hold that fires needlessly is a wasted
    # sentence, one that fails to fire is hardware that stops being described --
    # so the guarantee lives in who gets DESCRIBED, not in that set.
    hw = run_node('A cell.\n\nDan says: "McKenna, put the cuffs on."'
                  '\n\nDan waits.', plan_only=True,
                  character_memory="Dan: he, 40, uniform.\n"
                                   "McKenna: she, 27, green dress.")[3]
    first = hw.split("[Shot ")[1].split("]", 1)[1]
    check("a spoken instruction describes nobody new",
          "McKenna:" not in first, first[:200])
    check("...and puts no hardware on her",
          "cuffs stay" not in first and "cuffs are" not in first, first[:200])


def test_a_verb_is_not_a_room():
    """Found in the same shot as the bug above: "McKenna steps out of the far
    room" read as the flight of STEPS, and "Ana walks into the room" produced
    "This shot is in the room, not the room the scene text names" -- a sentence
    that contradicts itself in eight words.

    first_place searches free text with no preposition in front of it, so it
    cannot tell a noun from a verb. The words that collide do not get to win."""
    print("\n=== a verb is not a room ===")
    for t_ in ("McKenna steps out of the far room.", "She steps forward.",
               "He lounges on the sofa.", "They study the map."):
        check(f"not a place: {t_!r}", S.first_place(t_) not in
              ("steps", "lounge", "study"), repr(S.first_place(t_)))
    check("a bare room names nowhere", not S.first_place("Ana walks into the room."))
    # ...but a qualified one is somewhere.
    check("the back room does", S.first_place("Ana walks into the back room.")
          == "back room")
    check("the far room does", S.first_place("She steps out of the far room.")
          == "far room")
    for room, want in (("A kitchen.", "kitchen"), ("A long hallway.", "hallway"),
                       ("A basement.", "basement"), ("The corridor is dark.",
                                                     "corridor")):
        check(f"{want} still reads", S.first_place(room) == want)


def test_entering_a_room_is_a_journey_not_a_cut():
    """Reported: "Dana continues to walk through the home and enters McKenna's
    bedroom" cut instantly from the living room to the bedroom.

    travel_in returned ('', '', '') for it, so the journey guard said nothing and
    the shot was free to cut. Two reasons, both in the destination pattern:

      - ENTERING is a verb, and the pattern held prepositions only. "into the
        bedroom" read; "enters the bedroom" did not, and entering is how arrival
        is usually written.
      - "McKenna's bedroom" is the ordinary way to say whose room it is, and the
        determiner list was the/her/his/their/a, which no possessive name matches.

    The origin was never the problem: travel_anchor already falls back to the room
    the film is in. It just never learned there was a destination to go to."""
    print("\n=== entering a room is a journey ===")
    for beat, want_to in (
            ("Dana continues to walk through the home and enters McKenna's "
             "bedroom.", "bedroom"),
            ("Dana walks through the house and enters the bedroom.", "bedroom"),
            ("Dana enters McKenna's bedroom.", "bedroom"),
            ("Dana walks down the hall and steps into the kitchen.", "kitchen"),
            ("Dana walks from the living room into the bedroom.", "bedroom")):
        _f, _v, to = S.travel_in(beat)
        check(f"destination read: {beat[:44]!r} -> {to!r}", to == want_to)
    # A named place with no movement is still not a journey.
    check("looking somewhere is not travel",
          S.travel_in("She looks to the bedroom.") == ("", "", ""))
    check("...nor is standing in a room",
          S.travel_in("Dana stands in the bedroom.") == ("", "", ""))

    # END TO END: the walk is one move, and every shot after it is in the new room.
    P = ("A living room in a suburban home.\n\n"
         "Dana stands by the sofa.\n\n"
         "Dana continues to walk through the home and enters McKenna's bedroom.\n\n"
         "Dana looks around the bedroom.")
    script = run_node(P, plan_only=True,
                      character_memory="Dana: she, 31, blue jacket.")[3]
    sh = [b.split("]", 1)[1] for b in script.split("[Shot ")[1:]]
    check("the walk is one continuous move",
          "played out on screen" in sh[1], sh[1][:220])
    check("...starting where she was", "opens in the living room" in sh[1])
    check("...and ending where she goes", "arrives in the bedroom" in sh[1])
    # The room the film is in follows her, or the scene paragraph keeps saying
    # living room over a shot standing in the bedroom.
    check("the next shot is in the bedroom",
          "takes place in the bedroom" in sh[2], sh[2][:220])
    # The shot before the journey is untouched.
    check("the shot before it says nothing",
          "played out on screen" not in sh[0] and "takes place in the" not in sh[0])


NEGATION = re.compile(
    r"\b(?:not|never|no|none|nobody|nothing|without|cannot|can't|won't|"
    r"doesn't|isn't|aren't)\b", re.I)


def test_no_guard_sentence_carries_a_negation():
    """At cfg 1 H3 evaluates NO negative prompt, so a negation in the positive
    prompt is just the thing it names. This file says so in eleven places -- "no
    leggings" would be read as leggings -- and then shipped, in the clause
    written to prevent a cut:

        one continuous move, not a cut

    ...which puts the word CUT in the prompt of the only shot that must not cut.
    Reported twice as an instant cut across a house, the second time after the
    destination reader was fixed and the clause was demonstrably in the prompt.
    The wording was the bug, and no amount of reading the shot text catches it,
    because the clause looks exactly like what you meant.

    Two more went with it: "in the {room}, NOT the room the scene text names"
    pointed at the room being overridden, and "fully removed and NO LONGER on the
    body" names the body it is clearing.

    This sweeps what the node actually SENDS. A negation in a comment is fine and
    there are hundreds; one in an emitted sentence is a bug by construction."""
    print("\n=== no guard sentence carries a negation ===")
    mem = ("Ana: she, 28, blue coat, wool scarf, grey shirt, handcuffs.\n"
           "Dan: he, 40, uniform.")
    scenes = [
        "A living room.\n\nAna stands by the sofa.\n\n"
        "Ana walks into the bedroom.\n\nAna looks around.",
        "A cell.\n\nDan handcuffs Ana's wrists behind her back and chains her "
        "to the wall.\n\nAna pulls at the chain.\n\nDan walks out.",
        "A hallway.\n\nAna takes off her coat and hangs it up.\n\n"
        'Ana sits down.\n\nAna says: "Where is he?"',
        "A yard.\n\nRain on the corrugated roof.\n\nAna runs for the gate."
        "\n\nAna falls.",
        "A room.\n\nAna pulls her scarf aside.\n\nAna puts the scarf back on."
        "\n\nAna stands up.",
    ]
    # A scene may bring its own sheet, because the shared one cannot cover
    # every shape. This one is an UNDERGARMENT under an outer layer with both
    # coming off, and it went unswept for two commits: the shared memory has
    # no jeans and no belt, so the beats matched nothing and the agentless
    # branch of the removal clause was never reached. It held a live negation
    # -- "fully removed and no longer on the body", naming the body it is
    # clearing. Three copies of that sentence exist and the fix caught two.
    scenes = [(sc, mem) for sc in scenes] + [
        ("A room.\n\nAna stands.\n\nAna takes off her jeans.\n\n"
         "Ana takes off the chastity belt.\n\nAna sits.",
         "Ana: she, 28, blue jeans, a steel chastity belt."),
    ]
    bad = {}
    for sc, sheet in scenes:
        script = run_node(sc, plan_only=True, character_memory=sheet)[3]
        beats = set(sc.split("\n\n"))
        for block in script.split("[Shot ")[1:]:
            body = block.split("]", 1)[1]
            for sent in re.split(r"(?<=[.!?])\s+", body):
                sent = sent.strip()
                # The author's own beat and sheet go through verbatim and are
                # theirs to phrase; only what the NODE writes is in scope.
                if not sent or any(sent in b for b in beats):
                    continue
                if NEGATION.search(sent) and "<d>" not in sent:
                    bad[sent[:90]] = True
    check("every guard sentence is positively phrased: "
          + ("; ".join(sorted(bad)[:2]) or "all clear"), not bad)


def test_hardware_waits_for_the_beat_that_puts_it_on():
    """Reported: a handcuff on McKenna's arm before she is handcuffed.

    A sheet says WHAT somebody has and never WHEN. "McKenna: she, 27, green
    dress, handcuffs" beside a script that cuffs her in beat 3 put them on from
    shot 1 -- twice over. The engine declared them worn, so shot 1 went out
    saying "The handcuffs stay closed and fastened AS THEY WERE PUT ON" two shots
    before anybody put them on; and the sheet itself, which this node re-stamps
    into every shot, listed handcuffs on her from the opening frame. A described
    item is a drawn item.

    Where the script stages the fastening, the script owns the moment. The sheet
    still supplies the description -- it just does not start the clock."""
    print("\n=== hardware waits for its beat ===")
    P = ("A kitchen.\n\n"
         "McKenna stands by the counter.\n\n"
         "McKenna turns to face Dana.\n\n"
         "Dana handcuffs McKenna's wrists behind her back.\n\n"
         "McKenna pulls at the cuffs.")
    mem = "Dana: she, 31, blue jacket.\nMcKenna: she, 27, green dress, handcuffs."
    sh = [b.split("]", 1)[1] for b in
          run_node(P, plan_only=True, character_memory=mem)[3].split("[Shot ")[1:]]
    # THE SHEET IS NOT EDITED. This asserted the cuffs were absent from shots 1
    # and 2 entirely, which was achieved by scrubbing them out of the character
    # memory -- the wrong lever, and the user said so plainly: "Stop removing
    # items from the character memory!" It was also destructive beyond the item,
    # because scrub_removed drops the whole comma-separated entry and took the
    # person's line with it.
    #
    # What the node owes is narrower and is what this checks now: it must not
    # ASSERT the hardware is fastened before the beat that fastens it. The
    # author's own words stay exactly as written, and the clash is reported.
    check("shot 1 keeps the author's sheet line",
          "handcuffs" in sh[0], sh[0][:200])
    check("...but claims nothing is fastened yet",
          "closed and fastened" not in sh[0], sh[0][:200])
    check("...and shot 2 likewise",
          "handcuffs" in sh[1] and "closed and fastened" not in sh[1], sh[1][:200])
    check("the applying shot has them", "cuff" in sh[2].lower())
    check("...with both ends of the change",
          "off the body at the first frame" in sh[2] or "open and off" in sh[2],
          sh[2][:240])
    check("and they stay after", "cuff" in sh[3].lower())
    check("...held, not re-applied",
          "closed and fastened" in sh[3] and "first frame" not in sh[3], sh[3][:240])

    # A sheet with hardware the script NEVER stages is somebody who walks in
    # already wearing it. That still has to hold from shot 1 -- hardware nobody
    # mentions is hardware the model stops drawing.
    worn = [b.split("]", 1)[1] for b in
            run_node("A cell.\n\nKate sits on the bunk.\n\nKate waits.",
                     plan_only=True,
                     character_memory="Kate: she, 30, handcuffs."
                     )[3].split("[Shot ")[1:]]
    # ...and the disagreement is REPORTED, not silently resolved.
    info = run_node(P, plan_only=True,
                    character_memory="Dana: she, 31, blue jacket.\n"
                                     "McKenna: she, 27, green dress, handcuffs.")[2]
    check("the sheet/script clash is reported",
          "already lists it as worn" in info, info[-400:])
    check("...and says the wording is untouched",
          "Your wording is never edited" in info)

    check("already-worn hardware holds from shot 1",
          "handcuffs" in worn[0] and "closed and fastened" in worn[0], worn[0][:220])
    check("...and keeps holding", "closed and fastened" in worn[1])


def test_the_applying_shot_says_where_the_limbs_finish():
    """Reported: "the handcuffs broke in the next beat -- her arms were at her
    sides and not handcuffed behind her back".

    Nothing broke. The going-on clause said what the HARDWARE does across the
    shot -- open at the first frame, closed by the last -- and nothing about the
    body, and the limb position was deliberately withheld there on the grounds
    that the author's own words sit right beside it. They do, but they describe
    the ACT, and the next shot does not inherit the act. It inherits the last
    frame. So the cuffs could close with the arms wherever they happened to be,
    and the following shot opened on a picture of somebody with their arms at
    their sides while its text insisted the wrists were behind the back.

    Text loses to an inherited picture, every time. The frame the next shot
    started from never had them behind her back."""
    print("\n=== the applying shot says where the limbs finish ===")
    P = ("A kitchen.\n\nMcKenna stands by the counter.\n\n"
         "Dana handcuffs McKenna's wrists behind her back.\n\n"
         "McKenna pulls at the cuffs.\n\nMcKenna turns around.")
    sh = [b.split("]", 1)[1] for b in
          run_node(P, plan_only=True,
                   character_memory="Dana: she, 31.\nMcKenna: she, 27, green dress."
                   )[3].split("[Shot ")[1:]]
    check("the applying shot names the end position",
          "By the last frame the wrists are behind the back" in sh[1], sh[1][:300])
    check("...as well as the hardware's two ends",
          "open and off the body at the first frame" in sh[1])
    check("the next shot still holds the position",
          "wrists together at the small of the back" in sh[2], sh[2][:240])
    check("...and the one after that",
          "wrists together at the small of the back" in sh[3])
    # A shot that stages no fastening must not claim a last-frame position.
    check("an ordinary shot says nothing about it",
          "By the last frame" not in sh[0], sh[0][:200])
    # A collar has no limb position, and must not be handed one.
    neck = [b.split("]", 1)[1] for b in
            run_node("A cell.\n\nDan locks a steel collar around Ana's neck.\n\n"
                     "Ana sits down.", plan_only=True,
                     character_memory="Dan: he, 40.\nAna: she, 28."
                     )[3].split("[Shot ")[1:]]
    check("a collar gets no limb position",
          "By the last frame the wrists" not in neck[0], neck[0][:240])


def test_a_collar_in_the_sheet_is_held_like_hardware():
    """Reported: the collar was missing from her neck when she was seen in her
    room. It was not the room -- the collar never latched at all.

    restraint_present needs an ambiguous noun plus a binding verb or a body part
    in the SAME line, and a sheet entry reading "green dress, steel collar" has
    neither. So no hold ever fired for it in any shot, and hardware nobody holds
    is hardware the model drops -- it just took a change of room for it to show.

    Bare "collar" stays ambiguous, because a shirt has one. The MATERIAL settles
    it: a shirt's collar is stiff or starched, never steel, and it does not
    lock."""
    print("\n=== a collar in the sheet is held like hardware ===")
    for line in ("McKenna: she, 27, green dress, steel collar.",
                 "McKenna: 27, a leather collar.",
                 "McKenna: 27, a locked collar.",
                 "McKenna: 27, a collar and a padlock.",
                 "McKenna: she, 27, a collar at her throat."):
        check(f"a restraint: {line[:44]!r}", S.restraint_present(line))
    for line in ("McKenna: 27, white shirt with a stiff collar.",
                 "Dan: 40, uniform with a starched collar.",
                 "Dan: 40, a jacket with a fur collar.",
                 "McKenna: she, 27, green dress."):
        check(f"clothing, not hardware: {line[:40]!r}",
              not S.restraint_present(line))
    # END TO END, and across a change of room, which is where it was noticed.
    sh = [b.split("]", 1)[1] for b in
          run_node("A living room.\n\nMcKenna stands by the sofa.\n\n"
                   "McKenna walks into her bedroom.\n\n"
                   "McKenna sits on the bed.", plan_only=True,
                   character_memory="McKenna: she, 27, green dress, steel collar."
                   )[3].split("[Shot ")[1:]]
    for i, s in enumerate(sh, 1):
        check(f"shot {i} names the collar", "collar" in s.lower(), s[:180])
        check(f"shot {i} holds it", "closed and fastened" in s, s[:180])


def test_an_undergarment_keeps_its_words_and_waits_for_its_picture():
    """The two halves of layering an under-garment, which are NOT the same
    question and were got wrong in both directions before this settled.

      WORDS   stay. Deleting the author's item is what put their own wording out
              of the prompt, reported three times, the last a chastity belt with
              a reference attached to it.
      PICTURE waits. A reference is an instruction to reproduce an image and at
              near-clean ref_noise_aug it outweighs any sentence about what is on
              top of what -- so a picture of the belt, in a shot where the belt is
              under the jeans, draws the belt. Reported as it poking through the
              clothes, one commit after the picture was made to travel.

    The tag goes with the picture rather than the image merely being withheld,
    because refs are routed BY the tags and a tag naming a picture the shot does
    not carry is its own bug.

    Nothing here comes from a beat: the sheet is the only place the belt is
    named, so this fails if the sheet stops carrying it."""
    print("\n=== an undergarment keeps its words, its picture waits ===")
    mem = "Ana: <Picture 1>, she, 28, blue jeans, a chastity belt <Picture 2>."
    P = ("A room.\n\nAna stands by the window.\n\n"
         "Ana takes off her jeans.\n\nAna waits.")
    img = torch.rand(1, H, W, 3)
    sh = [b.split("]", 1)[1] for b in
          run_node(P, plan_only=True, character_memory=mem,
                   ref_image_1=img, ref_image_2=img)[3].split("[Shot ")[1:]]
    covered, bare = sh[0], sh[2]
    # On the SHEET LINE, not merely somewhere in the shot: the under-clause names
    # the belt too, so checking the whole shot passed even with the sheet scrubbed
    # -- green for the wrong reason, and the wrong reason was the actual report.
    sheet_line = [l for l in covered.split(chr(10)) if l.strip().startswith("Ana:")]
    check("the character memory line survives", bool(sheet_line), covered[:200])
    check("...with the rest of it intact",
          bool(sheet_line) and "blue jeans" in sheet_line[0].lower()
          and "28" in sheet_line[0], (sheet_line[0] if sheet_line else covered)[:200])
    check("...and only the belt held back",
          bool(sheet_line) and "chastity belt" not in sheet_line[0].lower(),
          (sheet_line[0] if sheet_line else covered)[:200])
    check("...and the jeans are described as covering it",
          "cover the hips and waist completely" in covered.lower(),
          covered[:260])
    check("...and describes the jeans as covering it",
          "whole, opaque and unbroken" in covered.lower(), covered[:260])
    check("...while its picture waits for the cover to move",
          "<Picture 2>" not in covered, covered[:200])
    check("...and the person's own picture does not",
          "<Picture 1>" in covered, covered[:200])
    check("the uncovered shot has both", "chastity belt" in bare.lower()
          and "<Picture 2>" in bare, bare[:220])
    check("...and stops calling it covered",
          "whole, opaque and unbroken" not in bare.lower(), bare[:220])


def test_an_under_layer_belongs_to_somebody():
    """Reported from a real shot: a scene with Dana in jeans and McKenna in a
    skirt and a chastity belt put

        The chastity belt is worn under the skirt...

    into a shot describing only DANA, who has neither. The belt does not go on
    Dana. A described garment is a drawn garment and it is drawn on whoever is in
    the frame.

    implied_layers was read off the WHOLE sheet at once, so it paired a garment
    with a cover and lost whose they were. A sheet line is one person, so the
    layers are read line by line and the owner kept. Per line the same sheet
    answers correctly -- Dana {}, McKenna {chastity belt: skirt} -- which is what
    made this findable."""
    print("\n=== an under-layer belongs to somebody ===")
    mem = ("Dana: she, 35, black t-shirt, blue jeans.\n"
           "McKenna: she, 27, a skirt, a chastity belt.")
    P = ("A home.\n\nA camera tracks Dana moving around the home.\n\n"
         "McKenna steps out of the back room.\n\nDana looks at her.")
    sh = [b.split("]", 1)[1] for b in
          run_node(P, plan_only=True, character_memory=mem)[3].split("[Shot ")[1:]]
    check("Dana's shot says nothing about a belt",
          "chastity belt" not in sh[0].lower(), sh[0][:220])
    check("...nor about a skirt she does not own",
          "skirt" not in sh[0].lower(), sh[0][:220])
    check("McKenna's shot describes the skirt covering it",
          "skirt covers the hips and waist" in sh[1].lower(), sh[1][:240])
    check("...and Dana's later shot is clean again",
          "chastity belt" not in sh[2].lower(), sh[2][:220])
    # Both in one shot: the belt has to say WHOSE, or it lands on either woman.
    two = run_node("A home.\n\nDana and McKenna stand in the hall.",
                   plan_only=True, character_memory=mem)[3]
    body = two.split("]", 1)[1]
    # Attribution moved to the cover, which is the garment now named.
    check("with both described, the cover is attributed",
          "McKenna's skirt covers" in body, body[:300])
    check("...and the name is not mangled", "Mckenna" not in body, body[:300])
    # One person, and no attribution needed.
    solo = run_node("A home.\n\nMcKenna waits.", plan_only=True,
                    character_memory="McKenna: she, 27, a skirt, a chastity belt.")[3]
    solo_body = solo.split("]", 1)[1]
    check("alone, it is just the skirt",
          "The skirt covers the hips and waist completely" in solo_body,
          solo_body[:240])


def test_the_picture_arrives_when_the_cover_moves():
    """The rule, given directly: "when the skirt has been lifted up to show the
    chastity belt, that's when it should be shown, or when the clothing on top has
    been removed."

    Two things stopped that. LIFTING was not a displacement at all -- lift, raise,
    hoist, gather and bunch were missing from the verb list, so the beat that
    uncovers the belt moved nothing. And the verbs that carry their own direction
    were thrown away for not stating one: "lifts her skirt" says which way by
    saying lift, and the pattern wanted an "up" that nobody writes.

    Then an off-by-one on top of it. The layering read `displaced` before the beat
    had been added to it, so even a recognised displacement only took effect on
    the NEXT shot -- the belt came out from under the skirt one shot late."""
    print("\n=== the picture arrives when the cover moves ===")
    mem = "McKenna: <Picture 1>, she, 27, a skirt, a chastity belt <Picture 2>."
    FACE, BELT = torch.rand(1, H, W, 3), torch.rand(1, H, W, 3)

    def refs(prompt):
        rows, ob = [], S.build_conditioning

        def spy(clip, vae, av, p, *a, **k):
            rows.append(len(k.get("refs") or []))
            return ob(clip, vae, av, p, *a, **k)
        S.build_conditioning = spy
        try:
            run_node(prompt, character_memory=mem, ref_image_1=FACE,
                     ref_image_2=BELT)
        finally:
            S.build_conditioning = ob
        return rows

    for how, beat in (("lifted", "McKenna lifts her skirt."),
                      ("raised", "McKenna raises her skirt."),
                      ("held up", "McKenna holds her skirt up."),
                      ("pulled aside", "McKenna pulls her skirt aside."),
                      ("removed", "McKenna takes off her skirt.")):
        got = refs("A home.\n\nMcKenna waits.\n\n" + beat + "\n\nMcKenna waits.")
        check(f"{how}: face only until the cover moves", got[0] == 1, str(got))
        check(f"{how}: and the belt from THAT shot, not the next",
              got[1] == 2, str(got))
    # Nothing touches it: the picture never arrives, and the words are there all
    # along -- that is the half that must never wait.
    quiet = refs("A home.\n\nMcKenna waits.\n\nMcKenna sits.\n\nMcKenna stands.")
    check("untouched: the picture never arrives", quiet == [1, 1, 1], str(quiet))
    words = run_node("A home.\n\nMcKenna waits.\n\nMcKenna sits.", plan_only=True,
                     character_memory=mem)[3]
    check("...and the belt waits, in every covered shot",
          words.lower().count("chastity belt") == 0, words[:200])
    # A lift is not everything that moves: a chin is not a garment.
    check("lifting a chin displaces nothing",
          not S.displaced_garments("McKenna lifts her chin.",
                                   "McKenna: she, 27, a skirt, a chastity belt."))


def test_letting_the_cover_fall_puts_it_back():
    """A lifted skirt has to come back down, or it stays lifted for the rest of the
    film and whatever was under it stays on show.

    The only restores recognised were pull/tug/hitch/hike/yank/push with a PRONOUN
    and a direction -- "pulls them back up". What actually gets written is the
    opposite: a lifted skirt is LET FALL, DROPPED, LOWERED, SMOOTHED DOWN,
    STRAIGHTENED or LET GO of, and none of those carries a direction word. Nor
    could any of them name the garment; "lets the skirt fall" matched nothing."""
    print("\n=== letting the cover fall puts it back ===")
    sheet = "McKenna: she, 27, a long grey skirt, a chastity belt."
    for beat in ("McKenna lets the skirt fall.", "McKenna lets her skirt drop.",
                 "McKenna lets go of the skirt.", "McKenna smooths her skirt down.",
                 "McKenna lowers her skirt.", "McKenna straightens her skirt.",
                 "McKenna drops her skirt.", "McKenna puts her skirt back."):
        got = S.restored_garments(beat, sheet)
        # The SHEET's name comes back, not the beat's -- a displacement is stored
        # under "long grey skirt" and a restore keyed on "skirt" has to clear it.
        check(f"restores: {beat[:38]!r}", got == ["long grey skirt"], str(got))
    for beat in ("McKenna lifts her skirt.", "McKenna lets the door close.",
                 "McKenna waits."):
        check(f"not a restore: {beat[:34]!r}",
              not S.restored_garments(beat, sheet), str(S.restored_garments(beat, sheet)))
    # Pronoun forms, which name nothing.
    for beat in ("McKenna lets it fall.", "McKenna lets them fall.",
                 "McKenna pulls them back up.", "McKenna covers herself up."):
        check(f"pronoun restore: {beat[:34]!r}", S.puts_it_back(beat))
    check("...and an ordinary beat is not one", not S.puts_it_back("McKenna waits."))

    # END TO END: the picture follows the cover, both ways, on the shot that moves it.
    mem = "McKenna: <Picture 1>, she, 27, a long grey skirt, a chastity belt <Picture 2>."
    FACE, BELT = torch.rand(1, H, W, 3), torch.rand(1, H, W, 3)

    def refs(prompt):
        rows, ob = [], S.build_conditioning

        def spy(clip, vae, av, p, *a, **k):
            rows.append(len(k.get("refs") or []))
            return ob(clip, vae, av, p, *a, **k)
        S.build_conditioning = spy
        try:
            run_node(prompt, character_memory=mem, ref_image_1=FACE, ref_image_2=BELT)
        finally:
            S.build_conditioning = ob
        return rows

    got = refs("A home.\n\nMcKenna waits.\n\nMcKenna lifts her skirt.\n\n"
               "McKenna lets it fall.\n\nMcKenna waits.")
    check(f"lifted, then covered again: {got}", got == [1, 2, 1, 1], str(got))
    stays = refs("A home.\n\nMcKenna waits.\n\nMcKenna lifts her skirt.\n\n"
                 "McKenna waits.\n\nMcKenna waits.")
    check(f"...and stays shown if nothing puts it back: {stays}",
          stays == [1, 2, 2, 2], str(stays))
    named = refs("A home.\n\nMcKenna waits.\n\nMcKenna lifts her skirt.\n\n"
                 "McKenna smooths her skirt down.\n\nMcKenna waits.")
    check(f"...and a named restore works too: {named}", named == [1, 2, 1, 1],
          str(named))


def test_somebody_else_can_lift_your_skirt():
    r"""Reported from a real shot: "Dana lifts up McKenna's skirt to check the
    chastity belt" left the belt covered and its picture withheld.

    The rule was right and the parse was not. _DISPLACE took a determiner of
    the/her/his/their/a/an, which matches no POSSESSIVE NAME, and the apostrophe
    is not in the [\w\- ] the garment itself is read with -- so the whole match
    failed and the lift was invisible. One person lifting another's clothing is
    the ordinary way this gets written.

    Same fault as "enters McKenna's bedroom" on travel, in a different reader, so
    both now use the same _DET_POSS rather than a second copy that can drift."""
    print("\n=== somebody else can lift your skirt ===")
    sheet = ("McKenna: she, 22, a chastity belt, a mini-skirt.\n"
             "Dana: she, 35, black t-shirt, blue jeans.")
    for beat in ("Dana lifts up McKenna's skirt to check the chastity belt.",
                 "Dana lifts McKenna's skirt.",
                 "Dana pulls McKenna's skirt aside."):
        got = S.displaced_garments(beat, sheet)
        check(f"displaced: {beat[:44]!r}", bool(got), str(got))
    for beat in ("Dana lets McKenna's skirt fall.",
                 "Dana smooths McKenna's skirt down."):
        check(f"restored: {beat[:44]!r}",
              bool(S.restored_garments(beat, sheet)),
              str(S.restored_garments(beat, sheet)))
    # A possessive is not a licence to match anything: a chin is still not a
    # garment, and a name with no garment after it displaces nothing.
    check("a chin is not displaced",
          not S.displaced_garments("Dana lifts McKenna's chin.", sheet))

    # END TO END on the reported shot: the belt shows while the skirt is up, and
    # goes back under when it falls.
    mem = ("McKenna: she, <Picture 1>, 22, a chastity belt <Picture 2>, "
           "a mini-skirt.\nDana: she, 35, black t-shirt, blue jeans.")
    FACE, BELT = torch.rand(1, H, W, 3), torch.rand(1, H, W, 3)
    rows, ob = [], S.build_conditioning

    def spy(clip, vae, av, p, *a, **k):
        rows.append((p, len(k.get("refs") or [])))
        return ob(clip, vae, av, p, *a, **k)
    S.build_conditioning = spy
    try:
        run_node("A home.\n\nDana lifts up McKenna's skirt to check the "
                 "chastity belt.\n\nMcKenna waits.", character_memory=mem,
                 ref_image_1=FACE, ref_image_2=BELT)
    finally:
        S.build_conditioning = ob
    check("the lifting shot carries the belt's picture", rows[0][1] == 2,
          str([n for _, n in rows]))
    check("...and does not call the skirt opaque over it",
          "whole, opaque" not in rows[0][0], rows[0][0][-200:])
    # ...and it STAYS up, because nothing in the next beat puts it back. That is
    # the documented behaviour of a displacement and the reason restore verbs
    # exist at all -- not an oversight to assert away.
    check("...and stays up while nothing puts it back",
          rows[1][1] == 2 and "whole, opaque" not in rows[1][0],
          str([n for _, n in rows]))


def test_timing_report():
    print("\n=== the timing breakdown ===")
    P = "A room.\n\nOne.\n\nTwo."
    info = run_node(P)[2]
    for want in ("sampling", "decode", "per shot"):
        check(f"info reports {want}", want in info)
    check("...with a wall-clock total", "rendered" in info and "s --" in info)
    # plan_only does no work, so it must not claim any timings.
    check("plan_only reports no timings", "per shot" not in run_node(P, plan_only=True)[2])


def test_upscale_paths():
    print("\n=== upscale wiring ===")
    # Neither pack is installed here, so both fall back. What is under test is that
    # the calls are wired correctly and a render still completes -- the failure mode
    # that shipped twice was a signature mismatch, not a bad upscale.
    P = "A room.\n\nOne.\n\nTwo."
    try:
        imgs, _a, info, _s, _p, total, _sh, _sec = run_node(
            P, latent_upscale="off", upscale="lanczos", upscale_target_short_edge=128)
        ok, detail = imgs.ndim == 4 and imgs.shape[0] == total, str(tuple(imgs.shape))
    except Exception as e:
        ok, detail = False, f"{type(e).__name__}: {e}"
    check("a pixel upscale pass runs and returns frames", ok, detail)
    try:
        imgs2 = run_node(P, latent_upscale="off")[0]
        check("no upscale leaves the frames alone", imgs2.ndim == 4)
    except Exception as e:
        check("no upscale leaves the frames alone", False, f"{type(e).__name__}: {e}")
    # The latent handoff must come from the SAMPLED latent, never the upscaled one,
    # or the chain inherits the upscaler's guess and it compounds.
    src = open(os.path.join(_HERE, "sampler.py"), encoding="utf-8").read()
    # The chain must not inherit the upscaler's reinterpretation: the shot's own
    # frames stay upscaled, but the handoff is decoded from the SAMPLED latent.
    # Eleven boundaries of upscaled-then-downscaled frames compounds into colour
    # cast and mush.
    check("the handoff comes from the pre-upscale latent",
          "pre_up[:, :, -n:]" in src and "hand_src = tail" in src)
    check("...and is clamped before it is re-encoded",
          "clamp(0.0, 1.0)" in src)


def _shots_of(result):
    return [sh.split("] ", 1)[1] if "] " in sh else sh for sh in result[3].split("\n---\n")]


def test_the_opening_does_not_name_people_a_shot_leaves_out():
    """An opening that names people rode into every shot unscoped: "Maya and Owen wait
    in a train station." headed a shot the node had cut down to Owen alone, so the
    model was told two people stand there and given one person to draw."""
    print("\n=== the opening paragraph names nobody a shot leaves out ===")
    memory = "Maya: she, 38, green sweater.\nOwen: he, 42, blue shirt."
    shots = _shots_of(run_node("Maya and Owen wait in a train station.\n\n"
                               "Owen checks the departure board.\n\n"
                               "Maya reads a newspaper on a bench.",
                               character_memory=memory, plan_only=True))
    check("two solo shots", len(shots) == 2)
    check("Owen's shot does not name Maya", "Maya" not in shots[0])
    check("Maya's shot does not name Owen", "Owen" not in shots[1])
    check("...and both keep the station", all("train station" in sh for sh in shots))
    # A pronoun after a cut sentence goes with it.
    shots = _shots_of(run_node("Maya sits at her desk in an office. She types a report.\n\n"
                               "Owen knocks and walks in.",
                               character_memory=memory, plan_only=True))
    check("no stray 'She' about an absent Maya", "She types" not in shots[-1]
          and "Maya" not in shots[-1] and "office" in shots[-1])
    # The anchor is scoped the same way.
    shots = _shots_of(run_node("A kitchen.\n\nOwen chops onions.", character_memory=memory,
                               anchor="Maya and Owen are in a bright kitchen.", plan_only=True))
    check("an anchor naming an absent person does not reach his solo shot",
          "Maya" not in shots[-1] and "kitchen" in shots[-1])
    # Control: everyone the opening names is in the shot, so it is untouched.
    shots = _shots_of(run_node("Maya and Owen wait in a train station.\n\n"
                               "They look at the board together.",
                               character_memory=memory, plan_only=True))
    check("an opening about the whole cast is kept word for word",
          "Maya and Owen wait in a train station." in shots[0])


def test_a_sheet_written_first_is_a_sheet():
    """Opening a script with who is in it made the sheet the SCENE: every person
    described in every shot as prose, with no scoping, count or mouth guard."""
    print("\n=== a character sheet written first is a sheet, not the scene ===")
    shots = _shots_of(run_node("Maya: she, 38, green sweater.\nOwen: he, 42, blue shirt.\n\n"
                               "A park.\n\nMaya sits on a bench.\n\nOwen feeds the ducks.",
                               plan_only=True))
    check("the sheet is not a shot of its own", len(shots) == 2)
    check("Owen's shot does not describe Maya", "Maya:" not in shots[1] and "Owen:" in shots[1])
    check("...and counts one person", "There is one person in the shot" in shots[1])
    check("the park is still the scene", all(sh.startswith("A park.") for sh in shots))
    # Control: a heading with a colon is not a person and stays the scene.
    shots = _shots_of(run_node("Interior: a kitchen at night.\n\nMaya: she, 38, green sweater.\n\n"
                               "Maya makes tea.", plan_only=True))
    check("'Interior: ...' stays the scene", shots[0].startswith("Interior: a kitchen at night."))


def test_a_two_word_name_makes_a_sheet():
    """"Mistress Vale:" did not match the sheet-line pattern, so the sheet rendered as
    a shot and every later shot described nobody."""
    print("\n=== a sheet entry with a two-word name is still a sheet ===")
    shots = _shots_of(run_node("A library.\n\nMistress Vale: she, 45, black dress.\n"
                               "Owen: he, 42, blue shirt.\n\nMistress Vale shelves a book.\n\n"
                               "Owen reads at a table.", plan_only=True))
    check("no shot is spent on the sheet", len(shots) == 2)
    check("Mistress Vale is described in her shot", "Mistress Vale: she, 45, black dress." in shots[0])
    check("Owen is described in his", "Owen: he, 42, blue shirt." in shots[1])
    check("'Both women: tired' is still not a sheet line",
          not S.is_character_sheet("Both women: tired"))


def test_one_person_under_two_names_is_described_once():
    """"Maya Brooks" in character_memory and "Maya:" in the prompt were two keys, so
    one woman went into every shot twice, in two different outfits."""
    print("\n=== one person under two forms of her name is described once ===")
    result = run_node("Maya: she, 38, red coat.\n\nA park.\n\nMaya sits on a bench.",
                      character_memory="Maya Brooks: she, 38, green sweater.", plan_only=True)
    shots = _shots_of(result)
    check("one entry for her", all(sh.count("she, 38") == 1 for sh in shots))
    check("character_memory's entry is the one kept", all("green sweater" in sh and "red coat" not in sh
                                                          for sh in shots))
    check("...and the author is told", any(isinstance(part, str) and "described more than once" in part
                                           for part in result))
    # Control: a shared word with a different age is somebody else.
    shots = _shots_of(run_node("A garden.\n\nMay: she, 24, yellow dress.\n\nMay and Aunt May pick apples.",
                               character_memory="Aunt May: she, 60, grey cardigan.", plan_only=True))
    check("May and Aunt May stay two people", "May: she, 24" in shots[-1]
          and "Aunt May: she, 60" in shots[-1] and "two people" in shots[-1])


def test_a_bare_chest_belongs_to_whoever_undressed():
    """Two entries listing a sweater: the first entry was taken as the wearer, so Lena
    taking hers off put the bare chest on Maya, still listed in her own sweater."""
    print("\n=== a bare region goes to the person who undressed, not the first entry ===")
    shots = _shots_of(run_node("A bedroom.\n\nMaya and Lena talk.\n\nLena takes off her sweater.",
                               character_memory="Maya: she, 38, green sweater.\n"
                                                "Lena: she, 30, yellow sweater.", plan_only=True))
    last = shots[-1]
    check("the bare chest is Lena's", "Lena's chest" in last)
    check("...not Maya's", "Maya's chest" not in last)
    check("Maya keeps her sweater", "Maya: she, 38, green sweater." in last)


def main():
    test_independent_adult_arm_actions()
    test_plan()
    test_render()
    test_a_cut_keeps_its_own_first_frame()
    test_a_room_change_with_no_walk_is_a_cut()
    test_extras_are_not_forbidden_by_the_body_count()
    test_a_walk_into_an_unlisted_room_is_still_walked()
    test_a_move_to_any_place_is_performed()
    test_an_unstated_frame_becomes_a_portrait()
    test_an_interrupt_releases_the_frame_buffer()
    test_a_posture_denied_is_not_a_posture_taken()
    test_the_face_plays_the_feeling_the_author_named()
    test_a_pronoun_pointing_away_keeps_the_person_it_means()
    test_a_feeling_belongs_to_the_face_the_beat_pins_it_on()
    test_a_bare_region_says_whose_body_it_is()
    test_a_plural_removal_still_comes_off()
    test_a_removal_undresses_only_its_wearer()
    test_a_pairing_the_beat_wrote_is_said_back()
    test_keyframe_handoff()
    test_references_and_silence()
    test_first_frame()
    test_upscale_paths()
    test_aug_protects_the_keyframe()
    test_beat_reviving_a_garment()
    test_restart_after_removal()
    test_auto_removal()
    test_fall_keeps_the_hardware()
    test_anchor_is_the_scene()
    test_guard_and_layers_end_to_end()
    test_removing_shot_without_a_keyframe()
    test_hardware_anchor_end_to_end()
    test_chain_hold_end_to_end()
    test_every_paragraph_accounted_for()
    test_av_stays_in_sync()
    test_person_described_once_end_to_end()
    test_references_ride_with_the_keyframe()
    test_undressing_completely_end_to_end()
    test_a_tagged_object_comes_off_and_goes_back_on()
    test_hardware_stays_on_its_owner()
    test_a_state_in_the_scene_is_not_reasserted()
    test_the_cuffs_stay_in_the_picture()
    test_a_sheet_that_claims_hardware_too_early()
    test_the_audit_findings_stay_fixed()
    test_no_cuffs_described_without_their_wearer()
    test_caught_first_then_restrained()
    test_a_television_keeps_its_own_voice()
    test_a_shifted_workflow_stops_before_rendering()
    test_the_removal_shot_says_what_is_under()
    test_a_beat_can_name_what_the_layering_hid()
    test_a_removal_says_whose_hands()
    test_the_only_tag_being_on_a_covered_thing()
    test_an_object_tag_works_without_a_face_picture()
    test_an_untagged_picture_defeats_the_layering()
    test_underwear_is_hidden_until_it_is_not()
    test_a_garment_moved_is_not_a_garment_gone()
    test_undressing_does_not_drop_her()
    test_an_unbound_fall_is_told_what_catches_it()
    test_a_named_look_target_is_restated()
    test_a_line_with_no_look_turns_the_faces_to_each_other()
    test_a_walk_along_a_place_arrives_in_it()
    test_a_rooms_description_waits_outside_that_room()
    test_a_covered_object_does_not_send_its_picture()
    test_the_anchor_survives_a_close_shot()
    test_mouths_stay_shut_with_no_line()
    test_a_grin_is_not_a_closed_mouth()
    test_nothing_tells_the_cast_to_hold_still()
    test_a_face_under_duress_is_not_a_portrait()
    test_her_whimper_does_not_free_his_mouth()
    test_a_line_is_locked_to_the_person_who_says_it()
    test_her_look_does_not_land_on_him()
    test_a_look_survives_the_next_beat()
    test_the_bed_survives_an_anchor()
    test_the_audio_branch_gets_a_soft_landing()
    test_an_anchor_says_when_it_has_taken_the_scenes_place()
    test_a_grim_film_is_grim_in_every_shot()
    test_a_kidnapping_reads_as_one_without_being_declared()
    test_script_is_what_was_sent()
    test_every_reference_is_claimed()
    test_the_demoted_handoff_is_claimed()
    test_a_gapped_socket_still_sends_its_image()
    test_a_modified_state_is_not_read_as_an_act()
    test_a_staged_change_gets_both_ends()
    test_dialogue_headroom()
    test_introducing_somebody_already_in_position()
    test_back_after_a_shot_away()
    test_a_name_with_no_entry_end_to_end()
    test_the_soundtrack_is_the_models_own()
    test_shot_one_is_the_only_unpinned_shot()
    test_a_first_frame_can_be_the_set_instead_of_frame_one()
    test_the_age_reaches_the_shot_and_the_floor_holds()
    test_sound_survives_silencing()
    test_auto_sound_end_to_end()
    test_room_tone_under_every_shot()
    test_the_decode_keeps_the_vae_it_is_about_to_use()
    test_an_exclusive_sound_clause_never_denies_the_beat()
    test_a_vocal_shot_says_what_fills_the_gaps()
    test_the_chain_is_never_held_twice()
    test_the_position_may_only_be_written_once_in_the_scene()
    test_finished_shots_are_held_in_half_precision()
    test_detail_trend()
    test_the_chain_does_not_burn_in()
    test_timing_report()
    test_no_garment_is_ever_invented()
    test_nothing_wearable_is_ever_added()
    test_a_garment_keeps_its_description()
    test_a_removal_always_names_hands()
    test_hands_and_holds_follow_the_beat()
    test_one_line_is_one_voice()
    test_the_plan_says_whether_silence_can_be_applied()
    test_the_silent_latent_looks_like_silence()
    test_silence_pins_the_generated_audio()
    test_a_softened_handoff_is_not_a_keyframe()
    test_a_journey_reaches_the_shot()
    test_the_scene_does_not_reset()
    test_a_described_room_still_holds()
    test_the_sound_clause_is_inside_the_budget()
    test_the_anchor_is_not_read_as_a_room()
    test_a_garment_does_not_appear_at_the_first_frame()
    test_one_picture_two_people_is_reported()
    test_a_collar_chained_to_a_wall_stays_on()
    test_every_way_of_chaining_a_collar_to_a_wall()
    test_moving_towards_something_is_not_being_chained_to_it()
    test_two_restraints_put_on_together_both_survive()
    test_a_name_that_is_only_spoken_is_not_in_the_shot()
    test_a_verb_is_not_a_room()
    test_entering_a_room_is_a_journey_not_a_cut()
    test_no_guard_sentence_carries_a_negation()
    test_hardware_waits_for_the_beat_that_puts_it_on()
    test_the_applying_shot_says_where_the_limbs_finish()
    test_a_collar_in_the_sheet_is_held_like_hardware()
    test_an_undergarment_keeps_its_words_and_waits_for_its_picture()
    test_an_under_layer_belongs_to_somebody()
    test_the_picture_arrives_when_the_cover_moves()
    test_letting_the_cover_fall_puts_it_back()
    test_somebody_else_can_lift_your_skirt()
    test_pacing_reaches_the_thin_shots()
    test_a_line_is_spoken_in_one_language()
    test_undressing_does_not_spread()
    test_a_working_character_is_not_still_lying_down()
    test_an_instruction_is_not_the_action()
    test_a_breath_does_not_hold_the_branch_open()
    test_camera_framing_is_read_from_the_anchor()
    test_a_tight_frame_drops_wardrobe_it_cannot_contain()
    test_a_hidden_garments_lettering_goes_with_it()
    test_audio_sigma_reads_the_scheduler()
    test_audio_sigma_falls_back_to_the_closed_form()
    test_appearing_is_not_arriving()
    test_a_line_is_marked_however_it_is_punctuated()
    test_a_two_word_sheet_name_does_not_duplicate_her()
    test_dan_is_not_instantiated_twice()
    test_a_carried_room_is_not_a_second_picture_of_somebody()
    test_a_bare_region_is_said_on_every_shot()
    test_a_squat_survives_speech_and_undressing()
    test_what_comes_off_in_the_bathroom_stays_off()
    test_the_thong_comes_off_however_it_is_written()
    test_the_opening_does_not_name_people_a_shot_leaves_out()
    test_a_sheet_written_first_is_a_sheet()
    test_a_two_word_name_makes_a_sheet()
    test_one_person_under_two_names_is_described_once()
    test_a_bare_chest_belongs_to_whoever_undressed()
    print()
    if _fails:
        print(f"RESULT: {len(_fails)} FAILURE(S): " + "; ".join(_fails))
        sys.exit(1)
    print("RESULT: ALL PASSED")


if __name__ == "__main__":
    main()
