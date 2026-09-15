"""Tests for the prompt engine.

Every case here is a bug that was REPORTED against the old engine. The engine was
replaced; the reports are the specification and they carry over.

Run: python test_engine.py
"""

import io
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import engine as E

_fails = []


def check(label, ok, extra=""):
    print(("  PASS  " if ok else "  FAIL  ") + label + ("" if ok else f"   {extra}"))
    if not ok:
        _fails.append(label)


def scene(beats, cast=("Ana", "Guard")):
    """Run beats through the state and return the continuity text per shot."""
    st = E.SceneState()
    out = []
    for i, b in enumerate(beats, 1):
        ch = st.read(b, cast=cast, shot=i)
        out.append(st.continuity(described=[n for n in cast], changed=ch))
    return st, out


def test_two_things_in_one_beat():
    """REPORTED: "she is still breaking handcuffs".

    A beat that cuffs the wrists and locks on a collar recorded whichever phrase
    was longer and dropped the other. From the next shot the handcuffs were not
    in the prompt at all -- not held, not named -- and hardware nobody mentions
    is hardware the model stops drawing."""
    print("\n=== two things in one beat ===")
    hw = [(c, p, w) for c, p, w, _at in E.hardware_spans(
        "The guard handcuffs Ana's wrists behind her back and "
        "locks a steel collar around her neck.")]
    names = [c for c, _p, _w in hw]
    check(f"both are read: {names}", "handcuffs" in names and "collar" in names)
    check("the material is kept", any(w == "steel collar" for _c, _p, w in hw))
    _st, shots = scene([
        "The guard handcuffs Ana's wrists behind her back and locks a steel "
        "collar around her neck, chained to the wall.",
        "Ana sits against the wall.",
        "Ana pulls at the chain.",
        "Ana looks at the door.",
    ])
    for i, s in enumerate(shots[1:], 2):
        check(f"shot {i} names the handcuffs", "handcuffs" in s, s)
        check(f"shot {i} names the collar", "collar" in s, s)


def test_a_neck_is_not_behind_a_back():
    """REPORTED: nonsense in the hold clause.

    A limb position describes where the ARMS are. The old engine read the part
    from the item list and the position from the beat and joined them, giving
    "holding the neck behind the back"."""
    print("\n=== a neck is not behind a back ===")
    _st, shots = scene([
        "The guard handcuffs Ana's wrists behind her back and locks a steel "
        "collar around her neck, chained to the wall.",
        "Ana sits against the wall.",
    ])
    s = shots[1]
    check("no neck behind a back", "neck behind the back" not in s, s)
    check("the wrists take the position", "wrists behind the back" in s, s)
    check("the collar takes the neck", "the neck" in s, s)
    check("...and the anchor", "fast to the wall" in s, s)
    # The constructor refuses it even if a caller tries.
    r = E.Restraint("collar", "neck", position="behind the back")
    check("a collar cannot hold a position", r.position == "")


def test_a_door_is_not_a_room():
    """REPORTED: "camera views are all screwed up".

    "door" was a place, so "Ana looks at the door" relocated the shot into a
    door. A place has to be somewhere a person can be, and behind a preposition
    that puts them there."""
    print("\n=== a door is not a room ===")
    for t in ("Ana looks at the door.", "Ana opens the door.",
              "Ana stares at the door.", "Ana knocks on the door."):
        check(f"no room in {t!r}", not E.place_in(t), repr(E.place_in(t)))
    check("a doorway is a place", E.place_in("Ana waits in the doorway.") == "doorway")
    for room in ("kitchen", "basement", "hallway", "bathroom", "cell"):
        check(f"{room} reads", E.place_in(f"Ana walks into the {room}.") == room)
    check("an adjective does not hide it",
          E.place_in("Ana walks into the tiled bathroom.") == "tiled bathroom")
    check("shallow depth of field is not a hall",
          not E.place_in("shallow depth of field, 35mm"))


def test_movement_is_not_fastening():
    """REPORTED indirectly: hardware latched over furniture somebody walked to.

    "to the <thing>" on its own is movement. The fastening verb is required, and
    a noun that is also a verb ("the chain", "the clip") must not supply it."""
    print("\n=== movement is not fastening ===")
    for t in ("chains it to the wall", "chained to the wall",
              "chains her collar to the wall", "secures the chain to the floor",
              "bolted to the ceiling", "clips the chain to a ring",
              "a chain runs from her collar to the wall",
              "a short chain holds her collar to the wall"):
        check(f"anchors: {t!r}", E.anchor_in(t), repr(E.anchor_in(t)))
    for t in ("she sinks to the floor", "he walks to the table",
              "she is dragged to the bed", "she falls to the floor",
              "she runs to the wall", "they move to the bed",
              "She drops the rope to the floor.", "The clip fell to the floor.",
              "He throws the chain to the floor.", "The chains hang by the door."):
        check(f"not an anchor: {t!r}", not E.anchor_in(t), repr(E.anchor_in(t)))


def test_the_agent_is_not_the_wearer():
    """REPORTED: a second figure invented to wear the hardware.

    "The guard cuffs Ana" puts them on Ana. Describing them on the guard gives
    the model wrists that belong to nobody the text put them on."""
    print("\n=== the agent is not the wearer ===")
    st, _ = scene(["The guard handcuffs Ana."])
    check("Ana is restrained", st.person("Ana").restrained())
    check("the guard is not", not st.person("Guard").restrained())


def test_it_comes_off_when_the_text_takes_it_off():
    """The latch has to release, or a script that unlocks the cuffs in its own
    prose goes on insisting they are fastened over hardware on the floor."""
    print("\n=== released when released ===")
    st, shots = scene([
        "The guard handcuffs Ana.",
        "Ana sits down.",
        "The guard unlocks the handcuffs and takes them off.",
        "Ana stands up.",
    ])
    check("held while on", "handcuffs" in shots[1], shots[1])
    check("gone after the release", "handcuffs" not in shots[3], shots[3])
    check("state agrees", not st.person("Ana").restrained())


def test_the_shot_that_applies_says_both_ends():
    """A standing hold on the shot that STAGES the fastening is a lie about the
    first frame: it says closed before it was closed, and the struggle then
    happens in whatever order is left over."""
    print("\n=== both ends on the applying shot ===")
    _st, shots = scene(["The guard handcuffs Ana.", "Ana sits down."])
    check("shot 1 says both ends", "off the body at the first frame" in shots[0],
          shots[0])
    check("shot 2 does not", "off the body at the first frame" not in shots[1],
          shots[1])
    check("shot 2 holds instead", "closed and fastened" in shots[1], shots[1])


def test_a_chain_is_a_tether_not_a_second_restraint():
    """"a collar chained to the wall" is ONE restraint with an anchor. Recording
    the chain separately gives the shot a chain holding the wrists beside a
    collar holding the neck, which is two things to draw where there is one."""
    print("\n=== a chain is a tether ===")
    st, shots = scene([
        "The guard locks a steel collar around Ana's neck and chains it to the wall.",
        "Ana sits down."])
    hw = st.person("Ana").kinds()
    check(f"one item, not two: {hw}", hw == ["collar"], str(hw))
    # Guarded: with the item reader broken there is no collar to index, and a
    # KeyError reports as a crashed suite rather than as the failure it is.
    check("it carries the anchor",
          bool(st.person("Ana").hw("collar"))
          and st.person("Ana").hw("collar").anchor == "wall")
    check("the shot says so", "fast to the wall" in shots[1], shots[1])
    check("no loose chain in the text", "chain" not in shots[1], shots[1])


def test_the_material_survives_as_written():
    """REPORTED: a mirrored steel collar rendering as a black leather one.

    The modifier list was twenty-odd words, so "mirrored" and "nickel" were not
    modifiers at all: "a mirrored steel collar" was recorded as "steel collar"
    and "a brushed nickel collar" as plain "collar". The guard repeats that
    shortened name once a shot, every shot after the first, and the prior for a
    bare collar is a black leather one -- so the node was asking for the thing
    that was reported."""
    print("\n=== the material survives as written ===")

    def hw(t):
        return [w for _c, _p, w, _a in E.hardware_spans(t)]

    check("a finish is part of the name",
          hw("a mirrored steel collar") == ["mirrored steel collar"],
          str(hw("a mirrored steel collar")))
    check("...and a metal it never listed",
          hw("a brushed nickel collar") == ["brushed nickel collar"],
          str(hw("a brushed nickel collar")))
    check("...and three words of it",
          hw("a mirrored stainless steel collar")
          == ["mirrored stainless steel collar"],
          str(hw("a mirrored stainless steel collar")))
    check("a hyphenated finish passes whole, unlisted",
          hw("a mirror-finish steel collar") == ["mirror-finish steel collar"],
          str(hw("a mirror-finish steel collar")))
    check("what already worked still works",
          hw("a black leather collar") == ["black leather collar"])
    check("garments keep theirs too",
          E.garments_in("a mirrored PVC skirt") == ["mirrored pvc skirt"],
          str(E.garments_in("a mirrored PVC skirt")))
    # A VERB IS NOT A MODIFIER. "-ed" is a verb far more often than an adjective,
    # and capturing one would put an action into the name of the thing.
    for t, want in (("She grabbed her collar.", ["collar"]),
                    ("He unlocked the collar.", ["collar"]),
                    ("He dropped the steel collar.", ["steel collar"])):
        check(f"no verb in the name: {t!r}", hw(t) == want, str(hw(t)))


def test_a_two_word_name_is_still_a_name():
    """REPORTED: a duplicate Mistress, present in the very first beat.

    The sheet's name parser took ONE word, so "Mistress Vale:" parsed as no name
    at all -- and an unlabelled sheet line belongs to everyone and is never
    dropped. Her whole physical description rode into every shot with no name on
    it, beside the character it was meant to describe. Two women, one of them
    anonymous, in a beat that named one person.

    The other half is here: a sheet name is often longer than what the beats call
    her, and matching only the whole name put her line in NO shot instead."""
    print("\n=== a two-word name is still a name ===")
    check("a longer name answers to one of its words",
          E.names_in("The Mistress stands at the window.",
                     ["Mistress Vale", "Ana"]) == ["Mistress Vale"])
    check("...or to the other one",
          E.names_in("Vale looks up.", ["Mistress Vale", "Ana"]) == ["Mistress Vale"])
    # Ambiguous: the word belongs to somebody else outright, so picking either is
    # a guess, and guessing is how the wrong person gets into a shot.
    check("a word another character owns stands in for nobody",
          E.names_in("The Mistress stands at the window.",
                     ["Mistress", "Mistress Vale", "Ana"]) == ["Mistress"])
    # A capital at the start of a sentence is free.
    check("an opening 'May' is not Aunt May",
          E.names_in("May I come in?", ["Aunt May", "Ana"]) == [])
    check("...but a mid-sentence one is",
          "Aunt May" in E.names_in("Ana looks at May.", ["Aunt May", "Ana"]))
    check("an opening 'Will' is not Will Barnes",
          E.names_in("Will you wait?", ["Will Barnes", "Ana"]) == [])
    check("...and a word that is nobody's verb still opens a sentence",
          E.names_in("Vale looks up.", ["Mistress Vale"]) == ["Mistress Vale"])


def test_a_bare_region_stays_bare():
    """REPORTED: a bra comes back on somebody topless -- and the character had no
    bra anywhere on the sheet.

    Nothing was restoring it. The clause saying a region is uncovered fired only
    on the beat that uncovered it, so every later shot left that region
    unspecified, and an unspecified region is filled by the model's own prior.
    The prior put a bra there and the keyframe carried it on."""
    print("\n=== a bare region stays bare ===")
    st = E.SceneState()
    st.declare("Kate", "Kate: she, 24, a shirt, jeans.")
    for i, b in enumerate(["Kate is topless, sitting on the crate.",
                           "Kate looks at the door."], 1):
        st.read(b, ["Kate"], i)
    k = st.person("Kate")
    check("being topless takes the shirt off", "shirt" not in str(k.worn), str(k.worn))
    check("...and latches the region", k.bare == ["torso"], str(k.bare))
    # The sheet is re-read every shot and is never edited, so it used to put the
    # shirt straight back on and the bare clause went silent from shot 2.
    st.declare("Kate", "Kate: she, 24, a shirt, jeans.")
    check("the sheet does not put it back on", "shirt" not in str(k.worn), str(k.worn))
    check("...so the region is still bare", k.bare == ["torso"], str(k.bare))
    # Dressing again releases it, or she is told the chest is bare over a shirt.
    st.read("Kate puts on her shirt.", ["Kate"], 3)
    check("dressing releases the latch", k.bare == [], str(k.bare))
    # Removal reaches the same state as description.
    st2 = E.SceneState()
    st2.declare("Kate", "Kate: she, 24, a shirt, jeans.")
    st2.read("Kate takes off her shirt.", ["Kate"], 1)
    check("a removal latches it too", st2.person("Kate").bare == ["torso"],
          str(st2.person("Kate").bare))
    check("naked reaches every region",
          E.nudity_in("Kate is naked.") == ["torso", "legs", "feet"])
    check("...and a naked flame is not a person", E.nudity_in("a naked flame") == [])
    _torso = next((s for _rx, r, s in E._REGION_RX if r == "torso"), "")
    check("the torso sentence names the CHEST -- where the bra was invented",
          "chest" in _torso, _torso)


def test_a_squat_is_held():
    """REPORTED: she does not stay squatting, she stands up on her own.

    Two causes, both in posture_cleared. It read the whole beat including quoted
    speech, so `Kate says: "Someone is coming."` cleared the pose on "coming" --
    a travel verb, inside the line, about somebody else. And "takes off her
    shirt" matched the travel list on "takes", so undressing cleared it too."""
    print("\n=== a squat is held ===")
    check("a squat is a squat, not a crouch",
          E.posture_in("Kate squats down.") == "squatting",
          E.posture_in("Kate squats down."))
    check("...and a crouch is still a crouch",
          E.posture_in("Kate crouches by the door.") == "crouching")
    st = E.SceneState()
    for i, b in enumerate(["Kate squats down beside the crate.",
                           "Kate says: \"Someone is coming.\"",
                           "Kate takes off her shirt.",
                           "Kate listens."], 1):
        st.read(b, ["Kate"], i)
        check(f"shot {i} still has her squatting",
              st.person("Kate").posture == "squatting", st.person("Kate").posture)


def test_chains_do_not_interfere():
    """Reported as chains interfering with each other.

    "chain" is a verb as often as it is a noun, and its part is whatever the
    beat says -- neither of which the table could express. So every chain
    landed on the WRISTS, where it collided with the cuffs already there, and a
    verb put one on somebody with nothing on their wrists at all."""
    print("\n=== chains do not interfere ===")
    P = E.hardware_spans

    def kinds(beat):
        return [(c, pt) for c, pt, _w, _a in P(beat)]

    # THE VERB IS NOT AN ITEM.
    check("a verb introduces nothing to draw",
          kinds("Sam chains her collar to the ring.") == [("collar", "neck")],
          str(kinds("Sam chains her collar to the ring.")))
    check("...even with no object to fasten",
          kinds("He chains it shut.") == [], str(kinds("He chains it shut.")))
    check("...and it does not reach the wrists",
          "wrists" not in str(kinds(
              "Sam cuffs her wrists behind her back and chains her collar to "
              "the ring.")[1:]))
    # THE PART COMES FROM THE BEAT.
    check("a chain goes where the beat puts it",
          kinds("Sam locks a chain around her ankles.") == [("chain", "ankles")],
          str(kinds("Sam locks a chain around her ankles.")))
    check("...and a verb still fastens what it names",
          kinds("Sam chains her ankles together.") == [("chain", "ankles")],
          str(kinds("Sam chains her ankles together.")))
    # TWO CHAINS ARE TWO RESTRAINTS. Keyed by name alone, the second overwrote
    # the first and one of them was never drawn again.
    two = kinds("Sam chains Kate's collar to the ring and chains her ankles "
                "together.")
    check("a second chain is not eaten by the first",
          two == [("collar", "neck"), ("chain", "ankles")], str(two))
    both = kinds("Sam locks a chain around her ankles and a chain around her "
                 "wrists.")
    check("...two of a kind on two parts both survive",
          both == [("chain", "ankles"), ("chain", "wrists")], str(both))
    # A TETHER IS NOT A RESTRAINT OF ITS OWN -- but a material is not a tether.
    check("a chain running TO something is that thing's tether",
          kinds("Sam clips a chain to her collar.") == [("collar", "neck")],
          str(kinds("Sam clips a chain to her collar.")))
    check("...while a material is kept",
          ("tape", "wrists") in kinds("Dan gags her with duct tape."),
          str(kinds("Dan gags her with duct tape.")))
    # ONE OF TWO RINGS IS STILL A RING. The determiner list was six words, so a
    # collar chained to "one ring" was not a restraint at all and nothing about
    # it survived the shot it went on in.
    check("an anchor takes any determiner",
          E.anchor_in("Sam chains Kate's collar to one ring.") == "ring")
    check("...and a qualifier before it",
          E.anchor_in("Sam chains Mara's collar to the other ring.") == "ring")
    check("...without swallowing ordinary prose",
          E.anchor_in("Kate walks to the far side of the room.") == "")


def test_a_modifier_belongs_to_its_own_item():
    """Two items and two modifiers in one beat. Giving both modifiers to both
    items produced handcuffs chained to a wall they were never near, and a
    collar held behind a back."""
    print("\n=== a modifier belongs to its own item ===")
    st, shots = scene([
        "The guard handcuffs Ana's wrists behind her back and locks a steel "
        "collar around her neck, chained to the wall.",
        "Ana sits down."])
    ana = st.person("Ana")
    # Guarded: with the item reader broken one of these is missing entirely, and
    # a KeyError reports as a crashed suite rather than as the failure it is.
    check(f"both items recorded: {ana.kinds()}",
          "handcuffs" in ana.kinds() and "collar" in ana.kinds())
    c, k = ana.hw("handcuffs"), ana.hw("collar")
    check("cuffs take the position", bool(c) and c.position == "behind the back")
    check("...and not the wall", bool(c) and c.anchor == "")
    check("the collar takes the wall", bool(k) and k.anchor == "wall")
    check("...and not the position", bool(k) and k.position == "")
    s = shots[1]
    check("the shot reads correctly",
          "the wrists behind the back and the neck fast to the wall" in s, s)


def test_a_spoken_name_is_not_a_staged_one():
    """REPORTED: a character turned up in a scene they were not in, because a
    beat called out to them --

        Dana opens the door and calls out: "McKenna where are you?"

    Calling for somebody is how absence gets written, and it was reading as
    presence. The state has to see the staged half only, or an instruction said
    aloud puts hardware on somebody who is not in the room."""
    print("\n=== a spoken name is not a staged one ===")
    check("speech is stripped",
          "McKenna" not in E._outside_speech(
              'Dana calls out: <d>McKenna where are you?</d>'))
    check("...and plain quotes too",
          "McKenna" not in E._outside_speech('Dana calls out: "McKenna?"'))
    check("staging survives it",
          "McKenna" in E._outside_speech(
              'Dana turns to McKenna and says: <d>Wait.</d>'))
    # An order given aloud names the person it is given to. That is not the
    # person the hardware goes on -- nobody in this beat is touched at all.
    st = SceneState_for('Dan says: <d>McKenna, put the cuffs on.</d>')
    check("nobody spoken to is restrained",
          not st.person("McKenna").restrained(),
          str(st.person("McKenna").hardware))
    # ...while the same sentence staged puts them on for real.
    st2 = SceneState_for("Dan puts the cuffs on McKenna.")
    check("staged, they go on", st2.person("McKenna").restrained())


def SceneState_for(beat):
    st = E.SceneState()
    st.read(beat, cast=("Dan", "McKenna"), shot=1)
    return st


def test_fastening_something_to_a_collar_does_not_date_the_collar():
    """REPORTED: "the collar is still being removed".

    "Dana clips a lead to the steel collar" puts a LEAD on. The collar is where
    it clips and has been round her neck all along -- but staged_applications
    counted every piece of hardware in the beat, so it dated the COLLAR to that
    beat too, and everything before it was treated as before she had one."""
    print("\n=== fastening to a thing does not date that thing ===")
    beats = ["McKenna sits on the sofa.",
             "Dana clips a lead to the steel collar."]
    got = E.staged_applications(beats)
    check(f"the collar is not dated: {got}", "collar" not in got, str(got))
    check("...and so nothing is staged at all here", got == {}, str(got))
    # A leash clipped ON is dated, because it is the thing going on.
    leashed = E.staged_applications(["McKenna waits.",
                                     "Dana clips a leash to the steel collar."])
    check("the leash is dated", leashed.get("leash") == 2, str(leashed))
    check("...and still not the collar", "collar" not in leashed, str(leashed))
    # ...while actually putting a collar on does date it.
    on = E.staged_applications(["McKenna waits.",
                                "Dana locks a steel collar around her neck."])
    check("locking one on dates it", on.get("collar") == 2, str(on))
    # A beat that only MENTIONS hardware dates nothing.
    check("a mention dates nothing",
          E.staged_applications(["McKenna looks at the collar."]) == {})
    # KNOWN GAP, recorded rather than guessed at: "lead" as a noun is not in the
    # hardware table. It is the British word for a leash and it is also a very
    # common verb ("Dana leads her down the hall"), and no cheap pattern told
    # them apart without false-firing on the verb. Write "leash" for now.
    check("a bare 'lead' is not read as hardware (known gap)",
          not E.hardware_spans("Dana clips a lead to it"))


def test_a_name_is_matched_case_sensitively():
    """Found by comparing the two files rather than by a report.

    The sampler matched names case-SENSITIVELY, with the reason written beside
    it: prose capitalises a name, and matching without case makes the word "will"
    find a character called Will and "grace" find Grace. This file matched
    case-insensitively and had that bug sitting in it, unreported, because the
    two files each worked out "who does this beat name" separately.

    One reader now, names_in, and both use it."""
    print("\n=== a name is matched case-sensitively ===")
    check("a lowercase 'will' is not Will",
          E.names_in("The guard will lock the door.", ["Will", "Guard"]) == [],
          str(E.names_in("The guard will lock the door.", ["Will", "Guard"])))
    check("...and the real Will still is",
          E.names_in("Will locks the door.", ["Will"]) == ["Will"])
    check("'says grace' is not Grace",
          E.names_in("She says grace.", ["Grace"]) == [])
    # The two things it already had to do, kept.
    check("a spoken name is not staged",
          E.names_in("Dana calls: <d>McKenna?</d>", ["Dana", "McKenna"]) == ["Dana"])
    check("...and the order is the sentence's",
          E.names_in("Dana handcuffs McKenna.", ["McKenna", "Dana"])
          == ["Dana", "McKenna"])


def test_the_passive_voice_puts_hardware_on():
    """Also found by comparison. "McKenna is handcuffed by Dana" is the ordinary
    way to write it, and this file read NO hardware in it at all -- the table had
    nouns only, and "handcuffed" is not "handcuffs". The sampler's restraint
    reader has carried the participles for a long time.

    The participle finds it and never names it: an item recorded as "handcuffed"
    renders as "The handcuffed stay closed and fastened"."""
    print("\n=== the passive voice puts hardware on ===")
    for beat, cast, wearer, item in (
            ("McKenna is handcuffed by Dana.", ("Dana", "McKenna"), "McKenna",
             "handcuffs"),
            ("Ana is collared by the guard.", ("Ana", "Guard"), "Ana", "collar"),
            ("Ana is gagged.", ("Ana",), "Ana", "gag")):
        st = E.SceneState()
        st.read(beat, cast=cast, shot=1)
        got = [r.item for r in st.person(wearer).hardware.values()]
        check(f"{beat[:34]!r} -> {got}", item in got, str(got))
    # ...and the wording is the noun, not the participle.
    st = E.SceneState()
    st.read("McKenna is handcuffed by Dana.", cast=("Dana", "McKenna"), shot=1)
    said = st.continuity(described=["McKenna"])
    check("the sentence says handcuffs", "handcuffed stay" not in said, said[:120])
    check("...and the agent wears nothing",
          not st.person("Dana").restrained())


def test_nothing_is_said_twice():
    """The old engine restated the same fact from several readers at once and
    the guards reached 65% of a shot against a 12% beat."""
    print("\n=== nothing said twice ===")
    _st, shots = scene([
        "The guard handcuffs Ana's wrists behind her back.",
        "Ana sits against the wall.",
        "Ana twists her wrists in the cuffs.",
    ])
    for i, s in enumerate(shots[1:], 2):
        check(f"shot {i}: cuffs named once", s.lower().count("cuffs") <= 1, s)
        check(f"shot {i}: under 45 words", len(s.split()) < 45,
              f"{len(s.split())}w: {s}")


def test_a_garment_change_says_both_ends():
    """REPORTED: "a diaper instantly changed into shorts before the beat where
    she puts her shorts back on".

    A shot told only the RESULT is free to open with the result already true.
    Both ends, or the change happens whenever the model feels like it."""
    print("\n=== a garment change says both ends ===")
    st, shots = scene([
        "Ana stands in the room.",
        "Ana takes off her blue shorts.",
        "Ana sits down.",
        "Ana puts her blue shorts back on.",
        "Ana walks out.",
    ], cast=("Ana",))
    check("the removal shot says both ends",
          "on the body as the shot opens and fully off it by the last frame"
          in shots[1], shots[1])
    check("the next shot says it stays off",
          "off the body and stays where it was put" in shots[2], shots[2])
    check("the wearing shot says both ends",
          "off the body as the shot opens and fully on by the last frame"
          in shots[3], shots[3])
    check("...and not still-off as well",
          "off the body and stays where it was put" not in shots[3], shots[3])
    check("state agrees at the end",
          [g for g in st.person("Ana").worn if "shorts" in g]
          and not st.person("Ana").removed)
    check("one pair of shorts, not two",
          shots[3].lower().count("shorts") == 1, shots[3])


def test_pulled_aside_is_not_taken_off():
    """A garment moved aside is still ON. Recording it as removed loses it."""
    print("\n=== pulled aside is not off ===")
    st, shots = scene(["Ana pulls her skirt aside."], cast=("Ana",))
    check("it stays on the body", "stays on the body" in shots[0], shots[0])
    check("it is not removed", not st.person("Ana").removed)
    check("it is displaced", st.person("Ana").displaced)


def test_each_clause_owns_its_change():
    print("\n=== each clause owns its change ===")
    st = E.SceneState("studio")
    st.declare("Ana", "coat and hat")
    st.declare("Bea", "coat")
    st.read("Ana removes her coat and puts on her hat.", ("Ana", "Bea"), 1)
    check("a later wearing verb does not turn into a second removal",
          st.person("Ana").removed == ["coat"] and st.person("Ana").worn == ["hat"])

    st = E.SceneState()
    st.read("Bea is handcuffed.", ("Bea",), 0)
    changed = st.read("Ana unlocks Bea's handcuffs and fastens a collar around her neck.",
                      ("Ana", "Bea"), 1)
    check("one beat can release one item and apply another",
          not st.person("Bea").hw("handcuffs") and bool(st.person("Bea").hw("collar")),
          str(changed))


def test_descriptions_and_places_have_owners():
    print("\n=== descriptions and places have owners ===")
    st = E.SceneState("studio")
    st.declare("Ana", "coat")
    st.declare("Bea", "coat")
    st.read("Ana looks at topless Bea.", ("Ana", "Bea"), 1)
    check("topless applies only to the person it describes",
          not st.person("Ana").bare and st.person("Bea").bare == ["torso"])
    st.read("Ana points to the kitchen.", ("Ana", "Bea"), 2)
    check("pointing at a room does not move the scene", st.place == "studio", st.place)


def main():
    test_two_things_in_one_beat()
    test_a_neck_is_not_behind_a_back()
    test_a_door_is_not_a_room()
    test_movement_is_not_fastening()
    test_the_agent_is_not_the_wearer()
    test_it_comes_off_when_the_text_takes_it_off()
    test_the_shot_that_applies_says_both_ends()
    test_a_chain_is_a_tether_not_a_second_restraint()
    test_the_material_survives_as_written()
    test_a_two_word_name_is_still_a_name()
    test_a_bare_region_stays_bare()
    test_a_squat_is_held()
    test_chains_do_not_interfere()
    test_a_modifier_belongs_to_its_own_item()
    test_a_spoken_name_is_not_a_staged_one()
    test_fastening_something_to_a_collar_does_not_date_the_collar()
    test_a_name_is_matched_case_sensitively()
    test_the_passive_voice_puts_hardware_on()
    test_nothing_is_said_twice()
    test_a_garment_change_says_both_ends()
    test_pulled_aside_is_not_taken_off()
    test_each_clause_owns_its_change()
    test_descriptions_and_places_have_owners()
    print()
    if _fails:
        print(f"RESULT: {len(_fails)} FAILURE(S): " + "; ".join(_fails))
    else:
        print("RESULT: ALL PASSED")


if __name__ == "__main__":
    main()
