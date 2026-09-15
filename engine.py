# H3-LongVideos -- https://github.com/Smite79/MiniMax-H3-LongVideos
# Copyright (c) 2026 Smite79. All rights reserved.
# Redistribution, in whole or in part, requires written permission.
# This notice may not be removed or altered. See LICENSE.
"""The prompt engine: scene state, beat by beat.

WHY THIS REPLACED WHAT WAS HERE BEFORE
--------------------------------------
The old engine was about sixty independent readers, each searching the beat for
its own thing and each appending its own sentence to the shot. `limb_anchor`
found "behind the back", `held_part` found "neck", and neither could see the
other, so the shot went out saying "holding the neck behind the back" -- a neck
behind a back. `hardware_named` returned one item, so a beat that cuffed the
wrists and locked on a collar recorded the collar and the handcuffs were never
mentioned again; hardware nobody mentions is hardware the model stops drawing,
and that read as her breaking out of them. `_PLACE` contained "door", so "Ana
looks at the door" moved the camera into a door.

Every one of those is the same failure: a clause derived on its own, with nothing
holding the facts together and nothing able to notice a contradiction.

So the shape here is different. A beat is parsed ONCE into events. Events update
one explicit state. The state renders ONE paragraph. There is exactly one place
that knows what is on whom, one that knows where anybody is, and one that turns
that into English -- so a contradiction is a bug in a value you can print, not an
emergent property of sixty regexes that never met.

WHAT THE STATE GUARANTEES
    - hardware is remembered per person, with the part it holds, the position it
      holds that part in, and what it is anchored to. All four together or not at
      all, so they cannot disagree.
    - every piece of hardware on a person is named in every shot until something
      takes it off. Not the newest, not the most specific: all of it.
    - a garment is on, off, or displaced, and the shot that changes it says both
      ends of the change.
    - a place is a room somebody can be in. A door is not a room.
"""

import re

# ---------------------------------------------------------------------------
# Vocabulary. One table per KIND of thing, and a word appears in exactly one of
# them. The old engine had "chain" as a noun in one list and a verb in another,
# which let a chain-link fence satisfy both halves of a rule by itself.
# ---------------------------------------------------------------------------

# Hardware, and the part each kind holds. The part is a property of the ITEM --
# this is the table whose absence produced "holding the neck behind the back".
HARDWARE = (
    (r"hand\s?cuffs?|handcuffed", "handcuffs", "wrists"),
    (r"leg\s?irons?", "leg irons", "ankles"),
    (r"ankle\s+(?:cuffs?|chains?|straps?)", "ankle cuffs", "ankles"),
    (r"shackles?|shackled", "shackles", "ankles"),
    (r"manacles?|manacled", "manacles", "wrists"),
    (r"zip\s?ties?|cable\s?ties?", "zip ties", "wrists"),
    (r"collars?|chokers?|collared", "collar", "neck"),
    (r"leash(?:es)?|leashed", "leash", "neck"),
    (r"gags?|gagged", "gag", "mouth"),
    (r"blindfolds?|blindfolded", "blindfold", "eyes"),
    (r"harness(?:es)?", "harness", "body"),
    (r"spreader\s+bars?", "spreader bar", "ankles"),
    (r"straitjackets?", "straitjacket", "arms"),
    (r"ropes?|cords?|twine", "rope", "wrists"),
    (r"straps?", "straps", "wrists"),
    (r"chains?", "chain", "wrists"),
    (r"cuffs?|cuffed", "cuffs", "wrists"),
    (r"tape", "tape", "wrists"),
)
# Preserve visual modifiers in continuity text. Hyphenated compounds pass whole;
# arbitrary participles do not, because they are more often verbs than modifiers.
_ADJ = (r"(?:[A-Za-z]+-[A-Za-z]+|"
        # materials
        r"steel|stainless|iron|metal|metallic|nickel|chrome|chromed|brass|"
        r"bronze|copper|pewter|gunmetal|titanium|alumini?um|gold|golden|silver|"
        r"platinum|leather|pleather|suede|velvet|satin|silk|lace|mesh|nylon|"
        r"plastic|rubber|latex|silicone|neoprene|vinyl|pvc|canvas|denim|cotton|"
        r"wool|woollen|linen|rope|wood|wooden|ceramic|glass|resin|"
        # finishes
        r"mirrored|mirror|polished|brushed|burnished|hammered|plated|anodi[sz]ed|"
        r"matte|matt|gloss|glossy|shiny|dull|tempered|hardened|welded|riveted|"
        r"studded|spiked|lined|padded|quilted|ribbed|textured|smooth|"
        # colours
        r"black|white|red|blue|green|grey|gray|brown|pink|purple|tan|cream|navy|"
        r"crimson|scarlet|ivory|amber|olive|"
        # size and build
        r"heavy|light|thin|thick|wide|narrow|short|long|small|large|broad|slim|"
        r"duct|packing|electrical|zip)")

# Body parts, for the hardware whose part is NOT a property of the item.
#
# A collar is the neck and handcuffs are the wrists, and those never need
# looking up. A chain, a rope, straps and tape go wherever the beat puts them,
# and reading their part off the table gave "locks a chain around her ankles"
# as a chain on the WRISTS -- where it then collided with the cuffs already
# there, two things drawn in one place. That is chains interfering.
PARTS = (
    (r"wrists?", "wrists"),
    (r"ankles?", "ankles"),
    (r"necks?|throats?", "neck"),
    (r"mouths?", "mouth"),
    (r"eyes?", "eyes"),
    (r"elbows?", "elbows"),
    (r"knees?", "knees"),
    (r"thighs?", "thighs"),
    (r"waists?", "waist"),
    (r"arms?", "arms"),
    (r"legs?", "legs"),
    (r"hands?", "hands"),
    (r"feet|foot", "feet"),
)
PART_VARIES = frozenset({"chain", "rope", "straps", "tape"})

# Which region a garment leaves uncovered when it comes off. Only what can be
# placed with certainty; a garment that cannot be placed gets no clause, because
# a wrong region is worse than none.
#
# Lives here, with the other vocabularies, because the BARE state is state -- it
# outlives the beat that caused it, and the clause that says so has to be
# writable from any later shot.
REGION_OF = (
    # UNDERWEAR IS IN HERE TOO, on both halves of the body. The torso row has
    # listed a bra since the day it was written -- that is the report it exists
    # for, "a bra coming back on somebody topless" -- and the leg row never got
    # its counterpart, so region_of("thong") answered "". A garment that cannot
    # be placed latches no bare region, so underwear coming off said nothing
    # about the hips in that shot or in any shot after it, and an unspecified
    # region is filled by the model's own prior -- which for a hip is underwear.
    #
    # Worse, a beat saying somebody is NAKED looks each worn garment's region up
    # to take it off the body, so the one garment it could not place stayed
    # "worn" in the state while the text said she was nude. Reported as a thong
    # restored a beat after she undressed to get in the shower.
    #
    # No hardware. A chastity belt is in the layering vocabulary, but it is a
    # restraint: it is latched and held by its own mechanism, and a bare region
    # read off it would argue with that.
    (r"shorts|trousers|jeans|slacks|chinos|skirt|kilt|leggings|joggers|tights|"
     r"pantyhose|jeggings|culottes|tracksuit\s+bottoms|"
     r"panties|knickers|thong|g-?string|briefs|boxers|underwear|undies|"
     r"jockstrap|loincloth", "legs",
     "The legs are bare from the hip down"),
    (r"socks|stockings|hold-?ups|boots|shoes|trainers|sneakers|sandals|heels",
     "feet", "The feet and ankles are bare"),
    # THE CHEST IS THE POINT. This said "The arms and shoulders are bare" and
    # stopped there, so a shirt coming off left the one region a bra occupies
    # unspecified -- and an unspecified region is filled by the model's own
    # prior. Reported as a bra coming back on somebody topless, on a character
    # whose sheet never listed a bra: it was never restored, it was invented.
    (r"top|shirt|blouse|t-?shirt|tee|jumper|sweater|sweatshirt|hoodie|cardigan|"
     r"jacket|coat|tunic|bra|bralette|camisole|vest", "torso",
     "The chest, shoulders and arms are bare skin"),
    (r"gloves|mittens", "hands", "The hands are bare"),
)
# Being in that state rather than arriving at it. "Kate is topless" takes nothing
# off, so every removal path had nothing to remove and no shot ever said what was
# on her chest. "naked eye" and "naked flame" are not people.
NUDITY = (
    (r"topless|bare-?chested|bare-?breasted|shirtless|"
     r"stripped\s+to\s+the\s+waist|strips\s+to\s+the\s+waist", ("torso",)),
    (r"bottomless|bare\s+from\s+the\s+waist\s+down", ("legs",)),
    (r"naked(?!\s+(?:eye|flame))|nude|in\s+the\s+nude|wearing\s+nothing|"
     r"with\s+no\s+clothes|stark\s+naked", ("torso", "legs", "feet")),
)

# Where a limb is held. These all describe the ARMS -- that is why a limb
# position may never be attached to a collar.
POSITIONS = (
    (r"behind\s+(?:her|his|their|the)\s+backs?", "behind the back"),
    (r"(?:above|over)\s+(?:her|his|their|the)\s+heads?|overhead", "above the head"),
    (r"in\s+front\s+of\s+(?:her|his|their)\s+(?:body|chest|waist)",
     "in front of the body"),
    (r"(?:out\s+)?to\s+the\s+sides?|spread\s+wide", "out to the sides"),
    (r"at\s+(?:her|his|their|the)\s+waists?", "at the waist"),
)

# Fixed things hardware can be anchored to. A thing you cannot pick up and walk
# away with.
ANCHORS = (r"walls?|floors?|grounds?|ceilings?|pillars?|columns?|posts?|rails?|"
           r"railings?|bars?|rings?|hooks?|pipes?|radiators?|beams?|girders?|"
           r"struts?|stakes?|eye\s?bolts?|brackets?|cages?|fences?|grates?|"
           r"grilles?|bed\s?frames?|bed\s?posts?|headboards?|bedsteads?|beds?|"
           r"bunks?|benches?|chairs?|tables?|desks?|ladders?|anchors?|loops?")

# Verbs, as VERBS. Participles and -ing forms are unambiguous. The -s forms are
# also plural nouns, so they carry a lookbehind: "the guard chains her collar" is
# a verb and "the chains on the floor" is not.
_DET = (r"(?<!\bthe\s)(?<!\ba\s)(?<!\ban\s)(?<!\bthese\s)(?<!\bthose\s)"
        r"(?<!\btwo\s)(?<!\bsome\s)(?<!\bmore\s)(?<!\bhis\s)(?<!\bher\s)")
APPLY_VERB = (
    r"(?:handcuffed|cuffed|chained|shackled|manacled|locked|padlocked|fastened|"
    r"secured|tethered|bound|tied|strapped|clipped|hooked|bolted|attached|"
    r"anchored|leashed|roped|gagged|blindfolded|collared|taped|trussed|lashed|"
    r"buckled|fettered|"
    r"handcuffing|cuffing|chaining|locking|fastening|securing|tethering|tying|"
    r"strapping|clipping|bolting|attaching|gagging|blindfolding|collaring|"
    r"taping|buckling|binding|shackling|"
    # The particle can sit four words from its verb, exactly as it can for
    # garments: "puts the cuffs on her" is the ordinary way to write it, and
    # requiring "puts on" adjacent read that as no application at all.
    r"(?:puts?|putting|slips?|slipped|snaps?|snapped|clicks?|clicked|clamps?|"
    r"clamped)(?:\s+\S+){0,4}?\s+(?:on|onto|around|shut|closed)|"
    r"closes?\s+around|clicks?\s+shut)"
    r"|" + _DET + r"(?:handcuffs|cuffs|chains|shackles|locks|padlocks|fastens|"
    r"secures|tethers|ties|straps|clips|hooks|bolts|attaches|anchors|leashes|"
    r"ropes|gags|blindfolds|collars|tapes|buckles|binds)")
RELEASE_VERB = (
    r"(?:unlocks?|unlocked|unlocking|uncuffs?|uncuffed|unbinds?|unbound|"
    r"unties?|untied|untying|unbuckles?|unbuckled|unstraps?|unstrapped|"
    r"unclips?|unclipped|unfastens?|unfastened|unshackles?|unshackled|"
    r"ungags?|ungagged|unchains?|unchained|releases?|released|releasing|"
    r"frees?|freed|freeing|cuts?\s+(?:off|away|free)|cut\s+(?:off|away|free)|"
    r"slips?\s+off|slipped\s+off|takes?\s+off|took\s+off|pulls?\s+off|"
    r"lifts?\s+(?:off|away)|removes?|removed|undoes|undid|opens?\s+the)")

# Rooms. A place is somewhere a scene can BE. "door" is not on this list and must
# not go back on it: a door is a thing inside a room, and putting it here moved
# the camera into a door whenever anybody looked at one.
# THE PLACE VOCABULARY. One list, because there were two and they disagreed --
# the engine knew a cell and a warehouse, the sampler did not, so a scene set in
# either was a room to one reader and nowhere to the other. Same fault the
# garment lists had, waiting to be reported.
# MULTI-WORD ROOMS COME BEFORE BARE "room". _MOD is non-greedy, so it tries no
# modifier first and the longest place wins -- but only if the long form is here to
# win with. Without "locker\s+room", "heads to the locker room" read its destination
# as "room": _MOD swallowed "locker" and the capture took what was left, so the film
# was reported as entering "room" and the shot was told to arrive in one.
#
# A gym, a locker room and a court were in no list at all, which is worse than vague:
# with the ORIGIN unknown, travel_anchor emitted nothing, so a beat walking out of a
# gym was never told to walk and the set simply changed under the characters.
PLACES = (r"hallway|hall|corridor|passage|landing|stairwell|staircase|stairs|"
          r"steps|bedroom|bathroom|washroom|kitchen|living\s+room|lounge|"
          r"dining\s+room|locker\s+rooms?|changing\s+rooms?|dressing\s+rooms?|"
          r"waiting\s+rooms?|utility\s+rooms?|gymnasium|gym|classroom|library|"
          r"cafeteria|canteen|reception|laundry|pantry|sauna|balcony|terrace|"
          r"rooftop|elevator|court|pool|showers|shower|store|shop|studio|"
          r"study|office|garage|basement|cellar|attic|loft|porch|"
          r"veranda|garden|yard|driveway|street|alley|car\s?park|lobby|foyer|"
          r"doorway|cell|warehouse|barn|shed|van|truck|room")
# Place words that are also ordinary verbs or everyday nouns. A reader with a
# preposition in front of it ("in the study") can tell which sense is meant; the
# free-text one cannot, and "she steps out", "they study the map" and "he lands
# badly" are all commoner than the rooms they collide with.
# "bar" and "lift" are deliberately NOT places in this file at all: bars are
# restraint hardware here ("chained to the bars") and lifting is what happens to a
# garment or a body, so behind a preposition they would both read as journeys.
PLACE_ALSO_A_VERB = {"steps", "landing", "study", "lounge", "garage", "porch",
                     "court", "pool", "shower", "showers", "store", "shop",
                     "studio", "reception"}
# A room is usually described, not just named -- "the tiled bathroom", "the long
# hallway". Up to three adjectives, non-greedy so the NEAREST room still wins.
_ROOM_MOD = (r"(?:(?!(?:of|the|an?|and|or|to|in|into|from|with|on|at|by|for|her|"
             r"his|their|its|my|our|your)\b)[A-Za-z][A-Za-z-]*\s+){0,3}?")

# Multi-word undergarments come FIRST, so the alternation prefers "chastity belt"
# over the bare "belt" further down -- otherwise the item was recorded as a belt
# and lost the half that says which kind.
# THE GARMENT VOCABULARY. One list, because there were two and they disagreed --
# the sampler's had thong and no chastity belt, the engine's had chastity belt and
# no thong and matched the bare "belt" inside it, so the item came back as a belt.
# Both were fixed on the same day from opposite ends, which is what a second copy
# of one idea costs.
#
# MULTI-WORD FIRST, always. Alternation is leftmost-first at each position, so
# "chastity belt" has to be offered before "belt" or the shorter one wins and the
# half that says which kind is lost.
GARMENT_PHRASES = (r"chastity[\s-]*(?:belts?|devices?|cages?)|g[\s-]?strings?|"
                   r"boxer[\s-]+shorts?|sports?[\s-]+bras?|body[\s-]?suits?|"
                   r"suspender[\s-]+belts?|garter[\s-]+belts?")
GARMENT_WORDS = (r"shirt|blouse|top|t-shirt|tshirt|dressing-gown|vest|waistcoat|"
                 r"gilet|jumper|sweater|"
                 r"sweatshirt|hoodie|cardigan|jacket|blazer|coat|anorak|parka|"
                 r"poncho|cloak|dress|gown|skirt|kilt|sari|kimono|trousers|pants|"
                 r"jeans|slacks|chinos|shorts|leggings|joggers|tracksuit|tights|"
                 r"stockings|socks|shoes|boots|trainers|sneakers|sandals|heels|"
                 r"slippers|gloves|mittens|scarf|hat|cap|beanie|tie|apron|"
                 r"overalls|dungarees|uniform|robe|pyjamas|pajamas|nightdress|"
                 r"nightie|swimsuit|bikini|trunks|romper|jumpsuit|"
                 r"bra|bralette|brassiere|camisole|undershirt|corset|bustier|"
                 r"slip|lingerie|knickers|panties|thong|briefs|boxers|underwear|"
                 r"undies|undercloth|jockstrap|loincloth|nappy|diaper|"
                 r"clothes|clothing|outfit")
_GARMENT = GARMENT_PHRASES + r"|" + GARMENT_WORDS

# Garment verbs. A garment has three states -- on, off, pulled aside -- and the
# shot that CHANGES one has to say both ends of the change, or the model is free
# to open the shot with it already done. Reported as a diaper turning into shorts
# a beat before the beat that put the shorts on.
# THE PARTICLE CAN BE FOUR WORDS AWAY. English puts the object between the verb
# and its particle as happily as after it: "takes off her shorts" and "takes her
# blue shorts off" are the same act, and "puts her blue shorts back on" has four
# words in the gap. Requiring them adjacent read that last one as no change at
# all, so the garment silently stayed off.
_GAP = r"(?:\s+\S+){0,4}?\s+"
TAKES_OFF = (r"(?:takes?|took|taking|pulls?|pulled|peels?|peeled|strips?|"
             r"stripped|shrugs?|slips?|slipped|steps?|gets?|got|kicks?|"
             r"kicked)" + _GAP + r"(?:off|out\s+of)\b"
             r"|\b(?:removes?|removed|removing|discards?|discarded|sheds?|shedding|"
             r"undresses|undressed)")
PUTS_ON = (r"(?:puts?|putting|pulls?|pulled|slips?|slipped|tugs?|tugged|"
           r"steps?|stepped|climbs?|climbed|gets?|got|wriggles?)" + _GAP +
           r"(?:on|into|back\s+on)\b"
           r"|\b(?:dresses?\s+in|dressed\s+in|buttons?|zips?\s+up|fastens?)")
# UNZIPPING A JACKET LEAVES IT ON. These were removals, so "Owen unzips his jacket"
# took the jacket out of every later shot and called his chest bare -- and "rolls up
# his sleeves" a beat later rolled the sleeves of a shirt that had gone with it. They
# open a garment; a beat that also takes it off says so ("and takes it off").
DISPLACES = (r"(?:pulls?|pulled|pushes?|pushed|tugs?|tugged|hikes?|hiked|"
             r"rolls?|rolled|lifts?|lifted|yanks?|yanked|shoves?|shoved)"
             + _GAP + r"(?:aside|up|down|open)\b"
             r"|\b(?:unzips?|unzipped|unbuttons?|unbuttoned|unfastens?|unfastened|"
             r"undoes|undid)\b")

# POSTURES and the _POSTURE list built from it used to live here. Nothing read
# _POSTURE -- it was a second, dead copy of the posture vocabulary, and it had
# already drifted from the live one: it never learned squatting as distinct from
# crouching, and it would not have learned rolling onto a side either. That is the
# divergence the note above _POSTURE_OF warns about, sitting in the same file.
# The live table is _POSTURE_OF, which sampler.py imports by reference
# (sampler._POSTURE_OF IS engine._POSTURE_OF), so there is one of these now.


def _rx(pattern):
    return re.compile(pattern, re.I)


_SPOKEN_SPAN = re.compile(r"<d>.*?</d>|[\"“][^\"“”]{1,400}?[\"”]",
                          re.S)


def spoken_text(text):
    """Only what people SAY, with the markers stripped. "" when nobody speaks."""
    said = []
    for m in _SPOKEN_SPAN.finditer(text or ""):
        s = m.group(0)
        s = s[3:-4] if s.startswith("<d>") else s[1:-1]
        if s.strip():
            said.append(s.strip())
    return " ".join(said)


# WHICH LANGUAGE A LINE IS IN, read off the line itself.
#
# The node used to name English and only English. That clause is not decoration
# -- H3 is joint and multilingual, and an audio branch told a line is spoken but
# never told in WHAT picks one, which is where "sounds like gibberish" came from
# -- but the language it named was hard-coded, so a script written in any other
# language was told its own line is spoken in English and the delivery fought the
# words. Naming nothing is not the way out of that. Naming what the author
# actually wrote is.
#
# A script settles it outright; a Latin alphabet is shared, so common words vote.
_BY_SCRIPT = (
    # Kana before Han: Japanese uses both, so Han alone is what makes it Chinese.
    ("Japanese", r"[぀-ヿ]"),
    ("Korean", r"[가-힯ᄀ-ᇿ]"),
    ("Chinese", r"[一-鿿㐀-䶿]"),
    ("Greek", r"[Ͱ-Ͽἀ-῿]"),
    ("Hebrew", r"[֐-׿]"),
    ("Arabic", r"[؀-ۿݐ-ݿ]"),
    ("Hindi", r"[ऀ-ॿ]"),
    ("Thai", r"[฀-๿]"),
    # Letters Russian lacks, or words it spells differently -- Ukrainian written
    # without і/ї/є still says "що" where Russian says "что".
    ("Ukrainian", r"[ЄЇєіїґ]|\b(?:що|це|ти|але|дуже|треба|дякую|немає)\b"),
    ("Russian", r"[Ѐ-ӿ]"),
)
_BY_SCRIPT_RX = tuple((n, re.compile(p)) for n, p in _BY_SCRIPT)
# Function words. Content words are what a translator changes; these are what
# stay, and a line or two of dialogue carries several.
_BY_WORDS = (
    ("English", "the and is are you that not it to of in for with but what have"),
    ("Spanish", "el la los las que de y no se es por con para pero muy sí está"),
    ("French", "le la les des que de et ne pas est vous je pour avec au ça"),
    ("German", "der die das und nicht ist ich du sie wir mit für auf ein aber"),
    ("Italian", "il lo la che di non è sono per con questo come più sei ma"),
    ("Portuguese", "os as que de não é para com você isso mais está eu sou"),
    ("Dutch", "de het een en niet is ik je dat van voor met maar hij zijn"),
    ("Polish", "nie jest to się na że do co jak ale jestem tak mnie"),
    ("Turkish", "bir bu ve için ne değil çok ben sen var yok ama beni"),
    ("Swedish", "och att det är inte jag du en för med men han hon"),
)
_BY_WORDS_SET = tuple((n, frozenset(w.split())) for n, w in _BY_WORDS)


# THE AUTHOR SAYING WHICH LANGUAGE IT IS.
#
# The word vote needs two function words before it will name a language, and
# ordinary speech often carries one:
#
#     "Oh, du siehst heute toll aus, Schatz!"      -- du
#     "OK, das reicht mir jetzt wirklich!"         -- das
#
# Both are unmistakably German to a reader and both scored 1, so both fell back --
# and in a script whose other lines are English, the fallback is English, so the
# German line was actively TOLD it is English. The line then fights its own
# delivery, which is the failure the whole language hold exists to prevent.
#
# But the author had already said which language it is, in the stage direction,
# where anybody writing this puts it: "says in German". spoken_text() strips
# everything outside the quotes before the vote ever sees it, so the one
# unambiguous statement in the beat was the one thing thrown away.
#
# REQUIRES A SPEECH FRAME, so a nationality is not a language: "in German" and
# "speaks German" match, "the German soldier" and "a German car" do not. Adjectives
# people actually write are allowed between ("in broken German").
_LANG_NAMES = tuple(n for n, _ in _BY_SCRIPT) + tuple(n for n, _ in _BY_WORDS)
_LANG_ADJ = (r"(?:fluent|broken|perfect|rapid|halting|accented|flawless|bad|"
             r"basic|simple|quiet|loud|slow|fast)\s+")
_NAMED_LANG = re.compile(
    r"\b(?:in|into|speaks?|speaking|spoke|spoken|"
    r"switch(?:es|ed|ing)?\s+to|repl(?:y|ies|ied)\s+in|answers?\s+in|"
    r"says?\s+in|said\s+in|ask(?:s|ed)?\s+in)\s+"
    r"(?:" + _LANG_ADJ + r")?"
    r"(" + "|".join(_LANG_NAMES) + r")\b", re.I)


def language_named(text):
    """The language the TEXT ITSELF says is being spoken, or ''.

    The author's own statement, read from the stage direction rather than voted
    for out of the line. See _NAMED_LANG for why a speech frame is required."""
    m = _NAMED_LANG.search(str(text or ""))
    if not m:
        return ""
    said = m.group(1).lower()
    return next((n for n in _LANG_NAMES if n.lower() == said), "")


def language_of(text, fallback="English", named=""):
    """The language `text` is written in, or `fallback` when it cannot tell.

    Conservative on purpose: naming the WRONG language is worse than naming the
    one the author most likely wanted, so a Latin-alphabet guess has to win by a
    clear margin before it displaces the fallback."""
    t = str(text or "")
    if not t.strip():
        return fallback
    for name, rx in _BY_SCRIPT_RX:
        if rx.search(t):
            return name
    words = set(re.findall(r"[^\W\d_]+", t.lower(), re.UNICODE))
    if not words:
        return fallback
    scores = sorted(((len(words & ws), n) for n, ws in _BY_WORDS_SET), reverse=True)
    best, runner = scores[0], scores[1]
    # Two hits, and ahead of everything else. One shared word ("no" is Spanish and
    # English both) is not a language.
    if best[0] >= 2 and best[0] > runner[0]:
        return best[1]
    # The author said so, and the vote could not tell. A statement beats a guess
    # that abstained -- but NOT a script: Cyrillic is not a matter of opinion, and
    # that branch has already returned above.
    if named:
        return named
    return fallback


def _outside_speech(text):
    """The beat with everything anybody SAYS taken out.

    What a character says is not stage direction. The commonest thing to talk
    about is something that is NOT in the room -- "McKenna where are you?" is how
    absence gets written -- and reading a spoken name as a staged one put a full
    description of the missing person into the shot, so the model drew her.

    Both markers, because both exist in the pipeline: <d> once mark_dialogue has
    run, plain quotes before it."""
    return _SPOKEN_SPAN.sub(" ", text or "")


# THREE modifiers, not two: "mirrored stainless steel collar" is three words and
# a noun, and the third was the first to be dropped.
_HW_ONE = _rx(r"\b(" + _ADJ + r"(?:\s+" + _ADJ + r"){0,2}\s+)?("
              + "|".join(p for p, _n, _pt in HARDWARE) + r")\b")
_PART_ONE = _rx(r"\b(" + "|".join(p for p, _n in PARTS) + r")\b")
# A NOUN carries a determiner, a number or an adjective; a VERB follows its
# subject. "Sam chains her collar to the ring" introduces nothing to draw -- it
# fastens the collar that is already named -- and reading that verb as an item
# put a chain on the wrists of somebody with nothing on their wrists.
#
# "and" is deliberately absent: "...to the ring and chains her ankles together"
# is a second verb, and letting a conjunction vouch for a noun brought the
# phantom straight back.
_NOUN_BEFORE = _rx(r"(?:\b(?:a|an|the|her|his|its|their|my|your|our|this|that|"
                   r"these|those|one|two|three|several|more|another|in|with|by|"
                   r"of|on|from)\b|[,;:(])\s*(?:" + _ADJ + r"\s+){0,3}$")
_POSITION = [(_rx(r"\b" + p + r"\b"), name) for p, name in POSITIONS]
# What can stand in front of an anchor. "one ring", "the other ring", "a second
# hook" are the same fixture as "the ring", and the six-word list read them as no
# anchor at all -- so a collar chained to ONE OF TWO rings was not a restraint at
# all, nothing latched, and every later shot forgot it. Two people chained to two
# rings lost both.
ANCHOR_DET = (r"(?:the|a|an|her|his|its|their|one|another|each|either|that|"
              r"this|both)\s+(?:(?:other|second|third|first|far|near|nearest|"
              r"opposite|left|right|upper|lower|top|bottom|same|nearby|steel|"
              r"iron|metal|heavy|small|large|wooden|old|thick)\s+){0,2}")
_ANCHOR_AT = _rx(r"\bto\s+" + ANCHOR_DET + r"(" + ANCHORS + r")\b")
_APPLY = _rx(r"\b(?:" + APPLY_VERB + r")\b")
_RELEASE = _rx(r"\b(?:" + RELEASE_VERB + r")\b")
_PLACE_IN = _rx(r"\b(?:in|into|inside|through|down|along|across|to|onto|at)\s+"
                r"(?:the|a|an|her|his|their)\s+(" + _ROOM_MOD + r"(?:" + PLACES
                + r"))\b")
_PLACE_WORD = _rx(r"\b(?:" + PLACES + r")\b")
_GARMENT_ONE = _rx(r"\b(" + _ADJ + r"(?:\s+" + _ADJ + r"){0,2}\s+)?("
                   + _GARMENT + r"s?)\b")
_TAKES_OFF = _rx(r"\b" + TAKES_OFF + r"\b")
_PUTS_ON = _rx(r"\b" + PUTS_ON + r"\b")
_DISPLACES = _rx(r"\b" + DISPLACES + r"\b")
# ...and the sentence can still finish the job after the garment is named. "Kate
# unzips the denim skirt and steps out of it": the unzip opens it, the rest of the
# sentence takes it off, and the removal verb comes after the item where the reader
# above does not look.
_OPENS_GARMENT = _rx(r"\b(?:unzips?|unzipped|unbuttons?|unbuttoned|unfastens?|unfastened|"
                     r"undoes|undid|unhooks?|unhooked|unclasps?|unclasped)\b")
_COMPLETES_OFF = _rx(r"\b(?:off|out\s+of|away|lets?\s+(?:it|them)\s+(?:fall|drop|slide)|"
                     r"drops?\s+(?:it|them)|falls?\s+(?:to|down|away|off))\b")
_MOVES = _rx(r"\b(?:walks?|walked|walking|goes|go|went|going|runs?|ran|running|"
             r"steps?|stepped|stepping|moves?|moved|moving|enters?|entered|"
             r"leaves?|left|leaving|crosses|crossed|crossing|climbs?|climbed|"
             r"heads?|headed|returns?|returned|arrives?|arrived)\b")


def hardware_spans(text):
    """Every piece of hardware named, as (canonical, part, as-written, at).

    `at` is where it sits in the text, and it is not decoration: a beat that
    cuffs the wrists BEHIND THE BACK and locks a collar CHAINED TO THE WALL has
    two modifiers and two items, and attaching either modifier to both gives
    handcuffs chained to a wall they were never near. Modifiers bind to the
    nearest item, which needs positions to work out.

    ALL of it, too. The old reader returned only the longest single match, so a
    beat that put on cuffs and a collar recorded one and lost the other for the
    rest of the film."""
    text = text or ""
    parts = part_spans(text)
    raw, verb_ats = [], []
    for m in _HW_ONE.finditer(text):
        adj, noun = (m.group(1) or "").strip(), m.group(2)
        canon, part = next((n, pt) for p, n, pt in HARDWARE
                           if re.fullmatch(p, noun, re.I))
        # A VERB IS NOT AN ITEM. Only the words that are also verbs need asking,
        # and only "chain" is one that the table would otherwise turn into a
        # restraint on a part the beat never mentions.
        if canon == "chain" and not noun.lower().endswith("ed") \
                and not _NOUN_BEFORE.search(text[:m.start()]):
            verb_ats.append(m.start())
            continue
        written = f"{adj} {noun}".strip().lower()
        # A PARTICIPLE FINDS IT AND DOES NOT NAME IT. "is handcuffed" is how the
        # passive voice writes hardware, but an item recorded as "handcuffed"
        # renders as "The handcuffed stay closed and fastened". Quote the noun.
        if noun.lower().endswith("ed"):
            written = f"{adj} {canon}".strip().lower()
        raw.append([canon, part, written, m.start()])
    # WHERE IT GOES, for the things that go anywhere. Bound after every item is
    # known, so the part attaches to the nearest one and a beat naming two of
    # them does not give both the same place.
    _ats = [(c, at) for c, _pt, _w, at in raw]
    _tether = []
    for row in raw:
        if row[0] not in PART_VARIES:
            continue
        _pt = _nearest_part(parts, row[3], _ats)
        if _pt:
            row[1] = _pt
        elif any(c != row[0] for c, _a in _ats) and _runs_to(text, row[3]):
            # NO PART OF ITS OWN, beside something that has one, and joined by
            # "to": it is that thing's TETHER, not a restraint holding a pair of
            # wrists nobody mentioned. "clips a chain to her collar" was a chain
            # on the wrists AND a collar on the neck -- two things to draw where
            # the beat put one.
            #
            # "to" matters. "gags her with duct tape" names the gag's MATERIAL
            # by the same shape, and folding that away lost the tape entirely.
            _tether.append(row)
    raw = [r for r in raw if r not in _tether]
    # A VERB STILL FASTENS SOMETHING, and what it fastens is either an item the
    # beat names or a part of the body. "...chains her collar to the ring and
    # chains her ankles together" is both, in that order: the first verb belongs
    # to the collar and introduces nothing, the second puts a chain on the
    # ankles. Recorded one and lost the other, which is two restraints becoming
    # one -- chains interfering.
    for _vat in verb_ats:
        _pt = _nearest_part(parts, _vat, _ats)
        if _pt and not any(c == "chain" and pt == _pt for c, pt, _w, _a in raw):
            raw.append(["chain", _pt, "chain", _vat])
    # Nothing named at all: somebody is chained somewhere and the beat never
    # says where on them. The anchor is still real, so it holds the body rather
    # than inventing a pair of wrists to hold.
    if verb_ats and not raw and anchor_in(text):
        raw.append(["chain", "body", "chain", verb_ats[0]])
    out, seen = [], {}
    for canon, part, written, at in raw:
        # Keyed by the PAIR. Two chains on two parts are two restraints -- a
        # beat chaining a collar and the ankles recorded one and lost the other
        # -- while "collar" then "steel collar" is one collar, same part, and
        # keeps the fuller wording.
        key = (canon, part)
        if key in seen:
            i = seen[key]
            if len(written) > len(out[i][2]):
                out[i] = (canon, part, written, out[i][3])
            continue
        seen[key] = len(out)
        out.append((canon, part, written, at))
    return out


_REGION_RX = tuple((_rx(r"\b(?:" + p + r")\b"), region, said)
                   for p, region, said in REGION_OF)
_NUDITY_RX = tuple((_rx(r"\b(?:" + p + r")\b"), regions) for p, regions in NUDITY)


def region_of(garment):
    """The region a garment covers, or "" when it cannot be placed."""
    for rx, region, _said in _REGION_RX:
        if rx.search(str(garment or "")):
            return region
    return ""


def nudity_in(text):
    """The regions a beat says are bare BY DESCRIPTION, widest match first."""
    out = []
    for rx, regions in _NUDITY_RX:
        if rx.search(text or ""):
            for r in regions:
                if r not in out:
                    out.append(r)
    return out


def _bare_on(p, regions):
    for r in ([regions] if isinstance(regions, str) else regions):
        if r and r not in p.bare:
            p.bare.append(r)


def _bare_off(p, regions):
    for r in ([regions] if isinstance(regions, str) else regions):
        if r in p.bare:
            p.bare.remove(r)


def part_spans(text):
    """Every body part named, as (name, at)."""
    out = []
    for m in _PART_ONE.finditer(text or ""):
        out.append((next(n for p, n in PARTS
                         if re.fullmatch(p, m.group(1), re.I)), m.start()))
    return out


_RUNS_TO = _rx(r"^\s*\w*\s*(?:to|onto|from)\b")


def _runs_to(text, at):
    """Does the item at `at` run TO something -- is it a tether?

    Read just past the word, so "a chain to her collar" and "a chain running to
    the ring" both answer yes and "duct tape" answers no."""
    return bool(_RUNS_TO.search(text[at:][_first_gap(text[at:]):])) \
        or bool(anchor_in(text))


def _first_gap(s):
    """Index just past the first word of `s`."""
    m = re.search(r"\s", s)
    return m.start() if m else len(s)


def _nearest_part(parts, at, ats):
    """The part belonging to the item at `at`, or "".

    English puts it after: "a chain around her ankles", "chains her ankles
    together". So the first part named AFTER this item wins, unless another
    item is named in between -- that one owns it instead."""
    later = [(p, q) for p, q in parts if q > at]
    for name, q in later:
        if any(at < other < q for _c, other in ats):
            break
        return name
    return ""


def position_spans(text):
    """Every limb position named, as (name, at)."""
    out = []
    for rx, name in _POSITION:
        m = rx.search(text or "")
        if m:
            out.append((name, m.start()))
    return sorted(out, key=lambda x: x[1])


def position_in(text):
    """Where the arms are held. '' when the text does not say."""
    got = position_spans(text)
    return got[0][0] if got else ""


def anchor_spans(text):
    """Every fixed thing hardware is fastened to, as (name, at).

    The VERB is required. "to the <thing>" on its own is movement -- "she sinks
    to the floor", "he walks to the table" -- and reading those as fastenings
    latched a restraint over furniture somebody merely walked towards."""
    t, out = text or "", []
    for m in _ANCHOR_AT.finditer(t):
        # The fastening verb has to be in THIS clause, not somewhere earlier in
        # the paragraph.
        clause = re.split(r"[.;!?]", t[:m.start()])[-1]
        if _APPLY.search(clause) or re.search(
                r"\b(?:chains?|ropes?|cords?|cables?|leash(?:es)?|straps?|"
                r"tethers?|links?|lines?)\s+(?:\S+\s+){0,4}?"
                r"(?:runs?|holds?|leads?|stretch(?:es)?|extends?|goes|hangs?)\b",
                clause, re.I):
            out.append((re.sub(r"\s+", " ", m.group(1).lower()), m.start()))
    return out


def anchor_in(text):
    """What hardware is fastened to. '' when the text fastens nothing."""
    got = anchor_spans(text)
    return got[0][0] if got else ""


def place_in(text):
    """The room this text puts the shot in. '' when it names none.

    Behind a preposition, so a room has to be somewhere somebody IS. "Ana looks
    at the door" names no room -- and a door is not on the list in any case."""
    text = text or ""
    m = _PLACE_IN.search(text)
    if not m:
        return ""
    clause = re.split(r"[.;!?]", text[:m.end()])[-1]
    present = re.search(r"\b(?:is|are|was|were|stands?|sits?|waits?|lies?|remains?)\b",
                        clause, re.I)
    if not _MOVES.search(clause) and not present:
        return ""
    got = re.sub(r"\s+", " ", m.group(1).lower()).strip()
    # A BARE "room" NAMES NOWHERE. "Ana walks into the room" says she goes
    # inside, not which room -- and taking it as a place produced "The shot is
    # in the room, not the room the scene text names", which contradicts itself
    # in one sentence. Modified, it is a real place: "the far room", "the back
    # room" and "the next room" all distinguish themselves from where we were.
    return "" if got == "room" else got


def garments_in(text):
    """Garments named, as written, in order."""
    out, seen = [], set()
    for m in _GARMENT_ONE.finditer(text or ""):
        adj, noun = (m.group(1) or "").strip(), m.group(2)
        phrase = f"{adj} {noun}".strip().lower()
        key = noun.lower().rstrip("s")
        if key not in seen:
            seen.add(key)
            out.append(phrase)
    return out


_CLAUSE_BOUNDARY = re.compile(r"[,;]|\b(?:and|while)\b", re.I)


def _clause_at(text, at, boundaries=None):
    """Return the clause containing character offset `at` and its start offset."""
    if boundaries is None:
        boundaries = list(_CLAUSE_BOUNDARY.finditer(text or ""))
    lo = max((m.end() for m in boundaries if m.end() <= at), default=0)
    hi = min((m.start() for m in boundaries if m.start() > at), default=len(text))
    return text[lo:hi], lo



# Hardware, not clothing. Taking clothes off does not unlock anything, so these
# are kept out of the garment answer -- the standing rule is that hardware is
# cleared by an explicit `remove:` and by nothing else.
_NOT_CLOTHING = re.compile(
    r"^(?:handcuffs?|cuffs?|shackles?|manacles?|chains?|ropes?|cords?|straps?|"
    r"collars?|gags?|blindfolds?|restraints?|bindings?|tape|ties?|harness|"
    r"straitjacket|spreader|hogtie|clamps?|clips?)$", re.I)
_PHRASE_ONE = _rx(r"\b(?:" + GARMENT_PHRASES + r")\b")
_WORD_ONE = _rx(r"^(?:" + GARMENT_WORDS + r")s?$")
# The same list with NO optional plural, which is what says whether a trailing "s"
# belongs to the word or was added to it. The vocabulary is clean on this: garments that
# are inherently plural are listed only in the plural (boots, jeans, shorts, panties,
# tights, leggings, socks, knickers, trousers, gloves) and the rest only in the singular
# (skirt, vest, top), so "stem is itself a garment" is an exact test and not a guess.
_WORD_EXACT = _rx(r"^(?:" + GARMENT_WORDS + r")$")


def singular_garment(word):
    """A garment word as the VOCABULARY spells it, so two readers cannot disagree.

    Reported: several women take their skirts off and the skirts are back in the next
    beat. "their skirts" yields the token "skirts" while the sheet says "a denim
    skirt", and every reader downstream looks the token up in the sheet -- the scrub
    by pattern, infer_removals by entry head -- so a plural garment matched nothing
    and the removal silently did nothing at all. One woman undressing wrote "her
    skirt" and worked; the moment the subject went plural so did the garment.

    A trailing "s" comes off only when the stem is ITSELF a garment word, which is an
    exact test here and not a guess: the vocabulary lists inherently plural garments
    only in the plural (boots, jeans, shorts, panties, tights, leggings, socks,
    knickers, trousers, gloves) and the rest only in the singular."""
    low = str(word or "").lower().strip("-")
    if low.endswith("s") and _WORD_EXACT.match(low[:-1]):
        return low[:-1]
    return low


def garment_words(text):
    """Every garment named, as HEAD WORDS, restraints excluded.

    What the wardrobe logic tracks and compares: ["jeans", "chastity belt"]. Its
    sister garments_in keeps the adjectives, because a shot has to SAY "blue
    jeans" while the tracking only has to know they are jeans.

    Multi-word entries are read first and their words removed, so the single-word
    pass cannot also find "belt" inside "chastity belt" and record the item twice
    under two names."""
    text = text or ""
    out = []
    for m in _PHRASE_ONE.finditer(text):
        phrase = re.sub(r"[\s-]+", " ", m.group(0)).strip().lower()
        if phrase not in out:
            out.append(phrase)
    text = _PHRASE_ONE.sub(" ", text)
    for word in re.findall(r"\b[\w-]{3,}\b", text):
        low = word.lower().strip("-")
        if _NOT_CLOTHING.match(low):
            continue
        if not _WORD_ONE.match(low):
            continue
        # The token the sheet wrote, not the one the beat happened to inflect. See
        # singular_garment -- shared with infer_removals so the two cannot disagree.
        low = singular_garment(low)
        if low not in out:
            out.append(low)
    return out

# A POSTURE DENIED IS NOT A POSTURE TAKEN.
#
# Reported: a woman chained by the ankles and forced into a squat stood up anyway.
# The chain clauses were all correct -- the metal "already drawn to its full length,
# so the position it fixes is the position that keeps" was on every shot after the
# squat. What sat beside it was the posture latch saying she was STANDING, because
# "She cannot stand." matched `stand` and nothing looked at the `cannot`. The latch
# then carried that forward, so every later shot asserted, flatly and positively,
# the one thing the chains were there to prevent. At cfg 1 a positive statement wins.
#
# The same shape as _in_a_request, which already suppresses a posture that is ASKED
# for rather than taken. Attempts are here too: "tries to stand", "struggles to get
# up", "strains to rise" are all bodies that have NOT got there, and reading them as
# arrival is the same error in a friendlier disguise.
#
# A SHORT WINDOW on purpose -- the five words before the verb. The cue always sits
# immediately in front of it ("cannot stand", "no longer able to stand"), and a wider
# reach would let a `cannot` from a different clause silence a real posture.
_POSTURE_DENIED = _rx(
    r"\b(?:cannot|can\s*not|can['\u2019]?t|could\s*not|could\s*n['\u2019]?t|"
    r"unable|never|not\s+able|no\s+longer\s+able|"
    r"does\s*n['\u2019]?t|does\s+not|do\s*n['\u2019]?t|did\s*n['\u2019]?t|did\s+not|"
    r"will\s+not|wo\s*n['\u2019]?t|fail(?:s|ed|ing)?|"
    r"tr(?:y|ies|ied|ying)|attempt(?:s|ed|ing)?|struggl(?:e|es|ed|ing)|"
    r"strain(?:s|ed|ing)?|fight(?:s|ing)?|want(?:s|ed)?|need(?:s|ed)?|"
    r"told|ordered|asked|begs?|begged)\b")


def denied_posture(text, at):
    """Is the posture verb at `at` negated, or only attempted, by what precedes it?"""
    before = str(text or "")[:max(0, int(at))]
    # A cue belongs to its OWN clause. Without stopping at the boundary, "McKenna
    # cannot kneel, so she sits" reached back past the comma and silenced the sitting
    # -- suppressing a posture the beat plainly states, which is the same class of
    # error in the other direction.
    cut = 0
    for _m in re.finditer(r"[,;:.!?]|\b(?:so|and|but|then|yet|while|as|before|after)\b",
                          before, re.I):
        cut = _m.end()
    window = " ".join(re.findall(r"[\w'\u2019]+", before[cut:])[-5:])
    return bool(_POSTURE_DENIED.search(window))


def posture_in(text):
    """The posture this beat puts a body in. '' when it does not.

    Reads the one table. sampler.posture_in answers a different question -- which
    PERSON each posture belongs to, clause by clause -- and keeps its own reader
    for that, but off the same vocabulary."""
    t = text or ""
    hits = sorted((m.start(), name) for name, rx in _POSTURE_OF
                  for m in rx.finditer(t) if not denied_posture(t, m.start()))
    return hits[0][1] if hits else ""



# A determiner, INCLUDING A POSSESSIVE NAME. "Dana lifts up McKenna's skirt" and
# "enters McKenna's bedroom" both failed on a list of the/her/his/their/a/an, in
# two different readers, fixed weeks apart. One constant, so the next reader that
# needs it cannot get a narrower copy.
DET_POSS = r"(?:the|her|his|their|its|a|an|\w+['’]s)"
_DET_POSS = DET_POSS

# ---------------------------------------------------------------------------
# WARDROBE: what moved, and what was put back.
#
# Moved here from sampler.py, whole, because the pair went wrong three separate
# ways while it was split across two files: a possessive name ("Dana lifts up
# McKenna's skirt") matched neither reader, the restore verbs covered only the
# direction nobody writes, and the layering read displacements a beat before the
# latch recorded them. They name the same garments, they have to agree with the
# same sheet, and they now sit beside the vocabulary they both read.
# ---------------------------------------------------------------------------

# Shared with the removal readers still in sampler.py, which is why it is here
# rather than moved: this is the copy, and the sampler imports it.
_STRIP_VERB = (r"take[sn]?|took|taking|pull(?:s|ed|ing)?|peel(?:s|ed|ing)?|"
               r"strip(?:s|ped|ping)?|cut(?:s|ting)?|rip(?:s|ped|ping)?|tear[s]?|tore|"
               r"slip(?:s|ped)?|shrug(?:s|ged)?|yank(?:s|ed)?|tug(?:s|ged)?|"
               r"toss(?:es|ed)?|throw[s]?|threw|"
               # How clothes actually come off, in the words people write it in.
               # Without these a beat took the garment off on screen while the scene
               # kept saying it was worn -- and the scene is re-stamped into every
               # later shot, so it came back on and stayed on.
               r"kick(?:s|ed|ing)?|step(?:s|ped|ping)?|lift(?:s|ed|ing)?|"
               r"slide[s]?|slid|wriggle[sd]?|wiggle[sd]?|work(?:s|ed)?")
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
_UNDO_VERB = (r"remove[sd]?|removing|undress(?:es|ed)?|shed(?:s|ding)?|unzip(?:s|ped)?|"
              r"unbutton(?:s|ed)?|unhook(?:s|ed)?|unclasp(?:s|ed)?|unfasten(?:s|ed)?|"
              # Hardware comes off by being UNDONE, and these were missing: a beat
              # saying "unlocks the belt" left it described as worn for the rest of
              # the film, because nothing here read as a removal at all.
              r"unlock(?:s|ed)?|unbuckle[sd]?|unclip(?:s|ped)?|unstrap(?:s|ped)?|"
              r"unlace[sd]?|untie[sd]?|unties|unwrap(?:s|ped)?|"
              r"undo(?:es)?|undid")


# WHERE A GARMENT ENDS UP ONCE IT IS OFF. A thing on the floor is not on a body,
# and this is how a beat says so -- by DESTINATION, not by verb. Two readers need
# the same list, for opposite reasons: the removal reader to call it a removal,
# the restore reader to stop calling it one. "Lets the skirt fall" drops a lifted
# skirt back over her legs; "lets the thong fall to the floor" is the thong coming
# off, and the only difference between those two sentences is this list.
FLOOR = (r"floor|ground|tiles?|tiling|lino|mat|bath\s*mat|rug|carpet|deck|boards|"
         r"concrete|grass|sand|bed|sofa|couch|chair|seat|stool|bench|basket|"
         r"hamper|laundry|pile|heap")
TO_THE_FLOOR = (r"(?:to|on|onto|into|in)\s+(?:the|a|an|her|his|their)?\s*"
                r"(?:" + FLOOR + r")\b")
_LANDS_OFF = _rx(r"\s*(?:fall(?:s|ing)?|drop(?:s|ping)?|land(?:s|ing)?)?\s*"
                 + TO_THE_FLOOR)

_DISPLACE_WAY = (r"back\s+up|back\s+down|down|up|aside|open|back|"
                 r"off\s+(?:one|her|his|their)\s+shoulders?")
_DISPLACE = re.compile(
    r"\b(?:" + _STRIP_VERB + r"|push(?:es|ed|ing)?|shove[sd]?|roll(?:s|ed|ing)?|"
    r"hitch(?:es|ed)?|hike[sd]?|open(?:s|ed)?|undo(?:es)?|undid|unzip(?:s|ped)?|"
    r"unbutton(?:s|ed)?|unfasten(?:s|ed)?|unhook(?:s|ed)?|unclasp(?:s|ed)?|"
    # LIFTING A SKIRT IS DISPLACING IT, and none of these were here. Asked
    # for directly: "when the skirt has been lifted up to show the chastity
    # belt, that's when it should be shown". Lifting was not read as moving
    # anything, so the belt stayed covered through the shot that uncovers it.
    r"lift(?:s|ed|ing)?|raise[sd]?|rais(?:es|ed|ing)|hoist(?:s|ed|ing)?|"
    r"hold(?:s|ing)?|held|gather(?:s|ed|ing)?|bunch(?:es|ed|ing)?)\s+"
    r"(?:(" + _DISPLACE_WAY + r")\s+)?"
    r"(" + _DET_POSS + r"\s+)?([\w][\w\- ]{0,28}?)"
    r"(?:\s+(" + _DISPLACE_WAY + r"))?"
    r"(?=[.,;:!?]|\s+(?:and|to|so|while|as|then)\b|$)", re.I)


def scene_name_for(head, scene):
    """The sheet's OWN full name for a garment, found by its head noun. "" if absent.

    A beat calls a thing whatever is convenient -- "the shorts" for what the sheet
    dressed her in as "blue jeans shorts". Anything the node then says about it has
    to use the SHEET's words: a shot carrying both names is a shot describing two
    garments, and the model draws the bare one however it likes. That is a garment
    invented out of the node's own text, which is the worst kind.

    The entry is read back from the sheet: its modifiers are the words before the
    head noun in the same comma-separated item, and no further -- a name from the
    entry before it would attach one garment's colour to another."""
    head = (head or "").strip().lower()
    if not head or not scene:
        return ""
    best = ""
    for line in str(scene).split("\n"):
        # Only the wardrobe side of "Name: she, 22, blue jeans shorts".
        line = line.split(":", 1)[-1]
        for item in re.split(r"[,;.]", line):
            # A <Picture N> tag is not part of the garment's NAME. "chastity belt
            # <Picture 2>" ends in "2", so the head-noun match failed and the belt
            # fell back to the beat's bare word -- while an untagged garment in the
            # same sheet expanded correctly. The tagged garment is exactly the one
            # a reference is pinning, so it is the worst one to describe loosely.
            item = re.sub(r"<\s*picture\s+\d+\s*>", " ", item, flags=re.I)
            item = re.sub(r"\s+", " ", item).strip()
            # ONE ENTRY CAN HOLD SEVERAL GARMENTS, and the name is the garment's own
            # part of it. "navy jacket over a white shirt" ends in "shirt", so the
            # whole entry came back as the shirt's name and "the navy jacket over a
            # white shirt open" was said about a shirt being unbuttoned. Split on the
            # words that join garments, and cut a "with ..." tail, which describes a
            # garment rather than naming it.
            for part in re.split(r"\s+(?:over|under|beneath|underneath|on\s+top\s+of|and)\s+",
                                 item, flags=re.I):
                part = re.split(r"\s+with\s+", part, flags=re.I)[0].strip()
                if not part or part.split()[-1].lower() != head:
                    continue
                # Drop a leading article or possessive; they are not description.
                part = re.sub(r"^(?:a|an|the|her|his|their|its)\s+", "", part, flags=re.I)
                # The longest entry wins: a sheet that names it twice described it
                # most fully once, and the fuller name is the one worth carrying.
                if len(part) > len(best):
                    best = part
    # The author's OWN capitalisation. Lowercasing turned "PVC" into "pvc" and
    # "Shiny white crop top" into all-lowercase -- a different token sequence than
    # was written, for a brand or material name that is capitalised for a reason.
    # Only the matching above is case-insensitive; what comes back is what they typed.
    return best


def displaced_garments(beat, scene):
    """[(garment, how)] this beat MOVES without taking off. [] when none.

    Same two conditions infer_removals uses, and for the same reason: the beat has
    to stage it, and the scene has to already say the thing is worn. A displacement
    invented for something nobody is wearing describes a garment into existence."""
    if not beat or not scene:
        return []
    out, seen = [], set()
    low = scene.lower()
    for m in _DISPLACE.finditer(beat):
        way = (m.group(1) or m.group(4) or "").lower().strip()
        thing = re.sub(r"\s+", " ", (m.group(3) or "")).strip().lower()
        # SOME VERBS CARRY THEIR OWN DIRECTION. "lifts her skirt" says which way
        # by saying lift, and the direction word this pattern wants is simply not
        # written -- so the match was thrown away for having no `way`, and the one
        # beat that uncovers the layer beneath did nothing. Read on the matched
        # text rather than a new capture group, which would renumber the rest.
        if not way and re.match(r"\s*(?:lift|rais|hoist|gather|bunch)", m.group(0),
                                re.I):
            way = "up"
        # ...and UNDOING a garment opens it. "Owen unzips his jacket" was a removal,
        # then (once it was not) nothing at all -- the jacket went back to being
        # described closed on the next shot, which is a jacket zipping itself up
        # across a cut.
        if not way and re.match(r"\s*(?:unzip|unbutton|unfasten|unhook|unclasp|undo|undid)",
                                m.group(0), re.I):
            way = "open"
        if not way or not thing or thing in seen:
            continue
        # The garment has to be one the scene already dresses them in, and the head
        # noun is what matches: "her denim shorts" is the scene's "blue denim shorts".
        head = thing.split()[-1]
        if len(head) < 3 or head not in low:
            continue
        seen.add(thing)
        # ...and it is the SCENE'S name that gets carried forward, not the beat's.
        # A beat says "pulls the shorts back up" for what the sheet calls "blue
        # jeans shorts", and the guard echoed the beat: the shot then carried a
        # bare "the shorts" beside the sheet's full name, and a model handed two
        # differently-named garments draws two different garments. The shorts came
        # back in a different colour and cut -- invented, from the node's own text.
        thing = scene_name_for(head, scene) or thing
        # "back up" and "back down" say the direction in their second word.
        way = re.sub(r"^back\s+", "", re.sub(r"\s+", " ", way))
        out.append((thing, "pulled " + way if way in ("down", "up", "aside", "back")
                    else way))
    return out


# Putting it right without naming it: "pulls them back up". A pronoun cannot be
# matched against the wardrobe, but if exactly one garment is displaced there is only
# one thing it can mean -- and leaving it displaced is the error that shows.
# PUTTING IT BACK. A displaced garment is still worn and the node keeps saying
# where the beat left it -- so the beat that puts it right has to be read, or the
# skirt stays lifted for the rest of the film and whatever was under it stays on
# show. Reported: "McKenna lets it fall" did nothing, because the only restores
# recognised were pull/tug/hitch/hike/yank/push with a pronoun and a direction.
#
# What actually gets written is mostly the opposite: a lifted skirt is LET FALL,
# DROPPED, LOWERED, SMOOTHED DOWN, STRAIGHTENED, FIXED or simply LET GO of, and
# none of those has a direction word in it at all.
_RESTORE_VERB = (r"(?:let(?:s|ting)?(?:\s+go\s+of)?|drop(?:s|ped|ping)?|"
                 r"lower(?:s|ed|ing)?|smooth(?:s|ed|ing)?|straighten(?:s|ed|ing)?|"
                 r"fix(?:es|ed|ing)?|rearrang(?:e|es|ed|ing)|"
                 r"replac(?:e|es|ed|ing)|put(?:s|ting)?|tidy|tidies|tidied|"
                 r"cover(?:s|ed|ing)?\s+(?:herself|himself|themselves|up)|"
                 r"pull|tug|hitch|hike|yank|push)")
# The old pronoun form, plus the new verbs, still with no garment named.
_PUT_BACK = re.compile(
    r"\b(?:pull|tug|hitch|hike|yank|push)(?:s|ed|ing)?\s+"
    r"(?:it|them|these|those)\s+(?:back\s+)?(?:up|down|closed|shut|together)\b"
    r"|\b(?:pull|tug|hitch|hike|yank|push)(?:s|ed|ing)?\s+"
    r"(?:it|them)\s+back\b"
    r"|\b" + _RESTORE_VERB + r"(?:s|ed|ing)?\s+"
    r"(?:it|them|these|those)\s+(?:fall|drop|go|back|down|straight)\b"
    r"|\blet(?:s|ting)?\s+(?:it|them)\s+fall\b"
    r"|\b(?:cover(?:s|ed|ing)?\s+(?:herself|himself|themselves)\s+(?:back\s+)?up)\b",
    re.I)
# ...and the same act with the garment NAMED: "lets the skirt fall", "smooths her
# skirt down". The garment has to be one the sheet already dresses them in, which
# is the same condition displaced_garments uses.
_PUT_BACK_NAMED = re.compile(
    r"\b" + _RESTORE_VERB + r"\s+"
    + _DET_POSS + r"\s+([\w][\w\- ]{0,28}?)"
    r"(?:\s+(?:fall|drop|down|back|straight|up|closed|shut|together))?"
    r"(?=[.,;:!?]|\s+(?:and|to|so|while|as|then|over|again)\b|$)", re.I)


def puts_it_back(beat):
    """Does this beat put a displaced garment right without naming it?"""
    return bool(_PUT_BACK.search(beat or ""))


def restored_garments(beat, scene):
    """[garment] this beat puts back, by name. [] when it names none.

    Same two conditions as displaced_garments: the beat has to stage it, and the
    sheet has to already dress them in the thing. The sheet's own name is what
    comes back, so a restore keyed on "her skirt" clears a displacement stored as
    "long grey skirt"."""
    if not beat or not scene:
        return []
    out, low = [], scene.lower()
    for m in _PUT_BACK_NAMED.finditer(beat):
        # ...UNLESS IT LANDS ON THE FLOOR. These verbs are the restore vocabulary
        # because that is what people write for a lifted skirt -- let fall, drop,
        # lower, let go of -- and the identical words take a garment OFF when the
        # sentence says where it lands. "Lets the thong fall to the floor" was read
        # as putting the thong back on: the author's removal, enacted backwards, and
        # from there the sheet described it as worn for the rest of the film.
        # Reported as a thong restored after she undressed. See FLOOR.
        if _LANDS_OFF.match(beat[m.end():]):
            continue
        thing = re.sub(r"\s+", " ", (m.group(1) or "")).strip().lower()
        if not thing:
            continue
        # ...AND THE CAPTURE CANNOT RUN THROUGH A PREPOSITION. The group takes
        # spaces so a sheet's "long grey skirt" comes back whole, and on "drops the
        # thong on the floor" it swallowed "thong on the floor" instead -- head
        # "floor" -- so the restore was keyed to the ROOM. The scene named a wet
        # floor, scene_name_for handed back "tiled bathroom with a wet floor", and
        # the beat was recorded as putting the bathroom back on.
        if re.search(r"\b(?:on|onto|to|into|in|at|over|under|from|with|and)\b", thing):
            continue
        head = thing.split()[-1]
        if len(head) < 3 or head not in low:
            continue
        name = scene_name_for(head, scene) or thing
        if name not in out:
            out.append(name)
    return out




# ---------------------------------------------------------------------------
# LAYERING: which garment goes under which.
#
# Moved here to sit beside the vocabulary it reads. It is the last piece of the
# wardrobe that was living apart from the list of what a garment IS, and that
# separation is what let "chastity belt" be underwear to one file and a bare
# "belt" to the other.
#
# The CLAUSES stay in sampler.py -- under_clause, reveal_clause, bare_clause.
# Knowing a belt is under a skirt belongs here; saying so in a sentence belongs
# where a shot is assembled. Same split the restraint work settled on.
# ---------------------------------------------------------------------------

_UNDER_BY_REGION = {
    # NO HARDWARE HERE. A chastity belt is a restraint, and a restraint left out of
    # the text renders absent -- that is the bug the hardware latch exists for, and
    # putting the belt in this list rebuilt it from the other side. Reported as the
    # belt disappearing a few beats in, right after layering shipped.
    #
    # Cloth can be hidden and recovered from a description. Hardware cannot: a belt
    # that stops being drawn does not come back looking slightly wrong, it is gone,
    # and so is every beat that depended on it being there.
    # Hyphens and the other names for it. "chastity-belt" and "chastity device" were
    # not matched, so a sheet that spelled it either of those way showed it through
    # the jeans while "chastity belt" was correctly hidden -- the fix looked done
    # because the one spelling I tested worked.
    "lower": (r"panties|knickers|thong|g-?string|briefs|boxers|boxer\s+shorts|"
              r"underwear|undies|jockstrap|loincloth|"
              r"chastity[\s-]*(?:belts?|devices?|cages?)"),
    "upper": (r"bra|bralette|brassiere|camisole|undershirt|vest|corset|bustier"),
}
_OUTER_BY_REGION = {
    # Tights and pantyhose DO cover a waistband; stockings and hold-ups do not --
    # they stop at the thigh. Listing them together hid a chastity belt under a
    # pair of stockings, which covers nothing of it.
    "lower": (r"shorts|trousers|jeans|slacks|chinos|skirt|kilt|leggings|joggers|"
              r"tights|pantyhose|jeggings|culottes|"
              r"tracksuit\s+bottoms|dungarees|overalls|dress|gown|robe"),
    "upper": (r"top|shirt|blouse|t-?shirt|tee|jumper|sweater|sweatshirt|hoodie|"
              r"cardigan|jacket|coat|dress|gown|robe|dungarees|overalls|tunic"),
}


def implied_layers(scene):
    """{under: over} for underwear the scene lists beneath outer clothes it also lists.

    Only where BOTH are named: underwear with nothing over it is on show, and saying
    it is hidden would be describing away something the author dressed them in."""
    covers = {}
    # ONE PERSON AT A TIME. This read the whole scene as a single wardrobe, so one
    # character's jeans covered another character's belt -- and which garment won
    # depended on the ORDER the sheet lines happened to be written in. A sheet that
    # put the man second hid her belt under his trousers.
    #
    # Split on lines so each entry is judged alone. Text that is not an entry -- the
    # scene paragraph -- is still read as one block, since a location describing
    # clothing is describing whoever is in it.
    for line in (scene or "").split("\n"):
        text = line.strip()
        if not text:
            continue
        for region, unders in _UNDER_BY_REGION.items():
            over = None
            # The HEAD noun, which in English is the last one: "blue jeans shorts" is
            # a pair of shorts, not a pair of jeans. Taking the first match recorded
            # the cover as "jeans" while a removal names it "shorts", so the two never
            # lined up -- the belt was hidden correctly and then never uncovered,
            # because the garment that came off was not the one it was held under.
            for m in re.finditer(r"\b(?:" + _OUTER_BY_REGION[region] + r")\b",
                                 text, re.I):
                over = m.group(0).lower()
            if not over:
                continue
            for m in re.finditer(r"\b(?:" + unders + r")\b", text, re.I):
                under = re.sub(r"\s+", " ", m.group(0).lower())
                if under != over:
                    covers.setdefault(under, over)
    return covers


def is_undergarment(item):
    """Is this one of the things that is ALWAYS worn under clothes?

    Panties, knickers, thongs, briefs, boxers, underwear, bras, corsets, and
    chastity belts, devices and cages -- the _UNDER_BY_REGION list, which is why
    it is read off that list rather than a second copy that could drift from it.

    These are named and placed rather than deleted. A locket under a coat is a
    different thing: it genuinely cannot be seen, nothing is lost by waiting for
    the coat to come off, and it keeps the older behaviour."""
    t = str(item or "").lower()
    return any(re.search(r"\b(?:" + pat + r")\b", t, re.I)
               for pat in _UNDER_BY_REGION.values())


def hidden_layers(covers, gone, moved=()):
    """Garments still underneath something that is still covering them.

    `moved` is outer garments the beats have DISPLACED -- pulled down, pushed
    aside. Those are still worn, so `gone` never learns about them, and the
    layer beneath stayed hidden while the beat was busy showing it off: "pulls
    her shorts down to show the thong" described the thong in that one shot,
    from the author's own words, and hid it again in the next."""
    # Compared on the HEAD NOUN. `covers` holds the outer garment as implied_layers
    # read it ("shorts") while a displacement is keyed by the sheet's full name
    # ("denim shorts"), and an exact match between the two never fires -- the layer
    # underneath stayed hidden on the very shot the beat pulled the cover off.
    aside = {str(m).lower().split()[-1] for m in (moved or ()) if str(m).strip()}
    return [u for u, o in (covers or {}).items()
            if o not in gone and u not in gone
            and str(o).lower().split()[-1] not in aside]




# ---------------------------------------------------------------------------
# POSTURE. Moved from sampler.py, which had the richer table -- "takes a
# seat", "gets to her feet", "goes down on her knees" -- and the engine had
# three the sampler lacked. Two tables, diverged, and the sampler's is the
# one that drives the guard, so a crouch set no posture at all.
# ---------------------------------------------------------------------------

_POSTURE_OF = (
    ("sitting", re.compile(r"\b(?:sits?|sat|sitting|seats?\s+(?:her|him|them)self|"
                           r"is\s+seated|takes?\s+a\s+seat|perch(?:es|ed)?)\b", re.I)),
    ("kneeling", re.compile(r"\b(?:kneels?|knelt|kneeling|"
                            r"(?:goes?|got|gets?)\s+down\s+on\s+(?:her|his|their)\s+knees)\b",
                            re.I)),
    ("lying down", re.compile(r"\b(?:lies?|lay|lays?|laid|lying|laying|"
                              r"stretches?\s+out|sprawls?|sprawled)\b", re.I)),
    ("standing", re.compile(r"\b(?:stands?|stood|standing|"
                            r"(?:gets?|got)\s+(?:up|to\s+(?:her|his|their)\s+feet)|"
                            r"rises?|rose|risen)\b", re.I)),
)
# A posture verb that is really about somewhere else: "the chair stands in the
# corner", "the case lies on the table". Those set nobody's pose.
_NOT_A_BODY = re.compile(r"\b(?:it|chair|table|box|case|bag|door|house|room|"
                         r"building|tree|bottle|glass|book|light|lamp)\s+\w{0,8}?\s*"
                         r"(?:stands?|lies?|sits?)\b", re.I)
# THE THREE THE ENGINE KNEW AND THIS DID NOT. Two tables, diverged, and this is
# the one that drives the posture guard -- so "Ana crouches" set no posture at
# all and the next shot was told nothing about how she was left.
_POSTURE_OF = _POSTURE_OF + (
    # SQUATTING IS NOT CROUCHING. Folded together, a script that said "squats"
    # was held as "still crouching" -- a different shape of body, and not the
    # word the author chose. The hold says back what was written.
    ("squatting", _rx(r"\b(?:squats?|squatting|squatted)\b")),
    ("crouching", _rx(r"\b(?:crouch(?:es|ing|ed)?)\b")),
    ("bent over", _rx(r"\b(?:bends?\s+over|bent\s+over|leans?\s+over|"
                      r"leaned\s+over|doubles?\s+over)\b")),
    ("curled up", _rx(r"\b(?:curled\s+up|curls?\s+up|foetal|fetal)\b")),
    # ROLLING ONTO A SIDE IS STILL LYING DOWN. "McKenna rolls onto her side" set no
    # posture at all -- the lying verbs are all lie/lay/sprawl and none of them is
    # how you write a body that is ALREADY down changing which way it faces. So a
    # beat that put her on her side left the hold saying nothing, and the shot after
    # it was told nothing about how she was left.
    #
    # It also decides whether the weight gets named: the pose clause only says what
    # is under a bound body when it knows the body is off its feet (see
    # POSE_LYING_WEIGHT in sampler.py), and that reads this posture.
    #
    # The possessive is required. "the barrel rolls onto its side" is not a person,
    # and _NOT_A_BODY does not cover roll.
    ("lying down", _rx(r"\broll(?:s|ed|ing)?\s+(?:over\s+)?(?:on)?to\s+"
                       r"(?:her|his|their)\s+"
                       r"(?:side|back|front|stomach|belly)\b")),
)


# Words that are capitalised at the start of a sentence whatever they mean, so
# their capital says nothing about whether they are a name. "May I come in?"
# staged a character called Aunt May. Mid-sentence the capital is informative
# again, and these are accepted there.
_SENTENCE_START_ALSO = frozenset("""
may will can must might shall should would could does did was were are is
let get go come take put look stop wait now then there here this that these
those one some all any no yes so but and or if when while after before both
say tell keep hold turn open close pull push move step down
""".split())


def _alias_at(word, staged):
    """Where a one-word stand-in for a longer name appears, or None.

    A capital at the start of a sentence is free, so a word that is ordinary
    English there has to earn its match somewhere else in the beat."""
    fallback = None
    for m in re.finditer(r"\b" + re.escape(word) + r"\b", staged):
        opens = re.search(r"(?:^|[.!?;:]\s*|[\"'“]\s*)$", staged[:m.start()])
        if not opens:
            return m.start()
        if word.lower() not in _SENTENCE_START_ALSO and fallback is None:
            fallback = m.start()
    return fallback


def names_in(beat, cast):
    """Names this beat STAGES, in the order the sentence puts them.

    Case-SENSITIVE, and that is not fussiness: prose capitalises a name, and
    matching without case makes the word "will" find a character called Will and
    "grace" find Grace. The sampler had that fixed and this file did not, which
    is what two copies of one idea buys you.

    Speech-stripped, for the same reason it is everywhere else -- "McKenna, where
    are you?" is how absence gets written, and reading it as presence put a whole
    sheet entry into a shot the person is not in."""
    staged = _outside_speech(beat or "")
    names = [str(n) for n in (cast or []) if n]
    hits, found = [], set()
    for n in names:
        m = re.search(r"\b" + re.escape(n) + r"\b", staged)
        if m:
            hits.append((m.start(), n))
            found.add(n)
    # A SHEET NAME IS OFTEN LONGER THAN WHAT THE BEATS CALL HER. "Mistress Vale"
    # on the sheet and "the Mistress" in every beat matched nothing, so her line
    # was in no shot at all and the model invented her from scratch each time.
    #
    # One word of the name, and only when that word is hers alone: with both
    # "Mistress" and "Mistress Vale" on the sheet, "Mistress" belongs to the
    # first and picking either would be a guess. Titles are short and shared, so
    # a word under three letters never stands in.
    for n in names:
        if n in found or " " not in n:
            continue
        for w in n.split():
            if len(w) < 3 or not w[:1].isupper():
                continue
            if any(w == o or w in o.split() for o in names if o != n):
                continue
            m = _alias_at(w, staged)
            if m is not None:
                hits.append((m, n))
                found.add(n)
                break
    return [n for _at, n in sorted(hits)]


# ---------------------------------------------------------------------------
# STATE. One object knows what is true, and everything a shot says is rendered
# from it -- so two clauses cannot contradict each other, because there is only
# one place a fact lives.
# ---------------------------------------------------------------------------

class Restraint:
    """One piece of hardware on one person.

    part, position and anchor travel TOGETHER, and that is the whole point. The
    old engine read the part off the item list and the position off the beat, in
    two functions that never met, and emitted "holding the neck behind the back".
    A position describes where the ARMS are, so a collar cannot carry one -- the
    constructor drops it rather than trusting the caller."""

    __slots__ = ("item", "part", "position", "anchor", "applied_in", "rigid")

    def __init__(self, item, part, position="", anchor="", applied_in=0,
                 rigid=False):
        self.item = item
        self.part = part
        self.position = position if part in ("wrists", "arms") else ""
        self.anchor = anchor
        self.applied_in = applied_in
        # RIGID metal keeps its shape. Steel decoded and re-encoded once a shot
        # has nothing in the text holding its links to a size, and it creeps --
        # a chain grows slack, cuffs turn into bracelets. Soft goods do not need
        # this and must not be given it: rope is tied, not held rigid.
        self.rigid = bool(rigid) or _is_rigid(item)

    def phrase(self):
        """Where this one holds, as English."""
        bits = [f"the {self.part}"]
        if self.position:
            bits.append(self.position)
        if self.anchor:
            bits.append(f"fast to the {self.anchor}")
        return " ".join(bits)

    def __repr__(self):
        return (f"Restraint({self.item!r},{self.part!r},"
                f"{self.position!r},{self.anchor!r})")


class Person:
    __slots__ = ("name", "hardware", "worn", "removed", "displaced",
                 "posture", "place", "bare")

    def __init__(self, name):
        self.name = name
        self.hardware = {}      # (canonical, part) -> Restraint
        self.worn = []          # garments on the body, as written
        self.removed = []       # garments taken off
        self.displaced = []     # pulled aside but still on
        self.posture = ""
        self.place = ""
        # Regions with nothing on them. A LATCH, not a one-shot fact: the beat
        # that uncovered a region is the only shot that used to say so, and every
        # shot after it left that region unspecified -- which the model fills
        # from its own prior. Reported as a bra coming back on a topless
        # character who never had one on the sheet.
        self.bare = []

    def restrained(self):
        return bool(self.hardware)

    def kinds(self):
        """What is on, by canonical name, in the order it went on.

        The key is a (name, part) pair so that a chain on the ankles and a chain
        on the wrists can both exist. Almost nothing cares about the part, so it
        asks here instead of unpacking the key."""
        return [c for c, _pt in self.hardware]

    def hw(self, canon, part=None):
        """One restraint by name, or None. Give a part to pick between two."""
        for (c, pt), r in self.hardware.items():
            if c == canon and (part is None or pt == part):
                return r
        return None

    def __repr__(self):
        return (f"Person({self.name!r},hw={list(self.hardware)},"
                f"posture={self.posture!r})")


class SceneState:
    """What is true right now, and what each shot is told because of it."""

    def __init__(self, place=""):
        self.place = place
        # The room the SCENE paragraph names. A shot only needs telling where it
        # is once the film has moved somewhere else.
        self.opened_in = place
        self.people = {}
        self.shot = 0

    def person(self, name):
        if name not in self.people:
            self.people[name] = Person(name)
        return self.people[name]

    def declare(self, name, description, staged_later=()):
        """Take what a character sheet already says as read.

        A sheet entry is a STATE, not an event: "Kate: she, 30, coat, handcuffs"
        says the handcuffs are already on before any beat puts them there. The
        engine read only beats at first, so a scene that opened with somebody
        already restrained had no hardware in it at all until a beat happened to
        mention some -- and a hold that never fires is hardware the model is free
        to leave off.

        `staged_later` is what stops that becoming its own bug. A sheet says WHAT
        somebody has and never says WHEN, so a sheet reading "McKenna: she, 27,
        green dress, handcuffs" beside a script that cuffs her in beat 3 declared
        them on from shot 1 -- and shot 1 went out saying "The handcuffs stay
        closed and fastened AS THEY WERE PUT ON", two shots before anybody put
        them on. Reported as a handcuff on her arm before she is handcuffed.

        Where the script stages the fastening, the script decides the moment. The
        sheet still supplies the description; it just does not get to start the
        clock. Declared, never applied: nothing here is a change, so no shot is
        told anything goes on during it."""
        p = self.person(name)
        for canon, part, written, _at in hardware_spans(description or ""):
            if canon in staged_later:
                continue
            if (canon, part) not in p.hardware:
                # applied_in = 0, so this never reads as "goes on during this
                # shot" -- shots are numbered from 1.
                p.hardware[(canon, part)] = Restraint(written or canon, part,
                                              position_in(description or ""),
                                              anchor_in(description or ""), 0)
        # ...unless it has already come OFF. The sheet is re-read every shot and
        # the character memory is never edited, so a garment removed in shot 2
        # was put straight back on the body by the sheet in shot 3 -- and the
        # clause saying that region is bare then went silent, because something
        # "still worn" covered it. What the script did outranks what the sheet
        # lists; the sheet says what she has, not what is on her now.
        _off = [_garment_key(x) for x in p.removed]
        for g in garments_in(description or ""):
            key = _garment_key(g)
            if key not in [_garment_key(x) for x in p.worn] and key not in _off:
                p.worn.append(g)
        return p

    # -- reading a beat ----------------------------------------------------
    def read(self, beat, cast=(), shot=0):
        """Update the state from one beat, and report what CHANGED.

        The change matters separately from the result: the shot that puts the
        cuffs on has to say both ends of that, and every shot after it says only
        the result."""
        self.shot = shot
        beat = beat or ""
        changed = {"applied": [], "released": [], "moved_to": "", "posture": {},
                   "removed": [], "worn": [], "displaced": []}

        here = place_in(beat)
        if here:
            self.place = here
            changed["moved_to"] = here

        # IN SENTENCE ORDER, not cast order. Ordering by the sheet put "Ana"
        # before "Guard" in "The guard handcuffs Ana", so the cuffs went on the
        # guard -- the agent wearing what he is applying, which is the invented
        # second figure all over again.
        #
        # ...and read from the STAGED half only. A name inside a line of dialogue
        # is being said, not staged: "Dan says: 'McKenna, put the cuffs on'"
        # would otherwise hand McKenna hardware in a shot she is not in.
        who = names_in(beat, cast)
        subject = who[0] if who else next(iter(list(self.people) or list(cast)
                                               or [""]))

        spans = hardware_spans(beat)
        garments = list(_GARMENT_ONE.finditer(beat))
        boundaries = list(_CLAUSE_BOUNDARY.finditer(beat)) if spans or garments else []
        applying = bool(spans) and bool(_APPLY.search(beat))
        releasing = bool(_RELEASE.search(beat))

        if applying or releasing:
            # MODIFIERS BIND TO THE NEAREST ITEM. "handcuffs her wrists behind
            # her back and locks a steel collar around her neck, chained to the
            # wall" carries two modifiers and two items; giving both modifiers
            # to both items produced handcuffs chained to a wall they were never
            # near, and a collar held behind a back.
            for canon, part, written, at in spans:
                clause, lo = _clause_at(beat, at, boundaries)
                item_at = at - lo
                apply_at = max((m.start() for m in _APPLY.finditer(clause)
                                if m.start() <= item_at), default=-1)
                release_at = max((m.start() for m in _RELEASE.finditer(clause)
                                  if m.start() <= item_at), default=-1)
                if apply_at < 0 and release_at < 0:
                    continue
                local_who = names_in(clause, cast)
                wearer = _wearer(clause, local_who or who, subject)
                p = self.person(wearer)
                if release_at >= 0:
                    keys = [k for k in list(p.hardware) if k[0] == canon]
                    for key in keys:
                        changed["released"].append((wearer, p.hardware.pop(key)))
                    continue
                # KEYED BY THE PAIR. A chain on the ankles and a chain on the
                # wrists are two restraints; keyed by name alone the second
                # overwrote the first and one of them was never drawn again.
                p.hardware[(canon, part)] = Restraint(
                    written or canon, part,
                    _nearest(position_spans(beat), at, spans),
                    _nearest(anchor_spans(beat), at, spans), shot)
                changed["applied"].append((wearer, p.hardware[(canon, part)]))
            # A chain named beside another item is that item's TETHER, and it
            # is folded in by hardware_spans now, where the parts are known.
            # Doing it here meant popping "chain" whenever one was named with an
            # anchor -- which also popped a chain that had a part of its OWN, so
            # "chains her collar to the ring and chains her ankles together"
            # kept the collar and lost the ankles.
            # Its anchor needs no transferring either: with the tether gone from
            # the spans, the anchor binds to the nearest remaining item, which
            # is the one it was always describing.
        if releasing and not spans:
            # Whoever is actually wearing it. "The guard unlocks the handcuffs"
            # names only the agent, and taking the subject there tried to
            # release hardware from the man holding the key.
            held = [n for n, q in self.people.items() if q.restrained()]
            wearer = next((n for n in who if n in held),
                          held[0] if len(held) == 1 else subject)
            p = self.person(wearer)
            # Released by NAME, whatever part it is on: an unlocking beat says
            # "unlocks the chain", not which of two chains, and matching the
            # pair left one fastened forever.
            if re.search(r"\b(?:them|it|her|him|everything|all\s+of\s+it)\b",
                         beat, re.I):
                # "the guard releases her" names no item, so all of it comes off.
                while p.hardware:
                    changed["released"].append((wearer, p.hardware.popitem()[1]))

        # Garments. The verb decides which way the change runs, and the item has
        # to be named -- a bare "she undresses" says nothing about which garment,
        # and guessing is how a garment came off a beat before the beat that
        # took it off.
        if subject:
            for m in garments:
                g = f"{(m.group(1) or '').strip()} {m.group(2)}".strip().lower()
                key = _garment_key(g)
                clause, lo = _clause_at(beat, m.start(), boundaries)
                item_at = m.start() - lo
                actions = [(x.start(), "off") for x in _TAKES_OFF.finditer(clause)
                           if x.start() <= item_at]
                actions += [(x.start(), "on") for x in _PUTS_ON.finditer(clause)
                            if x.start() <= item_at]
                actions += [(x.start(), "aside") for x in _DISPLACES.finditer(clause)
                            if x.start() <= item_at]
                action = max(actions, default=(-1, ""))[1]
                if action == "aside" and _OPENS_GARMENT.search(clause[:item_at]):
                    if _COMPLETES_OFF.search(re.split(r"[.;!?]", beat[m.end():])[0]):
                        action = "off"
                local_who = names_in(clause, cast)
                wearer_g = _wearer(clause, local_who, subject)
                p = self.person(wearer_g)
                if action == "off":
                    if key not in [_garment_key(x) for x in p.removed]:
                        p.removed.append(g)
                        changed["removed"].append((wearer_g, g))
                    p.worn = [x for x in p.worn if _garment_key(x) != key]
                    p.displaced = [x for x in p.displaced
                                   if _garment_key(x) != key]
                    _bare_on(p, region_of(g))
                elif action == "on":
                    if key not in [_garment_key(x) for x in p.worn]:
                        p.worn.append(g)
                        changed["worn"].append((wearer_g, g))
                    p.removed = [x for x in p.removed if _garment_key(x) != key]
                    p.displaced = [x for x in p.displaced
                                   if _garment_key(x) != key]
                    # Covered again: the latch has to release, or a character who
                    # dresses is told for the rest of the film that the region is
                    # bare, over the garment she just put on.
                    _bare_off(p, region_of(g))
                elif action == "aside":
                    if key not in [_garment_key(x) for x in p.displaced]:
                        p.displaced.append(g)
                        changed["displaced"].append((wearer_g, g))

        # BEING in the state, rather than arriving at it. No garment is named and
        # nothing comes off, so every removal path had nothing to do and no shot
        # ever said what was on the chest.
        _nude = nudity_in(beat)
        if _nude:
            nude_at = min((m.start() for rx, _regions in _NUDITY_RX
                           for m in rx.finditer(beat)), default=len(beat))
            located = [(abs(beat.find(n) - nude_at), n) for n in who if beat.find(n) >= 0]
            owners = [min(located)[1]] if located else ([subject] if subject else [])
            for n in owners:
                q = self.person(n)
                _bare_on(q, _nude)
                # ...and it takes the garments OFF. Saying somebody is topless
                # names no garment, so nothing was removed and the sheet's shirt
                # stayed on the body -- which then suppressed the very clause
                # that says the chest is bare, because something "still worn"
                # covered the region. The state has to agree with itself.
                for g in list(q.worn):
                    if region_of(g) in _nude:
                        q.worn.remove(g)
                        if _garment_key(g) not in [_garment_key(x)
                                                   for x in q.removed]:
                            q.removed.append(g)
                            changed["removed"].append((n, g))

        pose = posture_in(beat)
        if pose and subject:
            self.person(subject).posture = pose
            changed["posture"][subject] = pose

        return changed

    # -- writing the shot --------------------------------------------------
    def continuity(self, described=(), changed=None):
        """ONE paragraph, rendered from state. Every fact said once.

        The order is fixed -- hardware, what it holds, posture -- so a reader of
        the output can tell at a glance whether something is missing. The old
        engine emitted clauses in whatever order its readers happened to fire,
        which is why nobody noticed the handcuffs had stopped appearing."""
        changed = changed or {}
        out = []
        names = [n for n in described if n in self.people] or list(self.people)
        applied_now = {r.item for _w, r in changed.get("applied", [])}

        for name in names:
            p = self.people.get(name)
            if not p or not p.hardware:
                continue
            items = [r.item for r in p.hardware.values()]
            subject = _join(items)
            many = len(items) > 1 or _plural(items[0])
            if applied_now & set(items):
                out.append(
                    f"The {subject} {'go' if many else 'goes'} on during this "
                    f"shot: open and off the body at the first frame, closed on "
                    f"it by the last.")
            else:
                # WHOSE. With more than one person described, a hold that does
                # not say whose hardware it is describes cuffs on wrists
                # belonging to nobody -- and the model draws a body to own them.
                who = f" on {name}" if len(names) > 1 else ""
                soft = all(not r.rigid for r in p.hardware.values())
                shut = "tied and holding" if soft else "closed and fastened"
                out.append(
                    f"The {subject}{who} {'stay' if many else 'stays'} {shut} "
                    f"as {'they were' if many else 'it was'} put on, the same "
                    f"object in the same material.")
            holds = [r.phrase() for r in p.hardware.values() if r.phrase()]
            if holds:
                out.append(f"{'They hold' if many else 'It holds'} {_join(holds)}.")
            if any(r.rigid for r in p.hardware.values()):
                out.append("The links keep their size and the run between them "
                           "stays taut.")

        # GARMENTS. The shot that changes one says BOTH ENDS of the change --
        # where it starts and where it finishes -- because a shot told only the
        # result is free to open with the result already true, which is a
        # garment coming off a beat before the beat that takes it off.
        off_now = {g for _w, g in changed.get("removed", [])}
        on_now = {g for _w, g in changed.get("worn", [])}
        aside_now = {g for _w, g in changed.get("displaced", [])}
        for name in names:
            p = self.people.get(name)
            if not p:
                continue
            for g in p.removed:
                if g in off_now:
                    out.append(f"The {g} is on the body as the shot opens and "
                               f"fully off it by the last frame, taken off "
                               f"during this shot.")
                else:
                    out.append(f"The {g} is off the body and stays where it "
                               f"was put.")
            for g in p.worn:
                if g in on_now:
                    out.append(f"The {g} is off the body as the shot opens and "
                               f"fully on by the last frame, put on during this "
                               f"shot.")
            for g in p.displaced:
                if g in aside_now:
                    out.append(f"The {g} is moved aside during this shot and "
                               f"stays on the body.")
                else:
                    out.append(f"The {g} is still pulled aside, still on the body.")

        for name in names:
            p = self.people.get(name)
            if p and p.posture and name not in (changed.get("posture") or {}):
                who = name if len(names) > 1 else "The body"
                out.append(f"{who} is still {p.posture}.")

        # WHERE. A journey moves the film, and every shot after it is in the new
        # room -- the scene paragraph still names the old one, and without this
        # the walk down the corridor arrives back in the room it left.
        if self.place and self.place != self.opened_in and not changed.get("moved_to"):
            out.append(f"The shot is in the {self.place}, "
                       f"not the room the scene text names.")

        return " ".join(out)


def _garment_key(g):
    """Two wordings of the same garment are one garment: "blue shorts" and
    "shorts" must not both be tracked, or the shot lists a spare pair."""
    return re.sub(r"\s+", " ", (g or "").lower()).split()[-1].rstrip("s")


_RIGID = _rx(r"\b(?:handcuffs?|cuffs?|chains?|shackles?|manacles?|leg\s?irons?|"
             r"spreader\s+bars?|steel|iron|metal|padlock|chrome|brass)\b")
_SOFT = _rx(r"\b(?:rope|ropes|cord|cords|twine|string|tape|scarf|stocking|"
            r"stockings|zip\s?ties?|cable\s?ties?|laces?)\b")


def _is_rigid(item):
    """Does this thing hold its shape? Soft goods never do, whatever else the
    phrase says -- "steel" in a sentence about rope does not make rope steel."""
    t = item or ""
    return bool(_RIGID.search(t)) and not _SOFT.search(t)


def _nearest(mods, at, spans):
    """The modifier belonging to the item at `at`, or ''.

    A modifier belongs to the last item mentioned before it -- English puts the
    qualifier after the thing it qualifies. So with items at 10 and 60 and a
    modifier at 75, the modifier is the second item's; a modifier at 20 is the
    first item's."""
    starts = sorted(s for _c, _p, _w, s in spans)
    mine = ""
    for name, where in mods:
        owner = max((s for s in starts if s <= where), default=starts[0])
        if owner == at:
            mine = mine or name
    return mine


def _wearer(beat, who, fallback):
    """Who the hardware goes ON. The agent is not the wearer.

    "The guard cuffs Ana" puts them on Ana; "Ana is cuffed by the guard" puts
    them on Ana too, and naive sentence order gets the second one backwards.
    Describing hardware on somebody the text never put it on gives the model
    wrists belonging to nobody, which is how a second figure gets invented to
    own them."""
    if len(who) < 2:
        return who[0] if who else fallback
    passive = re.search(r"\bby\s+(?:the\s+)?(\w+)", beat or "", re.I)
    agent = None
    if passive:
        agent = next((n for n in who
                      if n.lower() == passive.group(1).lower()), None)
    if agent is None:
        agent = who[0]          # active voice: the one doing it comes first
    return next((n for n in who if n != agent), fallback)


def held_part_of(items):
    """The body part these items hold, as a plural noun for a sentence.

    A limb position describes the ARMS, so this answers "wrists" for anything
    that holds them and defers to the item otherwise -- a collar's position is
    never a limb position, and the constructor already refuses to give it one."""
    text = " ".join(items or [])
    for pat, _n, part in HARDWARE:
        if re.search(r"\b(?:" + pat + r")\b", text, re.I) and part not in ("wrists",):
            return part
    return "wrists"


def staged_applications(beats):
    """{canonical hardware -> the 1-based beat that first puts it on}.

    Read once, before anything renders, because a sheet cannot say WHEN. Where
    the script stages a fastening, no earlier shot may be told that thing is
    already fastened -- that is a cuff on a wrist two shots before the cuffing."""
    out = {}
    for i, b in enumerate(beats or [], 1):
        b = b or ""
        if not _APPLY.search(b) or _RELEASE.search(b):
            continue
        for canon, _part, _w, at in hardware_spans(b):
            # WHAT IS BEING FASTENED TO WHAT. "Dana clips a lead to the steel
            # collar" puts a LEAD on; the collar is where it clips, and it has
            # been round her neck all along. Counting it as the collar's own
            # application dated the collar to that beat, and everything before
            # it was then treated as before she had one.
            if re.search(r"\bto\s+(?:the|a|an|her|his|their)\s*$", b[:at], re.I):
                continue
            out.setdefault(canon, i)
    return out


def _plural(item):
    return item.endswith("s") and not item.endswith("ss")


def _join(items):
    items = [i for i in items if i]
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    return ", ".join(items[:-1]) + " and " + items[-1]
