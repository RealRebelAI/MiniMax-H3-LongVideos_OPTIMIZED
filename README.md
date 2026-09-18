---
tags:
  - comfyui
  - comfyui-nodes
  - minimax
  - minimax-h3
  - video
  - text-to-video
  - audio
  - not-for-all-audiences
license: other
license_name: h3-longvideos-no-redistribution
license_link: LICENSE
---


**This node is a constant work in progress! If you are noticing bugs or features that do
not work, please ensure that you are pulling the most recent version and updating your
workflows.**

# H3-LongVideos

Long **MiniMax-H3 video with synchronised audio** from a single prompt, in ComfyUI.

H3 renders about 15 seconds at a time. This node turns a written scene into a chain of
shots and joins them into one continuous video. Your text reaches the model word for
word.

## Install

Copy this folder into `ComfyUI/custom_nodes/` and restart the ComfyUI **server**.

Requires ComfyUI 0.31+ with native MiniMax-H3 support (tested on 0.33).

## What to load

| | |
|---|---|
| **UNET** | a MiniMax-H3 model — a [hybrid fl2va/ref2va merge](https://huggingface.co/smhfacct/Minimax-H3-fl2va-ref2va-hybrid-models) works best |
| **CLIP** | H3's text encoder, loader type `minimax` |
| **VAE** | the H3 **video** VAE |
| **audio VAE** | the H3 **audio** VAE — a separate file, and it must be the *converted* one |

```
UNETLoader ─┐                     images ─> Video Combine / Save Video
CLIPLoader ─┼─> H3-LongVideos  ─> audio  ─┘
VAELoader ──┘                     info   ─> Show Text
```

`prompt` is an input socket — wire a multiline text node into it.

**Set `plan_only` first.** It reports the shot split, the lengths and every warning,
without rendering.

## Writing the prompt

**One paragraph = one shot**, separated by a blank line. The first paragraph is the
**scene**, prepended to every shot; the rest are beats.

```
Natural daylight, hard sun. A farm with a barn.

Dom drives a van down the driveway and stops in front of the barn.

Dom gets out and walks to the back of it.

Mara steps out of the barn and asks him: "Is that the last one?"
```

Dialogue goes in **double quotes**. Use the `anchor` widget if you want the framing
carried separately — then every paragraph is a beat.

### A line that must reach the model word for word

Put it on its own line inside the beat:

```
Mara walks Ana to the entrance.
exact: her wrists stay behind her back the whole way
```

It is placed straight after the beat, in your words, and nothing in the node reads,
scopes, scrubs, reorders or drops it. That matters because the node's own continuity
clauses compete for room: on a short beat they can be 70% of what the shot is told
against the beat's 8%, and `info` reports that balance every run. An `exact:` line is
not a guard and has no budget to lose.

Nothing reads it either, on purpose: a name in it puts nobody in the shot, a garment in
it removes nothing, and a door in it stages no change. Write what must be **said**, and
let the beat stage what happens. `exactly:` and `verbatim:` do the same thing.

To see your prompt entirely on its own, switch on the `verbatim` widget: a shot is then
your scene, your beat and the sheet entries for the people it names, and nothing this
node writes. Every failure the continuity clauses answer comes back with them —
duplicate characters, invented speech, a drifting camera, a door that shuts itself, a
walk played backwards — so it is most useful for proving whether a fault is the node's
doing or the model's. `info` still lists what each clause would have said.

### The character sheet

A paragraph of `Name: attributes` lines, or the `character_memory` widget. Each shot is
given the entries for the people its beat names, and nobody else.

```
Nora: <Picture 1>, 34, she, tall, red hair, green canvas jacket, brown boots.
Mike: he, 41, dark hair, navy overalls.
```

- **Declare a pronoun.** It is what lets *"he takes her coat off"* find the right two
  people.
- **Declare an age.** Where a shot describes a bare region, the body named is the age
  the sheet states — without one, an unstated attribute is filled from the model's prior,
  which is a twenty-something whatever you wrote. A declared age under 18 gets **no body
  described for it at all**, and a sheet declaring a minor alongside a script that stages
  nudity or sex refuses to render.
- **One name per person.** `Dan` in some beats and `Mike` in others reads as two.
- **Every name needs an entry**, or the model invents that person differently each shot.

`<Picture N>` means `ref_image_N`, the socket. Tag it onto the person or thing it
depicts and it follows them; untagged, a reference goes on every shot.

### LoRAs

A LoRA is the one input to a shot this node neither writes nor can read out of your
text, so `info` reports what is attached: how many are stacked, over how many weights,
at what strength, whether the **text encoder** carries them too, and the last one's
name if its metadata has one. Two runs whose prompts are identical can render
differently, and nothing else in the run says why.

## Settings

| setting | value |
|---|---|
| `cfg` | **1.0** — H3 is CFG-free; the negative prompt is never evaluated |
| `sampler_name` | `res_multistep`, or `euler` with PDD Acc |
| `scheduler` | `simple` |
| `shift_video` / `shift_audio` | **12 / 3** — keep them near 4:1 or the audio breaks |
| `steps` | 6–8 with a turbo/distill LoRA, 20+ without |
| `megapixels` | 1.0 is H3's native budget; lower is faster and leaner |
| `shot_seconds` | the cap on each shot |
| `shot_length` | `from the beat` sizes each shot from its own line; `fixed` gives every shot `shot_seconds` |
| `ambient_audio` | optional — wire a recording to play under the whole soundtrack. The node no longer builds one: the audio is the model's |
| `ambient_level` | how loud that recording plays. 0 is off; 0.15–0.3 is a bed you notice only when it stops. Does nothing with nothing wired |

Ambience is mixed, never prompted. Scoring a silent shot from text needs the audio
branch left open, and an open branch on a joint model invents a voice for the face to
lip-sync to. The bed is built from the room your scene names, so it needs no file,
and shaped noise cannot speak. It makes tone — air, rumble, hum, water, a clock — so
a scene whose ambience is birdsong gets the room, not the birds; `info` says when.

Everything else is a switch, on by default, and each has a tooltip explaining what it
does and what it costs. Hover before you change one.

## Outputs

| slot | what it is |
|---|---|
| `images` | the finished frames |
| `audio` | the synchronised soundtrack |
| `info` | what the node did, and every warning — **read this** |
| `script` | the exact per-shot text it sent |
| `frames_per_shot`, `total_frames`, `shots`, `video_seconds` | for downstream nodes |

When a shot renders something you did not expect, read `script`. It shows precisely what
that shot was told, and `info` says why.

## Other nodes here

- **H3 Shot Length** — seconds to a valid H3 frame count.
- **H3 Overlay** — watermark and intro title composited onto finished frames.
- **H3 Model Inspector** — checkpoint precision, and whether your card runs it natively.

## Notes

- No negative prompt: H3 is CFG-free at `cfg 1`.
- No denoise control: fixed at 1.0, because partial denoise desyncs the joint
  audio/video schedule.
- **Widgets are restored by position.** If they read NaN, right-click the node →
  **Fix node (recreate)**, set your values, and save the workflow again.

## Disclaimer

The owner of this repo will not be responsible for any copyright strikes incurred
because of use. You are responsible for your works. Use this node responsibly and
ethically.
