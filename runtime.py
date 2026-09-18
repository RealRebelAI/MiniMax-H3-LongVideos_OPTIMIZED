# H3-LongVideos -- https://github.com/Smite79/MiniMax-H3-LongVideos
# Copyright (c) 2026 Smite79. All rights reserved.
# Redistribution, in whole or in part, requires written permission.
# This notice may not be removed or altered. See LICENSE.
"""Sampling, decoding, resizing, memory handling, and frame assembly."""

import math

import torch
import nodes
import comfy.utils
import comfy.sample
import comfy.samplers
import comfy.nested_tensor
import comfy.model_management as mm
import latent_preview


class FrameAccumulator:
    """Build the final frame tensor once, retaining overflow only when necessary."""

    def __init__(self, capacity, dtype, store_on_cpu):
        self.capacity = int(capacity)
        self.dtype = dtype
        self.store_on_cpu = bool(store_on_cpu)
        self.tensor = None
        self.used = 0
        self.overflow = []

    def add(self, frames):
        count = int(frames.shape[0])
        if self.tensor is None and count:
            device = torch.device("cpu") if self.store_on_cpu else frames.device
            self.tensor = torch.empty(
                (max(count, self.capacity),) + tuple(frames.shape[1:]),
                dtype=self.dtype, device=device)
        if (not self.overflow and self.tensor is not None
                and self.used + count <= self.tensor.shape[0]):
            self.tensor[self.used:self.used + count].copy_(frames)
            self.used += count
            return
        self.overflow.append(frames.to("cpu", self.dtype, copy=True)
                             if self.store_on_cpu else frames)

    def release(self):
        """Drop every tensor held, now, rather than whenever the collector gets to it.

        On an interrupt the render unwinds through frames the collector tears down in
        its own order, and a large video buffer freed after the models it was sized
        against have already gone is a free the allocator cannot explain. Deliberately
        does NOT empty the cache: that is another CUDA call, and if the context is
        already in a sticky error state it is one more thing to abort inside."""
        self.tensor = None
        self.overflow = []
        self.used = 0

    def finish(self):
        if not self.overflow:
            if self.tensor is None:
                return torch.cat(self.overflow, dim=0)
            if self.used == self.tensor.shape[0]:
                out = self.tensor
            else:
                # COMPACT, never a slice. A slice of a larger buffer keeps the WHOLE
                # buffer's storage alive, which is the retention this class exists to
                # prevent -- and test_the_chain_is_never_held_twice measures exactly
                # that, demanding no unused bytes behind the returned tensor.
                #
                # There is slack because the capacity is now an upper bound: it can no
                # longer assume trim_seam drops a frame at every seam, since a shot that
                # opens on no keyframe keeps its first frame. Over-allocating by at most
                # one frame per seam and compacting once is the bounded cost. The
                # alternative -- an exact guess that can be too small -- drops into the
                # overflow list, which with cleanup_between_shots off holds every shot's
                # decoded frames live on the GPU until the end of the run.
                out = torch.empty((self.used,) + tuple(self.tensor.shape[1:]),
                                  dtype=self.dtype, device=self.tensor.device)
                out.copy_(self.tensor[:self.used])
            self.tensor = None
            return out

        extra = sum(int(piece.shape[0]) for piece in self.overflow)
        reference = self.tensor if self.tensor is not None else self.overflow[0]
        out = torch.empty((self.used + extra,) + tuple(reference.shape[1:]),
                          dtype=self.dtype, device=reference.device)
        if self.tensor is not None and self.used:
            out[:self.used].copy_(self.tensor[:self.used])
        at = self.used
        while self.overflow:
            piece = self.overflow.pop(0)
            count = int(piece.shape[0])
            out[at:at + count].copy_(piece)
            at += count
        self.tensor = None
        return out


H3_FPS = 24                    # H3 renders 24 fps, always


AUDIO_LATENT_FPS = 40          # audio latent frames per second


AUTO_TILE_T = 8                # temporal chunk for a tiled decode


MAX_FRAMES = 362               # H3's own ceiling (~15s)


CANVAS_MULTIPLE = 32


REF_IMAGE_SHORT_EDGE = 2048


def align_frame_count(n):
    """Up to the next valid H3 frame count. The grid is 17k+5."""
    n = max(5, int(n))
    while n % 17 != 5:
        n += 1
    return min(n, MAX_FRAMES)


def video_latent_t(fc):
    return 2 if fc <= 5 else ((fc - 5) // 17) * 5 + 2


def temporal_shape(length, fps=H3_FPS):
    """(frame count, video latent frames, audio latent frames) for a shot.

    `fps` is accepted but deliberately IGNORED: the audio latent has to line up
    with 24 fps video or the shot's sound is stretched against its picture."""
    fc = align_frame_count(length)
    return fc, video_latent_t(fc), round(fc / H3_FPS * AUDIO_LATENT_FPS)


def ref_image_canvas(w, h, gen_w, gen_h, mode="match"):
    """Pure: the (width, height) a reference image is encoded at.

    'match' scales it (DOWN only, aspect kept) to the generation's pixel area, so a
    reference costs about as much as one frame of the shot. 'max' goes to the
    reference pipeline's 2048 short edge for the best identity fidelity, which on a
    long chain is several times slower because the rows are re-attended every step
    of every shot. Never upscales: a small reference stays small."""
    w, h = max(1, int(w)), max(1, int(h))
    if mode == "max":
        scale = min(1.0, REF_IMAGE_SHORT_EDGE / min(w, h))
    else:
        scale = min(1.0, math.sqrt((int(gen_w) * int(gen_h)) / float(w * h)))
    snap = lambda v: max(CANVAS_MULTIPLE, round(v * scale / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
    return snap(w), snap(h)


def _resize(image, width, height, crop):
    s = image[..., :3].movedim(-1, 1)
    s = comfy.utils.common_upscale(s, width, height, "lanczos", crop)
    return s.movedim(1, -1)


def _empty_av_latent(width, height, length, fps, batch_size=1):
    fc, lt, at = temporal_shape(length, fps)
    video = torch.zeros([batch_size, 24, lt, height // 16, width // 16], device=mm.intermediate_device())
    audio = torch.zeros([batch_size, 32, 2, at], device=mm.intermediate_device())
    return {"samples": comfy.nested_tensor.NestedTensor((video, audio))}, fc


def _auto_tile_t(n_latent_frames, requested=None):
    """Temporal tile for a tiled decode. An explicit value wins.

    The decode_tile_frames widget is gone, so this is where the value comes from
    now. It has to come from somewhere: ComfyUI's decode_tiled_3d defaults tile_t
    to 999, i.e. SPATIAL tiles only, and expanding the whole clip's time axis at
    once is the single largest allocation in a run. A "tiled" decode that keeps the
    full temporal extent barely lowers the peak, so the OOM retry that switches
    tiling on was, without this, retrying with almost the same footprint."""
    if requested:
        return int(requested)
    n = int(n_latent_frames or 0)
    return AUTO_TILE_T if n > AUTO_TILE_T else None


def _decode_video(vae, out_latent, tiled, free_first=None, tile_t=None, tile_xy=None,
                  keep=()):
    """Decode the video latent.

    `free_first` is the diffusion model: sampling is finished, and the video VAE
    needs the room for THIS decode -- the free runs immediately before it, not to
    make room for the next shot. On a card where the DiT is most of the VRAM, the
    decode does not fit until it goes.

    `keep` is what must NOT be evicted on the way. It was `keep_loaded=[]`, which
    unloaded every resident model -- including the video VAE, which ComfyUI then
    reloaded three lines later to run the decode. An evict-and-reload of the thing
    about to be used, once per shot, on every card. Peak VRAM is identical either
    way, since the VAE has to be resident to decode; the round trip was pure cost.

    memory_required is ASKED FOR HONESTLY, which it was not. It was 1e30, and
    free_memory computes `memory_to_free = memory_required - get_free_memory(device)`
    (model_management.py:887), so 1e30 means "unload everything not in keep_loaded",
    every shot, in full -- skipping partially_unload entirely.

    What that evicts is the DiT, three lines before the next shot needs it again. On
    a machine whose RAM is already full of finished frames there is nowhere for it to
    go but disk, so the reload is a read from the drive, once per shot. Reported as
    thrashing that slows the preload, and it is exactly that: the same weights being
    read back at every boundary.

    The VAE knows what its own decode costs -- ComfyUI sizes it with
    memory_used_decode and uses that number everywhere else. Asked for that instead,
    a card with headroom frees NOTHING and the DiT simply stays. A card without
    headroom frees what it needs and no more, which is what partially_unload is for.
    1e30 remains the fallback for a VAE that cannot estimate itself."""
    latent = out_latent["samples"]
    if latent.is_nested:
        latent = latent.unbind()[0]
    # PERFORMANCE PATCH (Tritant-style residency): do not proactively call
    # model_management.free_memory() here. ComfyUI's VAE/model loaders already
    # request the memory they need and can partially unload as required. Forcing
    # an extra free at every shot boundary caused H3/TE/VAE residency churn on
    # low-VRAM cards. `free_first`/`keep` stay in the signature for full workflow
    # compatibility; no node sockets or calling code need to change.
    # A VAE THAT ALREADY TILES DOES NOT NEED TO BE ASKED TO, AND ASKING COSTS 3x.
    #
    # MiniMaxH3VideoVAE.decode_tiled is, in full:
    #
    #     def decode_tiled(self, z, **kwargs):
    #         return self.decode(z)
    #
    # Every tile_t/overlap_t/tile_x/tile_y this function computes is discarded, so
    # the tiling the widget promises is not happening here -- the model tiles
    # internally either way (256px spatial, 17-frame temporal), which is why
    # comfy/sd.py sets handles_tiling on it.
    #
    # What the detour costs is the OUTPUT BUFFER. comfy's VAE.decode preallocates
    # ONE result at vae_output_dtype and hands it to the model as output_buffer=,
    # and MiniMaxH3VideoVAE.decode_temporal writes finalized chunks straight into
    # it. Going through decode_tiled instead reaches _decode_tiled_owned, which
    # calls the model with output_buffer=None -- so decode_temporal allocates its
    # own at torch.float32 -- and then makes an fp16 `copy=True` of that. Two
    # buffers, the larger of them at double width:
    #
    #     tiled : fp32 2.60GB + fp16 copy 1.30GB = 3.90GB per shot
    #     decode: one preallocated fp16          = 1.30GB per shot
    #
    # at 362 frames of 1056x608. Every shot, on the node's own default.
    #
    # So: when the VAE owns its tiling AND can be written into, the un-tiled call IS
    # the tiled one, minus the copies. Anything else keeps the old path -- this is a
    # detour around a detour, not a claim that tiling is useless.
    _owns_tiling = bool(getattr(vae, "handles_tiling", False) and getattr(
        getattr(vae, "first_stage_model", None), "comfy_has_chunked_io", False))
    if tiled and _owns_tiling:
        imgs = vae.decode(latent)
    elif tiled:
        # Temporal + spatial tiling. Without tile_t the VAE expands the WHOLE latent
        # clip at once, which on a 243-frame 1344x768 shot is the single largest
        # allocation in the run -- and on an unpruned checkpoint that is already
        # streaming, it is what tips the card over. Decoding in temporal chunks
        # trades a little speed for a much lower peak; None keeps ComfyUI's defaults.
        args = {}
        tile_t = _auto_tile_t(latent.shape[2] if latent.ndim >= 5 else 0, tile_t)
        if tile_t:
            args["tile_t"] = int(tile_t)
            args["overlap_t"] = max(1, int(tile_t) // 8)
        if tile_xy:
            args["tile_x"] = int(tile_xy)
            args["tile_y"] = int(tile_xy)
        try:
            imgs = vae.decode_tiled(latent, **args) if args else vae.decode_tiled(latent)
        except TypeError:
            imgs = vae.decode_tiled(latent)      # older signature without tile_t
    else:
        imgs = vae.decode(latent)
    if len(imgs.shape) == 5:
        imgs = imgs.reshape(-1, imgs.shape[-3], imgs.shape[-2], imgs.shape[-1])
    return imgs


def _decode_audio(audio_vae, out_latent):
    latent = out_latent["samples"]
    if latent.is_nested:
        latent = latent.unbind()[-1]
    audio = audio_vae.decode(latent).movedim(-1, 1)
    std = torch.std(audio, dim=[1, 2], keepdim=True) * 5.0
    std[std < 1.0] = 1.0
    audio = audio / std
    sr = getattr(audio_vae, "audio_sample_rate_output", getattr(audio_vae, "audio_sample_rate", 44100))
    return {"waveform": audio, "sample_rate": sr}


def _is_oom(e):
    return isinstance(e, torch.cuda.OutOfMemoryError) or "out of memory" in str(e).lower()


def _deep_cleanup():
    """Release cached VRAM between shots so a long chain does not accumulate and OOM.

    It unloads NOTHING. soft_empty_cache(force) ignores `force` in current ComfyUI
    (model_management.py:2050) -- the body only reaches empty_cache() and
    ipc_collect() -- so this drops cached blocks, not models. The `True` is kept
    only for older builds that read it; the older comment here claimed this took an
    unload_all_models path, and it does not."""
    try:
        mm.soft_empty_cache(True)
    except TypeError:
        mm.soft_empty_cache()
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass


DECODE_HEADROOM = 1.25          # over ComfyUI's own estimate, for working allocations


SAMPLE_HEADROOM = 1.35          # likewise for sampling, which is the longer stretch


def _decode_headroom(vae, latent):
    """VRAM this decode actually needs, by the VAE's own estimate. 1e30 if unknown.

    ComfyUI sizes every VAE with memory_used_decode and uses that number itself, so
    it is the honest figure to hand free_memory. The alternative -- and what was here
    -- is 1e30, which means "unload everything" and evicts the DiT before every
    decode, three lines before the next shot reloads it.

    1e30 on failure rather than 0: a bad estimate that frees too little turns a slow
    render into an OOM, and a wrong guess should fall back to the behaviour that has
    been running, not to no freeing at all."""
    try:
        dtype = getattr(vae, "vae_dtype", None) or latent.dtype
        need = float(vae.memory_used_decode(tuple(latent.shape), dtype))
        if need > 0:
            return need * DECODE_HEADROOM
    except Exception:
        pass
    return 1e30


def _resident(models):
    """The LoadedModel entries ComfyUI currently holds for `models`.

    That is the form free_memory's keep_loaded wants: it compares against the
    entries in current_loaded_models, not against the ModelPatcher objects a node
    is holding. Anything not matched is simply not kept, so a model that is not
    resident costs nothing here."""
    out = []
    for lm in list(getattr(mm, "current_loaded_models", [])):
        for m in models or ():
            if m is None:
                continue
            try:
                if lm.model is m or getattr(lm, "model", None) is getattr(m, "model", None):
                    if lm not in out:
                        out.append(lm)
            except Exception:
                pass
    return out


def _image_out_dtype():
    """The dtype ComfyUI itself hands between nodes on THIS install.

    The join used to end in a hard-coded .float(), commented "back to what every
    downstream node expects". That was true when it was written and is not a
    constant: ComfyUI has --fp16-intermediates, and on an install running it the
    VAE's own decode already returns fp16 -- VAE.vae_output_dtype() IS
    model_management.intermediate_dtype() (comfy/sd.py) -- as do EmptyLatentImage
    and the rest of nodes.py. So on that install the node was taking frames the
    VAE handed it in fp16, widening them to fp32 nothing had asked for, and
    handing them to nodes whose own convention is fp16.

    It is the largest thing this node holds, so the widening is not free: the
    2580-frame chain costed at the join is 9.3GB as fp16 and 18.5GB as fp32,
    against 44.6GB of staged weights on a 62GB machine -- which is the difference
    between the render finishing and the OOM killer taking the server. Reported as
    exactly that, twice.

    Asked, not assumed, and never widened: whatever ComfyUI says it wants between
    nodes is what the chain is built in. An install with the flag off is told
    float32 and gets float32, byte for byte what it got before. Older builds have
    no intermediate_dtype at all, so the fallback is the old constant."""
    try:
        return mm.intermediate_dtype()
    except Exception:
        return torch.float32


def _evict_all_but(keep_model, latent=None):
    """Tritant-style residency: leave eviction decisions to ComfyUI.

    H3-LongVideos calls this immediately before every sampling pass. The previous
    implementation explicitly called model_management.free_memory(), which can
    force the text encoder / VAEs through partial unload/reload cycles between
    shots. Tritant's steady multi-clip path does not issue explicit free_memory,
    unload_all_models, or soft_empty_cache calls around sampling; it lets ComfyUI
    load the component being requested and reclaim only what that load requires.

    Kept as a no-op instead of deleting/changing the call so the node interface,
    sampler flow, saved workflows, and all public behavior remain unchanged.
    """
    return None

def _sample_on_sigmas(model, seed, cfg, sampler_name, positive, negative, latent, sigmas):
    """common_ksampler, driven by an EXTERNAL sigma schedule.

    common_ksampler derives its sigmas from (sampler_name, scheduler, steps, denoise)
    and takes no schedule argument, so a schedule computed anywhere else cannot
    reach it. Under PDD that is fatal rather than merely inconvenient: the heads
    accept only their nine trained boundaries, and re-deriving the grid from
    widgets means hitting it by coincidence and losing it again the moment a step
    count changes.

    Mirrors nodes.common_ksampler's noise / mask / callback handling exactly -- the
    only substitution is comfy.sample.sample_custom for comfy.sample.sample."""
    latent_image = latent["samples"]
    latent_image = comfy.sample.fix_empty_latent_channels(
        model, latent_image,
        latent.get("downscale_ratio_spacial", None),
        latent.get("downscale_ratio_temporal", None))
    noise = comfy.sample.prepare_noise(latent_image, seed, latent.get("batch_index"))
    # `steps` here only sizes the progress bar -- the schedule is `sigmas`, whose
    # step count is one less than its length (the trailing 0.0 is an endpoint).
    callback = latent_preview.prepare_callback(model, max(len(sigmas) - 1, 1))
    samples = comfy.sample.sample_custom(
        model, noise, cfg, comfy.samplers.sampler_object(sampler_name), sigmas,
        positive, negative, latent_image,
        noise_mask=latent.get("noise_mask"), callback=callback,
        disable_pbar=not comfy.utils.PROGRESS_BAR_ENABLED, seed=seed)
    out = latent.copy()
    out.pop("downscale_ratio_spacial", None)
    out.pop("downscale_ratio_temporal", None)
    out["samples"] = samples
    return out


RESIZE_CHUNK = 32


def _stream_chunks(total):
    """A collector that writes upscaled chunks into ONE destination as they land.

    Both chunk loops in _upscale_frames used `out.append(...)` then
    `frames = torch.cat(out, dim=0)`. That is the shape the finished-chain join was
    rebuilt to stop, at a LARGER size: the list holds the whole upscaled chain and
    the cat allocates a second one, both live at the cat, and `out` is a local that
    is never cleared -- so it survives the cat, survives the trailing resize, and is
    still bound at the return. Meanwhile the CALLER's pre-upscale chain cannot be
    dropped either, because `part = frames[s:s+batch]` is a view into it.

    At 2580 frames of 1056x608 that is 9.26GB per copy per doubling: 37GB x2 at 2x,
    and 148GB x2 with the RealESRGAN_x4plus that is sitting in models/upscale_models.
    Preallocating from the first chunk and copying into it removes exactly one of
    those two, and drops the list at the same time.

    The destination is sized from the FIRST chunk, so the model's scale factor does
    not have to be known in advance, and the frame count is the caller's own -- an
    upscaler changes width and height, never the number of frames."""
    state = {"dst": None, "at": 0}

    def put(piece):
        if state["dst"] is None:
            state["dst"] = torch.empty((int(total),) + tuple(piece.shape[1:]),
                                       dtype=piece.dtype, device=piece.device)
        k = int(piece.shape[0])
        end = min(state["at"] + k, state["dst"].shape[0])
        if end > state["at"]:
            state["dst"][state["at"]:end].copy_(piece[:end - state["at"]])
        state["at"] = end

    def done():
        d, at = state["dst"], state["at"]
        if d is None:
            return None
        return d if at == d.shape[0] else d[:at]

    return put, done


def _resize_short_edge(frames, target, method="lanczos", chunk=0):
    """Resize a [B,H,W,C] frame batch so its short edge == target (keeping aspect,
    snapped to /32). Plain high-quality resize -- enlarges, doesn't add detail.

    IN CHUNKS, BECAUSE LANCZOS IS FOUR FULL-LENGTH COPIES. The whole chain went
    into one common_upscale call, and comfy.utils.lanczos is three successive list
    comprehensions over every frame at once:

        images = [Image.fromarray(...) for image in samples]        # N at source size
        images = [image.resize(...) for image in images]            # N at target size
        images = [torch.from_numpy(np.array(im).astype(np.float32)/255.) ...]
        result = torch.stack(images)
        return result.to(samples.device, samples.dtype)

    A comprehension builds the new list completely before rebinding the name, so at
    each rebind BOTH are live; then torch.stack allocates a full copy while its list
    still exists, and .to() allocates the result while the stack still exists. Note
    the astype(np.float32): the input is fp16 but the two largest transients are at
    DOUBLE its width. At 2580 frames to a 1080 short edge that peaked around 147GB
    to produce a 29GB result, and it fires on a DOWNSCALE too.

    Chunked, the peak is the result plus one chunk's worth of that machinery. It is
    bit-identical: PIL resizes each frame independently, so per-chunk and per-chain
    give the same pixels. The early return for an already-correct size is kept, so
    the common no-op case still allocates nothing."""
    b, h, w, c = frames.shape
    if min(h, w) == target:
        return frames
    if h <= w:
        nh = target; nw = max(32, int(round(target * w / h / 32) * 32))
    else:
        nw = target; nh = max(32, int(round(target * h / w / 32) * 32))
    step = max(1, int(chunk) or RESIZE_CHUNK)
    out = torch.empty((b, nh, nw, c), dtype=frames.dtype, device=frames.device)
    for i in range(0, b, step):
        part = comfy.utils.common_upscale(
            frames[i:i + step].movedim(-1, 1), nw, nh, method, "disabled")
        out[i:i + step].copy_(part.movedim(1, -1))
        del part
    return out


def _upscale_frames(frames, mode, model_name, target_short_edge, batch=4):
    """Optional post-pass upscale of the finished frames (on CPU).
      mode 'model'   : run a ComfyUI upscale model (Real-ESRGAN/UltraSharp class)
                       via the registered loader+apply nodes, chunked with cleanup
                       so 2000+ frames don't OOM; then fit to target short edge.
      mode 'rtx'     : NVIDIA RTX Video Super Resolution (Tensor Cores; fastest,
                       best quality for video -- needs Nvidia_RTX_Nodes_ComfyUI).
      mode 'lanczos' : plain high-quality resize to the target short edge.
    Any failure falls back to lanczos (or the raw frames), so it never breaks a
    render. Returns (frames, note). NOTE: this SHARPENS/ENLARGES; it does not
    reconstruct video detail the way a second-model (LTX 2.3) pass does."""
    if mode == "off" or frames is None or getattr(frames, "shape", [0])[0] == 0:
        return frames, ""
    note = ""
    if mode == "rtx":
        # NVIDIA RTX Video Super Resolution (Comfy-Org/Nvidia_RTX_Nodes_ComfyUI).
        # Runs on RTX Tensor Cores -- far faster than ESRGAN-class models and
        # generally cleaner on video, though like them it enhances/enlarges rather
        # than reconstructing detail (an LTX 2.3 re-generation does that).
        try:
            rtx = (_find_node(["rtx", "video", "super"]) or _find_node(["rtxvideosuperresolution"])
                   or _find_node(["rtx", "upscale"]))
            if rtx is None:
                raise RuntimeError("RTX node not installed (Nvidia_RTX_Nodes_ComfyUI)")
            scale = 2
            if target_short_edge and int(target_short_edge) > 0:
                cur = min(frames.shape[1], frames.shape[2])
                if cur > 0:
                    scale = max(1, min(4, int(round(int(target_short_edge) / cur))))
            _put, _done = _stream_chunks(frames.shape[0])
            n = frames.shape[0]
            step = max(1, int(batch))
            for st in range(0, n, step):
                part = frames[st:st + step]
                res = None
                for kw in ({"image": part, "scale": scale}, {"images": part, "scale": scale},
                           {"image": part, "scale_factor": scale}, {"image": part}):
                    try:
                        res = _invoke_node(rtx, **kw); break
                    except TypeError:
                        continue
                if res is None:
                    raise RuntimeError("RTX node signature not recognized")
                _put(res.detach().to("cpu"))
                del res, part
                _deep_cleanup()
            frames = _done()
            note = f"RTX Video Super Resolution x{scale}"
            if target_short_edge and int(target_short_edge) > 0:
                frames = _resize_short_edge(frames, int(target_short_edge))
                note += f"; fit to {int(target_short_edge)}px short edge"
            return frames, note
        except Exception as e:
            mode = "model"
            note = f"RTX upscale unavailable ({e}); fell back to model/lanczos"
    if mode == "model" and model_name and model_name != "none":
        try:
            loader = _find_node(["upscale", "model", "load"]) or _find_node(["loadupscalemodel"])
            applier = _find_node(["imageupscale", "model"]) or _find_node(["upscaleimageusingmodel"])
            if loader is None or applier is None:
                raise RuntimeError("upscale-model nodes not found")
            up_model = _invoke_node(loader, model_name=model_name)
            _put, _done = _stream_chunks(frames.shape[0])
            n = frames.shape[0]
            for s in range(0, n, max(1, int(batch))):
                part = frames[s:s + max(1, int(batch))]
                res = _invoke_node(applier, upscale_model=up_model, image=part)
                _put(res.detach().to("cpu"))
                del res, part
                _deep_cleanup()
            frames = _done()
            note = f"upscaled with {model_name}"
        except Exception as e:
            mode = "lanczos"
            note = f"model upscale unavailable ({e}); used lanczos"
    if target_short_edge and int(target_short_edge) > 0:
        try:
            frames = _resize_short_edge(frames, int(target_short_edge))
            note = (note + "; " if note else "") + f"fit to {int(target_short_edge)}px short edge"
        except Exception as e:
            note = (note + "; " if note else "") + f"resize failed ({e})"
    elif mode == "lanczos" and not note:
        note = "lanczos selected but no target set -> unchanged"
    return frames, note


def _find_node(substrings):
    """Find a registered node whose key contains all of `substrings` (lowercased)."""
    maps = getattr(nodes, "NODE_CLASS_MAPPINGS", {}) or {}
    for k, v in maps.items():
        kl = k.lower()
        if all(s in kl for s in substrings):
            return v
    return None


def _invoke_node(cls, **kwargs):
    """Call a registered ComfyUI node (V1 FUNCTION or V3 execute) with kwargs and
    return its first output. Used to reuse ComfyUI's own upscale-model loader/apply
    so we don't reimplement spandrel loading or tiled scaling."""
    inst = cls()
    fn = None
    if getattr(cls, "FUNCTION", None) and hasattr(inst, cls.FUNCTION):
        fn = getattr(inst, cls.FUNCTION)
    else:
        for cand in ("execute", "upscale", "load_model", "load"):
            if hasattr(inst, cand):
                fn = getattr(inst, cand); break
    if fn is None:
        raise RuntimeError("no callable entrypoint")
    out = fn(**kwargs)
    out = getattr(out, "result", out)
    return out[0] if isinstance(out, (tuple, list)) else out


# --- THE GRADE THE CHAIN ADDS TO ITSELF -------------------------------------
# Every shot boundary decodes a shot, hands its LAST frame over, and re-encodes that as
# the next shot's keyframe. The distill reproduces the keyframe faithfully enough to
# inherit whatever is already in it and SYNTHESISES frame 0 rather than copying it, so
# its own bias lands on top: S_next = a*S + b, a near 1, b above 0. Linear at best,
# geometric at worst, invisible shot to shot. And the VAE hard-clips every decode to
# 0..1, which makes the expansion a RATCHET -- headroom spent is not recoverable, so it
# shows as crushed blacks and blown highlights rather than merely as more contrast.
#
# These two are the measurement and the correction. Both work per colour channel,
# because the clip is per channel: the VAE un-whitens with ImageNet stds before it
# clamps, so the 0..1 rails sit at different distances in each channel and the blue
# floor and red ceiling bite first. A single luma number would miss the colour half.
LEVEL_POOL = 64                # cells per axis the level statistics are measured on


def frame_levels(img):
    """(mean, std) per colour channel for one frame, as 3-vectors, or (None, None).

    Area-pooled to LEVEL_POOL first, so a pre-upscale frame and an upscaled one can be
    compared: pooling measures the PICTURE's levels rather than its resolution. Measured
    across a 2x resize, std agrees to 0.28% on picture-like content -- and to only 15%
    on pure noise, because pooling cannot preserve variance that lives entirely at the
    pixel scale. Real frames are the former, and whatever residual there is cancels
    anyway: the caller measures the same pipeline difference separately and subtracts it.

    float32 throughout, deliberately: these frames are fp16 under
    --fp16-intermediates, and an fp16 mean accumulated over a 1344x768 frame biases
    badly enough to matter at the sizes being corrected here."""
    x = img
    if x.dim() == 4:
        x = x[0]
    if x.dim() != 3 or int(x.shape[-1]) < 3:
        return None, None
    if int(x.shape[0]) < 2 or int(x.shape[1]) < 2:
        return None, None
    x = x[..., :3].float().permute(2, 0, 1).unsqueeze(0)
    p = torch.nn.functional.adaptive_avg_pool2d(x, LEVEL_POOL)[0].reshape(3, -1)
    return p.mean(dim=1), p.std(dim=1)


# The per-shot motion envelope lived here, measured so the built footsteps could be
# timed off the picture. Nothing is built any more -- see the note at the top of
# audio.py -- so there is nothing left to time, and a measurement with no reader is a
# measurement that rots. Removed with the synthesiser it served.


def apply_levels(img, gain, offset):
    """Rescale a frame's contrast and level about its OWN per-channel mean.

    The pivot is the frame's own mean and never a target. That is the whole reason this
    can run on any scene: a beat that walks into a darker room keeps its darkness,
    because nothing here knows or cares what the level is -- only how much the last
    boundary expanded it. Anchoring to shot 1 instead would cancel every deliberate
    lighting change in the film, which is the opposite failure.

    Clamped into 0..1 because the next thing that happens to this frame is an 8-bit
    quantisation (comfy.utils.common_upscale goes through a uint8 PIL round trip even
    at the same size), so there is no headroom outside the range to borrow from."""
    x = img.float()
    c = min(3, int(x.shape[-1]))
    m = x[..., :c].reshape(-1, c).mean(dim=0)
    g = gain[:c].to(device=x.device, dtype=x.dtype)
    o = offset[:c].to(device=x.device, dtype=x.dtype)
    y = x.clone()
    y[..., :c] = ((x[..., :c] - m) * g + m + o).clamp(0.0, 1.0)
    return y.to(img.dtype)
