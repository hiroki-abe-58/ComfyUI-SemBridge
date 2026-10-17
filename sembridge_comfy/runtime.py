"""Loading and running SemBridge inside ComfyUI.

The model code is the official package (ASLP-lab/SemBridge, commit e8bee27), vendored unmodified in ``vendor/sembridge``.
This module only wraps it:

* ``ComfyPipeline`` subclasses the official ``SemBridge`` and overrides ONE method, ``_load_prompt``: the official one
  reads a file with torchaudio; ComfyUI hands over an in-memory AUDIO tensor, so the same steps (downmix, resample to
  the codec rate, left-pad to a whole patch, ``codec.encode``, ``codec.patch``) are applied to that tensor.
  ``comfy.audio.resample`` is ComfyUI's port of ``torchaudio.functional.resample`` (same defaults).
* ``generate()`` calls the official ``SemBridge.generate`` unchanged. A forward hook on the model's ``stop_head`` (it is
  called once per generated patch) provides ComfyUI's progress bar and the cancel check; the seed is applied inside
  ``torch.random.fork_rng`` so ComfyUI's own RNG state is restored afterwards.
"""

from __future__ import annotations

import contextlib
import gc
import json
import os
import threading
import time
from dataclasses import dataclass, field

import torch

UPSTREAM_COMMIT = "e8bee2777a0dececf87efb6705992c974b51c9b4"
MODEL_REVISION = "4fb6534f7c697f0aea90e71dc5a6f3123596ae45"
DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
DECODERS = ("auto", "native", "bigvgan")
APPROX_VRAM_BYTES = 8 * 1024**3  # what to ask ComfyUI to free before loading (measured peak is lower, see the README)

_LOCK = threading.RLock()
_SLOT: "Loaded | None" = None


@dataclass
class AudioPrompt:
    """An in-memory reference recording (mono or multi-channel, float, CPU)."""

    waveform: torch.Tensor  # [channels, samples]
    sample_rate: int


@dataclass
class Loaded:
    key: tuple
    pipeline: object
    info: dict = field(default_factory=dict)


def _resample(waveform: torch.Tensor, orig: int, new: int) -> torch.Tensor:
    try:
        from comfy.audio import resample  # ComfyUI >= 0.3x: port of torchaudio.functional.resample
    except ImportError:
        try:
            from torchaudio.functional import resample
        except ImportError as exc:
            raise RuntimeError("Resampling the reference audio needs a ComfyUI with comfy.audio.resample or torchaudio") from exc
    return resample(waveform, orig, new)


def _pipeline_class():
    from .vendor.sembridge.pipeline import SemBridge

    class ComfyPipeline(SemBridge):
        def _load_prompt(self, path, codec=None):
            if not isinstance(path, AudioPrompt):
                return super()._load_prompt(path, codec)
            codec = self.codec if codec is None else codec
            audio = path.waveform.float()
            if audio.size(0) > 1:
                audio = audio.mean(dim=0, keepdim=True)
            if path.sample_rate != codec.sample_rate:
                audio = _resample(audio, path.sample_rate, codec.sample_rate)
            patch_samples = codec.hop_length * self.model.config.patch_size
            remainder = audio.size(-1) % patch_samples
            if remainder:
                audio = torch.nn.functional.pad(audio, (patch_samples - remainder, 0))
            with torch.inference_mode():
                latent = codec.encode(audio.to(self.device, dtype=torch.float32), codec.sample_rate)
            return codec.patch(latent).to(self.dtype)

    return ComfyPipeline


def resolve_device(choice: str) -> torch.device:
    if choice == "cpu":
        return torch.device("cpu")
    if choice in ("auto", "cuda"):
        if torch.cuda.is_available():
            return torch.device("cuda", torch.cuda.current_device())
        if choice == "cuda":
            raise RuntimeError("device=cuda was requested but no CUDA device is available")
        return torch.device("cpu")
    raise ValueError(f"unknown device {choice!r}")


def make_handle(path: str, dtype: str, device: str, keep_loaded: bool) -> dict:
    dev = resolve_device(device)
    if dev.type == "cpu" and dtype != "float32":
        dtype = "float32"  # the official loader refuses CPU + half precision
    return {"path": path, "dtype": dtype, "device": str(dev), "keep_loaded": bool(keep_loaded)}


def _free_vram(dev: torch.device) -> None:
    if dev.type != "cuda":
        return
    try:
        import comfy.model_management as mm

        mm.free_memory(APPROX_VRAM_BYTES, dev)
    except Exception:  # best effort only
        pass


def release() -> None:
    global _SLOT
    with _LOCK:
        _SLOT = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def acquire(handle: dict) -> Loaded:
    global _SLOT
    key = (handle["path"], handle["dtype"], handle["device"])
    with _LOCK:
        if _SLOT is not None and _SLOT.key == key:
            return _SLOT
        release()
        dev = torch.device(handle["device"])
        _free_vram(dev)
        t0 = time.perf_counter()
        pipeline = _pipeline_class().from_pretrained(handle["path"], device=str(dev), dtype=handle["dtype"], local_files_only=True)
        seconds = time.perf_counter() - t0
        gen_cfg = dict(pipeline.generation_config)
        info = {
            "model_dir": os.path.basename(handle["path"]),
            "dtype": handle["dtype"],
            "device": handle["device"],
            "load_seconds": round(seconds, 2),
            "generation_config": gen_cfg,
            "decoders": pipeline.validation_summary(task="svs", decoder="auto")["available_decoders"],
            "sample_rate": pipeline.codec.sample_rate,
        }
        _SLOT = Loaded(key=key, pipeline=pipeline, info=info)
        return _SLOT


def _hook(progress):
    def stop_head_called(module, args, output):
        progress()

    return stop_head_called


@contextlib.contextmanager
def _seeded(seed: int, device: torch.device):
    devices = [device] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        yield


def generate(
    loaded: Loaded,
    *,
    task: str,
    decoder: str,
    seed: int,
    max_new_patches: int,
    min_new_patches: int,
    flow_steps: int,
    cfg_value: float,
    temperature: float,
    text: str | None = None,
    notes: list | None = None,
    segments: list | None = None,
    prompt: AudioPrompt | None = None,
    prompt_text: str | None = None,
    prompt_task: str | None = None,
    progress=None,
) -> dict:
    """Run one request; returns {"audio": ComfyUI AUDIO dict, "report": dict}."""
    pipeline = loaded.pipeline
    if decoder not in DECODERS:
        raise ValueError(f"decoder must be one of {DECODERS}")
    if min_new_patches > max_new_patches:
        raise ValueError("min_new_patches must not exceed max_new_patches")
    dev = pipeline.device
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats(dev)
    handle = pipeline.model.stop_head.register_forward_hook(_hook(progress or (lambda: None)))
    started = time.perf_counter()
    try:
        with _seeded(seed, dev):
            waveform, patches = pipeline.generate(
                task=task,
                decoder=decoder,
                text=text,
                notes=notes,
                segments=segments,
                prompt_audio=prompt,
                prompt_text=prompt_text,
                prompt_task=prompt_task,
                max_new_patches=int(max_new_patches),
                min_new_patches=int(min_new_patches),
                flow_steps=int(flow_steps),
                cfg_value=float(cfg_value),
                temperature=float(temperature),
                seed=int(seed),
            )
    finally:
        handle.remove()
    elapsed = time.perf_counter() - started
    codec = pipeline._codec_for_task(task, decoder)
    sample_rate = int(codec.sample_rate)
    wav = waveform.reshape(1, 1, -1).clamp(-1.0, 1.0).contiguous()  # clamp = what the official WAV writer does
    seconds = wav.shape[-1] / sample_rate
    report = {
        "task": task,
        "decoder": codec.config.decoder_type,
        "seed": int(seed),
        "patches_generated": int(patches.shape[1]),
        "stopped_early": int(patches.shape[1]) < int(max_new_patches),
        "audio_seconds": round(seconds, 3),
        "sample_rate": sample_rate,
        "generation_seconds": round(elapsed, 3),
        "realtime_factor": round(seconds / elapsed, 3) if elapsed > 0 else None,
        "flow_steps": int(flow_steps),
        "cfg_value": float(cfg_value),
        "temperature": float(temperature),
        "reference_audio": prompt is not None,
    }
    if dev.type == "cuda":
        report["peak_vram_allocated_gib"] = round(torch.cuda.max_memory_allocated(dev) / 1024**3, 3)
    return {"audio": {"waveform": wav, "sample_rate": sample_rate}, "report": report}


def report_json(info: dict, report: dict) -> str:
    return json.dumps({"model": {k: info[k] for k in ("model_dir", "dtype", "device")}, **report}, indent=1, ensure_ascii=False)
