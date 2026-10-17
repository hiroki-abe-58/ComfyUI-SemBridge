"""Compare this node pack's runtime with the official SemBridge package, request by request, in ONE Python environment.

    python scripts/parity_vs_official.py --upstream <SemBridge checkout> --model <model folder> --out <dir> [--examples id ...]

For every prepared example of the official repository the script runs
  A) the official ``SemBridge.generate`` (reference audio read by the official code path), and
  B) ``sembridge_comfy.runtime.generate`` (reference audio handed over as an in-memory tensor, as ComfyUI does),
with identical request fields, then compares the two waveforms (float and 16-bit PCM, which is what the official WAV
writer stores). Both pipelines are loaded from the same files on the same device. Nothing is skipped silently: an
example that cannot run makes the script exit non-zero.
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

ROOT = Path(__file__).resolve().parents[1]


def pcm16(wave: torch.Tensor) -> np.ndarray:
    """16-bit samples exactly as the official WAV writer (soundfile, PCM_16) stores them."""
    buffer = io.BytesIO()
    sf.write(buffer, wave.detach().float().cpu().clamp(-1, 1).flatten().numpy(), 44100, subtype="PCM_16", format="WAV")
    buffer.seek(0)
    return sf.read(buffer, dtype="int16")[0]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--examples", nargs="*")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--codec-numerics",
        choices=("reproducible", "official-torchscript"),
        default="official-torchscript",
        help="official-torchscript: both sides use the official TorchScript Snake (call-history dependent, so both sides see the same "
        "sequence of calls); reproducible: both sides use the plain-PyTorch Snake of runtime.codec_numerics (the official run is wrapped too)",
    )
    parser.add_argument(
        "--torchaudio-shim",
        type=Path,
        metavar="COMFYUI_DIR",
        help="for environments without torchaudio (a stock ComfyUI): give the OFFICIAL code path a torchaudio stand-in made of "
        "soundfile (load) and ComfyUI's comfy.audio.resample (a port of torchaudio.functional.resample)",
    )
    args = parser.parse_args(argv)

    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(args.upstream))
    if args.torchaudio_shim is not None:
        import types

        sys.path.insert(0, str(args.torchaudio_shim))
        from comfy.audio import resample as comfy_resample

        def shim_load(path):
            data, rate = sf.read(str(path), dtype="float32", always_2d=True)
            return torch.from_numpy(np.ascontiguousarray(data.T)), int(rate)

        import importlib.machinery

        shim = types.ModuleType("torchaudio")
        shim.__spec__ = importlib.machinery.ModuleSpec("torchaudio", None)
        shim.load = shim_load
        shim.functional = types.SimpleNamespace(resample=comfy_resample)
        sys.modules["torchaudio"] = shim
    from sembridge import SemBridge as Official
    from sembridge.requests import load_request

    import contextlib

    import sembridge.models.savae as official_savae

    from sembridge_comfy import runtime

    official_snake = official_savae.snake

    @contextlib.contextmanager
    def official_numerics(mode):
        """The official package is a different module object from the vendored copy, so switch its Snake here."""
        official_savae.snake = runtime._eager_snake if mode == "reproducible" else official_snake
        try:
            yield
        finally:
            official_savae.snake = official_snake

    index =json.loads((args.upstream / "examples" / "index.json").read_text(encoding="utf-8"))["examples"]
    if args.examples:
        index = [e for e in index if e["id"] in set(args.examples)]
    args.out.mkdir(parents=True, exist_ok=True)

    official = Official.from_pretrained(str(args.model), device=args.device, dtype=args.dtype, local_files_only=True)
    handle = runtime.make_handle(str(args.model), args.dtype, args.device, True)
    loaded = runtime.acquire(handle)

    rows, failed = [], 0
    for entry in index:
        request = load_request(args.upstream / "examples" / entry["request"])
        t0 = time.perf_counter()
        with official_numerics(args.codec_numerics):
            wave_a, patches_a = official.generate(decoder="auto", **request)
        t_a = time.perf_counter() - t0

        node_args = dict(request)
        prompt_path = node_args.pop("prompt_audio", None)
        prompt = None
        if prompt_path is not None:
            data, rate = sf.read(str(prompt_path), dtype="float32", always_2d=True)
            prompt = runtime.AudioPrompt(torch.from_numpy(np.ascontiguousarray(data.T)), int(rate))
        t0 = time.perf_counter()
        out = runtime.generate(
            loaded,
            task=node_args["task"],
            decoder="auto",
            seed=node_args["seed"],
            max_new_patches=node_args["max_new_patches"],
            min_new_patches=node_args["min_new_patches"],
            flow_steps=node_args["flow_steps"] or loaded.info["generation_config"]["flow_steps"],
            cfg_value=node_args["cfg_value"] if node_args["cfg_value"] is not None else loaded.info["generation_config"]["cfg_value"],
            temperature=node_args["temperature"] if node_args["temperature"] is not None else loaded.info["generation_config"]["temperature"],
            text=node_args.get("text"),
            notes=node_args.get("notes"),
            segments=node_args.get("segments"),
            prompt=prompt,
            prompt_text=node_args.get("prompt_text"),
            prompt_task=node_args.get("prompt_task"),
            numerics=args.codec_numerics,
        )
        t_b = time.perf_counter() - t0
        wave_b = out["audio"]["waveform"]

        a16, b16 = pcm16(wave_a), pcm16(wave_b)
        same_len = a16.shape == b16.shape
        row = {
            "id": entry["id"],
            "samples_official": int(a16.size),
            "samples_node": int(b16.size),
            "pcm16_identical": bool(same_len and np.array_equal(a16, b16)),
            "float_identical": bool(same_len and torch.equal(wave_a.reshape(-1).clamp(-1, 1), wave_b.reshape(-1))),
            "max_abs_pcm16_diff": int(np.abs(a16.astype(np.int32) - b16.astype(np.int32)).max()) if same_len else None,
            "seconds_official": round(t_a, 2),
            "seconds_node": round(t_b, 2),
            "audio_seconds": round(a16.size / 44100, 3),
        }
        rows.append(row)
        failed += 0 if row["pcm16_identical"] else 1
        sf.write(str(args.out / f"{entry['id']}.official.wav"), a16, 44100, subtype="PCM_16")
        sf.write(str(args.out / f"{entry['id']}.node.wav"), b16, 44100, subtype="PCM_16")
        print(json.dumps(row), flush=True)

    summary = {
        "torch": torch.__version__,
        "device": torch.cuda.get_device_name(0) if args.device.startswith("cuda") else "cpu",
        "dtype": args.dtype,
        "codec_numerics": args.codec_numerics,
        "examples": len(rows),
        "pcm16_identical": sum(r["pcm16_identical"] for r in rows),
    }
    (args.out / "parity.json").write_text(json.dumps({"summary": summary, "rows": rows}, indent=1), encoding="utf-8")
    print(json.dumps(summary))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
