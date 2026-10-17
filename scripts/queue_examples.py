"""Run the official SemBridge examples through a RUNNING ComfyUI queue with these nodes, and compare with official outputs.

    python scripts/queue_examples.py --server http://127.0.0.1:8188 --upstream <SemBridge checkout> \
        --comfy-input <ComfyUI/input> --out <dir> [--official-dir <dir with <id>.wav from the official CLI>] [--examples id ...]

Each prepared request of the official repository becomes one API workflow (LoadAudio -> loader -> TTS / Singing / Speech+Singing
-> SaveAudio) and is submitted with POST /prompt. The reference recordings are copied from the official checkout into the
ComfyUI input folder as `sembridge_ref_<name>.wav` (they stay local; they are not part of this repository and keep their
own terms). Saved FLAC output is decoded and, if --official-dir is given, compared with the official WAV.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent))
import queue_client  # noqa: E402


def build(entry: dict, request: dict, ref_name: str | None, model_dir: str, codec_numerics: str = "reproducible") -> dict:
    task = request["task"]
    gen = request.get("generation", {})
    wf = {
        "1": {
            "class_type": "SemBridgeLoader",
            "inputs": {"model_dir": model_dir, "dtype": "bfloat16", "device": "auto", "keep_loaded": True, "codec_numerics": codec_numerics},
        }
    }
    common = {
        "model": ["1", 0],
        "seed": gen.get("seed", 42),
        "decoder": "auto",
        "max_new_patches": gen.get("max_new_patches", 500),
        "min_new_patches": gen.get("min_new_patches", 2),
        "flow_steps": gen.get("flow_steps", 10),
        "cfg_value": gen.get("cfg_value", 2.0),
        "temperature": gen.get("temperature", 1.0),
    }
    if ref_name:
        wf["5"] = {"class_type": "LoadAudio", "inputs": {"audio": ref_name}}
        common.update({"prompt_audio": ["5", 0], "prompt_text": request["prompt_text"], "prompt_task": request.get("prompt_task", "auto")})
    if task == "tts":
        wf["2"] = {"class_type": "SemBridgeTTS", "inputs": {**common, "text": request["text"]}}
    elif task == "svs":
        wf["2"] = {"class_type": "SemBridgeSinging", "inputs": {**common, "score": json.dumps(request["notes"], ensure_ascii=False), "length_control": "manual"}}
    else:
        wf["2"] = {"class_type": "SemBridgeUnified", "inputs": {**common, "segments": json.dumps(request["segments"], ensure_ascii=False)}}
    wf["3"] = {"class_type": "SaveAudio", "inputs": {"audio": ["2", 0], "filename_prefix": f"audio/sembridge_ex_{entry['id']}"}}
    wf["4"] = {"class_type": "PreviewAny", "inputs": {"source": ["2", 1]}}
    return wf


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--server", default="http://127.0.0.1:8188")
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--comfy-input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--official-dir", type=Path)
    parser.add_argument("--model-dir", default="SemBridge", help="folder name in models/sembridge")
    parser.add_argument("--examples", nargs="*")
    parser.add_argument("--codec-numerics", choices=("reproducible", "official-torchscript"), default="reproducible")
    parser.add_argument("--reverse", action="store_true", help="run the examples in reverse order (call-order independence check)")
    args = parser.parse_args(argv)

    index = json.loads((args.upstream / "examples" / "index.json").read_text(encoding="utf-8"))["examples"]
    if args.examples:
        index = [e for e in index if e["id"] in set(args.examples)]
    args.out.mkdir(parents=True, exist_ok=True)
    rows, bad = [], 0
    for entry in (reversed(index) if args.reverse else index):
        request = json.loads((args.upstream / "examples" / entry["request"]).read_text(encoding="utf-8"))
        ref_name = None
        if request.get("prompt_audio"):
            src = (args.upstream / "examples" / entry["request"]).parent / request["prompt_audio"]
            ref_name = f"sembridge_ref_{src.name}"
            shutil.copyfile(src.resolve(), args.comfy_input / ref_name)
        result = queue_client.run(args.server, build(entry, request, ref_name, args.model_dir, args.codec_numerics), args.out, 900)
        row = {"id": entry["id"], "status": result["status"], "wall_seconds": result["wall_seconds"], "error": result.get("error")}
        if result["status"] == "success":
            report = json.loads(next(iter(result["text"].values()))[0])
            row.update({k: report.get(k) for k in ("decoder", "patches_generated", "audio_seconds", "generation_seconds", "peak_vram_allocated_gib")})
            decoded, rate = sf.read(result["files"][0], dtype="int16")
            row["saved_sample_rate"] = rate
            row["saved_samples"] = int(decoded.size)
            if args.official_dir and (args.official_dir / f"{entry['id']}.wav").is_file():
                official, _ = sf.read(args.official_dir / f"{entry['id']}.wav", dtype="int16")
                row["official_samples"] = int(official.size)
                if official.shape == decoded.shape:
                    diff = np.abs(official.astype(np.int32) - decoded.astype(np.int32))
                    row["pcm16_max_abs_diff"] = int(diff.max())
                    row["pcm16_identical"] = bool(diff.max() == 0)
                    o, d = official.astype(np.float64), decoded.astype(np.float64)
                    row["waveform_correlation"] = round(float(np.corrcoef(o, d)[0, 1]), 6)
        else:
            bad += 1
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
    (args.out / "queue_examples.json").write_text(json.dumps(rows, indent=1, ensure_ascii=False), encoding="utf-8")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
