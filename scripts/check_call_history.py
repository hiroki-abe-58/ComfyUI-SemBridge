"""Does the same request give the same audio on the first call after loading and on later calls?

    python scripts/check_call_history.py --upstream <SemBridge checkout> --model <model folder> --codec-numerics reproducible
    python scripts/check_call_history.py --upstream <SemBridge checkout> --model <model folder> --codec-numerics official-torchscript

Run each mode in a fresh process (the effect is about the first call). The request is the official `tts_zh_r001` example, so the
reference-audio encoder (which uses the codec's Snake activation) is exercised. With `reproducible` all calls must be identical (exit
code 0); with `official-torchscript` the first call differs from the later ones on CUDA (exit code 1 is the expected result there).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

ROOT = Path(__file__).resolve().parents[1]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--codec-numerics", choices=("reproducible", "official-torchscript"), required=True)
    parser.add_argument("--comfyui", type=Path, help="ComfyUI checkout (supplies comfy.audio.resample when torchaudio is not installed)")
    parser.add_argument("--repeat", type=int, default=4)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)

    sys.path.insert(0, str(ROOT))
    if args.comfyui:
        sys.path.insert(1, str(args.comfyui))
    from sembridge_comfy import runtime

    request = json.loads((args.upstream / "examples" / "tts" / "zh_r001.json").read_text(encoding="utf-8"))
    data, rate = sf.read(str(args.upstream / "examples" / "tts" / request["prompt_audio"]), dtype="float32", always_2d=True)
    prompt = runtime.AudioPrompt(torch.from_numpy(np.ascontiguousarray(data.T)), int(rate))
    loaded = runtime.acquire(runtime.make_handle(str(args.model), args.dtype, args.device, True, args.codec_numerics))

    hashes = []
    for call in range(args.repeat):
        out = runtime.generate(
            loaded, task="tts", decoder="auto", seed=42, max_new_patches=500, min_new_patches=2, flow_steps=10, cfg_value=2.0,
            temperature=1.0, text=request["text"], prompt=prompt, prompt_text=request["prompt_text"], prompt_task="speech",
            numerics=args.codec_numerics,
        )
        digest = hashlib.sha256(out["audio"]["waveform"].numpy().tobytes()).hexdigest()[:12]
        hashes.append(digest)
        print(json.dumps({"call": call + 1, "patches": out["report"]["patches_generated"], "samples": int(out["audio"]["waveform"].numel()), "sha256": digest}))
    identical = len(set(hashes)) == 1
    print(json.dumps({"codec_numerics": args.codec_numerics, "all_calls_identical": identical, "torch": torch.__version__}))
    return 0 if identical else 1


if __name__ == "__main__":
    sys.exit(main())
