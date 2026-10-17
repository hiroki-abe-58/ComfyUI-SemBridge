"""Batch test of the nodes' runtime on new prompts (no reference voice): speed, level and silence statistics, pitch for singing.

    python scripts/eval_generation.py --model <model folder> --out <dir> [--comfyui <ComfyUI checkout>] [--seeds 1 2 3]

Writes <out>/<id>.flac and <out>/results.jsonl (one row per clip). Then run scripts/eval_asr.py on the folder for the Whisper
round trip. These are automatic indicators only: they do not say whether a clip sounds good, and nobody listened to the clips.

Rows contain: seconds, patches, whether the cap was hit (`hit_cap`, a runaway or truncated output), generation seconds, realtime
factor, peak and RMS in dBFS, share of 20 ms frames below -50 dBFS (`silence_share`), leading/trailing silence, share of samples at
full scale, and for singing the pitch error per note (see scripts/eval_pitch.py).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]

TTS = {
    "en": [
        "Good morning.",
        "Please close the door.",
        "The quick brown fox jumps over the lazy dog near the river bank.",
        "She sells sea shells by the sea shore, and the shells she sells are surely sea shells.",
        "Our meeting is scheduled for three thirty on Tuesday afternoon, in room two hundred and four.",
        "Artificial intelligence is changing how we write, draw, and compose music, but people still decide what is worth making.",
        "In the early morning the village was quiet. A baker opened his shop, the smell of warm bread drifted along the street, "
        "and a few sleepy children walked slowly toward the school.",
        "Wait, what? You really did that on your own? That is amazing!",
    ],
    "zh": [
        "早上好。",
        "请把门关上。",
        "明天下午三点在二零四会议室开会，请大家准时参加。",
        "四是四，十是十，十四是十四，四十是四十。",
        "人工智能正在改变我们写作、绘画和创作音乐的方式，但决定什么值得创作的仍然是人。",
        "清晨的村庄很安静，面包师打开了店门，温暖的面包香气飘过街道，几个睡眼惺忪的孩子慢慢走向学校。",
        "你真的是自己做的吗？太了不起了！",
        "春眠不觉晓，处处闻啼鸟。夜来风雨声，花落知多少。",
    ],
}

SCORES = {
    "twinkle_en": "twin 60 0.4\nkle 60 0.4\ntwin 67 0.4\nkle 67 0.4\nlit 69 0.4\ntle 69 0.4\nstar 67 0.8\n<SP> 0 0.4",
    "scale_la": "la 60 0.5\nla 62 0.5\nla 64 0.5\nla 65 0.5\nla 67 0.5\nla 69 0.5\nla 71 0.5\nla 72 1.0\n<SP> 0 0.4",
    "twinkle_zh": "一 60 0.4\n闪 60 0.4\n一 67 0.4\n闪 67 0.4\n亮 69 0.4\n晶 69 0.4\n晶 67 0.8\n<SP> 0 0.4",
    "birthday_en": "hap 60 0.375\npy 60 0.125\nbirth 62 0.5\nday 60 0.5\nto 65 0.5\nyou 64 1.0\n<SP> 0 0.4",
    "low_la": "la 48 0.5\nla 50 0.5\nla 52 0.5\nla 55 1.0\nla 52 0.5\nla 48 1.0\n<SP> 0 0.4",
    "high_leaps_la": "la 72 0.5\nla 79 0.5\nla 72 0.5\nla 76 0.5\nla 79 1.0\nla 74 1.0\n<SP> 0 0.4",
}

UNIFIED = {
    "unified_en": "[speech]\nHere is a short song.\n[singing]\nla 60 0.5\nla 64 0.5\nla 67 1.0\n[speech]\nThank you for listening.",
    "unified_zh": "[speech]\n下面是一首短歌。\n[singing]\n啦 60 0.5\n啦 64 0.5\n啦 67 1.0\n[speech]\n谢谢大家。",
}


def level_stats(x: np.ndarray, rate: int) -> dict:
    peak = float(np.abs(x).max()) if x.size else 0.0
    rms = float(np.sqrt((x ** 2).mean())) if x.size else 0.0
    hop = int(rate * 0.02)
    frames = x[: len(x) // hop * hop].reshape(-1, hop) if len(x) >= hop else x.reshape(1, -1)
    frame_db = 20 * np.log10(np.sqrt((frames ** 2).mean(axis=1)) + 1e-9)
    quiet = frame_db < -50.0
    loud = np.flatnonzero(~quiet)
    lead = float(loud[0] * hop / rate) if loud.size else float(len(x) / rate)
    trail = float((len(frames) - 1 - loud[-1]) * hop / rate) if loud.size else float(len(x) / rate)
    return {
        "peak_dbfs": round(20 * np.log10(peak + 1e-9), 2),
        "rms_dbfs": round(20 * np.log10(rms + 1e-9), 2),
        "silence_share": round(float(quiet.mean()), 3),
        "leading_silence_s": round(lead, 3),
        "trailing_silence_s": round(trail, 3),
        "full_scale_share": round(float((np.abs(x) >= 0.999).mean()), 6),
        "dc_offset": round(float(x.mean()), 6),
        "finite": bool(np.isfinite(x).all()),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--comfyui", type=Path, help="ComfyUI checkout (supplies comfy.audio.resample when torchaudio is not installed)")
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--codec-numerics", default="reproducible", choices=("reproducible", "official-torchscript"))
    parser.add_argument("--only", choices=("tts", "svs", "unified"), nargs="+", default=["tts", "svs", "unified"])
    args = parser.parse_args(argv)

    sys.path.insert(0, str(ROOT))
    sys.path.insert(1, str(ROOT / "scripts"))
    if args.comfyui:
        sys.path.insert(2, str(args.comfyui))
    import torch
    from sembridge_comfy import runtime, scores
    import eval_pitch

    args.out.mkdir(parents=True, exist_ok=True)
    loaded = runtime.acquire(runtime.make_handle(str(args.model), args.dtype, "auto", True, args.codec_numerics))
    jobs = []
    if "tts" in args.only:
        for lang, texts in TTS.items():
            for i, text in enumerate(texts):
                jobs += [(f"tts_{lang}{i + 1}_s{s}", "tts", {"text": text}, {"lang": lang, "text": text}, s) for s in args.seeds]
    if "svs" in args.only:
        for name, score in SCORES.items():
            jobs += [(f"svs_{name}_s{s}", "svs", {"notes": scores.parse_notes(score, "score")}, {"score": score}, s) for s in args.seeds]
    if "unified" in args.only:
        for name, seg in UNIFIED.items():
            jobs += [(f"{name}_s{s}", "unified", {"segments": scores.parse_segments(seg)}, {"segments": seg}, s) for s in args.seeds[:2]]

    rows = []
    started = time.time()
    for n, (clip, task, kwargs, meta, seed) in enumerate(jobs, 1):
        extra = {}
        if task == "svs":
            min_p, max_p = scores.length_from_score(scores.score_seconds(kwargs["notes"]))
            extra = {"min_new_patches": min_p, "max_new_patches": max_p, "flow_steps": 20}
        elif task == "unified":
            extra = {"min_new_patches": 2, "max_new_patches": 500, "flow_steps": 20}
        else:
            extra = {"min_new_patches": 2, "max_new_patches": 500, "flow_steps": 10}
        out = runtime.generate(loaded, task=task, decoder="auto", seed=seed, cfg_value=2.0, temperature=1.0,
                               numerics=args.codec_numerics, **kwargs, **extra)
        wav = out["audio"]["waveform"].reshape(-1).float().numpy()
        rate = out["audio"]["sample_rate"]
        path = args.out / f"{clip}.flac"
        sf.write(str(path), wav, rate, subtype="PCM_16")
        rep = out["report"]
        row = {"clip": clip, "task": task, "seed": seed, **meta, "seconds": rep["audio_seconds"], "patches": rep["patches_generated"],
               "max_new_patches": extra["max_new_patches"], "hit_cap": rep["patches_generated"] >= extra["max_new_patches"],
               "generation_s": rep["generation_seconds"], "realtime_factor": rep["realtime_factor"],
               "peak_vram_gib": rep.get("peak_vram_allocated_gib"), **level_stats(wav, rate)}
        if task == "svs":
            row["pitch"] = eval_pitch.evaluate(path, kwargs["notes"])["summary"]
        rows.append(row)
        print(f"[{n}/{len(jobs)}] {clip}: {row['seconds']} s in {row['generation_s']} s, peak {row['peak_dbfs']} dBFS, hit_cap={row['hit_cap']}", flush=True)
    (args.out / "results.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
    print(json.dumps({"clips": len(rows), "wall_s": round(time.time() - started, 1), "torch": torch.__version__, "codec_numerics": args.codec_numerics}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
