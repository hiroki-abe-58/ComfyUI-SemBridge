"""Aggregate the rows written by scripts/eval_generation.py and scripts/eval_asr.py (results_asr.jsonl preferred).

    python scripts/summarize_eval.py <dir>

Prints, per group (TTS English, TTS Chinese, singing, unified), the count and min / median / max of the main indicators, and lists
the clips that hit the length cap, have more than 0.25 error rate, a peak above -1 dBFS or any full-scale samples, or a pitch median
error above one semitone.
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path


def spread(values):
    values = [v for v in values if v is not None]
    if not values:
        return None
    return {"min": round(min(values), 3), "median": round(statistics.median(values), 3), "max": round(max(values), 3)}


def main() -> int:
    folder = Path(sys.argv[1])
    path = folder / "results_asr.jsonl"
    if not path.exists():
        path = folder / "results.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    groups = {
        "tts_en": [r for r in rows if r["task"] == "tts" and r["lang"] == "en"],
        "tts_zh": [r for r in rows if r["task"] == "tts" and r["lang"] == "zh"],
        "svs": [r for r in rows if r["task"] == "svs"],
        "unified": [r for r in rows if r["task"] == "unified"],
    }
    for name, group in groups.items():
        if not group:
            continue
        summary = {
            "clips": len(group),
            "seconds": spread([r["seconds"] for r in group]),
            "realtime_factor": spread([r["realtime_factor"] for r in group]),
            "peak_dbfs": spread([r["peak_dbfs"] for r in group]),
            "rms_dbfs": spread([r["rms_dbfs"] for r in group]),
            "silence_share": spread([r["silence_share"] for r in group]),
            "leading_silence_s": spread([r["leading_silence_s"] for r in group]),
            "trailing_silence_s": spread([r["trailing_silence_s"] for r in group]),
            "peak_vram_gib": spread([r["peak_vram_gib"] for r in group]),
            "hit_cap": sum(r["hit_cap"] for r in group),
            "full_scale_clips": sum(r["full_scale_share"] > 0 for r in group),
            "non_finite": sum(not r["finite"] for r in group),
        }
        if name.startswith("tts") and "error_rate" in group[0]:
            summary["error_rate"] = {**spread([r["error_rate"] for r in group]),
                                     "mean": round(statistics.mean(r["error_rate"] for r in group), 3),
                                     "clips_with_error": sum(r["error_rate"] > 0 for r in group),
                                     "clips_over_0.25": sum(r["error_rate"] > 0.25 for r in group)}
        if name == "svs":
            pitched = [r["pitch"] for r in group]
            summary["notes_scored"] = sum(p["notes_scored"] for p in pitched)
            summary["notes_with_clear_pitch"] = sum(p["notes_with_clear_pitch"] for p in pitched)
            summary["median_abs_folded_semitone_error"] = spread([p["median_abs_folded_semitone_error"] for p in pitched])
            summary["share_within_1_semitone"] = spread([p["share_within_1_semitone_folded"] for p in pitched])
        print(name, json.dumps(summary, ensure_ascii=False))
    print("-- flagged clips")
    for r in rows:
        why = []
        if r["hit_cap"]:
            why.append("hit_cap")
        if r.get("error_rate", 0) > 0.25:
            why.append(f"error_rate={r['error_rate']}")
        if r["peak_dbfs"] > -1.0:
            why.append(f"peak={r['peak_dbfs']}")
        if r["full_scale_share"] > 0:
            why.append(f"full_scale={r['full_scale_share']}")
        pitch = r.get("pitch")
        if pitch and (pitch["median_abs_folded_semitone_error"] or 0) > 1.0:
            why.append(f"pitch_median_err={pitch['median_abs_folded_semitone_error']}")
        if why:
            print(r["clip"], ", ".join(why), "|", r.get("heard", ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
