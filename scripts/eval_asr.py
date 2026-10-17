"""Whisper round trip for the clips written by scripts/eval_generation.py (TTS and the spoken parts only through the TTS rows).

    python scripts/eval_asr.py --results <dir>/results.jsonl [--model openai/whisper-small]

Adds `heard` and `error_rate` to every TTS row (word error rate for English, character error rate for Chinese after converting
traditional to simplified characters; punctuation and case ignored) and writes <dir>/results_asr.jsonl plus a summary line per
language. Whisper is a rough proxy for intelligibility: a clip can score 0 and still sound bad, and one wrong word can be Whisper's
mistake. Needs transformers, opencc (opencc-python-reimplemented) and soundfile/scipy; the Whisper model is downloaded by transformers
unless it is already in the Hugging Face cache.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import warnings
from pathlib import Path


def norm(text: str, lang: str, t2s):
    text = text.lower()
    if lang == "zh":
        return list(re.sub(r"[^一-鿿]", "", t2s.convert(text)))
    return re.sub(r"[^a-z0-9 ]", "", text).split()


def edit_distance(a, b) -> int:
    d = list(range(len(b) + 1))
    for i in range(1, len(a) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(b) + 1):
            cur = d[j]
            d[j] = min(d[j] + 1, d[j - 1] + 1, prev + (a[i - 1] != b[j - 1]))
            prev = cur
    return d[-1]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--model", default="openai/whisper-small")
    args = parser.parse_args(argv)
    warnings.filterwarnings("ignore")
    import numpy as np
    import scipy.signal
    import soundfile as sf
    import torch
    from opencc import OpenCC
    from transformers import pipeline

    t2s = OpenCC("t2s")
    asr = pipeline("automatic-speech-recognition", model=args.model, device=0 if torch.cuda.is_available() else -1,
                   torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32)
    rows = [json.loads(line) for line in args.results.read_text(encoding="utf-8").splitlines() if line.strip()]
    for row in rows:
        if row["task"] != "tts":
            continue
        data, rate = sf.read(str(args.results.parent / f"{row['clip']}.flac"), dtype="float32", always_2d=True)
        audio16 = scipy.signal.resample_poly(data.mean(axis=1), 16000, rate).astype(np.float32)
        out = asr({"raw": audio16, "sampling_rate": 16000},
                  generate_kwargs={"language": "chinese" if row["lang"] == "zh" else "english", "task": "transcribe"})
        ref, hyp = norm(row["text"], row["lang"], t2s), norm(out["text"], row["lang"], t2s)
        row["heard"] = out["text"].strip()
        row["error_rate"] = round(edit_distance(ref, hyp) / max(1, len(ref)), 3)
        print(f"{row['clip']}: {row['error_rate']}  {row['heard']}", flush=True)
    (args.results.parent / "results_asr.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
    for lang in ("en", "zh"):
        errs = [r["error_rate"] for r in rows if r["task"] == "tts" and r["lang"] == lang]
        if errs:
            print(json.dumps({"lang": lang, "clips": len(errs), "mean_error_rate": round(statistics.mean(errs), 3),
                              "clips_with_error": sum(e > 0 for e in errs), "clips_over_0.25": sum(e > 0.25 for e in errs)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
