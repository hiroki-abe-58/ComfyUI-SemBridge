"""Does a sung result follow its score? Compare the fundamental frequency in each note's time slot with the note's MIDI pitch.

    python scripts/eval_pitch.py generated.flac "twin 60 0.4\nkle 60 0.4 ..."  (or --score-file score.txt)

Method (numpy only): for every non-rest note, take the middle 60 % of its slot (slots are the cumulative note durations from t=0),
estimate f0 by autocorrelation in 70-1000 Hz, convert to a MIDI number and report the error in semitones, once raw and once folded
to the nearest octave (a singer may sit one or two octaves away from the written pitch). This is a rough check, not a transcription:
it cannot tell breathy or unvoiced notes apart, so slots without a clear periodic peak are reported as `None` and not counted.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def f0_autocorr(segment: np.ndarray, rate: int, lo: float = 70.0, hi: float = 1000.0):
    segment = segment - segment.mean()
    if len(segment) < rate // 20 or float(np.abs(segment).max()) < 1e-3:
        return None
    n = len(segment)
    spectrum = np.fft.rfft(segment, 2 * n)
    corr = np.fft.irfft(spectrum * np.conj(spectrum))[:n]
    corr /= corr[0] + 1e-12
    lag_lo, lag_hi = int(rate / hi), min(int(rate / lo), n - 1)
    if lag_hi <= lag_lo:
        return None
    window = corr[lag_lo:lag_hi]
    lag = int(np.argmax(window)) + lag_lo
    if corr[lag] < 0.45:  # no clear periodicity
        return None
    return rate / lag


def midi_of(freq: float) -> float:
    return 69.0 + 12.0 * np.log2(freq / 440.0)


def evaluate(audio_path: Path, notes: list[dict]) -> dict:
    data, rate = sf.read(str(audio_path), dtype="float32", always_2d=True)
    mono = data.mean(axis=1)
    rows, t = [], 0.0
    for note in notes:
        start, end = t, t + note["duration"]
        t = end
        if note["pitch"] == 0 or note["text"] == "<SP>":
            continue
        lo, hi = start + 0.2 * note["duration"], start + 0.8 * note["duration"]
        f0 = f0_autocorr(mono[int(lo * rate): int(hi * rate)], rate)
        if f0 is None:
            rows.append({"note": note["text"], "target": note["pitch"], "midi": None})
            continue
        midi = midi_of(f0)
        diff = midi - note["pitch"]
        folded = (diff + 6) % 12 - 6
        rows.append({"note": note["text"], "target": note["pitch"], "midi": round(midi, 2), "error": round(diff, 2), "folded_error": round(folded, 2)})
    voiced = [r for r in rows if r["midi"] is not None]
    summary = {
        "notes_scored": len(rows),
        "notes_with_clear_pitch": len(voiced),
        "median_abs_folded_semitone_error": round(float(np.median([abs(r["folded_error"]) for r in voiced])), 2) if voiced else None,
        "share_within_1_semitone_folded": round(sum(abs(r["folded_error"]) <= 1.0 for r in voiced) / len(voiced), 2) if voiced else None,
        "median_octave_offset": round(float(np.median([round(r["error"] / 12) for r in voiced])), 1) if voiced else None,
    }
    return {"summary": summary, "notes": rows}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("audio", type=Path)
    parser.add_argument("score", nargs="?", help="score text (lines or JSON)")
    parser.add_argument("--score-file", type=Path)
    args = parser.parse_args(argv)
    from sembridge_comfy import scores

    text = args.score_file.read_text(encoding="utf-8") if args.score_file else (args.score or "").replace("\\n", "\n")
    result = evaluate(args.audio, scores.parse_notes(text))
    print(json.dumps(result, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
