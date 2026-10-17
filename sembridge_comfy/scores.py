"""Text inputs for singing and mixed requests -> the note / segment dictionaries the official package expects.

No torch and no model needed: everything here is pure Python so it runs (and is tested) without a GPU.

Note layout (identical to the official request JSON): ``text``, MIDI ``pitch`` (0 = rest, ``<SP>``), ``duration`` in
SECONDS and, optionally, ``note_type`` (3 = this note extends the previous lyric over another pitch).
"""

from __future__ import annotations

import json
import math
import re

PATCH_SECONDS = 0.08  # one generated patch = 2 latent frames at 25 Hz
MAX_NOTES = 2000
MAX_TEXT_CHARS = 20000
MAX_SEGMENTS = 64

_SECTION = re.compile(r"^\s*\[(speech|singing|tts|svs)\]\s*(.*)$", re.IGNORECASE)


class ScoreError(ValueError):
    """The user's text could not be turned into a valid request."""


def clean_text(value: str, what: str = "text") -> str:
    """Strip surrounding whitespace.

    transformers 4.x and 5.x tokenize *leading* whitespace differently (checked on 412 strings: every difference was a
    leading space or newline), so it is removed to give the same token IDs with either version.
    """
    if not isinstance(value, str):
        raise ScoreError(f"{what} must be text")
    value = value.strip()
    if not value:
        raise ScoreError(f"{what} is empty")
    if len(value) > MAX_TEXT_CHARS:
        raise ScoreError(f"{what} is longer than {MAX_TEXT_CHARS} characters")
    return value


def _note(raw: dict, where: str) -> dict:
    if not isinstance(raw, dict):
        raise ScoreError(f"{where}: a note must be an object with text, pitch and duration")
    missing = [k for k in ("text", "pitch", "duration") if k not in raw]
    if missing:
        raise ScoreError(f"{where}: missing {', '.join(missing)}")
    try:
        pitch = int(raw["pitch"])
        duration = float(raw["duration"])
    except (TypeError, ValueError) as exc:
        raise ScoreError(f"{where}: pitch must be an integer and duration a number of seconds") from exc
    if not 0 <= pitch <= 128:
        raise ScoreError(f"{where}: pitch {pitch} is outside 0..128 (0 = rest)")
    if not math.isfinite(duration) or duration <= 0:
        raise ScoreError(f"{where}: duration must be a positive number of seconds")
    frames = max(1, int(round(duration / 0.025)))
    if frames >= 500:
        raise ScoreError(f"{where}: a single note of {duration:g} s is too long (the model supports < 12.5 s per note)")
    out = {"text": str(raw["text"]), "pitch": pitch, "duration": duration}
    if raw.get("note_type") is not None:
        try:
            out["note_type"] = int(raw["note_type"])
        except (TypeError, ValueError) as exc:
            raise ScoreError(f"{where}: note_type must be an integer") from exc
    return out


def _notes_from_json(value, where: str) -> list[dict]:
    if isinstance(value, dict):
        if "notes" not in value:
            raise ScoreError(f"{where}: JSON object without a 'notes' list")
        value = value["notes"]
    if not isinstance(value, list) or not value:
        raise ScoreError(f"{where}: expected a non-empty list of notes")
    if len(value) > MAX_NOTES:
        raise ScoreError(f"{where}: more than {MAX_NOTES} notes")
    return [_note(n, f"{where} note {i + 1}") for i, n in enumerate(value)]


def _notes_from_lines(text: str, where: str) -> list[dict]:
    notes = []
    for number, line in enumerate(text.splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) not in (3, 4):
            raise ScoreError(f"{where} line {number}: expected 'lyric pitch duration [note_type]', got {line!r}")
        raw = {"text": parts[0], "pitch": parts[1], "duration": parts[2]}
        if len(parts) == 4:
            raw["note_type"] = parts[3]
        notes.append(_note(raw, f"{where} line {number}"))
    if not notes:
        raise ScoreError(f"{where}: no notes")
    if len(notes) > MAX_NOTES:
        raise ScoreError(f"{where}: more than {MAX_NOTES} notes")
    return notes


def parse_notes(text: str, where: str = "score") -> list[dict]:
    """A JSON list (or an official request object with 'notes') or one note per line: ``lyric pitch seconds [type]``."""
    text = clean_text(text, where)
    if text[0] in "[{":
        try:
            return _notes_from_json(json.loads(text), where)
        except json.JSONDecodeError as exc:
            raise ScoreError(f"{where}: not valid JSON ({exc.msg} at line {exc.lineno})") from exc
    return _notes_from_lines(text, where)


def _segment(raw: dict, index: int) -> dict:
    where = f"segment {index + 1}"
    if not isinstance(raw, dict):
        raise ScoreError(f"{where}: must be an object")
    task = str(raw.get("task", "")).lower()
    if task in ("speech", "tts"):
        return {"task": "speech", "text": clean_text(raw.get("text"), f"{where} text")}
    if task in ("singing", "svs"):
        return {"task": "singing", "notes": _notes_from_json(raw.get("notes"), where)}
    raise ScoreError(f"{where}: task must be 'speech' or 'singing'")


def parse_segments(text: str) -> list[dict]:
    """Ordered speech / singing segments.

    Either the official JSON (``[{"task": "speech", "text": "..."}, {"task": "singing", "notes": [...]}]``) or sections::

        [speech] Some words to say
        [singing]
        la 60 0.5
        la 62 0.5
        [speech]
        More words
    """
    text = clean_text(text, "segments")
    segments: list[dict] = []
    if not _SECTION.match(text.splitlines()[0]):
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ScoreError(f"segments: neither section syntax ([speech] / [singing]) nor valid JSON ({exc.msg})") from exc
        if isinstance(data, dict):
            data = data.get("segments")
        if not isinstance(data, list) or not data:
            raise ScoreError("segments: expected a non-empty list")
        segments = [_segment(s, i) for i, s in enumerate(data)]
    else:
        kind, body = None, []

        def flush():
            if kind is None:
                return
            joined = "\n".join(body).strip()
            where = f"section {len(segments) + 1}"
            if kind == "speech":
                segments.append({"task": "speech", "text": clean_text(" ".join(joined.split()), f"{where} text")})
            else:
                segments.append({"task": "singing", "notes": _notes_from_lines(joined, where)})

        for line in text.splitlines():
            m = _SECTION.match(line)
            if m:
                flush()
                kind = "speech" if m.group(1).lower() in ("speech", "tts") else "singing"
                body = [m.group(2)] if m.group(2).strip() else []
            else:
                body.append(line)
        flush()
    if not segments:
        raise ScoreError("segments: nothing to generate")
    if len(segments) > MAX_SEGMENTS:
        raise ScoreError(f"segments: more than {MAX_SEGMENTS} segments")
    return segments


def score_seconds(notes: list[dict]) -> float:
    return float(sum(n["duration"] for n in notes))


def length_from_score(seconds: float) -> tuple[int, int]:
    """(min_new_patches, max_new_patches) from the score length, as the official singing examples set them.

    The prepared SVS requests in the official repository use min = 0.5 x and max = 1.8 x the score length in patches
    (80 ms each); checked against all four of them (see docs/VERIFICATION.md).
    """
    patches = seconds / PATCH_SECONDS
    return max(2, int(round(0.5 * patches))), max(3, int(round(1.8 * patches)))
