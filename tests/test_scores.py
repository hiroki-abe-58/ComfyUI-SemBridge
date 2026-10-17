"""Score / segment parsing: pure Python, no model."""

import json

import pytest

from sembridge_comfy import scores


def test_line_format_with_rest_and_note_type():
    notes = scores.parse_notes("# comment\nla 60 0.5\nla 62 0.25 3\n<SP> 0 0.4 1\n")
    assert notes == [
        {"text": "la", "pitch": 60, "duration": 0.5},
        {"text": "la", "pitch": 62, "duration": 0.25, "note_type": 3},
        {"text": "<SP>", "pitch": 0, "duration": 0.4, "note_type": 1},
    ]


def test_json_list_and_official_request_object_give_same_notes():
    raw = [{"text": "a", "pitch": 60, "duration": 0.5, "note_type": 2}]
    assert scores.parse_notes(json.dumps(raw)) == scores.parse_notes(json.dumps({"task": "svs", "notes": raw}))


@pytest.mark.parametrize(
    "bad, message",
    [
        ("", "empty"),
        ("la 60", "expected 'lyric pitch duration"),
        ("la x 0.5", "pitch must be an integer"),
        ("la 200 0.5", "outside 0..128"),
        ("la 60 0", "positive"),
        ("la 60 -1", "positive"),
        ("la 60 nan", "positive"),
        ("la 60 20", "too long"),
        ("[1, 2]", "must be an object"),
        ("[]", "non-empty"),
        ('{"x": 1}', "'notes'"),
        ("[{", "not valid JSON"),
    ],
)
def test_bad_scores_are_rejected_with_a_reason(bad, message):
    with pytest.raises(scores.ScoreError, match=message):
        scores.parse_notes(bad)


def test_note_count_is_limited():
    with pytest.raises(scores.ScoreError, match="more than"):
        scores.parse_notes("\n".join(["a 60 .1"] * (scores.MAX_NOTES + 1)))


def test_segments_section_syntax_and_json_agree():
    sections = scores.parse_segments("[speech] Hello there\nsecond line\n[singing]\nla 60 0.5\nla 62 0.5\n[SPEECH]\nBye")
    assert sections == [
        {"task": "speech", "text": "Hello there second line"},
        {"task": "singing", "notes": [{"text": "la", "pitch": 60, "duration": 0.5}, {"text": "la", "pitch": 62, "duration": 0.5}]},
        {"task": "speech", "text": "Bye"},
    ]
    assert scores.parse_segments(json.dumps(sections)) == sections
    assert scores.parse_segments(json.dumps({"segments": sections})) == sections


@pytest.mark.parametrize("bad", ["", "[speech]", "[singing]\n", "[speech]\n  \n[singing]\nla 60 0.5", '[{"task": "dance"}]', "plain words"])
def test_bad_segments_are_rejected(bad):
    with pytest.raises(scores.ScoreError):
        scores.parse_segments(bad)


def test_leading_whitespace_is_removed():
    assert scores.clean_text("  \n hello  ") == "hello"
    with pytest.raises(scores.ScoreError):
        scores.clean_text("   ")
    with pytest.raises(scores.ScoreError):
        scores.clean_text(None)


def test_length_rule_matches_the_four_official_singing_examples():
    """(score seconds, min, max) copied from examples/svs/*.json of the official repository."""
    official = [(15.30, 96, 344), (7.83, 49, 176), (6.57, 41, 148), (4.31, 27, 97)]
    for seconds, lo, hi in official:
        got = scores.length_from_score(seconds)
        assert abs(got[0] - lo) <= 1 and abs(got[1] - hi) <= 1, (seconds, got, (lo, hi))
