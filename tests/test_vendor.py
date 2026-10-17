"""The vendored upstream package must stay byte-identical to the pinned upstream commit."""

import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / "sembridge_comfy" / "vendor"
UPSTREAM_COMMIT = "e8bee2777a0dececf87efb6705992c974b51c9b4"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_manifest_matches_every_vendored_file():
    listed = {}
    for line in (VENDOR / "MANIFEST.sha256").read_text(encoding="ascii").splitlines():
        digest, _, name = line.partition("  ")
        listed[name] = digest
    actual = {p.relative_to(VENDOR).as_posix(): _sha(p) for p in (VENDOR / "sembridge").rglob("*") if p.is_file() and "__pycache__" not in p.parts}
    actual["LICENSE"] = _sha(VENDOR / "LICENSE")
    assert actual == listed


def test_vendor_notes_pin_the_commit():
    assert UPSTREAM_COMMIT in (VENDOR / "UPSTREAM.md").read_text(encoding="utf-8")


def test_byte_identical_to_upstream_checkout(upstream_dir):
    """CI checks out the pinned commit; compare the files (LF line endings on every platform)."""
    upstream = Path(upstream_dir)
    for path in (VENDOR / "sembridge").rglob("*.py"):
        rel = path.relative_to(VENDOR).as_posix()
        assert path.read_bytes() == (upstream / rel).read_bytes(), rel
    assert (VENDOR / "LICENSE").read_bytes() == (upstream / "LICENSE").read_bytes()
    count = len([p for p in (upstream / "sembridge").rglob("*.py")])
    assert count == len(list((VENDOR / "sembridge").rglob("*.py")))
