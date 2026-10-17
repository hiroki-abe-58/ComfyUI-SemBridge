"""Rewrite sembridge_comfy/vendor/MANIFEST.sha256 after the vendored upstream copy was refreshed (maintainer tool)."""

import hashlib
from pathlib import Path

VENDOR = Path(__file__).resolve().parents[1] / "sembridge_comfy" / "vendor"


def main() -> None:
    files = sorted(p for p in (VENDOR / "sembridge").rglob("*") if p.is_file() and "__pycache__" not in p.parts)
    files.append(VENDOR / "LICENSE")
    lines = [f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(VENDOR).as_posix()}" for p in files]
    (VENDOR / "MANIFEST.sha256").write_text("\n".join(lines) + "\n", encoding="ascii", newline="\n")
    print(f"{len(lines)} files")


if __name__ == "__main__":
    main()
