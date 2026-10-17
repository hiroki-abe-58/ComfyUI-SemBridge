# Vendored upstream code

`sembridge/` is an UNMODIFIED copy of the `sembridge` package of https://github.com/ASLP-lab/SemBridge

- commit: `e8bee2777a0dececf87efb6705992c974b51c9b4` (`main`, 2026-10-13)
- license: Apache License 2.0 (`LICENSE` next to this file is the upstream file, including its third-party notices)
- `MANIFEST.sha256` lists the SHA-256 of every vendored file; `tests/test_vendor.py` checks it, and, when a checkout is
  available (CI), compares the files byte for byte with the pinned upstream commit.

All behaviour changes live outside this folder (`../runtime.py`).
