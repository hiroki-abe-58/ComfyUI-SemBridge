"""Download the official SemBridge model folder into ComfyUI's models/sembridge and verify it.

    python tools/download_model.py --models-dir <ComfyUI>/models/sembridge [--name SemBridge] [--revision <commit>]

Source: https://huggingface.co/ASLP-lab/SemBridge (model card license: Apache-2.0; about 4.3 GB). The default revision is the one
this node pack was verified with; the three weight files are checked against the SHA-256 values published by the Hub.
Needs `huggingface_hub` (installed with ComfyUI's transformers). Nothing is downloaded when you do not run this script.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

REPO = "ASLP-lab/SemBridge"
VERIFIED_REVISION = "4fb6534f7c697f0aea90e71dc5a6f3123596ae45"
FILES = [
    "config.json", "generation_config.json", "model.safetensors", "audiovae.safetensors", "unified_audiovae.safetensors",
    "savae_config.json", "unified_savae_config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
    "LICENSE", "README.md",
]
SHA256 = {
    "model.safetensors": "f157386839d5bda47e533a85d34bb0803cad5ab455f73cfdfc722a530379f7db",
    "audiovae.safetensors": "44c0d88956f8060745274f0048cebc1c0aa4ec1dd1c1573ecc584e014dab71ef",
    "unified_audiovae.safetensors": "c059ae970319b83284d5d5f4aca8fab2f2af97082072534c54f428dec54d336e",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 24), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models-dir", type=Path, required=True, help="ComfyUI's models/sembridge folder")
    parser.add_argument("--name", default="SemBridge", help="sub-folder name shown in the Loader node")
    parser.add_argument("--revision", default=VERIFIED_REVISION)
    args = parser.parse_args(argv)

    from huggingface_hub import snapshot_download

    target = args.models_dir / args.name
    snapshot_download(repo_id=REPO, revision=args.revision, local_dir=str(target), allow_patterns=FILES)
    failed = False
    for name in FILES:
        if not (target / name).is_file():
            print(f"MISSING {name}")
            failed = True
    if args.revision == VERIFIED_REVISION:
        for name, expected in SHA256.items():
            actual = sha256(target / name)
            ok = actual == expected
            failed |= not ok
            print(f"{'ok     ' if ok else 'MISMATCH'} {name}")
    else:
        print("note: other revision - weight hashes were not compared")
    print(f"model folder: {target}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
