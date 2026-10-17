"""Test setup. References come from environment variables, never from this repository:

SEMBRIDGE_COMFYUI_DIR    ComfyUI checkout (v0.38.0 or later)
SEMBRIDGE_UPSTREAM_DIR   ASLP-lab/SemBridge checkout (the commit in sembridge_comfy/vendor/UPSTREAM.md)
SEMBRIDGE_TOKENIZER_DIR  folder with the official tokenizer.json, tokenizer_config.json, special_tokens_map.json
SEMBRIDGE_REQUIRE_REFERENCES=1  fail instead of skip when one of them is missing (CI sets this)
SEMBRIDGE_REQUIRE_TORCHAUDIO=1   the same for torchaudio (it only exists for torch <= 2.9, so only the torch 2.8 CI legs require it)
No trained weights are used: the model tests run a tiny random-initialized model on the CPU.
"""

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
REQUIRE = os.environ.get("SEMBRIDGE_REQUIRE_REFERENCES") == "1"


def _dir(var):
    p = os.environ.get(var)
    return p if p and Path(p).exists() else None


COMFY = _dir("SEMBRIDGE_COMFYUI_DIR")
UPSTREAM = _dir("SEMBRIDGE_UPSTREAM_DIR")
TOKENIZER = _dir("SEMBRIDGE_TOKENIZER_DIR")
if COMFY:
    sys.path.insert(1, COMFY)
    import comfy.options

    comfy.options.enable_args_parsing(False)
    import comfy.cli_args

    comfy.cli_args.args.cpu = True


def _need(value, what):
    if value:
        return value
    if REQUIRE:
        pytest.fail(f"{what} is required (SEMBRIDGE_REQUIRE_REFERENCES=1)")
    pytest.skip(f"{what} not available")


@pytest.fixture(scope="session")
def comfy_dir():
    return _need(COMFY, "ComfyUI checkout (SEMBRIDGE_COMFYUI_DIR)")


@pytest.fixture(scope="session")
def upstream_dir():
    return _need(UPSTREAM, "SemBridge checkout (SEMBRIDGE_UPSTREAM_DIR)")


@pytest.fixture(scope="session")
def tokenizer_dir():
    return Path(_need(TOKENIZER, "official tokenizer files (SEMBRIDGE_TOKENIZER_DIR)"))


@pytest.fixture(scope="session")
def torchaudio_mod():
    """torchaudio has no release for the newest torch builds; the CI legs with torch 2.8 set SEMBRIDGE_REQUIRE_TORCHAUDIO=1."""
    try:
        import torchaudio
    except ImportError:
        if os.environ.get("SEMBRIDGE_REQUIRE_TORCHAUDIO") == "1":
            pytest.fail("torchaudio is required (SEMBRIDGE_REQUIRE_TORCHAUDIO=1)")
        pytest.skip("torchaudio not installed")
    return torchaudio


@pytest.fixture(scope="session")
def tiny_model(tmp_path_factory, tokenizer_dir):
    import tiny_model as builder

    return builder.build(tmp_path_factory.mktemp("models") / "sembridge" / "tiny", tokenizer_dir)
