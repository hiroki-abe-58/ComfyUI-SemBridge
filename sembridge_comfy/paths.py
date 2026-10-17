"""Model locations: `<ComfyUI models>/sembridge/<name>/` (plus any path an administrator adds for `sembridge` in
extra_model_paths.yaml). A model is the complete official folder (config.json, model.safetensors, tokenizer files, both
codecs). Workflows choose a folder name from this list only; nothing is downloaded at run time."""

from __future__ import annotations

import os

FOLDER = "sembridge"

# What the official package needs to load. The "unified" codec pair is optional in the official loader (a legacy
# single-codec folder still works) but both files must be present together.
REQUIRED_FILES = (
    "config.json",
    "model.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "savae_config.json",
    "audiovae.safetensors",
)


def register() -> None:
    import folder_paths

    default = os.path.join(folder_paths.models_dir, FOLDER)
    folder_paths.add_model_folder_path(FOLDER, default, is_default=False)


def model_dirs() -> list[str]:
    import folder_paths

    found = set()
    for base in folder_paths.get_folder_paths(FOLDER):
        if not os.path.isdir(base):
            continue
        for entry in os.scandir(base):
            if entry.is_dir() and all(os.path.isfile(os.path.join(entry.path, f)) for f in REQUIRED_FILES):
                found.add(entry.name)
    return sorted(found)


def model_path(name: str) -> str:
    import folder_paths

    if name not in model_dirs():
        raise FileNotFoundError(
            f"model folder {name!r} was not found in models/{FOLDER}. Download ASLP-lab/SemBridge into "
            f"models/{FOLDER}/{name} (see the README)."
        )
    for base in folder_paths.get_folder_paths(FOLDER):
        cand = os.path.realpath(os.path.join(base, name))
        if os.path.dirname(cand) == os.path.realpath(base) and os.path.isdir(cand):
            return cand
    raise FileNotFoundError(name)
