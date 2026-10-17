"""SemBridge speech and singing for ComfyUI (unofficial integration of https://github.com/ASLP-lab/SemBridge)."""

import logging

try:
    from .sembridge_comfy.nodes import comfy_entrypoint  # noqa: F401
except ImportError as e:  # ComfyUI without the V3 node API, or a missing dependency (soundfile)
    logging.error("[SemBridge] not loaded: %s", e)
    NODE_CLASS_MAPPINGS = {}
    NODE_DISPLAY_NAME_MAPPINGS = {}
