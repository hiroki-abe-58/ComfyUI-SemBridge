"""The node pack as ComfyUI loads it: registration, ComfyUI's own validation of the API workflows, and node execution
with a tiny random model on the CPU."""

import asyncio
import json
import shutil
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
NODE_IDS = {"SemBridgeLoader", "SemBridgeTTS", "SemBridgeSinging", "SemBridgeUnified", "SemBridgeScorePreview", "SemBridgeUnload"}


@pytest.fixture(scope="module")
def comfy_nodes(comfy_dir, tiny_model, tmp_path_factory):
    import folder_paths
    import nodes

    models = tmp_path_factory.mktemp("models") / "sembridge"
    shutil.copytree(tiny_model, models / "SemBridge")  # the name the shipped workflows use
    (models / "incomplete").mkdir()
    (models / "incomplete" / "config.json").write_text("{}", encoding="utf-8")
    folder_paths.add_model_folder_path("sembridge", str(models), is_default=True)
    asyncio.run(nodes.init_extra_nodes(init_custom_nodes=False, init_api_nodes=False))
    assert asyncio.run(nodes.load_custom_node(str(ROOT))) is True
    folder_paths.filename_list_cache.clear()
    return nodes


def test_registered(comfy_nodes):
    assert NODE_IDS <= set(comfy_nodes.NODE_CLASS_MAPPINGS)


def test_model_folder_lists_only_complete_models(comfy_nodes):
    from sembridge_comfy import paths

    assert "SemBridge" in paths.model_dirs()
    assert "incomplete" not in paths.model_dirs()
    for bad in ("..", "../x", "incomplete", "missing"):
        with pytest.raises(FileNotFoundError):
            paths.model_path(bad)


@pytest.mark.parametrize("name", sorted(p.name for p in (ROOT / "workflows" / "api").glob("*.json")))
def test_api_workflows_pass_comfy_validation(comfy_nodes, name):
    import execution

    prompt = json.loads((ROOT / "workflows" / "api" / name).read_text(encoding="utf-8"))
    valid, error, outputs, node_errors = asyncio.run(execution.validate_prompt("t", prompt, None))
    assert valid, (error, node_errors)


def test_validation_rejects_out_of_range_values(comfy_nodes):
    import execution

    prompt = json.loads((ROOT / "workflows" / "api" / "tts_basic.json").read_text(encoding="utf-8"))
    prompt["2"]["inputs"]["flow_steps"] = 1
    valid, _, _, node_errors = asyncio.run(execution.validate_prompt("t", prompt, None))
    assert not valid and "2" in node_errors


def _handle(comfy_nodes):
    out = comfy_nodes.NODE_CLASS_MAPPINGS["SemBridgeLoader"].execute("SemBridge", "bfloat16", "cpu", True)
    return out.args[0]


COMMON = dict(seed=1, decoder="auto", max_new_patches=5, min_new_patches=2, flow_steps=2, cfg_value=2.0, temperature=1.0)


def test_loader_handle_never_carries_half_precision_on_cpu(comfy_nodes):
    handle = _handle(comfy_nodes)
    assert handle["dtype"] == "float32" and handle["device"] == "cpu" and Path(handle["path"]).is_dir()


def test_tts_singing_unified_nodes_return_audio_and_a_report(comfy_nodes):
    n = comfy_nodes.NODE_CLASS_MAPPINGS
    handle = _handle(comfy_nodes)
    runs = [
        n["SemBridgeTTS"].execute(handle, "Hello there.", **COMMON),
        n["SemBridgeSinging"].execute(handle, "la 60 0.5\nla 62 0.5 3", "manual", **{**COMMON, "flow_steps": 2}),
        n["SemBridgeUnified"].execute(handle, "[speech] Hi.\n[singing]\nla 60 0.5", **COMMON),
    ]
    for out in runs:
        audio, report = out.args
        assert audio["sample_rate"] == 44100 and audio["waveform"].dim() == 3 and torch.isfinite(audio["waveform"]).all()
        assert json.loads(report)["audio_seconds"] > 0


def test_singing_from_score_sets_patch_limits_from_the_score(comfy_nodes):
    n = comfy_nodes.NODE_CLASS_MAPPINGS
    out = n["SemBridgeSinging"].execute(_handle(comfy_nodes), "la 60 0.16\nla 62 0.16", "from_score", **{**COMMON, "max_new_patches": 500})
    assert json.loads(out.args[1])["patches_generated"] <= 8  # 0.32 s = 4 patches -> max 1.8x = 7 (rounded); never the widget's 500


def test_reference_audio_requires_its_transcript_and_vice_versa(comfy_nodes):
    n = comfy_nodes.NODE_CLASS_MAPPINGS
    handle = _handle(comfy_nodes)
    audio = {"waveform": torch.zeros(1, 1, 4410), "sample_rate": 44100}
    with pytest.raises(ValueError, match="needs its matching transcript"):
        n["SemBridgeTTS"].execute(handle, "Hello.", **COMMON, prompt_audio=audio, prompt_text="  ")
    with pytest.raises(ValueError, match="without prompt_audio"):
        n["SemBridgeTTS"].execute(handle, "Hello.", **COMMON, prompt_audio=None, prompt_text="words")


def test_bad_inputs_fail_before_any_model_work(comfy_nodes):
    n = comfy_nodes.NODE_CLASS_MAPPINGS  # ScoreError is a ValueError (pytest and ComfyUI load the package twice, so match the base class)
    handle = {"path": "does-not-exist", "dtype": "float32", "device": "cpu", "keep_loaded": True}
    with pytest.raises(ValueError, match="pitch must be an integer"):
        n["SemBridgeSinging"].execute(handle, "la sixty 0.5", "manual", **COMMON)
    with pytest.raises(ValueError, match="nothing to generate|empty|no notes"):
        n["SemBridgeUnified"].execute(handle, "[speech]", **COMMON)
    with pytest.raises(ValueError, match="empty"):
        n["SemBridgeTTS"].execute(handle, "   ", **COMMON)


def test_score_preview_needs_no_model(comfy_nodes):
    out = comfy_nodes.NODE_CLASS_MAPPINGS["SemBridgeScorePreview"].execute("la 60 0.5\n<SP> 0 0.4")
    notes_json, tokens, summary = out.args
    assert json.loads(notes_json)[1]["text"] == "<SP>"
    assert tokens == "<|note_pitch_60|><|note_dur_20|>la<|note_pitch_0|><|note_dur_16|><SP>"
    assert json.loads(summary)["score_seconds"] == 0.9


def test_keep_loaded_false_releases_after_the_job(comfy_nodes):
    from sembridge_comfy import runtime

    n = comfy_nodes.NODE_CLASS_MAPPINGS
    handle = {**_handle(comfy_nodes), "keep_loaded": False}
    n["SemBridgeTTS"].execute(handle, "Hello there.", **COMMON)
    assert runtime._SLOT is None
    n["SemBridgeUnload"].execute()  # safe when nothing is loaded
