"""The runtime wrapper with a tiny random model on the CPU (float32): plumbing, determinism, hooks, prompts."""

import numpy as np
import pytest
import torch

from sembridge_comfy import runtime

BASE = dict(decoder="auto", seed=3, max_new_patches=6, min_new_patches=2, flow_steps=2, cfg_value=2.0, temperature=1.0)
NOTES = [{"text": "la", "pitch": 60, "duration": 0.5}, {"text": "la", "pitch": 62, "duration": 0.5, "note_type": 3}]


@pytest.fixture(scope="module")
def loaded(tiny_model):
    handle = runtime.make_handle(str(tiny_model), "bfloat16", "cpu", True)
    assert handle["dtype"] == "float32" and handle["device"] == "cpu"  # CPU forces float32 like the official loader demands
    return runtime.acquire(handle)


def test_load_reports_both_decoders_and_44k(loaded):
    assert loaded.info["decoders"] == ["bigvgan", "native"]
    assert loaded.info["sample_rate"] == 44100
    assert runtime.acquire(runtime.make_handle(str(loaded.pipeline.model_root), "float32", "cpu", True)) is loaded  # cached


def test_three_modes_produce_finite_mono_44k_audio(loaded):
    jobs = {
        "tts": dict(task="tts", text="Hello there."),
        "svs": dict(task="svs", notes=NOTES),
        "unified": dict(task="unified", segments=[{"task": "speech", "text": "Hi."}, {"task": "singing", "notes": NOTES}]),
    }
    decoders = {}
    for name, job in jobs.items():
        out = runtime.generate(loaded, **BASE, **job)
        wav = out["audio"]["waveform"]
        assert wav.shape[0] == 1 and wav.shape[1] == 1 and wav.shape[2] > 0
        assert out["audio"]["sample_rate"] == 44100
        assert torch.isfinite(wav).all() and wav.abs().max() <= 1.0
        assert 1 <= out["report"]["patches_generated"] <= 6
        decoders[name] = out["report"]["decoder"]
    assert decoders == {"tts": "causal", "svs": "bigvgan", "unified": "bigvgan"}  # the official auto rule


def test_decoder_can_be_forced_for_any_task(loaded):
    for decoder, expected in (("native", "causal"), ("bigvgan", "bigvgan")):
        out = runtime.generate(loaded, **{**BASE, "decoder": decoder}, task="tts", text="Hello.")
        assert out["report"]["decoder"] == expected


def test_same_seed_same_audio_different_seed_differs(loaded):
    a = runtime.generate(loaded, **BASE, task="tts", text="Hello there.")["audio"]["waveform"]
    b = runtime.generate(loaded, **BASE, task="tts", text="Hello there.")["audio"]["waveform"]
    c = runtime.generate(loaded, **{**BASE, "seed": 4}, task="tts", text="Hello there.")["audio"]["waveform"]
    assert torch.equal(a, b)
    assert a.shape != c.shape or not torch.equal(a, c)


def test_seed_does_not_leak_into_the_global_rng(loaded):
    torch.manual_seed(1234)
    expected = torch.rand(3)
    torch.manual_seed(1234)
    runtime.generate(loaded, **BASE, task="tts", text="Hello there.")
    assert torch.equal(torch.rand(3), expected)


def test_progress_hook_ticks_once_per_patch_and_is_removed(loaded):
    ticks = []
    out = runtime.generate(loaded, **BASE, task="tts", text="Hello there.", progress=lambda: ticks.append(1))
    assert len(ticks) == out["report"]["patches_generated"]
    assert not loaded.pipeline.model.stop_head._forward_hooks


def test_interrupt_from_the_hook_propagates_and_cleans_up(loaded):
    class Stop(Exception):
        pass

    def boom():
        raise Stop

    with pytest.raises(Stop):
        runtime.generate(loaded, **BASE, task="tts", text="Hello there.", progress=boom)
    assert not loaded.pipeline.model.stop_head._forward_hooks
    runtime.generate(loaded, **BASE, task="tts", text="Hello there.")  # still usable afterwards


def _reference():
    sr = 24000
    t = torch.arange(0, int(sr * 0.5)) / sr
    return runtime.AudioPrompt((0.1 * torch.sin(2 * torch.pi * 220 * t)).unsqueeze(0).repeat(2, 1), sr)  # stereo, 24 kHz


def test_reference_audio_changes_the_result_and_needs_matching_options(loaded, comfy_dir):
    plain = runtime.generate(loaded, **BASE, task="tts", text="Hello there.")["audio"]["waveform"]
    with_ref = runtime.generate(
        loaded, **BASE, task="tts", text="Hello there.", prompt=_reference(), prompt_text="la la", prompt_task="speech"
    )
    assert with_ref["report"]["reference_audio"] is True
    assert torch.isfinite(with_ref["audio"]["waveform"]).all()
    assert plain.shape != with_ref["audio"]["waveform"].shape or not torch.equal(plain, with_ref["audio"]["waveform"])
    with pytest.raises(ValueError, match="matching prompt_text"):
        runtime.generate(loaded, **BASE, task="tts", text="Hello there.", prompt=_reference())


def test_prompt_encoding_equals_the_official_steps(loaded, comfy_dir, torchaudio_mod):
    """The in-memory path must equal the official _load_prompt steps spelled out with torchaudio's resampler."""
    torchaudio = torchaudio_mod
    pipe = loaded.pipeline
    prompt = _reference()
    audio = prompt.waveform.mean(dim=0, keepdim=True)
    audio = torchaudio.functional.resample(audio, prompt.sample_rate, pipe.codec.sample_rate)
    patch_samples = pipe.codec.hop_length * pipe.model.config.patch_size
    if audio.size(-1) % patch_samples:
        audio = torch.nn.functional.pad(audio, (patch_samples - audio.size(-1) % patch_samples, 0))
    torch.manual_seed(5)
    with torch.inference_mode():
        expected = pipe.codec.patch(pipe.codec.encode(audio.float(), pipe.codec.sample_rate)).to(pipe.dtype)
    torch.manual_seed(5)
    got = pipe._load_prompt(prompt)
    assert got.shape == expected.shape
    np.testing.assert_allclose(got.numpy(), expected.numpy(), rtol=0, atol=1e-6)


def test_comfy_resample_equals_torchaudio(comfy_dir, torchaudio_mod):
    torchaudio = torchaudio_mod
    from comfy.audio import resample

    x = torch.randn(1, 24000 * 2)
    for src in (16000, 24000, 48000, 22050):
        np.testing.assert_allclose(resample(x, src, 44100).numpy(), torchaudio.functional.resample(x, src, 44100).numpy(), rtol=0, atol=1e-6)


def test_release_frees_the_slot(tiny_model):
    handle = runtime.make_handle(str(tiny_model), "float32", "cpu", False)
    first = runtime.acquire(handle)
    runtime.release()
    assert runtime._SLOT is None
    assert runtime.acquire(handle) is not first
    runtime.release()
