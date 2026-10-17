"""ComfyUI nodes for SemBridge speech and singing generation (in-process; no subprocess, no download at run time)."""

from __future__ import annotations

import json

from comfy_api.latest import ComfyExtension, io

from . import paths, runtime, scores

SemBridgeModel = io.Custom("SEMBRIDGE_MODEL")

DECODER_HELP = (
    "auto = the official default (native causal decoder for speech, BigVGAN-Large for singing and mixed). "
    "native / bigvgan force one decoder for any task."
)
SEED_HELP = "Fixes the reference-audio sampling and the generation. Same model, request, seed, settings and environment give the same audio."
PATCH_HELP = "One patch is 80 ms of audio. max_new_patches=500 is a 40 s upper bound, not a target length: generation stops earlier when the model decides the text is finished."


def _progress(total: int):
    import comfy.model_management as mm
    import comfy.utils

    bar = comfy.utils.ProgressBar(max(1, int(total)))

    def tick():
        mm.throw_exception_if_processing_interrupted()
        bar.update(1)

    return tick


def _prompt(prompt_audio, prompt_text, prompt_task):
    """AUDIO + transcript -> (AudioPrompt | None, transcript | None, prompt_task | None)."""
    if prompt_audio is None:
        if prompt_text and prompt_text.strip():
            raise ValueError("prompt_text was given without prompt_audio")
        return None, None, None
    if not prompt_text or not prompt_text.strip():
        raise ValueError("prompt_audio needs its matching transcript in prompt_text (the model conditions on both)")
    wav = prompt_audio["waveform"]
    if wav.dim() != 3 or wav.shape[-1] < 1:
        raise ValueError("prompt_audio is empty or malformed")
    return (
        runtime.AudioPrompt(wav[0].detach().cpu(), int(prompt_audio["sample_rate"])),
        prompt_text.strip(),
        None if prompt_task == "auto" else prompt_task,
    )


def _sampling_inputs(min_default: int = 2):
    return [
        io.Int.Input("seed", default=42, min=0, max=0xFFFFFFFFFFFFFFFF, control_after_generate=True, tooltip=SEED_HELP),
        io.Combo.Input("decoder", options=list(runtime.DECODERS), default="auto", tooltip=DECODER_HELP),
        io.Int.Input("max_new_patches", default=500, min=1, max=8000, tooltip=PATCH_HELP),
        io.Int.Input("min_new_patches", default=min_default, min=0, max=8000, tooltip="Generation cannot stop before this many patches (80 ms each)."),
        io.Int.Input("flow_steps", default=10, min=2, max=100, tooltip="Flow-matching steps per patch (model default 10; the official singing examples use 20)."),
        io.Float.Input("cfg_value", default=2.0, min=0.0, max=10.0, step=0.1, tooltip="Classifier-free guidance (model default 2.0)."),
        io.Float.Input("temperature", default=1.0, min=0.0, max=2.0, step=0.05, tooltip="Sampling temperature of the flow noise (model default 1.0)."),
    ]


def _reference_inputs(task_default: str = "auto"):
    return [
        io.Audio.Input("prompt_audio", optional=True, tooltip="Optional reference recording of a voice you have the right to use. Only the first item of a batch is used."),
        io.String.Input("prompt_text", multiline=True, default="", optional=True,
                        tooltip="The exact transcript of prompt_audio. For a sung reference use the score token string (see SemBridge Score Preview)."),
        io.Combo.Input("prompt_task", options=["auto", "speech", "singing"], default=task_default, optional=True,
                       tooltip="What the reference is: spoken or sung. auto = singing for the Singing node, speech otherwise."),
    ]


def _run(model, progress_total, **kwargs) -> io.NodeOutput:
    loaded = runtime.acquire(model)
    info = loaded.info
    try:
        out = runtime.generate(loaded, progress=_progress(progress_total), numerics=model.get("codec_numerics", "reproducible"), **kwargs)
    finally:
        del loaded  # drop this reference first, otherwise release() cannot free the weights
        if not model.get("keep_loaded", True):
            runtime.release()
    return io.NodeOutput(out["audio"], runtime.report_json(info, out["report"]))


class SemBridgeLoader(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SemBridgeLoader",
            display_name="SemBridge Loader",
            category="loaders/sembridge",
            description=("Loads the official ASLP-lab/SemBridge folder from models/sembridge (generator, tokenizer and both "
                         "44.1 kHz codecs). One model stays resident; loading another frees it. The weights are not included "
                         "in this node pack."),
            inputs=[
                io.Combo.Input("model_dir", options=paths.model_dirs(), tooltip="A folder in models/sembridge holding the complete official SemBridge files"),
                io.Combo.Input("dtype", options=list(runtime.DTYPES), default="bfloat16", tooltip="bfloat16 is the official GPU setting. The CPU path always runs float32."),
                io.Combo.Input("device", options=["auto", "cuda", "cpu"], default="auto"),
                io.Boolean.Input("keep_loaded", default=True, tooltip="Keep the model in memory after the job. Turn off to free VRAM for other models after every generation."),
                io.Combo.Input("codec_numerics", options=list(runtime.CODEC_NUMERICS), default="reproducible", optional=True,
                               tooltip="reproducible: the same request gives the same audio on every run (plain PyTorch ops for the codec's Snake activation). "
                                       "official-torchscript: the official TorchScript function; its first call after loading rounds differently from later calls, "
                                       "so the first run can differ from the rest."),
            ],
            outputs=[SemBridgeModel.Output(display_name="model")],
        )

    @classmethod
    def execute(cls, model_dir, dtype, device, keep_loaded, codec_numerics="reproducible") -> io.NodeOutput:
        handle = runtime.make_handle(paths.model_path(model_dir), dtype, device, keep_loaded, codec_numerics)
        return io.NodeOutput(handle)


class SemBridgeTTS(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SemBridgeTTS",
            display_name="SemBridge Text to Speech",
            category="audio/sembridge",
            description="Text to speech (Chinese or English), optionally in the voice of a reference recording. Output: ComfyUI AUDIO, mono 44.1 kHz.",
            inputs=[
                SemBridgeModel.Input("model"),
                io.String.Input("text", multiline=True, default="Reading between the lines requires understanding."),
            ] + _sampling_inputs() + _reference_inputs("auto"),
            outputs=[io.Audio.Output(display_name="audio"), io.String.Output(display_name="report")],
        )

    @classmethod
    def execute(cls, model, text, seed, decoder, max_new_patches, min_new_patches, flow_steps, cfg_value, temperature,
                prompt_audio=None, prompt_text="", prompt_task="auto") -> io.NodeOutput:
        text = scores.clean_text(text, "text")
        prompt, ptext, ptask = _prompt(prompt_audio, prompt_text, prompt_task)
        return _run(model, max_new_patches, task="tts", decoder=decoder, text=text, prompt=prompt, prompt_text=ptext,
                    prompt_task=ptask, seed=seed, max_new_patches=max_new_patches, min_new_patches=min_new_patches,
                    flow_steps=flow_steps, cfg_value=cfg_value, temperature=temperature)


class SemBridgeSinging(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SemBridgeSinging",
            display_name="SemBridge Singing",
            category="audio/sembridge",
            description=("Singing voice synthesis from an aligned score: one note per line 'lyric pitch seconds [note_type]' "
                         "(pitch = MIDI number, 0 or <SP> = rest, note_type 3 = continue the previous lyric), or the official JSON. "
                         "This is not a MIDI importer: lyrics must already be aligned to notes."),
            inputs=[
                SemBridgeModel.Input("model"),
                io.String.Input("score", multiline=True, default="la 60 0.5\nla 62 0.5\nla 64 1.0\n<SP> 0 0.4",
                                tooltip="One note per line: lyric pitch duration_seconds [note_type]. Lines starting with # are ignored. "
                                "A JSON list of {text, pitch, duration, note_type} also works."),
                io.Combo.Input("length_control", options=["from_score", "manual"], default="from_score",
                               tooltip="from_score: min/max patches = 0.5x / 1.8x the score length, as in the official singing examples. manual: use the two values below."),
            ] + _sampling_inputs(min_default=2)[:2] + [
                io.Int.Input("max_new_patches", default=500, min=1, max=8000, tooltip="Used when length_control = manual. " + PATCH_HELP),
                io.Int.Input("min_new_patches", default=2, min=0, max=8000, tooltip="Used when length_control = manual."),
                io.Int.Input("flow_steps", default=20, min=2, max=100, tooltip="Flow-matching steps per patch. The official singing examples use 20."),
                io.Float.Input("cfg_value", default=2.0, min=0.0, max=10.0, step=0.1),
                io.Float.Input("temperature", default=1.0, min=0.0, max=2.0, step=0.05),
            ] + _reference_inputs("auto"),
            outputs=[io.Audio.Output(display_name="audio"), io.String.Output(display_name="report")],
        )

    @classmethod
    def execute(cls, model, score, length_control, seed, decoder, max_new_patches, min_new_patches, flow_steps, cfg_value,
                temperature, prompt_audio=None, prompt_text="", prompt_task="auto") -> io.NodeOutput:
        notes = scores.parse_notes(score, "score")
        if length_control == "from_score":
            min_new_patches, max_new_patches = scores.length_from_score(scores.score_seconds(notes))
        prompt, ptext, ptask = _prompt(prompt_audio, prompt_text, prompt_task)
        return _run(model, max_new_patches, task="svs", decoder=decoder, notes=notes, prompt=prompt, prompt_text=ptext,
                    prompt_task=ptask, seed=seed, max_new_patches=max_new_patches, min_new_patches=min_new_patches,
                    flow_steps=flow_steps, cfg_value=cfg_value, temperature=temperature)


class SemBridgeUnified(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SemBridgeUnified",
            display_name="SemBridge Speech + Singing",
            category="audio/sembridge",
            description=("Mixed speech and singing in one continuous voice. Sections: '[speech] words', '[singing]' followed by "
                         "score lines (lyric pitch seconds [note_type]); or the official JSON segment list."),
            inputs=[
                SemBridgeModel.Input("model"),
                io.String.Input("segments", multiline=True,
                                default="[speech]\nOnce upon a time, a singer walked into town.\n[singing]\nla 60 0.5\nla 62 0.5\nla 64 1.0\n[speech]\nAnd the crowd went quiet.",
                                tooltip="Ordered [speech] / [singing] sections, or the official JSON list of {task, text | notes}."),
            ] + _sampling_inputs(min_default=2)[:2] + [
                io.Int.Input("max_new_patches", default=500, min=1, max=8000, tooltip=PATCH_HELP),
                io.Int.Input("min_new_patches", default=2, min=0, max=8000),
                io.Int.Input("flow_steps", default=20, min=2, max=100, tooltip="Flow-matching steps per patch. The official mixed examples use 20."),
                io.Float.Input("cfg_value", default=2.0, min=0.0, max=10.0, step=0.1),
                io.Float.Input("temperature", default=1.0, min=0.0, max=2.0, step=0.05),
            ] + _reference_inputs("auto"),
            outputs=[io.Audio.Output(display_name="audio"), io.String.Output(display_name="report")],
        )

    @classmethod
    def execute(cls, model, segments, seed, decoder, max_new_patches, min_new_patches, flow_steps, cfg_value, temperature,
                prompt_audio=None, prompt_text="", prompt_task="auto") -> io.NodeOutput:
        parsed = scores.parse_segments(segments)
        prompt, ptext, ptask = _prompt(prompt_audio, prompt_text, prompt_task)
        return _run(model, max_new_patches, task="unified", decoder=decoder, segments=parsed, prompt=prompt, prompt_text=ptext,
                    prompt_task=ptask, seed=seed, max_new_patches=max_new_patches, min_new_patches=min_new_patches,
                    flow_steps=flow_steps, cfg_value=cfg_value, temperature=temperature)


class SemBridgeScorePreview(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SemBridgeScorePreview",
            display_name="SemBridge Score Preview",
            category="audio/sembridge",
            description=("Checks a singing score without loading any model: returns it as canonical JSON, its length in seconds, the patch "
                         "limits 'from_score' would use, and the exact token string the model receives (usable as prompt_text for a sung reference)."),
            inputs=[io.String.Input("score", multiline=True, default="la 60 0.5\nla 62 0.5\nla 64 1.0\n<SP> 0 0.4")],
            outputs=[io.String.Output(display_name="notes_json"), io.String.Output(display_name="token_string"), io.String.Output(display_name="summary")],
        )

    @classmethod
    def execute(cls, score) -> io.NodeOutput:
        from .vendor.sembridge.tokenizer import format_svs_notes

        notes = scores.parse_notes(score, "score")
        seconds = scores.score_seconds(notes)
        lo, hi = scores.length_from_score(seconds)
        summary = json.dumps({"notes": len(notes), "score_seconds": round(seconds, 3), "from_score_min_patches": lo, "from_score_max_patches": hi}, indent=1)
        return io.NodeOutput(json.dumps(notes, ensure_ascii=False, indent=1), format_svs_notes(notes), summary)


class SemBridgeUnload(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SemBridgeUnload",
            display_name="SemBridge Unload",
            category="loaders/sembridge",
            description="Frees the resident SemBridge model (GPU and RAM).",
            inputs=[io.Audio.Input("after", optional=True, tooltip="Optional: connect audio so this runs after it")],
            outputs=[io.String.Output(display_name="report")],
            is_output_node=True,
            not_idempotent=True,
        )

    @classmethod
    def execute(cls, after=None) -> io.NodeOutput:
        import torch

        was = runtime._SLOT is not None
        runtime.release()
        text = json.dumps({"unloaded": was, "cuda_allocated_mib": torch.cuda.memory_allocated() // 2**20 if torch.cuda.is_available() else None}, indent=1)
        return io.NodeOutput(text, ui={"text": [text]})


class SemBridgeExtension(ComfyExtension):
    async def get_node_list(self):
        return [SemBridgeLoader, SemBridgeTTS, SemBridgeSinging, SemBridgeUnified, SemBridgeScorePreview, SemBridgeUnload]


async def comfy_entrypoint() -> SemBridgeExtension:
    paths.register()
    return SemBridgeExtension()
