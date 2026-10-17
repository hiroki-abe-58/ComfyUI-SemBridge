# SemBridge for ComfyUI

Speech, singing and mixed speech + singing generation in ComfyUI with the official **SemBridge** model
([ASLP-lab/SemBridge](https://github.com/ASLP-lab/SemBridge), paper [arXiv:2608.07462](https://arxiv.org/abs/2608.07462)).
The model code runs **in-process** (no subprocess, no server, no download at run time) and returns ComfyUI's standard `AUDIO`
type: mono, 44.1 kHz.

> **Unofficial.** This node pack is an independent integration. It is not affiliated with or endorsed by the SemBridge authors
> (ASLP-lab, Northwestern Polytechnical University) or by Comfy Org.
>
> 日本語の概要は [README.ja.md](README.ja.md) にあります。

## What it does, and how far it was checked

| Node | Purpose |
| --- | --- |
| **SemBridge Loader** | Picks a model folder from `models/sembridge`, dtype (bfloat16 / float16 / float32), device, keep-loaded switch, `codec_numerics` (see *Reproducibility*) |
| **SemBridge Text to Speech** | Chinese or English text → speech, optionally in the voice of a reference recording |
| **SemBridge Singing** | An aligned score (lyric, MIDI pitch, seconds per note) → singing voice |
| **SemBridge Speech + Singing** | Ordered speech and singing sections → one continuous voice |
| **SemBridge Score Preview** | Validates a score without loading the model; shows the exact token string the model receives |
| **SemBridge Unload** | Frees the resident model (GPU and RAM) |

Checked on Windows 11 with an RTX 5090 (32 GB), details and raw numbers in [docs/VERIFICATION.md](docs/VERIFICATION.md):

- **Identical to the official code.** For all 10 prepared examples of the official repository (4 speech, 4 singing, 2 mixed),
  the audio from these nodes equals the audio from the official `SemBridge.generate` sample for sample (16-bit PCM identical),
  when both run in the same Python environment: 10/10 with torch 2.8.0 (the range the official package supports) and
  10/10 with torch 2.14.1 + transformers 5.20 (what a current ComfyUI installs). The official command-line script, run
  separately, produced the same files as the official API. This holds for both `codec_numerics` settings of the Loader
  (see *Reproducibility* below): `official-torchscript` against the unmodified official code, `reproducible` (the default)
  against the official code with the one function swapped.
- **Runs through the real ComfyUI queue.** All 10 examples, executed as API workflows with `POST /prompt` in ComfyUI v0.38.0,
  succeeded. Interrupt, error messages, unload, all three dtypes and the CPU device were exercised with the real weights.
- **Singing follows the score.** On the demo clip below, the median pitch error against the written notes is 0.15 semitone
  (7 of 7 notes within one semitone, no octave offset); spoken demos are recognized by Whisper (see the verification page).
- CI (GitHub Actions, Ubuntu and Windows, torch 2.8 and 2.14) runs the CPU tests and a real ComfyUI queue with a tiny
  random-initialized model; it does not use the trained weights.

**Not checked:** Linux or macOS with a GPU, AMD/Apple GPUs, fp8 or other quantization, multi-GPU, languages other than Chinese
and English, and inputs beyond the 40 s default cap (see *Limits* below).

## Install

1. Needs ComfyUI with the V3 node API (**v0.38.0** was used), Python 3.10+, and, for reasonable speed, an NVIDIA GPU.
2. Put this folder into `ComfyUI/custom_nodes/` (or install it from the ComfyUI Manager / Comfy Registry once listed) and install
   its one extra requirement into ComfyUI's Python:

   ```
   python -m pip install -r custom_nodes/ComfyUI-SemBridge/requirements.txt
   ```

   That is `soundfile` only. Do **not** `pip install sembridge`: its metadata pins torch below 2.9, and the model code is already
   included here (`sembridge_comfy/vendor`, unmodified, see [NOTICE](NOTICE)).
3. Download the official model folder (about 4.3 GB, Apache-2.0 per its model card) into `models/sembridge/`:

   ```
   python custom_nodes/ComfyUI-SemBridge/tools/download_model.py --models-dir ComfyUI/models/sembridge
   ```

   The script uses the revision this pack was verified with and checks the SHA-256 of the three weight files. You can also
   download [ASLP-lab/SemBridge](https://huggingface.co/ASLP-lab/SemBridge) yourself; the folder must contain `config.json`,
   `model.safetensors`, `audiovae.safetensors`, `unified_audiovae.safetensors`, `savae_config.json`,
   `unified_savae_config.json`, `tokenizer.json`, `tokenizer_config.json`, `special_tokens_map.json`, `generation_config.json`.
   An `extra_model_paths.yaml` entry named `sembridge` works as well.
4. Restart ComfyUI. Load a workflow from `workflows/gui/` (or run one from `workflows/api/`).

Nothing is downloaded or installed by the nodes themselves.

## Using the nodes

**Text to Speech.** Type text, choose a seed, run. Without reference audio the model picks a voice of its own (one voice per
seed). With **prompt_audio** (a `Load Audio` node) you must also give **prompt_text**, the exact transcript of that recording;
the model conditions on both. Only use recordings you have the right to use.

**Singing.** The score is one note per line:

```
lyric  pitch  seconds  [note_type]
twin   60     0.4
kle    60     0.4
star   67     0.9
<SP>   0      0.3        # a rest is "<SP>" with pitch 0
```

`pitch` is a MIDI note number (60 = middle C), `seconds` the note length (the model quantizes to 25 ms), and `note_type` 3 means
"this note continues the previous lyric on another pitch" (what the official examples use for melisma). The official JSON note
list works too, including a whole official request object. **This is not a MIDI importer:** the model needs lyrics already
aligned to notes; a plain MIDI file does not contain that. With `length_control = from_score` the generation limits are set
from the score length (0.5x and 1.8x, the rule the four official singing examples follow). Use **Score Preview** to check a score first.

**Speech + Singing.** Sections, in order:

```
[speech]
Once upon a time, a singer walked into a quiet town.
[singing]
twin 60 0.4
kle 60 0.4
[speech]
And everyone stopped to listen.
```

The official JSON segment list is accepted as well.

**Settings** (defaults are the model's own): `decoder` auto picks the native causal decoder for speech and BigVGAN-Large for
singing / mixed (force either one for any task if you want to compare), `flow_steps` 10 (20 for singing / mixed, as in the official
examples), `cfg_value` 2.0, `temperature` 1.0. `max_new_patches` caps the length (one patch = 80 ms, 500 = 40 s); generation
usually stops earlier by itself. The **report** output is JSON with the decoder, patch count, audio seconds, generation time and
peak VRAM of that run.

## Measured on this setup

RTX 5090 32 GB, Windows 11, ComfyUI v0.38.0, torch 2.14.1+cu130, bfloat16 (the official examples with their reference voices):

| | Speech (native decoder) | Singing / mixed (BigVGAN-Large) |
| --- | --- | --- |
| Generation time per second of audio | about 1.0 to 1.6 s | about 2.2 to 2.7 s |
| Peak VRAM allocated by the model | 2.7 to 3.0 GiB | 2.9 to 3.9 GiB |

So generation is slower than real time on this GPU (the autoregressive loop is small-kernel bound). A 29 s utterance needed 4.3 GiB.
Loading takes about 6 s with a warm file cache. float32 needs 4.5 GiB, float16 and bfloat16 about 2.9 GiB. System commit charge rose by
at most about 9 GiB during a test sequence that included a float32 load (24.7 to 33.4 GiB on a 64 GB machine); the full table is in the
verification page.

## Limits and things to know

- **Reproducibility.** The same model, request, seed, settings *and environment* give the same audio. Across torch versions or GPUs
  the numbers differ slightly and the autoregressive stop point can move by a few patches (the official README says the same).
  **First run after loading.** The official codec computes its Snake activation in a `@torch.jit.script` function, and TorchScript
  re-optimizes such a function after its first calls: with the official code, the first generation after loading rounds differently from
  every later one (the same request and seed gave 69 patches on the first call and 67 afterwards on torch 2.14.1; the reference-audio
  encoder is where it enters, and the autoregressive loop amplifies it; torch 2.8.0 shows the same effect). The Loader's
  `codec_numerics = reproducible` (default) evaluates the same formula with ordinary PyTorch ops, so every call gives the same audio
  (checked: 4 calls identical on torch 2.8.0 and 2.14.1; cost about 10 ms per 16 s of decoded audio). `official-torchscript` keeps the
  official function, e.g. to match the output of the official command line exactly.
- **Long input.** For a 29 s test utterance (three long sentences) Whisper's transcript lacked the last six words; the model had
  stopped by itself (368 patches, below the 500 cap). Split very long text into sentences.
- **Loudness varies** between outputs (official examples range from -12 to -31 dBFS RMS). Normalize downstream if you need it.
- **Leading whitespace** of texts is removed, because transformers 4.x and 5.x tokenize a leading space differently (all other
  strings tokenized identically in a 412-string check).
- **Languages:** the model card lists Chinese and English.
- **Speed:** not real time on this GPU, see above. CPU works (float32) but is much slower.
- Synthetic voices: the model card asks users to use reference recordings only with permission and to disclose synthetic audio.
  Please do.

## Licenses

- This node pack's code: **MIT** ([LICENSE](LICENSE)).
- `sembridge_comfy/vendor/sembridge`: unmodified copy of the official package, **Apache-2.0** with its third-party notices
  (VoxCPM / OpenBMB Apache-2.0, BigVGAN MIT, alias-free-torch Apache-2.0, Descript Audio Codec MIT); see [NOTICE](NOTICE) and
  `sembridge_comfy/vendor/LICENSE`.
- Model weights are **not included**. The model card of ASLP-lab/SemBridge states Apache-2.0; the official LICENSE adds that
  external assets keep their own terms (the GLM-4-Voice tokenizer used in training is not needed or included).
- The reference recordings of the official examples are **not included** here and are not covered by any of these licenses.
  The demo clips in `docs/audio/` were generated by this pack with the model alone, with **no reference recording**.
- ComfyUI (GPL-3.0) hosts the nodes; PyTorch, transformers, SoundFile and NumPy keep their own licenses.

## Development

```
pip install pytest ruff soundfile                      # plus ComfyUI's requirements
SEMBRIDGE_COMFYUI_DIR=... SEMBRIDGE_UPSTREAM_DIR=... SEMBRIDGE_TOKENIZER_DIR=... python -m pytest tests
python scripts/ci_queue_check.py --comfyui <ComfyUI> --tokenizer-dir <tokenizer files>   # real queue, tiny model, CPU
python scripts/parity_vs_official.py --upstream <SemBridge checkout> --model <model folder> --out <dir>   # needs the real weights
```

`tools/queue_client.py` submits any API workflow to a running ComfyUI. `scripts/queue_examples.py` runs the official examples through
a live queue, `scripts/eval_pitch.py` checks a sung result against its score. Refreshing the vendored code: replace
`sembridge_comfy/vendor/sembridge` with the new upstream files, run `scripts/update_vendor_manifest.py`, update `UPSTREAM.md` and
the commit in `.github/workflows/ci.yml`.

## Citation

If you use SemBridge, please cite the authors:

```bibtex
@article{xie2026sembridge,
  title = {SemBridge: Semantic Token Anchoring for Continuous-Latent Autoregressive Speech Generation},
  author = {Hanke Xie and Haopeng Lin and Jiale Qian and Dake Guo and Yuepeng Jiang and Zhichao Wang and Wenxiao Cao and Jingbin Hu and Guobin Ma and Wenhao Li and Huakang Chen and Chengyou Wang and Ming Tao and Zhonghua Fu and Lei Xie and Xinsheng Wang},
  journal = {arXiv preprint arXiv:2608.07462},
  year = {2026}
}
```
