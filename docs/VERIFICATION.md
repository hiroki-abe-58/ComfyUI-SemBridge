# Verification record

What was run, on what, and what came out. Everything here was run on 2026-10-17 on one machine; numbers are single runs unless
stated. Raw per-example results are reproducible with the commands at the end.

## Environment

| | |
| --- | --- |
| OS / GPU | Windows 11 Pro (10.0.26200), NVIDIA GeForce RTX 5090 32 GB (compute capability 12.0), driver 610.88, 64 GB RAM |
| ComfyUI | v0.38.0 (`6b747c0428c343e1417219641db93a4fb7cb69ae`), Python 3.12.13, own virtual environment, 127.0.0.1 only |
| ComfyUI environment | torch 2.14.1+cu130, transformers 5.20.1, tokenizers 0.23.3, soundfile 0.14.0 |
| Official-reference environment | torch 2.8.0+cu128, transformers 4.57.6, safetensors 0.8.0 (the range the official package pins: torch 2.6 to 2.8, transformers below 5) |
| Upstream code | ASLP-lab/SemBridge `e8bee2777a0dececf87efb6705992c974b51c9b4` (vendored unmodified, checked by `tests/test_vendor.py`) |
| Model | ASLP-lab/SemBridge revision `4fb6534f7c697f0aea90e71dc5a6f3123596ae45` (model card: Apache-2.0) |

Weight file SHA-256 (equal to the values published by the Hub):

```
f157386839d5bda47e533a85d34bb0803cad5ab455f73cfdfc722a530379f7db  model.safetensors             3,270,433,760 bytes
44c0d88956f8060745274f0048cebc1c0aa4ec1dd1c1573ecc584e014dab71ef  audiovae.safetensors            345,969,784 bytes
c059ae970319b83284d5d5f4aca8fab2f2af97082072534c54f428dec54d336e  unified_audiovae.safetensors    654,223,052 bytes
```

## 1. Official reference on this machine

`scripts/infer_examples.py` of the official repository, bfloat16 on the RTX 5090, torch 2.8.0: all 10 prepared examples
(`tts_zh_r001/r002`, `tts_en_r001/r002`, `svs_zh_r001/r002`, `svs_en_r001/r002`, `unified_zh_robust_caihong`,
`unified_zh_novel_wukong`) were generated. One process for all ten: 166 s wall including model load; GPU memory in use across the
whole system peaked at 7.6 GiB (this includes the desktop and other applications, not only the model).

## 2. These nodes against the official code, sample for sample

`scripts/parity_vs_official.py` loads the official pipeline and this pack's runtime from the same files, runs every example with
identical fields through both, and compares the float waveforms and the 16-bit PCM the official WAV writer would store.
The Loader's `codec_numerics` (section 3) has two settings, so both are compared: `official-torchscript` against the unmodified
official code, `reproducible` against the official code with only the codec's Snake function replaced by the same formula in plain ops.

| Environment | `codec_numerics` | Examples | Identical PCM16 | Identical float |
| --- | --- | --- | --- | --- |
| torch 2.8.0+cu128, transformers 4.57.6 (official reader, torchaudio) | official-torchscript | 10 | **10** | 10 |
| same | reproducible | 10 | **10** | 10 |
| torch 2.14.1+cu130, transformers 5.20.1 (ComfyUI environment, `--torchaudio-shim`) | official-torchscript | 10 | **10** | 10 |
| same | reproducible | 10 | **10** | 10 |

`--torchaudio-shim` replaces the official reader by soundfile + `comfy.audio.resample` because a stock ComfyUI has no torchaudio. An earlier
run on torch 2.8 with the shim reproduced the real torchaudio path on all ten recordings (24, 44.1 and 48 kHz), and the
`official-torchscript` torch 2.8 files are byte for byte the files the official command-line script wrote in a separate process
(section 1). `tests/test_runtime_cpu.py` additionally compares `comfy.audio.resample` with `torchaudio.functional.resample` at 16, 22.05,
24 and 48 kHz (max difference below 1e-6) and the prompt encoding with the official steps (CI legs with torch 2.8).

## 3. A reproducibility finding in the official code: the first call after loading

While comparing queue runs with standalone runs, the same request and seed gave different audio depending on whether it was the first
generation after loading. This happens with the official package alone (no node code):

| Official `SemBridge.generate`, request `tts_zh_r001`, seed 42, 4 calls in one process | call 1 | calls 2 to 4 |
| --- | --- | --- |
| torch 2.8.0+cu128 | 69 patches, sha256 `e6caa13d3ce3` | 69 patches, sha256 `d63c4d2adad3` (all three equal) |
| torch 2.14.1+cu130 | 69 patches, sha256 `04059a1e4487` | 67 patches, sha256 `bfd58dbe55dd` (all three equal) |

Bisection on the official classes (torch 2.14.1): the reference-audio latents (`_load_prompt`) differ between call 1 and calls 2 and 3 with the same seed
(`4a7d814884` vs `9b237ae6d5`), and so do the first generated patches. The codec computes its Snake activation in a
`@torch.jit.script` function (`sembridge/models/savae.py`); with that single function replaced by the same formula in plain ops, all calls
give identical latents and audio. TorchScript's profiling executor re-optimizes a scripted function after its first calls; the
optimized version gives slightly different float32 results (I did not look inside the executor; the evidence is the on/off experiment).
The effect is amplified by the autoregressive loop and moves the stop point.

The Loader's `codec_numerics` setting exposes both behaviors:

| `codec_numerics` | 4 identical calls in a fresh process | Notes |
| --- | --- | --- |
| `official-torchscript` | no: call 1 differs (torch 2.8.0 and 2.14.1) | exactly the official function |
| `reproducible` (default) | **yes** (torch 2.8.0: 62 patches, `3aa3d4078fa7` four times; the official code with the swap, run in a second process, gives the same hash; torch 2.14.1: 69 patches, `6c5a926a815f` four times) | same formula, plain ops |

(`scripts/check_call_history.py`.) Cost of the plain-ops version, measured on the codec alone (RTX 5090, torch 2.8.0, bfloat16 model, 16 s decode /
5 s encode): decode 37.5 to 47.0 ms, encode 9.6 to 11.1 ms, peak memory unchanged; generation of the same audio takes 20 to 40 s.
The two settings produce different (equally valid) audio for the same seed, like two different torch versions do. Whisper-small
transcripts of the four official TTS examples generated with each variant had the same error rate (0.04 for `tts_zh_r001`, 0 for the other three; Chinese
compared after traditional-to-simplified conversion).

Through the queue (clean install of this pack in a fresh ComfyUI, default `reproducible`), the 10 examples were run in the official order and
again in reverse order: the decoded FLAC audio of the two runs is identical for 10 of 10 examples, and every file is within 1 LSB of the
in-environment official output (section 2, torch 2.14, same setting) for 10 of 10 (the FLAC writer's rounding).

## 4. Through the real ComfyUI queue

`scripts/queue_examples.py` turns each official request into an API workflow (Load Audio, Loader, node, Save Audio) and posts it
to the running server (`POST /prompt`). The server was a clean install: fresh ComfyUI v0.38.0 checkout and virtual environment, this
repository cloned into `custom_nodes`, `pip install -r requirements.txt`, the model fetched with `tools/download_model.py` (hashes
checked). All 10 examples succeeded, in both orders.

Per example, default settings, bfloat16 (generation seconds are the node's own measurement, loading excluded):

| Example | Decoder | Patches | Audio s | Generation s | Peak VRAM (GiB, torch allocated) |
| --- | --- | --- | --- | --- | --- |
| tts_zh_r001 | native | 69 | 5.52 | 8.60 | 2.99 |
| tts_zh_r002 | native | 79 | 6.32 | 8.71 | 2.88 |
| tts_en_r001 | native | 52 | 4.16 | 6.12 | 2.80 |
| tts_en_r002 | native | 54 | 4.32 | 5.20 | 2.83 |
| svs_zh_r001 | BigVGAN | 82 | 6.56 | 16.98 | 3.06 |
| svs_zh_r002 | BigVGAN | 54 | 4.32 | 11.90 | 2.86 |
| svs_en_r001 | BigVGAN | 193 | 15.44 | 36.82 | 3.83 |
| svs_en_r002 | BigVGAN | 97 | 7.76 | 18.11 | 3.16 |
| unified_zh_robust_caihong | BigVGAN | 187 | 14.96 | 38.24 | 3.78 |
| unified_zh_novel_wukong | BigVGAN | 205 | 16.40 | 37.01 | 3.91 |

Generation takes 1.2 to 1.6 s per second of audio for speech and 2.3 to 2.8 s for singing and mixed. Model load: 6.0 s (warm file cache),
2.5 GiB allocated after loading in bfloat16.

Queue output against the torch 2.8 official files is not equal (different torch, transformers and CUDA builds; the stop point of an
autoregressive model with sampling moves). That is the environment, not the nodes: with identical environments the result is bit-identical
(section 2).

## 5. Behavior with the real weights

| Check | Result |
| --- | --- |
| Interrupt a long generation (`POST /interrupt` 4 s in) | status `interrupted`, stopped about 2 s later; the next job ran normally (45 patches) |
| Reference audio without transcript | error: "prompt_audio needs its matching transcript in prompt_text ..." |
| Unparsable score `la sixty 0.5` | error: "score line 1: pitch must be an integer and duration a number of seconds" |
| Blank text | error: "text is empty" |
| Model folder that does not exist | rejected by ComfyUI's validation before queuing |
| dtype float32 / float16 / bfloat16 on CUDA | all succeed; peak allocated 4.53 / 2.85 / 2.87 GiB |
| `keep_loaded = false` | GPU memory in use fell from 4769 MiB to 2250 MiB after the job (server baseline 2241 MiB) |
| SemBridge Unload node | 4769 MiB to 2241 MiB |
| CPU device, float32, real weights (40 patch cap) | succeeded: 1.6 s of audio in 8.8 s |
| 29 s utterance (3 long sentences, up to 500 patches) | 368 patches, 29.4 s audio in 39.1 s, 4.33 GiB allocated; the model stopped before the cap and Whisper's transcript lacked the last six words |
| System commit charge during the sequence above | 24.7 to 33.4 GiB (limit 111.4 GiB); lowest free RAM 43 GiB; GPU in use peaked at 6.5 GiB with the float32 run |

(The interrupt, dtype, release and long-utterance rows were measured before `codec_numerics` was added; the option only changes which
Snake function the codec calls.)

## 6. Quality checks of the demo clips (`docs/audio/`)

Generated through the queue on the clean install, no reference audio, seed 7, default settings (`reproducible`):

| Clip | How it was made | Check |
| --- | --- | --- |
| `demo_tts_en.flac` (4.7 s) | `workflows/api/tts_basic.json`, text "Welcome to ComfyUI. This voice was generated locally with SemBridge, on a single graphics card." | Whisper small heard: "Welcome to Confia. This voice was generated locally with Sembridge on a single graphics card." (word error rate 0.07: one word, "ComfyUI" heard as "Confia") |
| `demo_tts_zh.flac` (3.5 s) | text 今天天气很好，我们一起去公园散步吧。 | Whisper small heard 今天天氣很好,我們一起去公園散步吧! (all characters; traditional forms, error rate 0 after conversion to simplified) |
| `demo_singing_en.flac` (3.6 s) | `workflows/api/singing_basic.json` with a seven-note "twin-kle" melody (see below) | `scripts/eval_pitch.py`: 7 of 7 notes with a clear pitch, median error 0.15 semitone, all within one semitone, octave offset 0 |
| `demo_unified_en.flac` (8.0 s) | `workflows/api/unified_basic.json`, speech + the same melody + speech | Whisper heard both spoken parts exactly; the sung syllables as "Twin kill, twin kill, lead to the star" |

Score of the singing demo: `twin 60 0.4`, `kle 60 0.4`, `twin 67 0.4`, `kle 67 0.4`, `lit 69 0.4`, `tle 69 0.4`, `star 67 0.9`, `<SP> 0 0.3`.
The pitch checker itself was validated on a synthetic sine melody (median error 0.06 semitone) and on the official example
`svs_en_r002` (median 0.54 semitone, 85 % within one semitone), so 0.15 is not a ceiling of the method. Whisper is a rough proxy for
intelligibility, not a listening test; the clips have **not** been judged by a human listener (HUMAN_AUDIO_REVIEW_PENDING).

The default level of the BigVGAN outputs is low (the unified demo is about -30 dBFS RMS); the official examples range from -12 to
-31 dBFS.

## 7. Tests

| Where | What | Result |
| --- | --- | --- |
| local, torch 2.14.1 CPU | ruff, `pytest tests` (scores, vendored-code manifest + byte comparison with the pinned upstream commit, runtime on a tiny random model, node registration, ComfyUI validation of the shipped workflows, node execution, numerics switch) | 55 passed, 2 skipped (torchaudio does not exist for torch 2.14) |
| local, torch 2.8.0 CPU + torchaudio 2.8.0 | same, torchaudio tests required | 57 passed |
| local, `scripts/ci_queue_check.py` | a second, fresh ComfyUI checkout + the node pack copied in, tiny model, `--cpu`: object_info, then score preview, TTS, singing, unified workflows through `POST /prompt` | passed |
| ComfyUI frontend | `scripts/export_gui_workflows.py` loads each API workflow with the frontend's `loadApiJson`: no missing node types, `graphToPrompt` returns the same classes and values, no page errors | 4 of 4 |
| Comfy Registry tooling | `comfy node validate` (comfy-cli 1.22.0) and `comfy node pack` | passed |

GitHub Actions results are linked from the repository's release notes.

## 8. Not verified

Linux or macOS with a GPU; AMD, Intel or Apple GPUs; torch builds other than 2.8.0+cu128 and 2.14.1+cu130 with the real weights;
other ComfyUI versions; quantized or offloaded loading; multi-GPU; Chinese-English code-switching; texts or scores beyond about 40 s;
listening tests by people; any comparison of quality with other speech or singing systems.

## Commands

```
python tools/download_model.py --models-dir <ComfyUI>/models/sembridge
python scripts/parity_vs_official.py --upstream <SemBridge checkout> --model <model folder> --out <dir> --codec-numerics reproducible          # section 2
python scripts/parity_vs_official.py ... --codec-numerics official-torchscript --torchaudio-shim <ComfyUI>                                      # stock ComfyUI environment
python scripts/check_call_history.py --upstream <SemBridge checkout> --model <model folder> --codec-numerics reproducible --comfyui <ComfyUI>   # section 3
python scripts/queue_examples.py --server http://127.0.0.1:8188 --upstream <SemBridge checkout> --comfy-input <ComfyUI>/input --out <dir> [--reverse]
python tools/queue_client.py workflows/api/singing_basic.json --server http://127.0.0.1:8188 --set 2.seed=7
python scripts/eval_pitch.py <clip.flac> --score-file <score.txt>
python scripts/ci_queue_check.py --comfyui <ComfyUI> --tokenizer-dir <tokenizer files>
```
