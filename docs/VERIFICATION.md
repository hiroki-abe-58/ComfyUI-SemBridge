# Verification record

What was run, on what, and what came out. Everything here was run on 2026-10-17 on one machine; numbers are single runs unless
stated. Raw per-example results are reproducible with the commands below.

## Environment

| | |
| --- | --- |
| OS / GPU | Windows 11 Pro (10.0.26200), NVIDIA GeForce RTX 5090 32 GB (compute capability 12.0), driver 610.88, 64 GB RAM |
| ComfyUI | v0.38.0 (`6b747c0428c343e1417219641db93a4fb7cb69ae`), Python 3.12.13, own virtual environment, port 8197 on 127.0.0.1 |
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

| Environment | Examples | Identical PCM16 | Identical float |
| --- | --- | --- | --- |
| torch 2.8.0+cu128, transformers 4.57.6 (official reader, torchaudio) | 10 | **10** | 10 |
| same, official reader replaced by the soundfile + `comfy.audio.resample` stand-in (`--torchaudio-shim`) | 10 | **10** | 10 |
| torch 2.14.1+cu130, transformers 5.20.1 (ComfyUI environment, stand-in reader) | 10 | **10** | 10 |

The second row shows the stand-in reader equals the real torchaudio path on all ten recordings (24, 44.1 and 48 kHz); the
first row's files are also byte-for-byte the files the official command-line script wrote in a separate process (section 1).
`tests/test_runtime_cpu.py` additionally compares `comfy.audio.resample` with `torchaudio.functional.resample` at 16, 22.05, 24 and
48 kHz (max difference below 1e-6) and the prompt encoding with the official steps (CI legs with torch 2.8).

## 3. Through the real ComfyUI queue

`scripts/queue_examples.py` turns each official request into an API workflow (Load Audio, Loader, node, Save Audio) and posts it
to the running server. All 10 succeeded in both server configurations:

| Server | Result against the in-environment official output (section 2, torch 2.14) |
| --- | --- |
| default (DynamicVRAM / comfy-aimdo on) | 9 of 10 equal in length and within 1 LSB (the FLAC writer); `tts_zh_r001` stopped at 67 patches instead of 69 |
| `--disable-dynamic-vram` | **10 of 10** equal in length and within 1 LSB |

Against the torch 2.8 official files the queue outputs are not equal (different torch, transformers and CUDA builds; the stop
point of an autoregressive model with sampling moves): 7 of 10 have the same length, and for those the waveform correlation is between 0.25 and 0.99. That is
the environment, not the nodes: with identical environments the result is bit-identical (section 2). The same request repeated in
the same server gave identical audio (4 runs alternating two seeds).

Per example, default server, bfloat16 (generation seconds are the node's own measurement, loading excluded):

| Example | Decoder | Patches | Audio s | Generation s | Peak VRAM (GiB, torch allocated) |
| --- | --- | --- | --- | --- | --- |
| tts_zh_r001 | native | 67 | 5.36 | 8.10 | 2.82 |
| tts_zh_r002 | native | 79 | 6.32 | 8.87 | 2.88 |
| tts_en_r001 | native | 52 | 4.16 | 4.10 | 2.80 |
| tts_en_r002 | native | 55 | 4.40 | 6.24 | 2.83 |
| svs_zh_r001 | BigVGAN | 82 | 6.56 | 17.94 | 3.06 |
| svs_zh_r002 | BigVGAN | 54 | 4.32 | 10.35 | 2.86 |
| svs_en_r001 | BigVGAN | 193 | 15.44 | 37.20 | 3.83 |
| svs_en_r002 | BigVGAN | 97 | 7.76 | 20.10 | 3.16 |
| unified_zh_robust_caihong | BigVGAN | 182 | 14.56 | 36.20 | 3.75 |
| unified_zh_novel_wukong | BigVGAN | 204 | 16.32 | 38.49 | 3.90 |

Model load: 6.0 s (warm file cache), 2.5 GiB allocated after loading in bfloat16.

## 4. Behavior with the real weights

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

## 5. Quality checks of the demo clips (`docs/audio/`)

Generated through the queue, no reference audio, seed 7, default server:

| Clip | How it was made | Check |
| --- | --- | --- |
| `demo_tts_en.flac` (4.7 s) | `workflows/api/tts_basic.json`, text "Welcome to ComfyUI. This voice was generated locally with SemBridge, on a single graphics card." | Whisper small heard: "Welcome to Confia. This voice was generated locally with Sembridge on a single graphics card." (word error rate 0.07: one word, "ComfyUI" heard as "Confia") |
| `demo_tts_zh.flac` (3.5 s) | text 今天天气很好，我们一起去公园散步吧。 | Whisper small heard 今天天氣很好,我們一起去公園散步吧! (all 16 characters, 3 of them in traditional form) |
| `demo_singing_en.flac` (3.6 s) | `workflows/api/singing_basic.json` with a seven-note "twin-kle" melody (see below) | `scripts/eval_pitch.py`: 7 of 7 notes with a clear pitch, median error 0.15 semitone, all within one semitone, octave offset 0 |
| `demo_unified_en.flac` (8.0 s) | `workflows/api/unified_basic.json`, speech + the same melody + speech | Whisper heard both spoken parts exactly; the sung syllables as "Twin kill, twin kill, lead to the star" |

Score of the singing demo: `twin 60 0.4`, `kle 60 0.4`, `twin 67 0.4`, `kle 67 0.4`, `lit 69 0.4`, `tle 69 0.4`, `star 67 0.9`, `<SP> 0 0.3`.
The pitch checker itself was validated on a synthetic sine melody (median error 0.06 semitone) and on the official example
`svs_en_r002` (median 0.54 semitone, 85 % within one semitone), so 0.15 is not a ceiling of the method. Whisper is a rough proxy for
intelligibility, not a listening test; the clips have **not** been judged by a human listener (HUMAN_AUDIO_REVIEW_PENDING).

The default level of the BigVGAN outputs is low (the unified demo is about -30 dBFS RMS); the official examples range from -12 to
-31 dBFS.

## 6. Tests

| Where | What | Result |
| --- | --- | --- |
| local, torch 2.14.1 CPU | ruff, `pytest tests` (scores, vendored-code manifest + byte comparison with the pinned upstream commit, runtime on a tiny random model, node registration, ComfyUI validation of the shipped workflows, node execution) | 50 passed, 2 skipped (torchaudio does not exist for torch 2.14) |
| local, torch 2.8.0 CPU + torchaudio 2.8.0 | same, torchaudio tests required | 52 passed |
| local, `scripts/ci_queue_check.py` | a second, fresh ComfyUI checkout + the node pack copied in, tiny model, `--cpu`: object_info, then score preview, TTS, singing, unified workflows through `POST /prompt` | passed |
| ComfyUI frontend | `scripts/export_gui_workflows.py` loads each API workflow with the frontend's `loadApiJson`: no missing node types, `graphToPrompt` returns the same classes and values, no page errors | 4 of 4 |

GitHub Actions results are linked from the repository's release notes.

## 7. Not verified

Linux or macOS with a GPU; AMD, Intel or Apple GPUs; torch builds other than 2.8.0+cu128 and 2.14.1+cu130 with the real weights;
other ComfyUI versions; quantized or offloaded loading; multi-GPU; Chinese-English code-switching; texts or scores beyond about 40 s;
listening tests by people; any comparison of quality with other speech or singing systems.

## Commands

```
python tools/download_model.py --models-dir <ComfyUI>/models/sembridge
python scripts/parity_vs_official.py --upstream <SemBridge checkout> --model <model folder> --out <dir>           # section 2
python scripts/parity_vs_official.py ... --torchaudio-shim <ComfyUI>                                               # stock ComfyUI environment
python scripts/queue_examples.py --server http://127.0.0.1:8188 --upstream <SemBridge checkout> --comfy-input <ComfyUI>/input --out <dir>
python tools/queue_client.py workflows/api/singing_basic.json --server http://127.0.0.1:8188 --set 2.seed=7
python scripts/eval_pitch.py <clip.flac> --score-file <score.txt>
python scripts/ci_queue_check.py --comfyui <ComfyUI> --tokenizer-dir <tokenizer files>
```
