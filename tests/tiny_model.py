"""Build a tiny random-initialized SemBridge model folder (no trained weights) for CPU tests.

The folder has the same layout as the official one (config.json, model.safetensors, savae/unified codecs, tokenizer files),
so the real loader, tokenizer, prompt encoder, flow-matching loop and both decoders all run; only the numbers are random.
The three tokenizer files are copied from `tokenizer_dir` (the official ones: vocabulary size and task-token IDs are checked
by the official loader, so a stand-in tokenizer is not possible).
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import torch
from safetensors.torch import save_file

TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json")


def build(dest: Path, tokenizer_dir: Path, seed: int = 0) -> Path:
    from sembridge_comfy.vendor.sembridge.models import SemBridgeGenerator
    from sembridge_comfy.vendor.sembridge.models.bigvgan import BigVGANDecoderConfig
    from sembridge_comfy.vendor.sembridge.models.configuration import GeneratorConfig
    from sembridge_comfy.vendor.sembridge.models.savae import AudioVAEConfig, SemanticAcousticVAE, save_savae_pretrained

    dest.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    config = GeneratorConfig(
        text_vocab_size=74_080,
        hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2,
        encoder_hidden_size=32, encoder_intermediate_size=64, encoder_num_layers=1, encoder_num_heads=2, encoder_num_key_value_heads=2,
        dit_hidden_size=32, dit_intermediate_size=64, dit_num_layers=1, dit_num_heads=2, dit_num_key_value_heads=2,
        semantic_head_enabled=False, semantic_anchor_layer=None,
        speech_task_token_id=74_078, singing_task_token_id=74_079,
        max_length=1024,
    )
    model = SemBridgeGenerator(config)
    state = {k: v.detach().clone().contiguous() for k, v in model.state_dict().items()}
    save_file(state, str(dest / "model.safetensors"))
    (dest / "config.json").write_text(json.dumps(config.model_dump(mode="json"), indent=1), encoding="utf-8")
    (dest / "generation_config.json").write_text(
        json.dumps({"cfg_value": 2.0, "flow_steps": 10, "latent_frame_rate_hz": 25.0, "patch_rate_hz": 12.5, "sample_rate": 44100, "temperature": 1.0}),
        encoding="utf-8",
    )

    rates = dict(encoder_rates=[2, 3, 6, 7, 7], decoder_rates=[7, 7, 6, 3, 2], latent_dim=64, sample_rate=44100)
    causal = SemanticAcousticVAE(AudioVAEConfig(encoder_dim=8, decoder_type="causal", decoder_dim=64, **rates))
    unified = SemanticAcousticVAE(
        AudioVAEConfig(encoder_dim=8, decoder_type="bigvgan", decoder_dim=64, bigvgan=BigVGANDecoderConfig(), **rates)
    )
    unified.encoder.load_state_dict(causal.encoder.state_dict())  # the official loader requires one shared encoder
    save_savae_pretrained(causal, dest, variant="default")
    save_savae_pretrained(unified, dest, variant="unified")

    for name in TOKENIZER_FILES:
        shutil.copyfile(tokenizer_dir / name, dest / name)
    return dest
