import json
import os

import torch
from transformers import Qwen3_5Model, Qwen3_5TextModel

from .._mixin import OstrisTransformersMixin
from .qwen3_vl import patch_qwen_vl_patch_embed

QWEN35_COMFY_REPO = "Comfy-Org/Qwen3.5"


def _load_sharded_state_dict(folder: str) -> dict:
    """All safetensors shards of a diffusers/transformers checkpoint folder."""
    from safetensors.torch import load_file

    index_path = os.path.join(folder, "model.safetensors.index.json")
    if os.path.isfile(index_path):
        with open(index_path) as f:
            files = sorted(set(json.load(f)["weight_map"].values()))
    else:
        files = [n for n in sorted(os.listdir(folder)) if n.endswith(".safetensors")]
    state_dict = {}
    for name in files:
        state_dict.update(load_file(os.path.join(folder, name)))
    return state_dict


class _DirectoryStateDictLoad:
    """Vendor Qwen3.5 folders hold a Qwen3_5Model checkpoint (``language_model.*``
    / ``visual.*`` keys). ``from_pretrained`` cannot map those onto the text or
    VL wrapper classes and silently leaves every weight missing, so local
    folders go through the state-dict path where ``convert_state_dict_on_load``
    applies."""

    @classmethod
    def aitk_from_pretrained(cls, path, subfolder=None, dtype=None, **kwargs):
        folder = os.path.join(path, subfolder) if subfolder else path
        if os.path.isdir(folder) and any(
            n.endswith(".safetensors") for n in os.listdir(folder)
        ):
            state_dict = _load_sharded_state_dict(folder)
            return cls.load_from_state_dict(
                state_dict,
                dtype if dtype is not None else torch.bfloat16,
                config_path=path,
                subfolder=subfolder,
            )
        return super().aitk_from_pretrained(
            path, subfolder=subfolder, dtype=dtype, **kwargs
        )


class Qwen35TextOnlyEncoder(_DirectoryStateDictLoad, Qwen3_5TextModel, OstrisTransformersMixin):
    """The Qwen3.5 language tower alone (moss_image). Text-only inputs, so the
    default arange positions match the multimodal model's text path. A stock
    Qwen repo id resolves to the Comfy-Org single-file repack under
    MODELS_PATH/text_encoders/ (config from the Qwen repo); its lm_head, mtp
    head and vision tower are dropped on load."""

    aitk_subfolder = "text_encoder"
    aitk_tokenizer_subfolder = "tokenizer"
    # the int8 repacks keep A_log / gated-norm weights in fp32; one dtype
    # throughout, like the bf16 path
    aitk_cast_quantized_load = True

    aitk_comfy_repo = QWEN35_COMFY_REPO
    aitk_comfy_weight_names = {
        "Qwen/Qwen3.5-2B": ["text_encoders/qwen3.5_2b_bf16.safetensors"],
        "Qwen/Qwen3.5-4B": ["text_encoders/qwen3.5_4b_bf16.safetensors"],
        "Qwen/Qwen3.5-9B": ["text_encoders/qwen3.5_9b_bf16.safetensors"],
    }

    _PREFIXES = ("model.language_model.", "language_model.", "model.")
    _DROPPED = ("lm_head.", "mtp.", "model.visual.", "visual.")

    @classmethod
    def get_transformer_block_names(cls):
        return ["layers"]

    @classmethod
    def aitk_load_config(cls, path, subfolder=None):
        config = super().aitk_load_config(path, subfolder=subfolder)
        # the Qwen repos carry the multimodal Qwen3_5Config
        return getattr(config, "text_config", config)

    @classmethod
    def convert_state_dict_on_load(cls, state_dict):
        out = {}
        for key, value in state_dict.items():
            if key.startswith(cls._DROPPED):
                continue
            for prefix in cls._PREFIXES:
                if key.startswith(prefix):
                    key = key[len(prefix) :]
                    break
            out[key] = value
        return out

    @classmethod
    def convert_state_dict_on_save(cls, state_dict):
        return {f"model.language_model.{k}": v for k, v in state_dict.items()}


class Qwen35VLEncoder(_DirectoryStateDictLoad, Qwen3_5Model, OstrisTransformersMixin):
    """Qwen3.5 language tower + vision tower (moss_image edit: reference images
    ride in the prompt). Same comfy files as Qwen35TextOnlyEncoder; only the
    lm_head and mtp head are dropped. Text-only inputs give the same states
    as the text-only class."""

    aitk_subfolder = "text_encoder"
    aitk_tokenizer_subfolder = "tokenizer"
    aitk_cast_quantized_load = True

    aitk_comfy_repo = QWEN35_COMFY_REPO
    aitk_comfy_weight_names = Qwen35TextOnlyEncoder.aitk_comfy_weight_names

    _DROPPED = ("lm_head.", "mtp.")

    @classmethod
    def get_transformer_block_names(cls):
        return ["language_model.layers", "visual.blocks"]

    @classmethod
    def convert_state_dict_on_load(cls, state_dict):
        out = {}
        for key, value in state_dict.items():
            if key.startswith(cls._DROPPED):
                continue
            if key.startswith("model."):
                key = key[len("model.") :]
            out[key] = value
        return out

    @classmethod
    def convert_state_dict_on_save(cls, state_dict):
        return {f"model.{k}": v for k, v in state_dict.items()}

    def patch_vision_patch_embed(self) -> int:
        """kernel==stride Conv3d patch_embed has no fast bf16 path; swap for a GEMM."""
        return patch_qwen_vl_patch_embed(self)
