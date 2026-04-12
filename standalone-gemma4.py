#!/usr/bin/env python
# coding: utf-8

# In[1]:


# pip install -r https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/refs/heads/main/ch05/07_gpt_to_llama/requirements-extra.txt


# In[2]:

from modelscope.hub.snapshot_download import snapshot_download
from importlib.metadata import version

pkgs = [
    "huggingface_hub",  # to download pretrained weights
    "safetensors",      # to load the checkpoint tensors
    "tokenizers",       # to implement the tokenizer
    "torch",            # to implement the model
]
for p in pkgs:
    print(f"{p} version: {version(p)}")


# In[3]:


CHOOSE_MODEL = "E2B"  # Options: "E2B", "E4B"
USE_INSTRUCT_MODEL = True


# In[4]:


import torch
import torch.nn as nn


def compute_rope_params(
    head_dim,
    theta_base=10_000.0,
    context_length=4096,
    rope_type="default",
    partial_rotary_factor=1.0,
    dtype=torch.float32,
):
    if rope_type == "proportional":
        rope_angles = int(partial_rotary_factor * head_dim // 2)
        inv_freq_rotated = 1.0 / (
            theta_base ** (torch.arange(0, 2 * rope_angles, 2, dtype=torch.float32) / head_dim)
        )
        nope_angles = head_dim // 2 - rope_angles
        if nope_angles > 0:
            inv_freq = torch.cat([inv_freq_rotated, torch.zeros(nope_angles, dtype=torch.float32)], dim=0)
        else:
            inv_freq = inv_freq_rotated
    else:
        inv_freq = 1.0 / (theta_base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))

    positions = torch.arange(context_length, dtype=torch.float32)
    angles = positions.unsqueeze(1) * inv_freq.unsqueeze(0)
    angles = torch.cat([angles, angles], dim=1)
    cos = torch.cos(angles).to(dtype)
    sin = torch.sin(angles).to(dtype)
    return cos, sin


def apply_rope(x, cos, sin):
    batch_size, num_heads, seq_len, head_dim = x.shape
    assert head_dim % 2 == 0, "Head dimension must be even"

    x1 = x[..., : head_dim // 2]
    x2 = x[..., head_dim // 2 :]

    cos = cos[:seq_len, :].unsqueeze(0).unsqueeze(0)
    sin = sin[:seq_len, :].unsqueeze(0).unsqueeze(0)
    rotated = torch.cat((-x2, x1), dim=-1)
    return ((x * cos) + (rotated * sin)).to(dtype=x.dtype)


def repeat_kv(x, repeats):
    if repeats == 1:
        return x
    return x.repeat_interleave(repeats, dim=1)


# In[5]:


class Gemma4RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6, with_scale=True):
        super().__init__()
        self.eps = eps
        self.with_scale = with_scale
        if with_scale:
            self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        x_float = x.float()
        mean_squared = x_float.pow(2).mean(dim=-1, keepdim=True) + self.eps
        x_norm = x_float * torch.pow(mean_squared, -0.5)
        if self.with_scale:
            x_norm = x_norm * self.weight.float()
        return x_norm.to(dtype=x.dtype)


class Gemma4FeedForward(nn.Module):
    def __init__(self, cfg, layer_idx):
        super().__init__()
        first_kv_shared_layer_idx = cfg["n_layers"] - cfg["num_kv_shared_layers"]
        is_kv_shared_layer = layer_idx >= first_kv_shared_layer_idx > 0
        use_double_wide_mlp = cfg["use_double_wide_mlp"] and is_kv_shared_layer
        intermediate_size = cfg["hidden_dim"] * (2 if use_double_wide_mlp else 1)
        self.gate_proj = nn.Linear(cfg["emb_dim"], intermediate_size, bias=False, dtype=cfg["dtype"])
        self.up_proj = nn.Linear(cfg["emb_dim"], intermediate_size, bias=False, dtype=cfg["dtype"])
        self.down_proj = nn.Linear(intermediate_size, cfg["emb_dim"], bias=False, dtype=cfg["dtype"])

    def forward(self, x):
        x_gate = self.gate_proj(x)
        x_up = self.up_proj(x)
        x = nn.functional.gelu(x_gate, approximate="tanh") * x_up
        return self.down_proj(x)


# In[6]:


class Gemma4Attention(nn.Module):
    def __init__(self, cfg, layer_idx):
        super().__init__()
        self.layer_type = cfg["layer_types"][layer_idx]
        self.is_sliding = self.layer_type == "sliding_attention"
        self.head_dim = cfg["head_dim"] if self.is_sliding else cfg["global_head_dim"]
        self.num_heads = cfg["n_heads"]
        self.num_kv_heads = cfg["n_kv_heads"]
        self.num_key_value_groups = self.num_heads // self.num_kv_heads
        self.q_proj = nn.Linear(cfg["emb_dim"], self.num_heads * self.head_dim, bias=False, dtype=cfg["dtype"])
        self.k_proj = nn.Linear(cfg["emb_dim"], self.num_kv_heads * self.head_dim, bias=False, dtype=cfg["dtype"])
        self.v_proj = nn.Linear(cfg["emb_dim"], self.num_kv_heads * self.head_dim, bias=False, dtype=cfg["dtype"])
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, cfg["emb_dim"], bias=False, dtype=cfg["dtype"])
        self.q_norm = Gemma4RMSNorm(self.head_dim, eps=cfg["layer_norm_eps"])
        self.k_norm = Gemma4RMSNorm(self.head_dim, eps=cfg["layer_norm_eps"])
        self.v_norm = Gemma4RMSNorm(self.head_dim, eps=cfg["layer_norm_eps"], with_scale=False)

        first_kv_shared_layer_idx = cfg["n_layers"] - cfg["num_kv_shared_layers"]
        self.is_kv_shared_layer = layer_idx >= first_kv_shared_layer_idx > 0
        prev_layers = cfg["layer_types"][:first_kv_shared_layer_idx]
        if self.is_kv_shared_layer:
            self.kv_shared_layer_index = len(prev_layers) - 1 - prev_layers[::-1].index(self.layer_type)
            self.store_full_length_kv = False
        else:
            self.kv_shared_layer_index = None
            self.store_full_length_kv = (
                first_kv_shared_layer_idx > 0
                and self.layer_type in prev_layers
                and layer_idx == len(prev_layers) - 1 - prev_layers[::-1].index(self.layer_type)
            )

    def forward(self, x, mask, cos, sin, shared_kv=None, return_kv=False):
        batch_size, seq_len, _ = x.shape
        query = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        query = self.q_norm(query)
        query = apply_rope(query, cos, sin)

        computed_kv = None
        if shared_kv is None:
            key = self.k_proj(x).view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
            value = self.v_proj(x).view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
            key = self.k_norm(key)
            value = self.v_norm(value)
            key = apply_rope(key, cos, sin)
            computed_kv = (key, value)
        else:
            key, value = shared_kv

        key_for_attn = repeat_kv(key, self.num_key_value_groups)
        value_for_attn = repeat_kv(value, self.num_key_value_groups)

        attn_scores = query @ key_for_attn.transpose(-1, -2)
        attn_scores = attn_scores.masked_fill(mask.unsqueeze(0).unsqueeze(0), torch.finfo(attn_scores.dtype).min)
        attn_weights = torch.softmax(attn_scores.float(), dim=-1).to(dtype=query.dtype)
        context = attn_weights @ value_for_attn
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.num_heads * self.head_dim)
        output = self.o_proj(context)
        if return_kv and computed_kv is not None:
            return output, computed_kv
        return output, None


# In[7]:


class Gemma4DenseBlock(nn.Module):
    def __init__(self, cfg, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        self.layer_type = cfg["layer_types"][layer_idx]
        self.att = Gemma4Attention(cfg, layer_idx)
        self.mlp = Gemma4FeedForward(cfg, layer_idx)
        self.input_layernorm = Gemma4RMSNorm(cfg["emb_dim"], eps=cfg["layer_norm_eps"])
        self.post_attention_layernorm = Gemma4RMSNorm(cfg["emb_dim"], eps=cfg["layer_norm_eps"])
        self.pre_feedforward_layernorm = Gemma4RMSNorm(cfg["emb_dim"], eps=cfg["layer_norm_eps"])
        self.post_feedforward_layernorm = Gemma4RMSNorm(cfg["emb_dim"], eps=cfg["layer_norm_eps"])
        self.register_buffer("layer_scalar", torch.ones(1), persistent=True)
        self.hidden_size_per_layer_input = cfg["hidden_size_per_layer_input"]
        if self.hidden_size_per_layer_input:
            self.per_layer_input_gate = nn.Linear(
                cfg["emb_dim"],
                self.hidden_size_per_layer_input,
                bias=False,
                dtype=cfg["dtype"],
            )
            self.per_layer_projection = nn.Linear(
                self.hidden_size_per_layer_input,
                cfg["emb_dim"],
                bias=False,
                dtype=cfg["dtype"],
            )
            self.post_per_layer_input_norm = Gemma4RMSNorm(cfg["emb_dim"], eps=cfg["layer_norm_eps"])

    def forward(
        self,
        x,
        per_layer_input,
        mask_local,
        mask_global,
        cos_local,
        sin_local,
        cos_global,
        sin_global,
        shared_kv=None,
        return_kv=False,
    ):
        mask = mask_local if self.layer_type == "sliding_attention" else mask_global
        cos = cos_local if self.layer_type == "sliding_attention" else cos_global
        sin = sin_local if self.layer_type == "sliding_attention" else sin_global

        residual = x
        x = self.input_layernorm(x)
        x_attn, cached_kv = self.att(x, mask, cos, sin, shared_kv=shared_kv, return_kv=return_kv)
        x_attn = self.post_attention_layernorm(x_attn)
        x = residual + x_attn

        residual = x
        x = self.pre_feedforward_layernorm(x)
        x = self.mlp(x)
        x = self.post_feedforward_layernorm(x)
        x = residual + x

        if self.hidden_size_per_layer_input:
            residual = x
            x_per_layer = self.per_layer_input_gate(x)
            x_per_layer = nn.functional.gelu(x_per_layer, approximate="tanh")
            x_per_layer = x_per_layer * per_layer_input
            x_per_layer = self.per_layer_projection(x_per_layer)
            x_per_layer = self.post_per_layer_input_norm(x_per_layer)
            x = residual + x_per_layer

        return x * self.layer_scalar.to(dtype=x.dtype), cached_kv


# In[8]:


class Gemma4DenseModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        assert cfg["layer_types"] is not None and len(cfg["layer_types"]) == cfg["n_layers"]
        self.cfg = cfg
        self.tok_emb = nn.Embedding(
            cfg["vocab_size"],
            cfg["emb_dim"],
            padding_idx=cfg.get("pad_token_id", 0),
            dtype=cfg["dtype"],
        )
        self.blocks = nn.ModuleList([Gemma4DenseBlock(cfg, i) for i in range(cfg["n_layers"])])
        self.final_norm = Gemma4RMSNorm(cfg["emb_dim"], eps=cfg["layer_norm_eps"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False, dtype=cfg["dtype"])
        if cfg.get("tie_word_embeddings", False):
            self.out_head.weight = self.tok_emb.weight

        self.hidden_size_per_layer_input = cfg["hidden_size_per_layer_input"]
        if self.hidden_size_per_layer_input:
            self.embed_tokens_per_layer = nn.Embedding(
                cfg["vocab_size_per_layer_input"],
                cfg["n_layers"] * self.hidden_size_per_layer_input,
                padding_idx=cfg.get("pad_token_id", 0),
                dtype=cfg["dtype"],
            )
            self.per_layer_model_projection = nn.Linear(
                cfg["emb_dim"],
                cfg["n_layers"] * self.hidden_size_per_layer_input,
                bias=False,
                dtype=cfg["dtype"],
            )
            self.per_layer_projection_norm = Gemma4RMSNorm(
                self.hidden_size_per_layer_input,
                eps=cfg["layer_norm_eps"],
            )

        rope_local_type = cfg.get("rope_local_type", "default")
        cos_local, sin_local = compute_rope_params(
            head_dim=cfg["head_dim"],
            theta_base=cfg["rope_local_base"],
            context_length=cfg["context_length"],
            rope_type=rope_local_type,
            dtype=torch.float32,
        )
        cos_global, sin_global = compute_rope_params(
            head_dim=cfg["global_head_dim"],
            theta_base=cfg["rope_global_base"],
            context_length=cfg["context_length"],
            rope_type=cfg["rope_global_type"],
            partial_rotary_factor=cfg["rope_global_partial_rotary_factor"],
            dtype=torch.float32,
        )
        self.register_buffer("cos_local", cos_local, persistent=False)
        self.register_buffer("sin_local", sin_local, persistent=False)
        self.register_buffer("cos_global", cos_global, persistent=False)
        self.register_buffer("sin_global", sin_global, persistent=False)

    def _create_masks(self, seq_len, device):
        ones = torch.ones((seq_len, seq_len), dtype=torch.bool, device=device)
        mask_global = torch.triu(ones, diagonal=1)
        far_past = torch.triu(ones, diagonal=self.cfg["sliding_window"]).T
        mask_local = mask_global | far_past
        return mask_global, mask_local

    def get_per_layer_inputs(self, input_ids):
        if not self.hidden_size_per_layer_input:
            return None
        return (self.embed_tokens_per_layer(input_ids) * (self.hidden_size_per_layer_input ** 0.5)).reshape(
            *input_ids.shape,
            self.cfg["n_layers"],
            self.hidden_size_per_layer_input,
        )

    def project_per_layer_inputs(self, inputs_embeds, per_layer_inputs=None):
        if not self.hidden_size_per_layer_input:
            return None
        projected = self.per_layer_model_projection(inputs_embeds) * (self.cfg["emb_dim"] ** -0.5)
        projected = projected.reshape(
            *inputs_embeds.shape[:-1],
            self.cfg["n_layers"],
            self.hidden_size_per_layer_input,
        )
        projected = self.per_layer_projection_norm(projected)
        if per_layer_inputs is None:
            return projected
        return (projected + per_layer_inputs) * (2.0 ** -0.5)

    def forward(self, input_ids, reuse_shared_kv=False):
        x = self.tok_emb(input_ids) * (self.cfg["emb_dim"] ** 0.5)
        per_layer_inputs = self.get_per_layer_inputs(input_ids)
        per_layer_inputs = self.project_per_layer_inputs(x, per_layer_inputs)
        mask_global, mask_local = self._create_masks(input_ids.size(1), input_ids.device)
        shared_layer_kv = {} if reuse_shared_kv else None

        for i, block in enumerate(self.blocks):
            per_layer_input = per_layer_inputs[:, :, i, :] if per_layer_inputs is not None else None
            shared_kv = None
            need_store_kv = False
            if reuse_shared_kv:
                shared_kv = shared_layer_kv.get(block.att.kv_shared_layer_index) if block.att.is_kv_shared_layer else None
                need_store_kv = block.att.store_full_length_kv and shared_kv is None
            x, cached_kv = block(
                x,
                per_layer_input,
                mask_local,
                mask_global,
                self.cos_local,
                self.sin_local,
                self.cos_global,
                self.sin_global,
                shared_kv=shared_kv,
                return_kv=need_store_kv,
            )
            if reuse_shared_kv and need_store_kv and cached_kv is not None:
                shared_layer_kv[i] = cached_kv

        x = self.final_norm(x)
        logits = self.out_head(x)
        if self.cfg.get("final_logit_softcap") is not None:
            logits = logits / self.cfg["final_logit_softcap"]
            logits = torch.tanh(logits)
            logits = logits * self.cfg["final_logit_softcap"]
        return logits


Gemma4Model = Gemma4DenseModel


# In[9]:


def get_gemma4_dense_config(model_size="E2B", dtype=torch.bfloat16):
    model_size = model_size.upper()

    if model_size == "E2B":
        return {
            "vocab_size": 262_144,
            "vocab_size_per_layer_input": 262_144,
            "emb_dim": 1536,
            "hidden_dim": 4 * 1536,
            "n_layers": 35,
            "n_heads": 8,
            "head_dim": 256,
            "n_kv_heads": 1,
            "num_global_kv_heads": None,
            "global_head_dim": 512,
            "context_length": 131_072,
            "sliding_window": 512,
            "layer_types": (["sliding_attention"] * 4 + ["full_attention"]) * 7,
            "hidden_size_per_layer_input": 256,
            "num_kv_shared_layers": 20,
            "use_double_wide_mlp": True,
            "attention_k_eq_v": False,
            "rope_local_base": 10_000.0,
            "rope_local_type": "default",
            "rope_global_base": 1_000_000.0,
            "rope_global_type": "proportional",
            "rope_global_partial_rotary_factor": 0.25,
            "layer_norm_eps": 1e-6,
            "final_logit_softcap": 30.0,
            "tie_word_embeddings": True,
            "pad_token_id": 0,
            "dtype": dtype,
        }

    if model_size == "E4B":
        return {
            "vocab_size": 262_144,
            "vocab_size_per_layer_input": 262_144,
            "emb_dim": 2560,
            "hidden_dim": 4 * 2560,
            "n_layers": 42,
            "n_heads": 8,
            "head_dim": 256,
            "n_kv_heads": 2,
            "num_global_kv_heads": None,
            "global_head_dim": 512,
            "context_length": 131_072,
            "sliding_window": 512,
            "layer_types": (["sliding_attention"] * 5 + ["full_attention"]) * 7,
            "hidden_size_per_layer_input": 256,
            "num_kv_shared_layers": 18,
            "use_double_wide_mlp": False,
            "attention_k_eq_v": False,
            "rope_local_base": 10_000.0,
            "rope_local_type": "default",
            "rope_global_base": 1_000_000.0,
            "rope_global_type": "proportional",
            "rope_global_partial_rotary_factor": 0.25,
            "layer_norm_eps": 1e-6,
            "final_logit_softcap": 30.0,
            "tie_word_embeddings": True,
            "pad_token_id": 0,
            "dtype": dtype,
        }

    raise ValueError(f"Unknown Gemma 4 dense size: {model_size}")


# In[10]:


if torch.cuda.is_available():
    device = torch.device("cuda")
    model_dtype = torch.bfloat16
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    model_dtype = torch.float32
else:
    device = torch.device("cpu")
    model_dtype = torch.float32

selected_cfg = get_gemma4_dense_config(CHOOSE_MODEL, dtype=torch.float32)
{
    "model": CHOOSE_MODEL,
    "instruct": USE_INSTRUCT_MODEL,
    "emb_dim": selected_cfg["emb_dim"],
    "n_layers": selected_cfg["n_layers"],
    "n_heads": selected_cfg["n_heads"],
    "n_kv_heads": selected_cfg["n_kv_heads"],
    "global_head_dim": selected_cfg["global_head_dim"],
    "num_kv_shared_layers": selected_cfg["num_kv_shared_layers"],
    "device": str(device),
    "dtype": str(model_dtype),
}


# In[11]:


def load_weights_into_gemma4_dense(model, cfg, params):
    def assign(left, right, tensor_name="unknown"):
        if right is None:
            return False
        if left.shape != right.shape:
            raise ValueError(
                f"Shape mismatch in tensor {tensor_name!r}. Left: {tuple(left.shape)}, Right: {tuple(right.shape)}"
            )
        with torch.no_grad():
            if isinstance(right, torch.Tensor):
                left.copy_(right.to(dtype=left.dtype, device=left.device))
            else:
                left.copy_(torch.as_tensor(right, dtype=left.dtype, device=left.device))
        return True

    prefixes = []
    for prefix in ("model.language_model.", "language_model.", "model.", ""):
        if any(name.startswith(prefix) for name in params):
            prefixes.append(prefix)
    if not prefixes:
        prefixes = [""]

    def get_tensor(*names):
        for name in names:
            if name in params:
                return params[name], name
        for prefix in prefixes:
            for name in names:
                key = f"{prefix}{name}"
                if key in params:
                    return params[key], key
        return None, None

    loaded = 0
    missing = []

    def assign_from(target, *names):
        nonlocal loaded
        tensor, name = get_tensor(*names)
        expected_name = names[0]
        if tensor is None:
            missing.append(expected_name)
            return
        loaded += int(assign(target, tensor, name or expected_name))

    assign_from(model.tok_emb.weight, "embed_tokens.weight")

    if getattr(model, "hidden_size_per_layer_input", 0):
        assign_from(model.embed_tokens_per_layer.weight, "embed_tokens_per_layer.weight")
        assign_from(model.per_layer_model_projection.weight, "per_layer_model_projection.weight")
        assign_from(model.per_layer_projection_norm.weight, "per_layer_projection_norm.weight")

    for layer_idx in range(cfg["n_layers"]):
        block = model.blocks[layer_idx]
        prefix = f"layers.{layer_idx}."

        assign_from(block.att.q_proj.weight, f"{prefix}self_attn.q_proj.weight")
        assign_from(block.att.k_proj.weight, f"{prefix}self_attn.k_proj.weight")
        assign_from(block.att.v_proj.weight, f"{prefix}self_attn.v_proj.weight")
        assign_from(block.att.o_proj.weight, f"{prefix}self_attn.o_proj.weight")
        assign_from(block.att.q_norm.weight, f"{prefix}self_attn.q_norm.weight")
        assign_from(block.att.k_norm.weight, f"{prefix}self_attn.k_norm.weight")

        assign_from(block.mlp.gate_proj.weight, f"{prefix}mlp.gate_proj.weight")
        assign_from(block.mlp.up_proj.weight, f"{prefix}mlp.up_proj.weight")
        assign_from(block.mlp.down_proj.weight, f"{prefix}mlp.down_proj.weight")

        assign_from(block.input_layernorm.weight, f"{prefix}input_layernorm.weight")
        assign_from(block.post_attention_layernorm.weight, f"{prefix}post_attention_layernorm.weight")
        assign_from(block.pre_feedforward_layernorm.weight, f"{prefix}pre_feedforward_layernorm.weight")
        assign_from(block.post_feedforward_layernorm.weight, f"{prefix}post_feedforward_layernorm.weight")

        if getattr(block, "hidden_size_per_layer_input", 0):
            assign_from(block.per_layer_input_gate.weight, f"{prefix}per_layer_input_gate.weight")
            assign_from(block.per_layer_projection.weight, f"{prefix}per_layer_projection.weight")
            assign_from(block.post_per_layer_input_norm.weight, f"{prefix}post_per_layer_input_norm.weight")

        assign_from(block.layer_scalar, f"{prefix}layer_scalar")

    assign_from(model.final_norm.weight, "norm.weight")
    assign_from(model.out_head.weight, "lm_head.weight", "embed_tokens.weight")

    if loaded == 0:
        raise KeyError(
            "No Gemma 4 language-model weights were loaded. Supported prefixes are "
            "'model.language_model.', 'language_model.', 'model.', and ''."
        )

    if missing:
        missing_preview = ", ".join(repr(name) for name in missing[:10])
        if len(missing) > 10:
            missing_preview += f", ... (+{len(missing) - 10} more)"
        raise KeyError(
            f"Missing {len(missing)} required Gemma 4 language-model tensors. "
            f"First missing tensors: {missing_preview}"
        )

    return loaded

load_weights_into_gemma4 = load_weights_into_gemma4_dense


# In[12]:


# Uncomment and run the following code if you are executing the notebook for the first time

# from huggingface_hub import login
# login()


# In[13]:


import json
from pathlib import Path
from safetensors.torch import load_file


repo_id = f"google/gemma-4-{CHOOSE_MODEL}-it" if USE_INSTRUCT_MODEL else f"google/gemma-4-{CHOOSE_MODEL}"
local_dir_name = Path(repo_id).parts[-1]


def resolve_local_model_dir(local_dir_name):
    candidates = [
        Path(local_dir_name),
        Path("17_gemma4") / local_dir_name,
        Path("ch05") / "17_gemma4" / local_dir_name,
    ]
    for candidate in candidates:
        if (candidate / "model.safetensors").exists() or (candidate / "tokenizer.json").exists():
            return candidate
    return Path(local_dir_name)


local_dir = resolve_local_model_dir(local_dir_name)
model_cfg = get_gemma4_dense_config(CHOOSE_MODEL, dtype=model_dtype)

single_file = Path(local_dir) / "model.safetensors"

if single_file.exists():
    weights_dict = load_file(single_file)
else:
    try:
        weights_path = snapshot_download(
            repo_id=repo_id,

            local_dir=str(local_dir),
        )
        weights_dict = load_file(weights_path)
    except Exception:
        repo_dir = snapshot_download(repo_id=repo_id, local_dir=str(local_dir))
        index_path = Path(repo_dir) / "model.safetensors.index.json"
        with open(index_path, "r") as f:
            index = json.load(f)

        weights_dict = {}
        for filename in sorted(set(index["weight_map"].values())):
            shard = load_file(Path(repo_dir) / filename)
            weights_dict.update(shard)
model = Gemma4DenseModel(model_cfg)

num_loaded_tensors = load_weights_into_gemma4_dense(model, model_cfg, weights_dict)
print(f"Using Gemma 4 files from: {local_dir}")
print(f"Loaded {num_loaded_tensors} Gemma 4 text tensors")

model.to(device)

del weights_dict

model.eval()


# In[14]:


from tokenizers import Tokenizer


class GemmaTokenizer:
    def __init__(self, tokenizer_file_path):
        tok_file = Path(tokenizer_file_path)
        self._tok = Tokenizer.from_file(str(tok_file))

        self.bos_token = "<bos>"
        self.eos_token = "<eos>"
        self.pad_token = "<pad>"
        self.turn_token = "<turn|>"

        self.bos_token_id = self._tok.token_to_id(self.bos_token)
        self.eos_token_id = self._tok.token_to_id(self.eos_token)
        self.pad_token_id = self._tok.token_to_id(self.pad_token)
        self.turn_token_id = self._tok.token_to_id(self.turn_token)

    def encode(self, text, add_special_tokens=True):
        return self._tok.encode(text, add_special_tokens=add_special_tokens).ids

    def decode(self, ids, skip_special_tokens=False):
        if isinstance(ids, int):
            ids = [ids]
        return self._tok.decode(ids, skip_special_tokens=skip_special_tokens)

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        text = self.bos_token
        for message in messages:
            role = "model" if message["role"] == "assistant" else message["role"]
            text += f"{self.turn_token}{role}\n{message['content']}{self.turn_token}\n"

        if add_generation_prompt:
            text += f"{self.turn_token}model\n"

        if tokenize:
            return self.encode(text, add_special_tokens=False)
        return text


# In[15]:


tokenizer_file_path = Path(local_dir) / "tokenizer.json"
if not tokenizer_file_path.exists():
    try:
        tokenizer_file_path = Path(
            snapshot_download(repo_id=repo_id, allow_file_pattern="tokenizer.json", local_dir=str(local_dir))
        )
    except Exception as e:
        print(f"Warning: failed to download tokenizer.json: {e}")

tokenizer = GemmaTokenizer(tokenizer_file_path=str(tokenizer_file_path))


# In[16]:


prompt = "Give me a short introduction to large language models."

if USE_INSTRUCT_MODEL:
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )
    input_token_ids = tokenizer.encode(prompt, add_special_tokens=False)
else:
    prompt = f"{prompt}\n\nAnswer:"
    input_token_ids = tokenizer.encode(prompt)

tokenizer.decode(input_token_ids, skip_special_tokens=False)


# In[17]:


# Optionally use torch.compile for an extra speed-up
# model = torch.compile(model)


# In[18]:


def generate_text_basic_stream(model, token_ids, max_new_tokens, eos_token_id=None):
    model.eval()

    with torch.no_grad():
        for _ in range(max_new_tokens):
            try:
                out = model(token_ids, reuse_shared_kv=True)[:, -1]
            except TypeError:
                out = model(token_ids)[:, -1]
            next_token = torch.argmax(out, dim=-1, keepdim=True)

            if eos_token_id is not None and torch.all(next_token == eos_token_id):
                break

            yield next_token
            token_ids = torch.cat([token_ids, next_token], dim=1)


# In[19]:


input_token_ids_tensor = torch.tensor(input_token_ids, device=device).unsqueeze(0)

if torch.cuda.is_available():
    torch.cuda.reset_peak_memory_stats()

stop_token_id = tokenizer.turn_token_id if USE_INSTRUCT_MODEL else tokenizer.eos_token_id

for token in generate_text_basic_stream(
    model=model,
    token_ids=input_token_ids_tensor,
    max_new_tokens=200,
    eos_token_id=stop_token_id,
):
    token_id = token.squeeze(0).tolist()
    print(tokenizer.decode(token_id), end="", flush=True)

if torch.cuda.is_available():
    def calc_gpu_gb(x):
        return f"{x / 1024 / 1024 / 1024:.2f} GB"

    print(f"\n\nGPU memory used: {calc_gpu_gb(torch.cuda.max_memory_allocated())}")

