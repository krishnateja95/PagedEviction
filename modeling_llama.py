import math
from typing import List, Optional, Tuple, Union
import os
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from transformers.activations import ACT2FN

from transformers.cache_utils import Cache, DynamicCache, StaticCache
# from models.cache_utils import Cache, DynamicCache, StaticCache

from transformers.generation import GenerationMixin
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from transformers.modeling_flash_attention_utils import _flash_attention_forward
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutputWithPast,
    TokenClassifierOutput,
)
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
from transformers.modeling_utils import PreTrainedModel
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
from transformers.utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_flash_attn_greater_or_equal_2_10,
    logging,
    replace_return_docstrings,
)
from transformers.models.llama.configuration_llama import LlamaConfig

logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "LlamaConfig"

class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


ALL_LAYERNORM_LAYERS.append(LlamaRMSNorm)


class LlamaRotaryEmbedding(nn.Module):
    def __init__(
        self,
        dim=None,
        max_position_embeddings=2048,
        base=10000,
        device=None,
        scaling_factor=1.0,
        rope_type="default",
        config: Optional[LlamaConfig] = None,
    ):
        super().__init__()
        # TODO (joao): remove the `if` below, only used for BC
        self.rope_kwargs = {}
        if config is None:
            self.rope_kwargs = {
                "rope_type": rope_type,
                "factor": scaling_factor,
                "dim": dim,
                "base": base,
                "max_position_embeddings": max_position_embeddings,
            }
            self.rope_type = rope_type
            self.max_seq_len_cached = max_position_embeddings
            self.original_max_seq_len = max_position_embeddings
        else:
            # BC: "rope_type" was originally "type"
            if config.rope_scaling is not None:
                self.rope_type = config.rope_scaling.get("rope_type", config.rope_scaling.get("type"))
            else:
                self.rope_type = "default"
            self.max_seq_len_cached = config.max_position_embeddings
            self.original_max_seq_len = config.max_position_embeddings

        self.config = config
        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]

        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device, **self.rope_kwargs)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    def _dynamic_frequency_update(self, position_ids, device):
        seq_len = torch.max(position_ids) + 1
        if seq_len > self.max_seq_len_cached:  # growth
            inv_freq, self.attention_scaling = self.rope_init_fn(
                self.config, device, seq_len=seq_len, **self.rope_kwargs
            )
            self.register_buffer("inv_freq", inv_freq, persistent=False)  # TODO joao: may break with compilation
            self.max_seq_len_cached = seq_len

        if seq_len < self.original_max_seq_len and self.max_seq_len_cached > self.original_max_seq_len:  # reset
            self.register_buffer("inv_freq", self.original_inv_freq, persistent=False)
            self.max_seq_len_cached = self.original_max_seq_len

    @torch.no_grad()
    def forward(self, x, position_ids):
        if "dynamic" in self.rope_type:
            self._dynamic_frequency_update(position_ids, device=x.device)

        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()

        cos = cos * self.attention_scaling
        sin = sin * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class LlamaLinearScalingRotaryEmbedding(LlamaRotaryEmbedding):
    def __init__(self, *args, **kwargs):
        kwargs["rope_type"] = "linear"
        super().__init__(*args, **kwargs)


class LlamaDynamicNTKScalingRotaryEmbedding(LlamaRotaryEmbedding):
    def __init__(self, *args, **kwargs):
        kwargs["rope_type"] = "dynamic"
        super().__init__(*args, **kwargs)


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def apply_rotary_pos_emb_single(x, cos, sin, position_ids, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    x_embed = (x * cos) + (rotate_half(x) * sin)
    return x_embed


def apply_rotary_pos_emb_single_streamingLLM(x, cos, sin, position_ids):
    # The first two dimensions of cos and sin are always 1, so we can `squeeze` them.
    cos = cos.squeeze(1).squeeze(0)  # [seq_len, dim]
    sin = sin.squeeze(1).squeeze(0)  # [seq_len, dim]
    cos = cos[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    sin = sin[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    x_embed = (x * cos) + (rotate_half(x) * sin)
    return x_embed



class LlamaMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=config.mlp_bias)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        if self.config.pretraining_tp > 1:
            slice = self.intermediate_size // self.config.pretraining_tp
            gate_proj_slices = self.gate_proj.weight.split(slice, dim=0)
            up_proj_slices = self.up_proj.weight.split(slice, dim=0)
            down_proj_slices = self.down_proj.weight.split(slice, dim=1)

            gate_proj = torch.cat(
                [F.linear(x, gate_proj_slices[i]) for i in range(self.config.pretraining_tp)], dim=-1
            )
            up_proj = torch.cat([F.linear(x, up_proj_slices[i]) for i in range(self.config.pretraining_tp)], dim=-1)

            intermediate_states = (self.act_fn(gate_proj) * up_proj).split(slice, dim=2)
            down_proj = [F.linear(intermediate_states[i], down_proj_slices[i]) for i in range(self.config.pretraining_tp)]
            down_proj = sum(down_proj)
        else:
            down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

        return down_proj


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class LlamaAttention(nn.Module):
    def __init__(self, config: LlamaConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing a `layer_idx` is not recommended and will "
                "lead to errors during the forward call if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )

        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = getattr(config, "head_dim", self.hidden_size // self.num_heads)
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=config.attention_bias)

        self.rotary_emb = LlamaRotaryEmbedding(config=self.config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:

        bsz, q_len, _ = hidden_states.size()

        if self.config.pretraining_tp > 1:
            key_value_slicing = (self.num_key_value_heads * self.head_dim) // self.config.pretraining_tp
            query_slices = self.q_proj.weight.split(
                (self.num_heads * self.head_dim) // self.config.pretraining_tp, dim=0
            )
            key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
            value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

            query_states = [F.linear(hidden_states, query_slices[i]) for i in range(self.config.pretraining_tp)]
            query_states = torch.cat(query_states, dim=-1)

            key_states = [F.linear(hidden_states, key_slices[i]) for i in range(self.config.pretraining_tp)]
            key_states = torch.cat(key_states, dim=-1)

            value_states = [F.linear(hidden_states, value_slices[i]) for i in range(self.config.pretraining_tp)]
            value_states = torch.cat(value_states, dim=-1)

        else:
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        if position_embeddings is None:
            cos, sin = self.rotary_emb(value_states, position_ids)
        else:
            cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attention_mask is not None:
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()

        attn_output = attn_output.reshape(bsz, q_len, -1)

        if self.config.pretraining_tp > 1:
            attn_output = attn_output.split(self.hidden_size // self.config.pretraining_tp, dim=2)
            o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.config.pretraining_tp, dim=1)
            attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.config.pretraining_tp)])
        else:
            attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


def get_token_score(KV_cache_method, key_state, value_state):
    if KV_cache_method in  ["Inverse_Key_L2_Norm_global", "Inverse_Key_L2_Norm_local"]:
        key_norms = torch.norm(key_state, dim=-1).mean(dim=[0, 1])  # Average across batch and heads
        inv_key_norms = 1.0 / (key_norms + 1e-6)  # Add small epsilon to avoid division by zero
        return inv_key_norms

    elif KV_cache_method in ["Value_L2norm_local", "Value_L2_Norm_global"]:
        value_norms = torch.norm(value_state, dim=-1).mean(dim=[0, 1])  # Average across batch and heads
        return value_norms

    elif KV_cache_method in ["Value_L2norm_Key_L2_norm_local", "Value_L2_Norm_Key_L2_norm_global"]:
        key_norms = torch.norm(key_state, dim=-1).mean(dim=[0, 1])  # Average across batch and heads
        value_norms = torch.norm(value_state, dim=-1).mean(dim=[0, 1])  # Average across batch and heads
        return key_norms + value_norms
    else:
        print(f"{KV_cache_method} not found exiting")
        exit()


def get_page_score(KV_cache_method, key_state, value_state):
    if KV_cache_method in  ["Inverse_Key_L2_Norm_global", "Inverse_Key_L2_Norm_local"]:
        page_key_norm = torch.norm(key_state, dim=-1).mean().item()
        page_inv_key_norm = 1.0 / (page_key_norm + 1e-6)
        return page_inv_key_norm

    elif KV_cache_method in ["Value_L2norm_local", "Value_L2_Norm_global"]:
        page_value_norm = torch.norm(value_state, dim=-1).mean().item()
        return page_value_norm

    elif KV_cache_method in ["Value_L2norm_Key_L2_norm_local", "Value_L2_Norm_Key_L2_norm_global"]:
        page_key_norm   = torch.norm(key_state, dim=-1).mean().item()
        page_value_norm = torch.norm(value_state, dim=-1).mean().item()
        return page_key_norm + page_value_norm
    
    else:
        print(f"{KV_cache_method} not found exiting")
        exit()


class LlamaSdpaAttention(LlamaAttention):
    def KV_cache_evict_params(self, cache_params_dict):
        self.cache_budget    = cache_params_dict["cache_budget"]
        self.page_size       = cache_params_dict["page_size"]
        self.KV_cache_method = cache_params_dict["KV_cache_method"]
        self.topk            = cache_params_dict["topk"]
        # self.dynamic_page_selection_method = cache_params_dict["dynamic_page_selection_method"]

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.46
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        
        if self.KV_cache_method == "baseline":
            return self.forward_baseline(hidden_states = hidden_states,
                    attention_mask = attention_mask,
                    position_ids = position_ids,
                    past_key_value = past_key_value,
                    output_attentions = output_attentions,
                    use_cache = use_cache,
                    cache_position = cache_position,
                    position_embeddings = position_embeddings, 
                    **kwargs)


        elif self.KV_cache_method == "streamingLLM":
            return self.forward_streamingLLM(hidden_states = hidden_states,
                    attention_mask = attention_mask,
                    position_ids = position_ids,
                    past_key_value = past_key_value,
                    output_attentions = output_attentions,
                    use_cache = use_cache,
                    cache_position = cache_position,
                    position_embeddings = position_embeddings, 
                    **kwargs)

        elif "local" in self.KV_cache_method:
            return self.forward_local_eviction(hidden_states = hidden_states,
                    attention_mask = attention_mask,
                    position_ids = position_ids,
                    past_key_value = past_key_value,
                    output_attentions = output_attentions,
                    use_cache = use_cache,
                    cache_position = cache_position,
                    position_embeddings = position_embeddings, 
                    **kwargs)

        elif "global" in self.KV_cache_method:
            return self.forward_global_eviction(hidden_states = hidden_states,
                    attention_mask = attention_mask,
                    position_ids = position_ids,
                    past_key_value = past_key_value,
                    output_attentions = output_attentions,
                    use_cache = use_cache,
                    cache_position = cache_position,
                    position_embeddings = position_embeddings, 
                    **kwargs)

        else:
            print(f"{self.KV_cache_method} Not found")
            exit()


    def forward_baseline(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.46
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if output_attentions:
            return super().forward(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )

        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        if position_embeddings is None:
            logger.warning_once(
                "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
                "through `position_ids` (2D tensor with the indexes of the tokens), to using externally computed "
                "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.46 `position_ids` will be "
                "removed and `position_embeddings` will be mandatory."
            )
            cos, sin = self.rotary_emb(value_states, position_ids)
        else:
            cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)
            
        key_states   = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        causal_mask = attention_mask
        if attention_mask is not None:
            causal_mask = causal_mask[:, :, :, : key_states.shape[-2]]

        if query_states.device.type == "cuda" and causal_mask is not None:
            query_states = query_states.contiguous()
            key_states = key_states.contiguous()
            value_states = value_states.contiguous()

        is_causal = True if causal_mask is None and q_len > 1 else False
        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=causal_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=is_causal,
        )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bsz, q_len, -1)

        attn_output = self.o_proj(attn_output)

        return attn_output, None, past_key_value
    




    def forward_streamingLLM(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if output_attentions:
            return super().forward(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )

        bsz, q_len, _ = hidden_states.size()

        print("q_len", q_len)

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        if position_embeddings is None:
            logger.warning_once(
                "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
                "through `position_ids` (2D tensor with the indexes of the tokens), to using externally computed "
                "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.46 `position_ids` will be "
                "removed and `position_embeddings` will be mandatory."
            )
            cos, sin = self.rotary_emb(value_states, position_ids)
        else:
            cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)
            

        # Implement KV cache compression
        # Determine if we're in prefill or decode phase
        is_prefill = q_len > 1
        
        # Get the current sequence length after update
        curr_seq_len = key_states.shape[2]
        
        if self.layer_idx == 0:
            print("curr_seq_len", curr_seq_len)
            print("self.page_size", self.page_size)
        
        # Check if we need to compress the cache
        if curr_seq_len > self.cache_budget:
            # Always keep the first page
            first_page_size = min(self.page_size, curr_seq_len)
            indices_to_keep = []
            
            # Always preserve the first page
            indices_to_keep.extend(range(first_page_size))
            
            if is_prefill:
                # Prefill phase: first page and as many tokens from the end as possible
                remaining_budget = self.cache_budget - first_page_size
                if remaining_budget > 0:
                    start_idx = max(first_page_size, curr_seq_len - remaining_budget)
                    indices_to_keep.extend(range(start_idx, curr_seq_len))
            else:
                # Decode phase: check if last page is full and evict oldest non-first page
                if curr_seq_len % self.page_size == 0:  # Last page just became full
                    # If we have more than 2 pages, skip the second page (oldest after first)
                    if curr_seq_len > 2 * self.page_size:
                        # Keep everything from third page onwards
                        indices_to_keep.extend(range(2 * self.page_size, curr_seq_len))
                else:
                    # Last page isn't full yet, keep all pages except when budget exceeded
                    if curr_seq_len <= self.cache_budget:
                        indices_to_keep.extend(range(first_page_size, curr_seq_len))
                    else:
                        # Keep as many pages as possible from the end
                        remaining_budget = self.cache_budget - first_page_size
                        start_idx = max(first_page_size, curr_seq_len - remaining_budget)
                        indices_to_keep.extend(range(start_idx, curr_seq_len))
            
            # Convert to tensor for indexing
            keep_indices = torch.tensor(indices_to_keep, device=key_states.device, dtype=torch.long)
            
            # Apply compression to key and value states
            key_states = torch.index_select(key_states, 2, keep_indices)
            value_states = torch.index_select(value_states, 2, keep_indices)
            
            # Update cache with compressed states
            past_key_value.key_cache[self.layer_idx] = key_states
            past_key_value.value_cache[self.layer_idx] = value_states
            
            # Update attention mask to match compressed sequence
            # if causal_mask is not None:
            #     causal_mask = causal_mask[:, :, :, :key_states.shape[2]]



        
        
        
        if self.layer_idx == 0:
            print("key_states.size()", key_states.size())

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        causal_mask = attention_mask
        if attention_mask is not None:
            causal_mask = causal_mask[:, :, :, : key_states.shape[-2]]

        if query_states.device.type == "cuda" and causal_mask is not None:
            query_states = query_states.contiguous()
            key_states = key_states.contiguous()
            value_states = value_states.contiguous()

        is_causal = True if causal_mask is None and q_len > 1 else False
        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=causal_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=is_causal,
        )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bsz, q_len, -1)

        attn_output = self.o_proj(attn_output)

        return attn_output, None, past_key_value


    def forward_local_eviction(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.46
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if output_attentions:
            return super().forward(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )

        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        if position_embeddings is None:
            logger.warning_once(
                "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
                "through `position_ids` (2D tensor with the indexes of the tokens), to using externally computed "
                "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.46 `position_ids` will be "
                "removed and `position_embeddings` will be mandatory."
            )
            cos, sin = self.rotary_emb(value_states, position_ids)
        else:
            cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)
            

        total_tokens = key_states.shape[2]
        num_pages = (total_tokens + self.page_size - 1) // self.page_size

        # Helper to get start/end indices for a page
        def page_indices(i):
            start = i * self.page_size
            end = min((i + 1) * self.page_size, total_tokens)
            return start, end

        is_prefill = q_len > 1

        # Prefill phase: keep first and last page, evict from middle
        if is_prefill:
            # Always keep first and last page
            keep_pages = {0, num_pages - 1}
            # Compute L2 norm per token in value cache
            
            # l2_norms = torch.norm(value_states, p=2, dim=-1)  # (bsz, heads, seq, head_dim) -> (bsz, heads, seq)
            # l2_norms = l2_norms.sum(dim=(0, 1))  # sum over batch and heads, shape (seq,)
            
            l2_norms = get_token_score(self.KV_cache_method, key_states, value_states)

            # Mark tokens in first and last page as keep
            keep_tokens = set()
            for idx in [0, num_pages - 1]:
                s, e = page_indices(idx)
                keep_tokens.update(range(s, e))
            # For middle tokens, rank by L2 norm and evict lowest until cache_budget is met
            middle_tokens = [i for i in range(total_tokens) if i not in keep_tokens]
            sorted_middle = sorted(middle_tokens, key=lambda i: l2_norms[i])
            tokens_to_keep = list(keep_tokens)
            tokens_to_keep += sorted_middle[-(self.cache_budget - len(keep_tokens)):]
            tokens_to_keep = sorted(tokens_to_keep)
            # Update cache
            key_states = key_states[:, :, tokens_to_keep, :]
            value_states = value_states[:, :, tokens_to_keep, :]
            # Save L2 norms per page for future
            self.cache_l2_norms = []
            for i in range(num_pages):
                s, e = page_indices(i)
                page_l2 = l2_norms[s:e].sum()
                self.cache_l2_norms.append(float(page_l2))

        # Decode phase: evict when last page is full
        else:
            # Check if last page is full
            total_tokens = key_states.shape[2]
            num_pages = (total_tokens + self.page_size - 1) // self.page_size
            last_page_start, last_page_end = page_indices(num_pages - 1)
            if (last_page_end - last_page_start) == self.page_size:
                # Update L2 norms for last page
                # l2_norms = torch.norm(value_states, p=2, dim=-1).sum(dim=(0, 1))
                
                l2_norms = get_token_score(self.KV_cache_method, key_states, value_states)
                
                self.cache_l2_norms[-1] = l2_norms[last_page_start:last_page_end].sum()
                # Evict one page (not first, among next topk)
                candidate_pages = [i for i in range(1, min(1 + self.topk, num_pages - 1))]
                min_norm = float('inf')
                evict_idx = None
                for i in candidate_pages:
                    if self.cache_l2_norms[i] < min_norm:
                        min_norm = self.cache_l2_norms[i]
                        evict_idx = i
                if evict_idx is not None:
                    s, e = page_indices(evict_idx)
                    keep_mask = [i for i in range(total_tokens) if not (s <= i < e)]
                    key_states = key_states[:, :, keep_mask, :]
                    value_states = value_states[:, :, keep_mask, :]
                    del self.cache_l2_norms[evict_idx]
                    # No need to update num_pages, as it's recomputed each time
        
        past_key_value.key_cache[self.layer_idx] = key_states
        past_key_value.value_cache[self.layer_idx] = value_states
        
        
        key_states   = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        causal_mask = attention_mask
        if attention_mask is not None:
            causal_mask = causal_mask[:, :, :, : key_states.shape[-2]]

        if query_states.device.type == "cuda" and causal_mask is not None:
            query_states = query_states.contiguous()
            key_states = key_states.contiguous()
            value_states = value_states.contiguous()

        is_causal = True if causal_mask is None and q_len > 1 else False
        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=causal_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=is_causal,
        )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bsz, q_len, -1)

        attn_output = self.o_proj(attn_output)

        return attn_output, None, past_key_value



    def forward_global_eviction(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:

        if output_attentions:
            return super().forward(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )

        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        if position_embeddings is None:
            cos, sin = self.rotary_emb(value_states, position_ids)
        else:
            cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)


        # Initialize page l2-norms tracking if needed
        if past_key_value is not None and self.cache_budget is not None and self.page_size is not None and use_cache:
            if not hasattr(past_key_value, 'page_norms'):
                past_key_value.page_norms = {}
            if self.layer_idx not in past_key_value.page_norms:
                past_key_value.page_norms[self.layer_idx] = []

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            
            # First update the cache with new tokens
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)
            

        # Get updated key and value cache
        key_cache   = past_key_value.key_cache[self.layer_idx]
        value_cache = past_key_value.value_cache[self.layer_idx]
        
        if self.layer_idx == 0:
            print("key_cache.size()", key_cache.size())


        seq_len = key_cache.shape[2]
        is_prefill = q_len > 1
        
        # Check if cache exceeds budget
        if seq_len > self.cache_budget:
            if is_prefill:
                # Prefill phase: keep first and last pages, evict middle tokens
                # Calculate l2-norm for key or value tokens

                token_score = get_token_score(self.KV_cache_method, key_cache, value_cache)
                
                # Reserve first page
                first_page_size = min(self.page_size, seq_len)
                keep_mask = torch.zeros(seq_len, dtype=torch.bool, device=key_cache.device)
                keep_mask[:first_page_size] = True  # First page
                
                # Reserve last page
                last_page_start = max(0, seq_len - (seq_len % self.page_size or self.page_size))
                keep_mask[last_page_start:] = True
                
                # Calculate how many additional tokens to keep
                num_reserved = keep_mask.sum().item()
                num_to_keep = self.cache_budget
                
                if num_to_keep > num_reserved:
                    # Get non-reserved tokens
                    non_reserved_mask = ~keep_mask
                    non_reserved_indices = torch.where(non_reserved_mask)[0]
                    
                    if len(non_reserved_indices) > 0:
                        # non_reserved_inv_norms = inv_key_norms[non_reserved_indices]
                        non_reserved_inv_norms = token_score[non_reserved_indices]
                        
                        # Select tokens with lowest inverse norm (highest key norm)
                        # Since we want to keep tokens with high key norms (low inverse norms)
                        additional_to_keep = min(len(non_reserved_indices), num_to_keep - num_reserved)
                        if additional_to_keep > 0:
                            _, top_indices = torch.topk(non_reserved_inv_norms, k=additional_to_keep, largest=False)
                            additional_keep_indices = non_reserved_indices[top_indices]
                            keep_mask[additional_keep_indices] = True
                
                # Get final indices to keep
                keep_indices = torch.where(keep_mask)[0]
                
                # Update KV cache with compressed version
                new_key_cache = torch.index_select(key_cache, dim=2, index=keep_indices)
                new_value_cache = torch.index_select(value_cache, dim=2, index=keep_indices)
                
                past_key_value.key_cache[self.layer_idx] = new_key_cache
                past_key_value.value_cache[self.layer_idx] = new_value_cache
                
                # Update key_states and value_states for attention calculation
                key_states = new_key_cache
                value_states = new_value_cache
                
                # Calculate and store inverse key l2-norm for each page
                past_key_value.page_norms[self.layer_idx] = []
                
                # Calculate new number of pages
                new_seq_len = new_key_cache.shape[2]
                new_num_pages = (new_seq_len + self.page_size - 1) // self.page_size
                
                for page_idx in range(new_num_pages):
                    page_start = page_idx * self.page_size
                    page_end = min((page_idx + 1) * self.page_size, new_seq_len)
                    
                    page_keys   = new_key_cache[:, :, page_start:page_end, :]
                    page_values = new_value_cache[:, :, page_start:page_end, :]

                    page_score = get_page_score(self.KV_cache_method, page_keys, page_values)

                    past_key_value.page_norms[self.layer_idx].append((page_idx, page_score))

            else:  # Decode phase (single token generation)
                # Check if last page is full
                last_page_size = seq_len % self.page_size
                last_page_full = (last_page_size == 0)
                
                if last_page_full:
                    # Calculate inverse key l2-norm of the current last page
                    last_page_idx = (seq_len // self.page_size) - 1
                    last_page_start = last_page_idx * self.page_size
                    
                    last_page_keys   = key_cache[:, :, last_page_start:last_page_start + self.page_size, :]
                    last_page_values = value_cache[:, :, last_page_start:last_page_start + self.page_size, :]

                    page_score = get_page_score(self.KV_cache_method, last_page_keys, last_page_values)

                    # Add last page inverse norm to tracking
                    past_key_value.page_norms[self.layer_idx].append((last_page_idx, page_score))
                    
                    # Find page with highest inverse key l2-norm (lowest key norm) excluding first and last pages
                    valid_pages = [(idx, inv_norm) for idx, inv_norm in past_key_value.page_norms[self.layer_idx]
                                if idx != 0 and idx != last_page_idx]
                    
                    if valid_pages:
                        # Find page with highest inverse norm (lowest key norm)
                        max_page_idx, max_page_inv_norm = max(valid_pages, key=lambda x: x[1])
                        
                        # Create a mask to keep all tokens except those in the evicted page
                        keep_mask = torch.ones(seq_len, dtype=torch.bool, device=key_cache.device)
                        evict_start = max_page_idx * self.page_size
                        evict_end = min((max_page_idx + 1) * self.page_size, seq_len)
                        keep_mask[evict_start:evict_end] = False
                        
                        # Get indices of tokens to keep
                        keep_indices = torch.where(keep_mask)[0]
                        
                        # Update KV cache
                        new_key_cache = torch.index_select(key_cache, dim=2, index=keep_indices)
                        new_value_cache = torch.index_select(value_cache, dim=2, index=keep_indices)
                        
                        past_key_value.key_cache[self.layer_idx] = new_key_cache
                        past_key_value.value_cache[self.layer_idx] = new_value_cache
                        
                        # Update key_states and value_states for attention
                        key_states = new_key_cache
                        value_states = new_value_cache
                        
                        # Update page norm tracking by removing the evicted page and adjusting indices
                        updated_norms = []
                        for idx, inv_norm in past_key_value.page_norms[self.layer_idx]:
                            if idx != max_page_idx:
                                # Adjust indices for pages after the evicted one
                                new_idx = idx if idx < max_page_idx else idx - 1
                                updated_norms.append((new_idx, inv_norm))
                        
                        past_key_value.page_norms[self.layer_idx] = updated_norms


        # if self.layer_idx == 0:
        #     print("key_states.size()", key_states.size())

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        causal_mask = attention_mask
        if attention_mask is not None:
            causal_mask = causal_mask[:, :, :, : key_states.shape[-2]]

        if query_states.device.type == "cuda" and causal_mask is not None:
            query_states = query_states.contiguous()
            key_states = key_states.contiguous()
            value_states = value_states.contiguous()

        is_causal = True if causal_mask is None and q_len > 1 else False
        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=causal_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=is_causal,
        )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bsz, q_len, -1)

        attn_output = self.o_proj(attn_output)

        return attn_output, None, past_key_value




    # def forward_local_eviction(
    #     self,
    #     hidden_states: torch.Tensor,
    #     attention_mask: Optional[torch.Tensor] = None,
    #     position_ids: Optional[torch.LongTensor] = None,
    #     past_key_value: Optional[Cache] = None,
    #     output_attentions: bool = False,
    #     use_cache: bool = False,
    #     cache_position: Optional[torch.LongTensor] = None,
    #     position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    #     **kwargs,
    # ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        
    #     """Forward pass with page‑based KV‑cache compression.

    #     Implements the algorithm described in the prompt:
    #         * first & last pages are always kept
    #         * cache_budget is the *total* number of tokens (key/value pairs) that
    #           can reside in the cache **including** first & last pages
    #         * during **prefill** (q_len > 1) we greedily evict individual tokens
    #           from the middle pages (lowest L2‑norm first) until the budget is
    #           met, then we pre‑compute and store a per‑page L2‑norm summary
    #         * during **decode** (q_len == 1) we append the new token; once the
    #           last page becomes full, its L2 summary is (re‑)computed and a
    #           *single* page is evicted: the one with the smallest L2 among the
    #           oldest `topk` pages (page‑idx ≥ 1).  Metadata and tensors are
    #           updated in‑place inside `past_key_value`.
    #     """

    #     if output_attentions:
    #         return super().forward(
    #             hidden_states=hidden_states,
    #             attention_mask=attention_mask,
    #             position_ids=position_ids,
    #             past_key_value=past_key_value,
    #             output_attentions=output_attentions,
    #             use_cache=use_cache,
    #             cache_position=cache_position,
    #             position_embeddings=position_embeddings,
    #         )

    #     # ────────────────────────────────────────────────────────────────────────
    #     #  Projections
    #     # ────────────────────────────────────────────────────────────────────────
    #     bsz, q_len, _ = hidden_states.size()

    #     query_states = self.q_proj(hidden_states)
    #     key_states   = self.k_proj(hidden_states)
    #     value_states = self.v_proj(hidden_states)

    #     query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    #     key_states   = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    #     value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

    #     # ────────────────────────────────────────────────────────────────────────
    #     #  Rotary positional embeddings
    #     # ────────────────────────────────────────────────────────────────────────
    #     if position_embeddings is None:
    #         logger.warning_once(
    #             "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
    #             "through `position_ids` (2D tensor with the indexes of the tokens), to using externally computed "
    #             "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.46 `position_ids` will be "
    #             "removed and `position_embeddings` will be mandatory."
    #         )
    #         cos, sin = self.rotary_emb(value_states, position_ids)
    #     else:
    #         cos, sin = position_embeddings
    #     query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    #     # ────────────────────────────────────────────────────────────────────────
    #     #  Cache update & *compression*
    #     # ────────────────────────────────────────────────────────────────────────
        
    #     if past_key_value is not None:
    #         cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
    #         key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

    #         # # ------------------------------------------------------------------
    #         # # Parameters (assumed to be attributes of `self`)
    #         # # ------------------------------------------------------------------
    #         # cache_budget: int = self.cache_budget          # total token budget
    #         # page_size:    int = self.page_size             # tokens / page
    #         # topk:         int = self.topk                  # eviction window

    #     # Access full cache for this layer AFTER adding the current step
    #     key_cache   = past_key_value.key_cache[self.layer_idx]   # (B, H_kv, L, D)
    #     value_cache = past_key_value.value_cache[self.layer_idx] # (B, H_kv, L, D)
    #     total_len   = key_cache.size(-2)

    #     # Helper: lazily initialise per‑page L2 summaries container
    #     if not hasattr(past_key_value, "page_l2_norms"):
    #         past_key_value.page_l2_norms = {}
    #     layer_page_norms = past_key_value.page_l2_norms.get(self.layer_idx, [])

    #     # Detect *prefill* vs *decode* by q_len (prefill: >1, decode: ==1)
    #     is_prefill = q_len > 1

    #     # ------------------------------------------------------------------
    #     #  PREFILL  ─────────────────────────────────────────────────────────
    #     # ------------------------------------------------------------------
    #     if is_prefill:
    #         # Keep first & last page, evict middle tokens (lowest L2 first)
    #         num_pages   = math.ceil(total_len / self.page_size)
    #         first_end   = self.page_size
    #         last_start  = (num_pages - 1) * self.page_size
            
    #         # Compute L2 for every token (mean over batch, sum over heads+dim)
    #         token_norms = get_token_score(self.KV_cache_method, key_cache, value_cache)

    #         keep_mask = torch.ones(total_len, dtype=torch.bool, device=key_cache.device)
            
    #         keep_mask[first_end:last_start] = False  # middle tokens candidate for eviction

    #         # Sort middle tokens by norm (ascending)
    #         middle_idx  = torch.arange(first_end, last_start, device=key_cache.device)
    #         middle_norm = token_norms[first_end:last_start]
    #         sorted_norm, sort_idx = torch.sort(middle_norm, stable=True)

    #         # Number of tokens to drop so that len == cache_budget
    #         tokens_to_evict = max(0, total_len - self.cache_budget)
    #         if tokens_to_evict > 0:
    #             evict_tokens = middle_idx[sort_idx[:tokens_to_evict]]
    #             keep_mask[evict_tokens] = True  # retain non‑evicted middle tokens
    #             keep_mask[first_end:last_start] = False
    #             keep_mask[evict_tokens] = False  # final eviction mask

    #         # Compress the tensors
    #         key_cache   = key_cache[..., keep_mask, :]
    #         value_cache = value_cache[..., keep_mask, :]
    #         past_key_value.key_cache[self.layer_idx]   = key_cache
    #         past_key_value.value_cache[self.layer_idx] = value_cache

    #         # Recompute page L2 summaries *after* eviction
    #         total_len  = key_cache.size(-2)
    #         num_pages  = math.ceil(total_len / self.page_size)
    #         layer_page_norms = []
    #         for p in range(num_pages):
    #             s = p * self.page_size
    #             e = min((p + 1) * self.page_size, total_len)
                
    #             k_slice = key_cache[..., s:e, :]
    #             v_slice = value_cache[..., s:e, :]

    #             norm_p = get_page_score(self.KV_cache_method, k_slice, v_slice)
                
    #             layer_page_norms.append(norm_p)

    #         past_key_value.page_l2_norms[self.layer_idx] = layer_page_norms

    #     # ------------------------------------------------------------------
    #     #  DECODE  ──────────────────────────────────────────────────────────
    #     # ------------------------------------------------------------------
    #     else:
    #         # We just added one token to the *last* page.  Check if full.
    #         num_pages  = math.ceil(total_len / self.page_size)
    #         last_start = (num_pages - 1) * self.page_size
    #         last_len   = total_len - last_start

    #         if last_len == self.page_size:
    #             # (1) update L2 for the now‑full last page

    #             k_last = key_cache[..., last_start:, :]
    #             v_last = value_cache[..., last_start:, :]
                
    #             last_norm = get_page_score(self.KV_cache_method, k_last, v_last)

    #             if len(layer_page_norms) == num_pages:  # already had placeholder
    #                 layer_page_norms[-1] = last_norm
    #             else:
    #                 layer_page_norms.append(last_norm)

    #             # (2) identify eviction candidate among the *oldest* `topk` pages
    #             oldest_end = min(1 + self.topk, len(layer_page_norms) - 1)  # exclude first (idx 0) & last
    #             candidate_idxs = list(range(1, oldest_end))

    #             if candidate_idxs:
    #                 cand_norms = [layer_page_norms[i] for i in candidate_idxs]
    #                 evict_idx  = candidate_idxs[int(torch.tensor(cand_norms).argmin().item())]

    #                 # Remove tokens belonging to page `evict_idx`
    #                 s = evict_idx * self.page_size
    #                 e = min((evict_idx + 1) * self.page_size, total_len)
    #                 keep_mask = torch.ones(total_len, dtype=torch.bool, device=key_cache.device)
    #                 keep_mask[s:e] = False

    #                 key_cache   = key_cache[..., keep_mask, :]
    #                 value_cache = value_cache[..., keep_mask, :]
    #                 past_key_value.key_cache[self.layer_idx]   = key_cache
    #                 past_key_value.value_cache[self.layer_idx] = value_cache

    #                 # Delete summary for evicted page
    #                 del layer_page_norms[evict_idx]

    #             # Store back the updated list
    #             past_key_value.page_l2_norms[self.layer_idx] = layer_page_norms

    #     # The local variables `key_states` / `value_states` must reflect the
    #     # possibly‑compressed cache (they are later repeated & consumed by
    #     # SDPA).
    #     key_states   = key_cache
    #     value_states = value_cache

    #     key_states   = repeat_kv(key_states, self.num_key_value_groups)
    #     value_states = repeat_kv(value_states, self.num_key_value_groups)

    #     causal_mask = attention_mask
    #     if attention_mask is not None:
    #         causal_mask = causal_mask[:, :, :, : key_states.shape[-2]]

    #     if query_states.device.type == "cuda" and causal_mask is not None:
    #         query_states = query_states.contiguous()
    #         key_states   = key_states.contiguous()
    #         value_states = value_states.contiguous()

    #     is_causal = True if causal_mask is None and q_len > 1 else False
    #     attn_output = torch.nn.functional.scaled_dot_product_attention(
    #         query_states,
    #         key_states,
    #         value_states,
    #         attn_mask=causal_mask,
    #         dropout_p=self.attention_dropout if self.training else 0.0,
    #         is_causal=is_causal,
    #     )

    #     attn_output = attn_output.transpose(1, 2).contiguous()
    #     attn_output = attn_output.view(bsz, q_len, -1)

    #     attn_output = self.o_proj(attn_output)

    #     return attn_output, None, past_key_value








LLAMA_ATTENTION_CLASSES = {
    "sdpa": LlamaSdpaAttention,
}


class LlamaDecoderLayer(nn.Module):
    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = LLAMA_ATTENTION_CLASSES[config._attn_implementation](config=config, layer_idx=layer_idx)

        self.mlp = LlamaMLP(config)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
    

    def KV_cache_evict_params(self, cache_params_dict):
        self.self_attn.KV_cache_evict_params(cache_params_dict)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


class LlamaPreTrainedModel(PreTrainedModel):
    config_class = LlamaConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["LlamaDecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_cache_class = True
    _supports_quantized_cache = True
    _supports_static_cache = True

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


class LlamaModel(LlamaPreTrainedModel):
    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [LlamaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = LlamaRotaryEmbedding(config=config)
        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def KV_cache_evict_params(self, cache_params_dict):
        for decoder_layer in self.layers:
            decoder_layer.KV_cache_evict_params(cache_params_dict)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        # kept for BC (non `Cache` `past_key_values` inputs)
        return_legacy_cache = False
        if use_cache and not isinstance(past_key_values, Cache):
            return_legacy_cache = True
            if past_key_values is None:
                past_key_values = DynamicCache()
            else:
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)
                logger.warning_once(
                    "We detected that you are passing `past_key_values` as a tuple of tuples. This is deprecated and "
                    "will be removed in v4.47. Please convert your cache or use an appropriate `Cache` class "
                    "(https://huggingface.co/docs/transformers/kv_cache#legacy-cache-format)"
                )

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
        )
        hidden_states = inputs_embeds

        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    causal_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    cache_position,
                    position_embeddings,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if return_legacy_cache:
            next_cache = next_cache.to_legacy_cache()

        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

    def _update_causal_mask(
        self,
        attention_mask: torch.Tensor,
        input_tensor: torch.Tensor,
        cache_position: torch.Tensor,
        past_key_values: Cache,
        output_attentions: bool,
    ):
        if self.config._attn_implementation == "flash_attention_2":
            if attention_mask is not None and 0.0 in attention_mask:
                return attention_mask
            return None

        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        using_static_cache = isinstance(past_key_values, StaticCache)

        if self.config._attn_implementation == "sdpa" and not using_static_cache and not output_attentions:
            if AttentionMaskConverter._ignore_causal_mask_sdpa(
                attention_mask,
                inputs_embeds=input_tensor,
                past_key_values_length=past_seen_tokens,
                is_training=self.training,
            ):
                return None

        dtype, device = input_tensor.dtype, input_tensor.device
        sequence_length = input_tensor.shape[1]
        if using_static_cache:
            target_length = past_key_values.get_max_cache_shape()
        else:
            target_length = (
                attention_mask.shape[-1]
                if isinstance(attention_mask, torch.Tensor)
                else past_seen_tokens + sequence_length + 1
            )

        causal_mask = self._prepare_4d_causal_attention_mask_with_cache_position(
            attention_mask,
            sequence_length=sequence_length,
            target_length=target_length,
            dtype=dtype,
            device=device,
            cache_position=cache_position,
            batch_size=input_tensor.shape[0],
        )

        if (
            self.config._attn_implementation == "sdpa"
            and attention_mask is not None
            and attention_mask.device.type == "cuda"
            and not output_attentions
        ):
            min_dtype = torch.finfo(dtype).min
            causal_mask = AttentionMaskConverter._unmask_unattended(causal_mask, min_dtype)

        return causal_mask

    @staticmethod
    def _prepare_4d_causal_attention_mask_with_cache_position(
        attention_mask: torch.Tensor,
        sequence_length: int,
        target_length: int,
        dtype: torch.dtype,
        device: torch.device,
        cache_position: torch.Tensor,
        batch_size: int,
    ):
        
        if attention_mask is not None and attention_mask.dim() == 4:
            causal_mask = attention_mask
        else:
            min_dtype = torch.finfo(dtype).min
            causal_mask = torch.full(
                (sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=device
            )
            if sequence_length != 1:
                causal_mask = torch.triu(causal_mask, diagonal=1)
            causal_mask *= torch.arange(target_length, device=device) > cache_position.reshape(-1, 1)
            causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, -1, -1)
            if attention_mask is not None:
                causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit
                mask_length = attention_mask.shape[-1]
                padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :]
                padding_mask = padding_mask == 0
                causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
                    padding_mask, min_dtype
                )

        return causal_mask


class LlamaForCausalLM(LlamaPreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = LlamaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def KV_cache_evict_params(self, cache_params_dict):
        self.model.KV_cache_evict_params(cache_params_dict)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        num_logits_to_keep: int = 0,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        hidden_states = outputs[0]
        if self.config.pretraining_tp > 1:
            lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
            logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
            logits = torch.cat(logits, dim=-1)
        else:
            logits = self.lm_head(hidden_states[:, -num_logits_to_keep:, :])

        loss = None
        if labels is not None:
            logits = logits.float()
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
    
    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        position_ids=None,
        use_cache=True,
        **kwargs,
    ):
        if past_key_values is not None:
            if inputs_embeds is not None:
                input_ids = input_ids[:, -cache_position.shape[0] :]
            elif input_ids.shape[1] != cache_position.shape[0]:
                input_ids = input_ids[:, cache_position]

        if attention_mask is not None and position_ids is None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

                position_ids = position_ids.clone(memory_format=torch.contiguous_format)

        if inputs_embeds is not None and cache_position[0] == 0:
            model_inputs = {"inputs_embeds": inputs_embeds, "input_ids": None}
        else:
            model_inputs = {"input_ids": input_ids.clone(memory_format=torch.contiguous_format), "inputs_embeds": None}

        if isinstance(past_key_values, StaticCache) and attention_mask.ndim == 2:
            if model_inputs["inputs_embeds"] is not None:
                batch_size, sequence_length, _ = model_inputs["inputs_embeds"].shape
                device = model_inputs["inputs_embeds"].device
            else:
                batch_size, sequence_length = model_inputs["input_ids"].shape
                device = model_inputs["input_ids"].device

            dtype = self.lm_head.weight.dtype
            min_dtype = torch.finfo(dtype).min

            attention_mask = _prepare_4d_causal_attention_mask_with_cache_position(
                attention_mask,
                sequence_length=sequence_length,
                target_length=past_key_values.get_max_length(),
                dtype=dtype,
                device=device,
                min_dtype=min_dtype,
                cache_position=cache_position,
                batch_size=batch_size,
            )

        model_inputs.update(
            {
                "position_ids": position_ids,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "use_cache": use_cache,
                "attention_mask": attention_mask,
            }
        )
        return model_inputs



# "baseline",
# "streamingLLM_local",
# "Inverse_Key_L2_Norm_global",
# "Inverse_Key_L2_Norm_local",
# "Value_L2norm_local",
# "Value_L2_Norm_global",
# "Value_L2norm_Key_L2_norm_local",
# "Value_L2_Norm_Key_L2_norm_global",



def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default="meta-llama/Llama-3.2-1B-Instruct")
    parser.add_argument('--batch_size', type=int, default=2)
    
    parser.add_argument('--cache_budget', type=int, default=128)
    parser.add_argument('--page_size',    type=int, default=16)
    parser.add_argument('--topk',         type=int, default=3)

    parser.add_argument('--KV_cache_method', type=str, default="Value_L2norm_local")

    parser.add_argument('--dataset', type=str, default="longbench_2wikimqa_e")

    return parser.parse_args(args)

if __name__ == '__main__':

    import argparse
    args = parse_args()

    from lm_eval.models.huggingface import HFLM
    from lm_eval.evaluator import evaluate
    from lm_eval.tasks import get_task_dict

    token = ""

    from huggingface_hub import login
    login(token=token)
    os.environ["HF_AUTH_TOKEN"] = token 
    os.environ["HUGGINGFACE_TOKEN"] = token
    os.environ["HF_TOKEN"] = token

    from transformers import AutoTokenizer 

    cache_dir = "/lus/grand/projects/datascience/krishnat/model_weights/LLaMA/llama_cache/"    

    tokenizer = AutoTokenizer.from_pretrained(args.model_name,
                                            trust_remote_code=True,
                                            cache_dir=cache_dir
                                            )

    model = LlamaForCausalLM.from_pretrained(args.model_name, 
                                            trust_remote_code=True,
                                            torch_dtype=torch.float16,
                                            cache_dir=cache_dir,
                                            device_map = "auto",
                                            _attn_implementation = "sdpa"
                                            )

    cache_params_dict = {}
    cache_params_dict["cache_budget"] = args.cache_budget          
    cache_params_dict["page_size"] = args.page_size
    cache_params_dict["KV_cache_method"] = args.KV_cache_method
    cache_params_dict["topk"] = args.topk
    
    model.KV_cache_evict_params(cache_params_dict)
    
    prompt = "Passage 1: Waldrada of Lotharingia Waldrada was the, and later the wife, of Lothair II of Lotharingia. Biography Waldrada's family origin is uncertain. The prolific 19th-century French writer Baron Ernouf suggested that Waldrada was of noble Gallo-Roman descent, sister of Thietgaud, the bishop of Trier, and niece of Gunther, archbishop of Cologne. However, these suggestions are not supported by any evidence, and more recent studies have instead suggested she was of relatively undistinguished social origins, though still from an aristocratic milieu. The Vita Sancti Deicoli states that Waldrada was related to Eberhard II, Count of Nordgau (included Strasbourg) and the family of Etichonids, though this is a late 10th-century source and so may not be entirely reliable on this question.In 855 the Carolingian king Lothar II married Teutberga, a Carolingian aristocrat and the daughter of Bosonid Boso the Elder. The marriage was arranged by Lothar's father Lothar I for political reasons. It is very probable that Waldrada was already Lothar II's mistress at this time.Teutberga was allegedly not capable of bearing children and Lothar's reign was chiefly occupied by his efforts to obtain an annulment of their marriage, and his relations with his uncles Charles the Bald and Louis the German were influenced by his desire to obtain their support for this endeavour. Lothair, whose desire for annulment was arguably prompted by his affection for Waldrada, put away Teutberga. However, Hucbert took up arms on his sister's behalf, and after she had submitted successfully to the ordeal of water, Lothair was compelled to restore her in 858. Still pursuing his purpose, he won the support of his brother, Emperor Louis II, by a cession of lands and obtained the consent of the local clergy to the annulment and to his marriage with Waldrada, which took place in 862. However, Pope Nicholas I was suspicious of this and sent legates to investigate at the Council of Metz in 863. The Council found in favour of Lothair's divorce, which led to rumours that the papal legates may have bribed and thus meant that Nicholas order Lothair to take Teutberga back or face excommunication. With the support of Charles the Bald and Louis the German, Teutberga appealed the annulment to Pope Nicholas. Nicholas refused to recognize the annulment and excommunicated Waldrada in 866, forcing Lothair to abandon Waldrada in favour of Teutberga. Lothair accepted this begrudgingly for a time, but shortly afterward at the end of 867 Pope Nicholas I died. Thus, Lothair began to seek the permission of the newly appointed Pope Adrian II to again put Teutberga aside and marry Waldrada, riding to Rome to speak with him on the matter in 869. However, on his way home, Lothair died. Children Waldrada and Lothair II had some sons and probably three daughters, all of whom were declared illegitimate: Hugh (c. 855–895), Duke of Alsace (867–885) Gisela (c. 865–908), who in 883 married Godfrey, the Viking leader ruling in Frisia, who was murdered in 885 Bertha (c. 863–925), who married Theobald of Arles (c. 854–895), count of Arles, nephew of Teutberga. They had two sons, Hugh of Italy and Boso of Tuscany. After Theobald's death, between 895 and 898 she married Adalbert II of Tuscany (c. 875–915) They had at least three children: Guy, who succeeded his father as count and duke of Lucca and margrave of Tuscany, Lambert succeeded his brother in 929, but lost the titles in 931 to his half-brother Boso of Tuscany, and Ermengard. Ermengarde (d. 90?) Odo (d. c.879) Passage 2: Francis I Rákóczi Francis I Rákóczi (February 24, 1645, Gyulafehérvár, Transylvania – July 8, 1676, Zboró, Royal Hungary) was a Hungarian aristocrat, elected prince of Transylvania and father of Hungarian national hero Francis Rákóczi II.Francis Rákóczi was the son of George Rákóczi II, prince of Transylvania, and Sophia Báthory. He was elected prince by the Transylvanian Diet in 1652, during his father's life. However, because of the disastrous Polish campaign of 1657 and its consequences, the Ottoman Empire removed his father from the throne in 1660, and prohibited any Rákóczi to ascend the Transylvanian throne. This left Francis unable to come by his father's legacy; he therefore withdrew to his estates in Royal Hungary. Notably, the Rákóczi family was Calvinist, and they were staunch supporters of the Reformed Church in Hungary. However, Francis' mother, Sophia Báthory, had converted to Calvinism merely for the sake of her marriage. After her husband's death, she returned to Catholicism and supported the Counter Reformation. Francis Rákóczi also became a Catholic, thus acquiring favour with the Catholic Habsburg Court. His mother converted him to Catholicism. He was made a count in 1664. In 1666 Francis married Jelena Zrinska (Hungarian: Zrínyi Ilona), a Croatian countess, and joined the Wesselényi conspiracy (Zrinski-Frankopan conspiracy in Croatia), one leader of which was Jelena's father, Petar Zrinski (Hungarian: Zrínyi Péter). Francis soon became the leader of the conspiracy, and, as a culmination of their anti-Habsburg stratagems, started an armed uprising of nobles in Upper Hungary, while the other conspirators were supposed to start the fight in Croatia. Due to poor organization and discord between the conspirators, however, the Austrian authorities were well informed; they quickly suppressed the Croatian branch of the revolt. When Rákóczi learned that Petar Zrinski had been captured by the Austrians, he laid down his arms and applied for mercy. All other leaders of the conspiracy were executed for high treason; Rákóczi, due to his mother's intervention, and for a ransom of 300,000 forints and several castles, was pardoned. Issue Francis I had three children: György (1667) Julianna Borbála (1672–1717), married Count Ferdinand Gobert von Aspremont-Lynden (1643-1708) Francis Rákóczi II (1676–1735)Francis II was born only three months before his father's death. He led a rebellion against Austrian rule (Rákóczi's War of Independence) and died in exile. Passage 3: Mary Fiennes (lady-in-waiting) Mary Fiennes (1495–1531) was an English courtier. She was the wife of Henry Norris. Norris was executed for treason as one of the alleged lovers of her cousin, Anne Boleyn, the second wife of King Henry VIII of England. Mary lived for six years at the French court as a Maid of Honour to queens consort Mary Tudor, wife of Louis XII; and Claude of France, wife of Francis I"
    inputs = tokenizer(prompt, return_tensors="pt")

    context_length = inputs["input_ids"].shape[-1]
    print("context_length", context_length)

    generate_ids = model.generate(inputs.input_ids, max_new_tokens=100)

    print(generate_ids)

    exit()

    model.eval()

    # lm_eval_tasks = [
        # "longbench"
        
        # "longbench_2wikimqa_e", # "qa_f1_score,none"
        # "longbench_gov_report_e", # "rouge_score,none"
        # "longbench_passage_retrieval_en_e", #'retrieval_score,none'
        # "longbench_qasper_e", #"qa_f1_score,none"
        # "longbench_samsum_e", #"rouge_score,none"
        # "longbench_hotpotqa_e", #"qa_f1_score,none"
        # "longbench_passage_count_e", #'count_score,none'

        # "longbench_repobench",
        # "longbench_lcc_e",
        # "longbench_multi_news_e",
        # "longbench_multifieldqa_en_e",
        
        # "longbench_trec_e",
        # "longbench_triviaqa_e",

        # ]

    map_dict = {
        "longbench_2wikimqa_e": "qa_f1_score,none",
        "longbench_gov_report_e": "rouge_score,none",
        "longbench_passage_retrieval_en_e": 'retrieval_score,none',
        "longbench_qasper_e": "qa_f1_score,none",
        "longbench_samsum_e": "rouge_score,none",
        "longbench_hotpotqa_e": "qa_f1_score,none",
        "longbench_passage_count_e": 'count_score,none'

    }

    lm_eval_tasks = [args.dataset]

    result = evaluate(
            HFLM(
                pretrained=model,
                tokenizer=tokenizer, 
                batch_size=args.batch_size, 
                max_length=None,
                trust_remote_code=True,
                cache_dir = cache_dir),
            get_task_dict(lm_eval_tasks),
            limit = None,
        )

    print(result)

    for task, res in result["results"].items():
        if task in lm_eval_tasks:
                
            print(f"{task}: {res}")

            list_1 = [
                "Dataset",
                "Model",
                "KV_Cache_method",
                "Cache Budget",
                "Page Size",
                "score"
                ]
        
            list_2 = [
                task,
                args.model_name,
                args.KV_cache_method,
                args.cache_budget,
                args.page_size,
                res[ map_dict[task]]
                ]

            assert len(list_1) == len(list_2)

            csv_file = f"Results/{args.model_name}_lm_eval_results.csv"
            file_exists = os.path.exists(csv_file)

            with open(csv_file, 'a', newline = '') as csvfile:
                writer = csv.writer(csvfile)
                
                if not file_exists:
                    writer.writerow(list_1)
                
                writer.writerow(list_2) 
                
            csvfile.close()



