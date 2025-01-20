from typing import List, Optional, Tuple, Union
import math
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers.activations import ACT2FN
from transformers.cache_utils import Cache
from transformers.modeling_outputs import (
    BaseModelOutputWithPast, 
    CausalLMOutputWithPast,
)
from transformers.modeling_utils import PreTrainedModel

from .configuration_evabyte import EvaByteConfig
from .multibyte_decoding_evabyte import MultiByteDecodingMixin
try:
    import triton
    USE_TRITON_IMPL = True
    from .eva import EvaAttention
    from .eva_agg_kernel import triton_eva_agg_fwd
    from .eva_prep_kv_kernel import triton_eva_prep_kv_fwd
except ImportError:
    USE_TRITON_IMPL = False
    print("WARNING: triton is not installed, using fallback EVA which might be slow and throw errors")
    from .eva_pt_ref import EvaAttention
from .eva_cache import EvaCache, EvaStaticCacheForTriton

MASK_MIN_VALUE = -10e10

def prepare_eva_attention_mask(
        seq_len, 
        device, 
        chunk_size, 
        window_size,
        use_cache=False, 
        cache=None
    ):
    """
    Prepare attention masks for EVA.
    
    """
    chunk_causal_mask  = None
    window_causal_mask = None
    if use_cache:
        cached_seq_len = cache.get_seq_length()
        total_seq_len = seq_len + cached_seq_len
        # cached_seq_len will be 0 during prefilling
        # padded_seq_len = chunk_size * math.ceil(total_seq_len / chunk_size)
        padded_seq_len = window_size * math.ceil(total_seq_len / window_size)
        num_chunks = padded_seq_len // chunk_size
    else:
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        assert seq_len % chunk_size == 0
        num_chunks = seq_len // chunk_size

        assert seq_len % window_size == 0

    # create causal mask
    ################################
    # generate chunked causal masks
    ################################
    # [b, h, j, c, c]
    chunks_per_window = window_size // chunk_size
    if num_chunks >= chunks_per_window:
        chunk_causal_mask = torch.ones(
            (chunk_size, num_chunks, num_chunks), 
            device=device,
            dtype=torch.bool
        ).triu(0)
        
        num_blocks = num_chunks // chunks_per_window
        chunk_causal_mask = chunk_causal_mask.reshape(
            chunk_size,
            num_blocks, 
            chunks_per_window, 
            num_blocks, 
            chunks_per_window
        ).transpose(-2, -3)

        block_diag_zero = (
            torch.eye(num_blocks, device=device, dtype=torch.bool)
            .unsqueeze(-1)
            .unsqueeze(-1)
            .unsqueeze(0)
        )

        # Set diagonal blocks to zero
        chunk_causal_mask = chunk_causal_mask.masked_fill(block_diag_zero, True)

        # Reshape back to original size
        chunk_causal_mask = (
            chunk_causal_mask
            .transpose(-2, -3)
            .reshape(chunk_size, num_chunks, num_chunks)
            .transpose(-2, -3)
            .reshape(chunk_size * num_chunks, num_chunks)
            .unsqueeze(0)
            .unsqueeze(0)
        )
    else:
        chunk_causal_mask = torch.ones(
            (1, 1, chunk_size, num_chunks, num_chunks), 
            device=device,
            dtype=torch.bool,
        ).triu(0).transpose(-2, -3) # [1, 1, c, j, c]
        chunk_causal_mask = chunk_causal_mask.reshape(
            1, 1, chunk_size * num_chunks, num_chunks
        ) # [1, 1, n, c]

    if use_cache:
        chunk_causal_mask = chunk_causal_mask[..., cached_seq_len : cached_seq_len + seq_len, :]

    window_causal_mask = torch.ones(
        (1, 1, 1, window_size, window_size), 
        device=device
    ).triu(1).to(torch.bool)
    return (chunk_causal_mask, window_causal_mask)

def pad_to_multiple(tensor, multiple, dim=-2, value=0, create_mask=False, left_padding=False):
    assert dim < 0 # only accept ``dim'' index in a reverse manner
    seqlen = int(tensor.shape[dim])
    m = seqlen / multiple
    if m.is_integer():
        if create_mask:
            return tensor, torch.ones(size=(tensor.shape[0], tensor.shape[dim]), dtype=torch.bool, device=tensor.device)
        else:
            return tensor
    remainder = math.ceil(m) * multiple - seqlen
    pad_offset = (0,) * (-1 - dim) * 2
    if left_padding:
        padded_res = F.pad(tensor, (*pad_offset, remainder, 0), value=value)
    else:
        padded_res = F.pad(tensor, (*pad_offset, 0, remainder), value=value)
    if create_mask:
        # assume dim 0 is the batch size
        padding_mask = torch.ones(size=(padded_res.shape[0], padded_res.shape[dim]), dtype=torch.bool, device=padded_res.device)
        if left_padding:
            padding_mask[:, :remainder] = False
        else:
            padding_mask[:, -remainder:] = False
        return padded_res, padding_mask
    else:
        return padded_res

class EvaByteRMSNorm(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.fp32_ln = config.fp32_ln
        self.variance_epsilon = config.rms_norm_eps
        self.add_unit_offset = config.norm_add_unit_offset
        if self.add_unit_offset:
            self.weight = nn.Parameter(torch.zeros(config.hidden_size))
        else:
            self.weight = nn.Parameter(torch.ones(config.hidden_size))

    def forward(self, hidden_states):
        if hasattr(self, 'config'):
            fp32_ln = self.config.fp32_ln
        else:
            fp32_ln = self.fp32_ln
        hidden_states = hidden_states.to(torch.float32 if fp32_ln else torch.bfloat16)

        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        if self.add_unit_offset:
            return (1 + self.weight) * hidden_states
        else:
            return self.weight * hidden_states

class EvaByteRotaryEmbedding(torch.nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        self._set_cos_sin_cache(seq_len=max_position_embeddings,
                                device=self.inv_freq.device,
                                dtype=torch.get_default_dtype())

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)

        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)


    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

        # return (
        #     self.cos_cached[:seq_len].to(dtype=x.dtype),
        #     self.sin_cached[:seq_len].to(dtype=x.dtype),
        # )
        if seq_len < self.max_seq_len_cached:
            cos_slice = self.cos_cached.split(seq_len, dim=0)[0]
            sin_slice = self.sin_cached.split(seq_len, dim=0)[0]
        else:
            cos_slice = self.cos_cached
            sin_slice = self.sin_cached

        return (
            cos_slice.to(dtype=x.dtype),
            sin_slice.to(dtype=x.dtype),
        )



class EvaByteLinearScalingRotaryEmbedding(EvaByteRotaryEmbedding):
    """EvaByteRotaryEmbedding extended with linear scaling. Credits to the Reddit user /u/kaiokendev"""
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None, scaling_factor=1.0):
        self.scaling_factor = scaling_factor
        super().__init__(dim, max_position_embeddings, base, device)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)
        t = t / self.scaling_factor

        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)


class EvaByteDynamicNTKScalingRotaryEmbedding(EvaByteRotaryEmbedding):
    """EvaByteRotaryEmbedding extended with Dynamic NTK scaling. Credits to the Reddit users /u/bloc97 and /u/emozilla"""
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None, scaling_factor=1.0):
        self.scaling_factor = scaling_factor
        super().__init__(dim, max_position_embeddings, base, device)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len

        if seq_len > self.max_position_embeddings:
            base = self.base * ((self.scaling_factor * seq_len / self.max_position_embeddings) -
                                (self.scaling_factor - 1))**(self.dim / (self.dim - 2))
            inv_freq = 1.0 / (base**(torch.arange(0, self.dim, 2).float().to(device) / self.dim))
            self.register_buffer("inv_freq", inv_freq, persistent=False)

        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)

        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)


class EvaByteMLP(nn.Module):
    def __init__(self, config, layer_idx: int = None):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]
        self.layer_idx = layer_idx
        self.config = config

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj

class EvaByteDecoderLayer(nn.Module):
    def __init__(self, config: EvaByteConfig, layer_idx: int = None):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.self_attn = EvaAttention(config=config, layer_idx=layer_idx)
        self.mlp = EvaByteMLP(config, layer_idx=layer_idx)
        self.input_layernorm = EvaByteRMSNorm(config)
        self.post_attention_layernorm = EvaByteRMSNorm(config)

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_value: Optional[Tuple[torch.Tensor]] = None,
            output_attentions: Optional[bool] = False,
            use_cache: Optional[bool] = False,
            cos: Optional[torch.Tensor] = None,
            sin: Optional[torch.Tensor] = None,
            multibyte_decoding: Optional[bool] = False,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        residual = hidden_states
        if self.config.fp32_skip_add:
            residual = residual.float()

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(hidden_states=hidden_states,
                                                                            attention_mask=attention_mask,
                                                                            position_ids=position_ids,
                                                                            past_key_value=past_key_value,
                                                                            output_attentions=output_attentions,
                                                                            use_cache=use_cache,
                                                                            cos=cos,
                                                                            sin=sin,
                                                                            multibyte_decoding=multibyte_decoding)
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        if self.config.fp32_skip_add:
            residual = residual.float()
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states, )

        if output_attentions:
            outputs += (self_attn_weights, )

        if use_cache:
            outputs += (present_key_value, )
        return outputs

class EvaBytePreTrainedModel(PreTrainedModel):
    config_class = EvaByteConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["EvaByteDecoderLayer"]
    _skip_keys_device_placement = "past_key_values"

    def _init_weights(self, module):
        std = getattr(self.config, "initializer_range", 0.02)
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, EvaByteModel):
            module.gradient_checkpointing = value

class EvaByteModel(EvaBytePreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`EvaByteDecoderLayer`]

    Args:
        config: EvaByteConfig
    """
    def __init__(self, config: EvaByteConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.max_position_embeddings = self.config.max_position_embeddings

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList([EvaByteDecoderLayer(config, layer_idx=layer_idx) for layer_idx in range(config.num_hidden_layers)])
        self.norm = EvaByteRMSNorm(config)

        self.gradient_checkpointing = False
        self.rope = config.rope_theta
        # Initialize weights and apply final processing
        self.post_init()
        self._init_rope()

    def _init_rope(self):
        if self.config.rope_scaling is None:
            self.rotary_emb = EvaByteRotaryEmbedding(self.head_dim,
                                                   max_position_embeddings=self.max_position_embeddings,
                                                   base=self.rope)
        else:
            scaling_type = self.config.rope_scaling["type"]
            scaling_factor = self.config.rope_scaling["factor"]
            if scaling_type == "linear":
                self.rotary_emb = EvaByteLinearScalingRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope)
            elif scaling_type == "dynamic":
                self.rotary_emb = EvaByteDynamicNTKScalingRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope)
            else:
                raise ValueError(f"Unknown RoPE scaling type {scaling_type}")

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def _helper_padding_mask(
            self,
            padding_mask,
            causal_mask
    ):
        padding_mask = torch.logical_or(padding_mask, padding_mask.transpose(-1, -2))
        return torch.logical_or(padding_mask, causal_mask)

    def _prepare_eva_generation_attn_mask_triton(
        self,
        attention_mask,
        input_ids,
        use_cache,
        past_key_values
    ):
        batch_size, seq_len = input_ids.shape
        if use_cache and past_key_values.get_seq_length() > 0:
            # decoding phase
            if past_key_values.rf_mask[0] is not None:
                cur_rf_mask = torch.zeros(
                    (batch_size, 1, seq_len, 1),
                    dtype=past_key_values.rf_mask[0].dtype,
                    device=past_key_values.rf_mask[0].device
                )
            else:
                cur_rf_mask = None
            
            if past_key_values.s_mask[0] is not None:
                cur_s_mask = torch.zeros(
                    (batch_size, 1, seq_len, 1),
                    dtype=past_key_values.s_mask[0].dtype,
                    device=past_key_values.s_mask[0].device
                )
            else:
                cur_s_mask = None
            
            seen_tokens = past_key_values.get_seq_length()
            if seen_tokens <= self.config.window_size:
                rfa_chunks_dummy_mask = None
            else:
                if cur_s_mask is not None: 
                    chunks_per_window = int(self.config.window_size // self.config.chunk_size)
                    # the ongoing decoding step would be (seen_seq_len + 1)-th token
                    num_windows_seen_so_far = seen_tokens // self.config.window_size
                    rfa_chunks_dummy_mask = torch.zeros(
                        (batch_size, 1, seq_len, num_windows_seen_so_far * chunks_per_window),
                        dtype=past_key_values.s_mask[0].dtype,
                        device=past_key_values.s_mask[0].device
                    )
                else:
                    rfa_chunks_dummy_mask = None
            # rf_mask and cur_mask are 0s because we do not want to mask them
            return (cur_s_mask, cur_rf_mask, rfa_chunks_dummy_mask)

        if attention_mask is not None and torch.any(attention_mask == 0.0):
            # convert 0 -> padding to 1 -> padding
            padded_attention_mask = pad_to_multiple(
                attention_mask, 
                self.config.window_size, 
                dim=-1,
                value=0, 
                create_mask=False,
                left_padding=False
            )
            # convert 0 -> padding to 1 -> padding
            padded_rf_mask = ~padded_attention_mask.unsqueeze(1).unsqueeze(-1).to(torch.bool) # [b, 1, n, 1]
            # [b, 1, w, j, 1]
            padded_w_attn_mask = padded_rf_mask.reshape(batch_size, 1, -1, self.config.window_size, 1).to(torch.bool)
            # [b, 1, w, j, 1] [b, 1, w, 1, j] -> [b, 1, w, j, j]
            w_padding_mask = torch.logical_or(padded_w_attn_mask, padded_w_attn_mask.transpose(-1, -2))
            w_causal_mask = torch.ones(
                (1, 1, 1, self.config.window_size, self.config.window_size),
                device=input_ids.device
            ).triu(1).to(torch.bool)
            s_mask = torch.logical_or(w_padding_mask, w_causal_mask)
            s_mask = s_mask.reshape(batch_size, 1, -1, self.config.window_size)
            s_mask = s_mask[..., :seq_len, :]
            # negate the attention mask to get the padding mask
            rf_mask = ~attention_mask.unsqueeze(1).unsqueeze(-1).to(torch.bool) # [b, 1, n, 1]
            return (s_mask, rf_mask)
        else:
            return (None, None)

    def _prepare_eva_generation_attn_mask(
        self,
        attention_mask,
        input_ids,
        use_cache,
        past_key_values
    ):
        batch_size, seq_len = input_ids.shape
        if use_cache and past_key_values.get_seq_length() > 0:
            # decoding phase
            if past_key_values.rf_mask[0] is not None:
                rf_mask = torch.zeros(
                    (batch_size, 1, seq_len, 1),
                    dtype=past_key_values.rf_mask[0].dtype,
                    device=past_key_values.rf_mask[0].device
                )
            else:
                rf_mask = None
            
            cur_causal_mask = torch.zeros(
                (batch_size, 1, seq_len, 1),
                dtype=torch.bool,
                device=input_ids.device
            )

            chunk_causal_mask = torch.ones(
                (batch_size, 1, seq_len, 1),
                dtype=torch.bool,
                device=input_ids.device
            )
            # chunk_causal_mask are 1s because we will mask them by default and
            # will be unmasked when the current singleton attention is processed over
            return (None, cur_causal_mask, chunk_causal_mask, rf_mask)

        true_num_chunks = seq_len // self.config.chunk_size
        chunk_causal_mask, _ = prepare_eva_attention_mask(
            seq_len, 
            input_ids.device, 
            self.config.chunk_size,
            self.config.window_size,
            use_cache=use_cache, 
            cache=past_key_values
        )
        chunk_causal_mask = chunk_causal_mask[..., :seq_len, :true_num_chunks]
        if attention_mask is not None and torch.any(attention_mask == 0.0):
            # convert 0 -> padding to 1 -> padding
            rf_mask = ~attention_mask.unsqueeze(1).unsqueeze(-1).to(torch.bool) # [b, 1, n, 1]
        else:
            rf_mask = None

        if seq_len < self.config.window_size:
            cur_window_mask = torch.ones(
                (1, 1, seq_len, seq_len),
                device=input_ids.device
            ).triu(1).to(torch.bool)
            if rf_mask is not None:
                cur_window_mask = self._helper_padding_mask(rf_mask, cur_window_mask)
            prev_window_mask = None
        else:
            if seq_len % self.config.window_size == 0:
                num_windows = seq_len // self.config.window_size
                cur_window_mask = None
                prev_window_mask = torch.ones(
                    (1, 1, num_windows, self.config.window_size, self.config.window_size),
                    device=input_ids.device
                ).triu(1).to(torch.bool)
                if rf_mask is not None:
                    prev_rf_mask = rf_mask.reshape(batch_size, 1, -1, self.config.window_size, 1)
                    prev_window_mask = self._helper_padding_mask(prev_rf_mask, prev_window_mask)
            else:
                num_windows = seq_len // self.config.window_size
                remainder_tokens = seq_len % self.config.window_size
                cur_window_mask = torch.ones(
                    (1, 1, remainder_tokens, remainder_tokens),
                    device=input_ids.device
                ).triu(1).to(torch.bool)
                prev_window_mask = torch.ones(
                    (1, 1, num_windows, self.config.window_size, self.config.window_size),
                    device=input_ids.device
                ).triu(1).to(torch.bool)
                if rf_mask is not None:
                    prev_rf_mask, cur_rf_mask = torch.split(rf_mask, [seq_len - remainder_tokens, remainder_tokens], dim=-2) 
                    cur_window_mask = self._helper_padding_mask(cur_rf_mask, cur_window_mask)
                    prev_rf_mask = prev_rf_mask.reshape(batch_size, 1, -1, self.config.window_size, 1)
                    prev_window_mask = self._helper_padding_mask(prev_rf_mask, prev_window_mask)
        
        return (prev_window_mask, cur_window_mask, chunk_causal_mask, rf_mask)

    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            multibyte_decoding: Optional[bool] = None,
    ) -> Tuple:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (output_hidden_states
                                if output_hidden_states is not None else self.config.output_hidden_states)
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )

        if self.gradient_checkpointing and self.training and use_cache:
            raise ValueError("`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`...")

        batch_size, seq_len = input_ids.shape
        #### Step 0. Hack
        if (not self.training) and (not use_cache) and (not multibyte_decoding):
            # forward-only inference mode. 
            # We tweak use_cache to be True to reuse code for generation
            use_cache = True
            device = input_ids.device if input_ids is not None else None
            if position_ids is None:
                position_ids = torch.arange(0, seq_len, device=device, dtype=int).reshape(1, -1).expand(batch_size, -1)

        #### Step 1. Prepare caches if in inference mode
        if use_cache:
            if past_key_values is not None:
                assert isinstance(past_key_values, Cache)
            else:
                if not USE_TRITON_IMPL:
                    past_key_values = EvaCache()
                else:
                    past_key_values = EvaStaticCacheForTriton(
                        input_ids.shape[0],
                        self.config.num_attention_heads,
                        self.config.window_size,
                        self.config.hidden_size // self.config.num_attention_heads,
                        self.config.num_hidden_layers,
                        self.embed_tokens.weight.dtype,
                        self.embed_tokens.weight.device,
                    )
        
        if not multibyte_decoding:
            if use_cache:
                if USE_TRITON_IMPL:
                    causal_mask = self._prepare_eva_generation_attn_mask_triton(
                        attention_mask,
                        input_ids,
                        use_cache,
                        past_key_values
                    )
                else:
                    causal_mask = self._prepare_eva_generation_attn_mask(
                        attention_mask,
                        input_ids,
                        use_cache,
                        past_key_values
                    )
            else:
                assert self.training
                assert seq_len % self.config.window_size == 0
                # for training, we need to pass in the attention mask
                # usually calculated by _prepare_training_attn_mask()
                causal_mask = attention_mask
        else:
            assert use_cache
            causal_mask = attention_mask

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        max_seq_length = past_seen_tokens + inputs_embeds.shape[1]

        hidden_states = inputs_embeds

        if position_ids is None:
            assert not use_cache, "during decoding we must explicitly pass position_ids to the model call"
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(past_seen_tokens, max_seq_length, device=device, dtype=int).reshape(1, -1).expand(batch_size, -1)

        cos, sin = self.rotary_emb(hidden_states, seq_len=max_seq_length)
        assert len(cos.shape) == 2, f"cos should be of shape (max_seq_len, head_dim), got {cos.shape} instead"
        assert sin.shape == cos.shape, f"sin should be of shape (max_seq_len, head_dim), got {sin.shape} instead"
        assert len(position_ids.shape) == 2, f"position_ids should be of 2D, got {position_ids.shape} instead"
        cos = cos[position_ids, :]
        sin = sin[position_ids, :]
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)

        if USE_TRITON_IMPL and (not multibyte_decoding):
            # the masks generated above for triton kernels are boolean. Convert them to floats
            if (
                (not use_cache) or
                (use_cache and past_seen_tokens == 0)
            ):
                window_mask, intra_chunk_mask = causal_mask
                
                if window_mask is not None:
                    assert window_mask.dtype == torch.bool
                    window_mask_float = window_mask.to(torch.float)
                    window_mask_float = window_mask_float.masked_fill(window_mask.to(torch.bool), MASK_MIN_VALUE)
                    window_mask_float = window_mask_float.reshape(batch_size, 1, -1, self.config.window_size)
                    window_mask = window_mask_float.to(hidden_states.dtype)

                if intra_chunk_mask is not None:
                    assert intra_chunk_mask.dtype == torch.bool
                    intra_chunk_mask_float = intra_chunk_mask.to(torch.float)
                    intra_chunk_mask_float = intra_chunk_mask_float.masked_fill(intra_chunk_mask.to(torch.bool), MASK_MIN_VALUE)
                    intra_chunk_mask = intra_chunk_mask_float.to(hidden_states.dtype)
                causal_mask = (window_mask, intra_chunk_mask)

        if self.config.fp32_skip_add:
            hidden_states = hidden_states.float()

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states, )

            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # None for past_key_value
                        return module(*inputs, output_attentions, use_cache=None)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(decoder_layer),
                    hidden_states,
                    causal_mask,
                    position_ids,
                    None,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cos=cos,
                    sin=sin,
                    multibyte_decoding=multibyte_decoding,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1], )

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states, )

        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class EvaByteForCausalLM(EvaBytePreTrainedModel, MultiByteDecodingMixin):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        EvaBytePreTrainedModel.__init__(self, config)

        self.model = EvaByteModel(config)
        self.vocab_size = config.vocab_size
        # define multibyte prediction heads
        if hasattr(config, "num_pred_heads") and config.num_pred_heads > 1:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size * config.num_pred_heads, bias=False)
        else:
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

    def _prepare_training_attn_mask(
        self, 
        target_token_type_ids,
        use_doc_boundary_attention, 
        EOS_TOKEN_TYPE_ID=None,
        PAD_TOKEN_TYPE_ID=None,
    ):
        '''
        This function prepares the attention mask for training byte models.
            target_token_type_ids:
                Tensor of shape (batch_size, seq_len), marking the token type ids 
                for the target sequence. In particular, we should have
                    - target_token_type_ids[i, j] = EOS_TOKEN_TYPE_ID 
                        if the j-th token in the i-th sequence is the end of an article.
                    - target_token_type_ids[i, j] = PAD_TOKEN_TYPE_ID 
                        if the j-th token in the i-th sequence is the padding token.
            use_doc_boundary_attention: bool, 
                whether to enable doc boundary attention.
            EOS_TOKEN_TYPE_ID: int, 
                the token type id for the end of an article.
            PAD_TOKEN_TYPE_ID: int, 
                the token type id for the padding token.
        '''
        assert self.training
        batch_size, num_tokens = target_token_type_ids.shape

        chunk_causal_mask, window_causal_mask = prepare_eva_attention_mask(
            num_tokens, 
            target_token_type_ids.device, 
            chunk_size=self.config.chunk_size, 
            window_size=self.config.window_size,
            use_cache=False,
            cache=None
        )
        if use_doc_boundary_attention:
            #### step 1: mark each document with a unique id
            end_token_ids = {EOS_TOKEN_TYPE_ID, PAD_TOKEN_TYPE_ID}
            token_types = torch.zeros(batch_size, num_tokens)
            for sequence_idx, sequence in enumerate(target_token_type_ids):
                num_articles = 0
                start_index = 0
                # for each sample in the batch, the collapsed attention mask looks like:
                # [1, 1, .... 1, 0, 2, 2, ... 2, 0, ... n, n ..... n], assuming there are n articles in the sequence.
                # Each of the n articles are separated by 0.
                for token_idx, token_type_id in enumerate(sequence):
                    if start_index is not None and token_type_id.item() in end_token_ids:
                        num_articles += 1
                        end_index = token_idx if token_type_id == PAD_TOKEN_TYPE_ID else token_idx + 1
                        token_types[sequence_idx][start_index:end_index] = num_articles
                        start_index = None
                    elif start_index is None and token_type_id not in end_token_ids:
                        start_index = token_idx + 1

            assert num_tokens % self.config.chunk_size == 0, "Number of tokens must be divisible by chunk size"
            assert num_tokens % self.config.window_size == 0, "Number of tokens must be divisible by window size"
            num_chunks = num_tokens // self.config.chunk_size
            num_windows = num_tokens // self.config.window_size

            article_separator = 0

            #### step 2: generate attention masks for each window
            #### NOTE: we perform exact attention within each window, 
            ####       so we only need to mask out different documents
            ####       for each window.
            token_types_windows = token_types.reshape(batch_size, num_windows, self.config.window_size, 1)
            token_types_windows_t = token_types_windows.transpose(-1, -2)
            # replace all elements in TOKEN_SEPS with -1
            token_types_windows = torch.where(token_types_windows == article_separator, -1, token_types_windows)
            window_3d_mask = (token_types_windows == token_types_windows_t)
            window_3d_mask = ~window_3d_mask

            #### step 3: generate chunk-level 3D masks
            #### NOTE: this is a bit tricky, as we aim to mask out different 
            ####       documents to avoid cross-doc attention across chunks.
            #### Example: suppose we have a sequence of length 12 with 3 documents:
            ####       [1, 1, 1, 1, 1, 2, 2, 3, 3, 3, 3, 3].
            ####       The chunk-size and window-size are both 4.
            ####       The chunk-level mask of shape (batch_size, seq_len, num_chunks) is:
            ####       [
            ####         [0, 0, 0],
            ####         [0, 0, 0],
            ####         [0, 0, 0],
            ####         [0, 0, 0],
            ####
            ####         [1, 0, 0],
            ####         [0, 0, 0],
            ####         [0, 0, 0],
            ####         [0, 0, 0],
            ####
            ####         [0, 1, 0],
            ####         [0, 1, 0],
            ####         [0, 1, 0],
            ####         [0, 1, 0],
            ####       ]
            ####       Explanation:
            ####       - Tokens will not attend to their own and future chunks.
            ####         (as tokens within a chunk are captured by the window-level exact attention)
            ####       - Tokens will attend to a chunk only if there are tokens 
            ####         from the same document in that chunk.
            ####       The mask within each chunk of shape (batch_size, num_chunks, chunk_size) is:
            ####       [
            ####         [1, 1, 1, 1],
            ####         [0, 0, 0, 1],
            ####         [1, 1, 1, 1],
            ####       ]
            ####       Explanation:
            ####       - If all tokens in a chunk are from the same document, 
            ####         no tokens will be masked out.
            ####       - If there are tokens from different documents in a chunk, 
            ####         only tokens from the rightmost document will be kept.
            ####         (b/c the future chunks might contain tokens from the rightmost document,
            ####         but all the remaining docs will never get attended by other docs)
            token_types_chunks = token_types.reshape(batch_size, num_chunks, self.config.chunk_size)
            inter_chunk_mask = torch.zeros((batch_size, num_tokens, num_chunks), dtype=torch.bool)
            intra_chunk_mask = torch.ones_like(token_types_chunks, dtype=torch.bool)
            
            for chunk_idx in range(num_chunks):
                for batch_idx in range(batch_size):
                    # Identify tokens in the current chunk belonging to each sequence
                    chunk = token_types_chunks[batch_idx, chunk_idx]
                    unique_elements = torch.unique(chunk, sorted=True).tolist()
                    
                    # Create a mask for whether each token can attend to the current chunk
                    for token_type in unique_elements:
                        if token_type == article_separator:
                            continue
                        token_mask = (token_types[batch_idx] == token_type)
                        inter_chunk_mask[batch_idx, :, chunk_idx] |= token_mask

                    # Create a mask within each chunk
                    unique_elements = [x for x in unique_elements if x != article_separator]
                    if len(unique_elements) > 1 and chunk[-1] != article_separator:
                        intra_chunk_mask[batch_idx, chunk_idx] = (chunk == unique_elements[-1])
            
            inter_chunk_mask = ~inter_chunk_mask
            intra_chunk_mask = ~intra_chunk_mask

            window_mask = torch.logical_or(window_causal_mask, window_3d_mask.unsqueeze(1))
            inter_chunk_mask = torch.logical_or(chunk_causal_mask, inter_chunk_mask.unsqueeze(1))
            intra_chunk_mask = intra_chunk_mask.unsqueeze(1).unsqueeze(-1)

            joint_mask = torch.cat([window_mask, inter_chunk_mask.reshape(*window_mask.shape)], dim=-1)
            attention_mask = (joint_mask, intra_chunk_mask)
        else:
            joint_mask = torch.cat([window_causal_mask, chunk_causal_mask.reshape(*window_causal_mask.shape)], dim=-1)
            attention_mask = (joint_mask, None)
        return attention_mask

    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            return_all_pred_logits: Optional[bool] = None,
            multibyte_decoding: Optional[bool] = None) -> Union[Tuple, CausalLMOutputWithPast]:

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (output_hidden_states
                                if output_hidden_states is not None else self.config.output_hidden_states)
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is None:
            assert past_key_values is None

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
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
            multibyte_decoding=multibyte_decoding,
        )

        hidden_states = outputs[0]

        logits = self.lm_head(hidden_states)
        if self.config.fp32_logits:
            logits = logits.float()

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss(reduction="none")
            if hasattr(self.config, "num_pred_heads") and self.config.num_pred_heads > 1:
                shift_logits = logits.view(logits.shape[0], logits.shape[1], self.config.num_pred_heads, self.config.vocab_size)
                # shift_logits = shift_logits.view(-1, logits.shape[1] * self.config.num_pred_heads, self.config.vocab_size)
                shift_logits = shift_logits.view(-1, self.config.vocab_size)
            else:
                shift_logits = logits.view(-1, self.config.vocab_size)
            shift_labels = labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if hasattr(self.config, "num_pred_heads") and self.config.num_pred_heads > 1:
            all_pred_logits = logits.reshape(logits.shape[0], logits.shape[1], self.config.num_pred_heads, self.config.vocab_size)
            
            if return_all_pred_logits:
                logits = all_pred_logits
            else:
                logits = all_pred_logits[..., 0, :]

        if not return_dict:
            output = (logits, ) + outputs[1:]
            return (loss, ) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


    def prepare_inputs_for_generation(self,
                                      input_ids,
                                      past_key_values=None,
                                      attention_mask=None,
                                      inputs_embeds=None,
                                      use_cache=True,
                                      **kwargs):
        # prefill phase:
        # input_ids:      b x s
        # attention_mask: None if no padding or b x s
        # position_ids :  b x s

        # token gen phase:
        # input_ids : b x 1
        # attention_mask: b x 1 x s
        # position_ids:  b x 1
        past_length = 0
        if past_key_values is not None:
            assert isinstance(past_key_values, Cache)
            past_length = past_key_values.get_seq_length()

            if attention_mask is not None and attention_mask.shape[1] > input_ids.shape[1]:
                input_ids = input_ids[:, -(attention_mask.shape[1] - past_length):]
            elif past_length < input_ids.shape[1]:
                input_ids = input_ids[:, past_length:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1]:]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        # must initialize position_ids at each step during GPU inference
        assert position_ids is not None
        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": use_cache,
                "attention_mask": attention_mask,
            }
        )
        return model_inputs

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (tuple(
                past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past), )
        return reordered_past
