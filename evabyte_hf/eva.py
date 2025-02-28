from typing import Dict, Optional, Tuple, List, Any, Union
import torch
from torch import nn
import torch.nn.functional as F
from .eva_agg_kernel import eva_agg_func_triton
from .eva_prep_kv_kernel import eva_prep_kv_func_triton
try:
    import triton
    USE_TRITON_IMPL = True
except ImportError:
    USE_TRITON_IMPL = False
    raise ImportError("Triton is not installed. Please install it by running `pip install triton`.")

def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """
    Rotates half the hidden dims (last dim) of the input.
    Args:
        x: Rotary embedded tensor
    Return:
        Tensor with half of last dim negated and rotated to the front.
    """
    x1, x2 = x.split(x.shape[-1] // 2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
                         position_ids: torch.Tensor) -> torch.Tensor:
    """
    Apply rotary embedding (cos, sin) to the query and key tensor on the sequence dimension.

    The legends for dimensions are defined as:
    num_heads: number of attention heads
    current_seq_len: the current batch's sequence length, should be either 1 or max_seq_len
    max_seq_len: the static sequence length, different from current_seq_len in cached inference case where it is always
                 maximum lenghth, e.g. the length of static sequence length of KV cache

                 
    Args:
        q: Query tensor, of size (batch_size, num_heads, current_seq_len, head_dim)
        k: Key tensor, of size (batch_size, num_key_value_heads, current_seq_len, head_dim)
        cos: Cosine base of rotary embedding, of size (max_seq_len, head_dim)
        sin: Sine base of rotary embedding, of size (max_seq_len, head_dim)
        position_ids: The position indices of the tokens corresponding to the query and key tensors. It has a size of
                      (batch_size, current_seq_len).

    Returns:
        Embedded query and key tensor of same size as input.
    
    """
    bs, nheads, cur_seq_len, head_dim = q.shape
    assert len(
        k.shape) == 4, f"k should be of shape (batch_size, num_heads, current_seq_len, head_dim), got {k.shape} instead"
    assert k.shape[0] == bs, f"k has a different batch_size {k.shape[0]} compared to q {bs}"
    assert list(k.shape[2:]) == [cur_seq_len,
                                 head_dim], f"k has different current_seq_len and/or head_dim compared to q"
    assert cos.shape[3] == head_dim, f"cos should have dim of head dim {head_dim}, got {cos.shape[3]} instead"
    assert list(position_ids.shape) in [[bs, cur_seq_len], [1, cur_seq_len]],\
            f"position_ids should be of shape {[bs, cur_seq_len]} or {[1, cur_seq_len]}, got {position_ids.shape} instead"

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

class EvaAttention(nn.Module):
    """
        Causal EVA for language modeling.
    """

    def __init__(self, config, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.head_dim_scaling = self.head_dim ** -0.5

        self.max_position_embeddings = config.max_position_embeddings

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        self.window_size = config.window_size
        
        self.num_chunks = config.num_chunks
        self.chunk_size = config.chunk_size
        if self.chunk_size is not None:
            assert self.window_size >= self.chunk_size and self.window_size % self.chunk_size == 0
            # chunk_size overrides the number of landmarks
            self.num_chunks = None

        self.chunks_per_window = int(self.window_size // self.chunk_size)
        self.adaptive_phi = nn.Parameter(
            torch.randn(
                1,
                self.num_heads,
                1,
                1,
                self.head_dim
            ).clamp(-1., 1.) * self.head_dim_scaling
        )
        self.adaptive_mu_k = nn.Parameter(
            torch.randn(
                1,
                self.num_heads,
                1,
                1,
                self.head_dim
            ).clamp(-1., 1.) * self.head_dim_scaling
        )

    def _triton_forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cos: Optional[torch.Tensor] = None,
        sin: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        assert not output_attentions
        bsz, q_len, _ = hidden_states.size()

        if use_cache:
            if past_key_value is None:
                raise ValueError
            assert isinstance(attention_mask, tuple)

        # infer the model's running mode
        is_prefilling = use_cache and past_key_value.get_seq_length(self.layer_idx) == 0
        is_decoding = use_cache and past_key_value.get_seq_length(self.layer_idx) > 0

        if is_prefilling:
            assert len(attention_mask) == 2
            window_mask, intra_chunk_mask = attention_mask
            chunk_mask = None
        elif is_decoding:
            assert len(attention_mask) == 3
            window_mask, intra_chunk_mask, chunk_mask = attention_mask
        else:
            if attention_mask is not None:
                assert isinstance(attention_mask, tuple) and len(attention_mask) == 3
                window_mask, chunk_mask, intra_chunk_mask = attention_mask
            else:
                window_mask, chunk_mask, intra_chunk_mask = None, None, None

        ############################################
        # compute q, k, v from hidden states
        ############################################
        # [b, h, q_len, d]
        q = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        # [b, h, kv_len, d]
        k = self.k_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        # [b, h, kv_len, d]
        v = self.v_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)

        if use_cache:
            past_key_value.update_past_len(q.shape[-2], self.layer_idx)

        ############################################
        # apply rotary positional embeddings to q, k
        ############################################
        q, k = apply_rotary_pos_emb(q, k, cos, sin, position_ids)

        ############################################
        # update and get cached singleton tokens
        # update and cache k and v for calculating chunk-level RFAs
        ############################################
        if use_cache:
            s_k, s_v, dump_k, dump_v = past_key_value.update_singletons_and_chunks(
                k, 
                v, 
                self.layer_idx,
                self.window_size,
            )
        else:
            s_k, s_v = k, v
            dump_k, dump_v = k, v

        if use_cache:
            singleton_mask, dump_rf_mask = past_key_value.update_mask(
                s_mask=window_mask,
                rf_mask=intra_chunk_mask,
                layer_idx=self.layer_idx,
                window_size=self.window_size,
            )
        else:
            singleton_mask = window_mask
            dump_rf_mask = intra_chunk_mask

        if dump_k is not None and dump_v is not None:
            # 1. in prefilling, the input shape is 
            #   dump_k/dump_v: [b, h, n, d]
            #   rfa_k/rfa_v: [b, h, n // c, d]
            # 2. in decoding, the input shape is 
            #   k/v: [b, h, w, d]
            #   rfa_k/rfa_v: [b, h, w//c, d]
            # 3. in forward inference; the seq_len is already divisible
            rfa_k, rfa_v = eva_prep_kv_func_triton(
                dump_k, dump_v, 
                self.adaptive_mu_k, self.adaptive_phi, 
                dump_rf_mask, self.head_dim_scaling, self.chunk_size
            )
            # rfa_mask = get_rfa_chunk_mask(dump_rf_mask)
            if use_cache:
                rfa_k, rfa_v = past_key_value.update_chunk_rfas(
                    rfa_k, rfa_v, self.layer_idx
                )
        elif use_cache:
            # if there are not enough elements within the last chunk,
            # we will only use the cached chunk-level RFAs
            rfa_k, rfa_v = past_key_value.get_chunk_rfas(self.layer_idx)
        else:
            rfa_k, rfa_v = None, None

        ############################################
        # compute the full attention output
        ############################################
        if is_prefilling:
            # prefilling
            # 1. in prefilling, the input shape is 
            #   q: [b, h, n, d]
            #   k/v: [b, h, n, d]
            #   rfa_k/rfa_v: [b, h, n // c, d]
            attn_output = eva_agg_func_triton(
                q, s_k, s_v, 
                rfa_k, rfa_v, 
                singleton_mask, chunk_mask,
                self.head_dim_scaling, self.window_size, self.chunks_per_window
            )
        elif is_decoding:
            # 2. in decoding, the input shape is 
            #   q: [b, h, 1, d] or [b, h, z, d] (for multi-byte prediction)
            #   k/v: [b, h, 1 + s, d]
            #   rfa_k/rfa_v: [b, h, n // c, d]
            if rfa_k is not None and rfa_v is not None:
                # we only take the chunk-level RFAs not in the current window
                seen_seq_len = past_key_value.get_seq_length(self.layer_idx)
                if seen_seq_len <= self.window_size:
                    agg_k = s_k
                    agg_v = s_v
                    attn_mask = singleton_mask
                else:
                    # NOTE: we already updated the cache so the length now 
                    # includes the current token 
                    # we subtract 1 from seen_seq_len because we want
                    # if seen_seq_len = 2048 -> num_windows_seen_so_far = 0
                    # if seen_seq_len = 4096 -> num_windows_seen_so_far = 1
                    # if seen_seq_len = 4097 -> num_windows_seen_so_far = 2
                    # NOTE the cat order should be taken care of;
                    # should align with the order based on which 
                    # the attention mask is constructed
                    num_windows_seen_so_far = (seen_seq_len - 1) // self.window_size
                    agg_k = torch.cat([s_k, rfa_k[..., :num_windows_seen_so_far * self.chunks_per_window, :]], dim=-2)
                    agg_v = torch.cat([s_v, rfa_v[..., :num_windows_seen_so_far * self.chunks_per_window, :]], dim=-2)
                    if singleton_mask is not None:
                        assert chunk_mask is not None
                        attn_mask = torch.cat([singleton_mask, chunk_mask], dim=-1)
                    else:
                        attn_mask = singleton_mask
            else:
                agg_k = s_k
                agg_v = s_v
                attn_mask = singleton_mask
            attn_output = F.scaled_dot_product_attention(
                q, agg_k, agg_v, 
                attn_mask=attn_mask,
                is_causal=False,
                dropout_p=0.0, 
                scale=self.head_dim_scaling
            )
        else:
            # 3. in single-forward inference
            attn_output = eva_agg_func_triton(
                q, s_k, s_v, 
                rfa_k, rfa_v, 
                singleton_mask, chunk_mask,
                self.head_dim_scaling, self.window_size, self.chunks_per_window
            )
        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )
        attn_output = attn_output.transpose(1, 2).reshape(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)
        attn_weights = None
        return attn_output, attn_weights, past_key_value

    def _multibyte_decoding_forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cos: Optional[torch.Tensor] = None,
        sin: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        # during multi-byte forwarding, we only read caches and do not update them
        assert not output_attentions
        bsz, q_len, _ = hidden_states.size()

        if use_cache and past_key_value is None:
            raise ValueError

        assert USE_TRITON_IMPL
        assert isinstance(attention_mask, torch.Tensor) and attention_mask.dtype == torch.bool

        assert use_cache and past_key_value.get_seq_length(self.layer_idx) > 0

        ############################################
        # compute q, k, v from hidden states
        ############################################
        # [b, h, q_len, d]
        q = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        # [b, h, kv_len, d]
        k = self.k_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        # [b, h, kv_len, d]
        v = self.v_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)

        ############################################
        # apply rotary positional embeddings to q, k
        ############################################
        q, k = apply_rotary_pos_emb(q, k, cos, sin, position_ids)

        ############################################
        # update and get cached singleton tokens
        ############################################
        input_len = k.shape[-2]
        window_pos = past_key_value.past_window_pos[self.layer_idx]
        new_window_pos = window_pos + input_len

        past_key_value.past_window_k[self.layer_idx][:, :, window_pos : new_window_pos, :] = k
        past_key_value.past_window_v[self.layer_idx][:, :, window_pos : new_window_pos, :] = v
        s_k = past_key_value.past_window_k[self.layer_idx][:, :, : new_window_pos, :]
        s_v = past_key_value.past_window_v[self.layer_idx][:, :, : new_window_pos, :]

        rfa_k, rfa_v = past_key_value.get_chunk_rfas(self.layer_idx)

        ############################################
        # compute the full attention output
        ############################################
        # 2. in decoding, the input shape is 
        #   q: [b, h, 1, d] or [b, h, z, d] (for multi-byte prediction)
        #   k/v: [b, h, 1 + s, d]
        #   rfa_k/rfa_v: [b, h, n // c, d]
        if rfa_k is not None and rfa_v is not None:
            # NOTE the cat order should be taken care of;
            # should align with the order based on which 
            # the attention mask is constructed
            # agg_k = torch.cat([s_k, rfa_k], dim=-2)
            # agg_v = torch.cat([s_v, rfa_v], dim=-2)
            agg_k = torch.cat([rfa_k, s_k], dim=-2)
            agg_v = torch.cat([rfa_v, s_v], dim=-2)
        else:
            agg_k = s_k
            agg_v = s_v
        attn_output = F.scaled_dot_product_attention(
            q, agg_k, agg_v, 
            attn_mask=attention_mask,
            is_causal=False,
            dropout_p=0.0, 
            scale=self.head_dim_scaling
        )

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )
        attn_output = attn_output.transpose(1, 2).reshape(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)
        attn_weights = None
        return attn_output, attn_weights, past_key_value

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cos: Optional[torch.Tensor] = None,
        sin: Optional[torch.Tensor] = None,
        multibyte_decoding: Optional[bool] = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        assert not output_attentions
        if use_cache and past_key_value is None:
            raise ValueError

        assert USE_TRITON_IMPL
        if use_cache and multibyte_decoding:
            return self._multibyte_decoding_forward(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cos=cos,
                sin=sin,
            )
        else:
            return self._triton_forward(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cos=cos,
                sin=sin,
            )
