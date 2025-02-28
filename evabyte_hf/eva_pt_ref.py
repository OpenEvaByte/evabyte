from typing import Optional, Tuple, Union
import torch
from torch import nn

MASK_MIN_VALUE = -10e10

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

def attention_op(
        q,
        k,
        v,
        attn_mask,
        mixedp_attn,
        head_dim_scaling
    ):
    attn = torch.matmul(q, k.transpose(-2, -1))
    if mixedp_attn:
        attn = attn.to(torch.float)
    attn = attn * head_dim_scaling
    if attn_mask is not None:
        attn = attn.masked_fill(attn_mask, MASK_MIN_VALUE)
    
    attn_weights = torch.softmax(attn, dim=-1).to(q.dtype)
    attn_output = torch.matmul(attn_weights, v)
    return attn_output

def prm_projection(
    x: torch.Tensor,
    projection_matrix: torch.Tensor,
    mixedp_attn: bool = False
    ):
    """
    Constructs nonnegative kernel features for fast softmax attention.
    Args:
    x: input for which features are computed
    projection_matrix: random matrix used to compute features
    Returns:
    Random features for fast attention.
    """
    # x : [..., m, d]
    # proj : [..., r, d]
    scaling_factor = (x.shape[-1] ** -0.5)
    proj_x = torch.matmul(projection_matrix, x.transpose(-1, -2)) # [..., r, m]
    norm = torch.sum(x ** 2, dim=-1).unsqueeze(-2) * 0.5 # [..., 1]
    if mixedp_attn:
        proj_x = proj_x.to(torch.float)
        norm = norm.to(torch.float)
    phi_x =  scaling_factor * (proj_x - norm)
    return phi_x

class EvaAttention(nn.Module):
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
        self.random_feature_dim = 1
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

    def _generate_feature_map(self, rf_q, rf_k, rf_v):
        rf_k_logits = torch.sum(self.adaptive_mu_k.to(rf_k.dtype) * rf_k, dim=-1, keepdim=True) # b h c m 1
        if self.config.mixedp_attn:
            rf_k_logits = rf_k_logits.to(torch.float)
        rf_k_weights = torch.softmax(rf_k_logits, dim=-2).to(rf_k.dtype)
        rf_k_bar = torch.sum(rf_k_weights * rf_k, dim=-2)
        weights = self.adaptive_phi.to(rf_k.dtype)
        return weights, rf_k_bar

    def _calculate_chunk_rfa_cache(self, rf_q, rf_k, rf_v, weights, rf_mask=None):
        proj_x = torch.sum(weights * rf_k, dim=-1, keepdim=True)
        norm = torch.sum(rf_k ** 2, dim=-1, keepdim=True) * 0.5 # [..., 1]
        if self.config.mixedp_attn:
            proj_x = proj_x.to(torch.float)
            norm = norm.to(torch.float)
        log_phi_k = self.head_dim_scaling * (proj_x - norm)

        if rf_mask is not None:
            log_phi_k = log_phi_k.masked_fill(rf_mask, MASK_MIN_VALUE)

        # [b, h, c, m, r]
        softmax_phi_k = torch.softmax(log_phi_k, dim=-2).to(rf_k.dtype)
        softmax_phi_k_v = torch.sum(softmax_phi_k * rf_v, dim=-2)
        # [b, h, c, r, m] [b, h, c, m, d] -> [b, h, c, r, d]
        # softmax_phi_k_v = torch.matmul(softmax_phi_k.transpose(-1, -2), rf_v).squeeze(-2)
        log_sum_phi_k = None
        return softmax_phi_k_v, log_sum_phi_k

    def _calculate_chunk_rfa(self, q, softmax_phi_k_v, log_sum_phi_k, weights):
        if self.random_feature_dim == 1:
            # when r = 1, the snis weights becomes 1, so this takes no effect 
            # [b, h, c, r, d] -> [b, h, c, d]
            return softmax_phi_k_v
        else:
            # [b, h, c, r, d] [b, h, 1, s, d] -> [b, h, c, r, s]
            log_phi_q = prm_projection(q.unsqueeze(-3), weights, self.config.mixedp_attn)
            # [b, h, c, r, s] [b, h, c, r, 1] -> [b, h, c, r, s]
            sniw = torch.softmax(log_phi_q + log_sum_phi_k, dim=-1).to(q.dtype)
            # [b, h, c, r, s] [b, h, c, r, d] -> [b, h, c, s, d] -> [b, h, s, c, d]
            rfa_per_chunk = torch.matmul(sniw.transpose(-1, -2), softmax_phi_k_v).transpose(-3, -2)
            return rfa_per_chunk

    def window_partition(self, x, window_size=None):
        window_size = window_size if window_size is not None else self.window_size

        gw, d = x.shape[-2:]
        leading_dims = x.shape[:-2]
        n_groups = gw // window_size
        return x.reshape(*leading_dims, n_groups, window_size, d)
    
    def window_merge(self, x, window_size=None):
        g, w, d = x.shape[-3:]
        leading_dims = x.shape[:-3]
        return x.reshape(*leading_dims, g * w, d)

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
        bsz, q_len, _ = hidden_states.size()

        ############################################
        # initialize past states if not provided
        ############################################
        if use_cache and past_key_value is None:
            raise ValueError
        if use_cache and multibyte_decoding:
            raise NotImplementedError("Multibyte decoding is not supported for PyTorch native implementation")
        # assert isinstance(attention_mask, tuple)
        if len(attention_mask) == 4:
            assert use_cache
            prev_causal_mask, cur_causal_mask, chunk_causal_mask, intra_chunk_mask = attention_mask
        elif len(attention_mask) == 3:
            assert not use_cache
            window_causal_mask, chunk_causal_mask, intra_chunk_mask = attention_mask
        else:
            raise NotImplementedError("Only attention-mask tuple with length 2 or 3 is supported")

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
        # compute q, k, v stats for the local window
        ############################################
        if use_cache:
            (prev_w_q, prev_w_k, prev_w_v), (cur_w_q, cur_w_k, cur_w_v) = past_key_value.update_singletons(
                q, 
                k, 
                v, 
                self.layer_idx,
                self.window_size,
            )
        else:
            prev_w_q = self.window_partition(q) # [b, h, w, i, d]
            prev_w_k = self.window_partition(k) # [b, h, w, j, d]
            prev_w_v = self.window_partition(v) # [b, h, w, j, d]
            # during training, we assume window_size divides seq_len so no remainders
            cur_w_q = cur_w_k = cur_w_v = None

        ############################################
        # compute q, k, v stats for chunk-level RFAs
        ############################################
        if use_cache:
            dump_q, dump_k, dump_v = past_key_value.update_chunks(q, k, v, self.layer_idx, self.chunk_size)
        else:
            dump_q, dump_k, dump_v = q, k, v

        if use_cache:
            prev_s_mask, cur_s_mask, prev_chunk_mask, cur_chunk_mask, dump_rf_mask = past_key_value.update_mask(
                prev_s_mask=prev_causal_mask,
                cur_s_mask=cur_causal_mask,
                chunk_mask=chunk_causal_mask,
                rf_mask=intra_chunk_mask,
                layer_idx=self.layer_idx,
                window_size=self.window_size,
                chunk_size=self.chunk_size,
            )
        else:
            prev_s_mask = self.window_partition(prev_causal_mask) # [1, 1, w, i, j]
            cur_s_mask = None
            prev_chunk_mask = self.window_partition(chunk_causal_mask)
            cur_chunk_mask = None
            dump_rf_mask = intra_chunk_mask
            if prev_s_mask.shape[-3] == 1:
                # need to expand
                prev_s_mask = prev_s_mask.expand(-1, -1, prev_chunk_mask.shape[-3], -1, -1)

        if (
            dump_q is not None and
            dump_k is not None and
            dump_v is not None
        ):
            # [b, h, c, j, d]
            rf_q = self.window_partition(dump_q, window_size=self.chunk_size)
            # [b, h, c, j, d]
            rf_k = self.window_partition(dump_k, window_size=self.chunk_size)
            # [b, h, c, j, d]
            rf_v = self.window_partition(dump_v, window_size=self.chunk_size)

            if dump_rf_mask is not None:
                rf_mask = self.window_partition(dump_rf_mask, window_size=self.chunk_size)
                rf_q = rf_q.masked_fill(rf_mask, 0.)
                rf_k = rf_k.masked_fill(rf_mask, 0.)
                rf_v = rf_v.masked_fill(rf_mask, 0.)
            else:
                rf_mask = None
        else:
            rf_q = None
            rf_k = None
            rf_v = None
            rf_mask = None


        if rf_q is not None:
            # import pdb; pdb.set_trace()
            weights, rf_k_bar = self._generate_feature_map(rf_q, rf_k, rf_v)
            softmax_phi_k_v, log_sum_phi_k = self._calculate_chunk_rfa_cache(rf_q, rf_k, rf_v, weights, rf_mask=rf_mask)
            if use_cache:
                softmax_phi_k_v, log_sum_phi_k, rf_k_bar = past_key_value.update_chunk_rfas(
                    softmax_phi_k_v, log_sum_phi_k, rf_k_bar, self.layer_idx, 1
                )
        elif use_cache:
            weights = None
            softmax_phi_k_v, log_sum_phi_k, rf_k_bar = past_key_value.get_chunk_rfas(self.layer_idx)
        else:
            weights = None
            softmax_phi_k_v = None
            log_sum_phi_k = None
            rf_k_bar = None

        if rf_k_bar is not None:
            rfa_per_chunk = self._calculate_chunk_rfa(q, softmax_phi_k_v, log_sum_phi_k, weights)
        ############################################
        # compute meta-attention weights for 
        # - group-wise RFAs and 
        # - singletons (equivalent to exact local attention)
        ############################################
        if prev_w_k is not None:
            if rf_k_bar is not None:
                num_windows = prev_w_k.shape[-3]
                # rf_k_bar and rfa_per_chunk take the shape [b, h, c, d]
                # -> [b, h, 1, c, d] -> [b, h, w, c, d]
                prev_rf_k_bar = rf_k_bar.unsqueeze(-3).expand(-1, -1, num_windows, -1, -1)
                prev_rfa_per_chunk = rfa_per_chunk.unsqueeze(-3).expand(-1, -1, num_windows, -1, -1)
                prev_agg_k = torch.cat([prev_w_k, prev_rf_k_bar], dim=-2)
                prev_agg_v = torch.cat([prev_w_v, prev_rfa_per_chunk], dim=-2)

                prev_attn_mask = torch.cat([prev_s_mask, prev_chunk_mask], dim=-1)
            else:
                prev_agg_k = prev_w_k
                prev_agg_v = prev_w_v
                prev_attn_mask = prev_s_mask

            prev_attn_output = attention_op(
                q=prev_w_q,
                k=prev_agg_k,
                v=prev_agg_v,
                attn_mask=prev_attn_mask,
                mixedp_attn=self.config.mixedp_attn,
                head_dim_scaling=self.head_dim_scaling
            )
            prev_attn_output = self.window_merge(prev_attn_output)

        if cur_w_k is not None:
            if rf_k_bar is not None:
                # rf_k_bar and rfa_per_chunk take the shape [b, h, c, d]
                # cur_w_k and cur_w_v also has shape [b, h, m, d]
                cur_agg_k = torch.cat([cur_w_k, rf_k_bar], dim=-2)
                cur_agg_v = torch.cat([cur_w_v, rfa_per_chunk], dim=-2)

                cur_attn_mask = torch.cat([cur_s_mask, cur_chunk_mask], dim=-1)
            else:
                cur_agg_k = cur_w_k
                cur_agg_v = cur_w_v
                cur_attn_mask = cur_s_mask

            cur_attn_output = attention_op(
                q=cur_w_q,
                k=cur_agg_k,
                v=cur_agg_v,
                attn_mask=cur_attn_mask,
                mixedp_attn=self.config.mixedp_attn,
                head_dim_scaling=self.head_dim_scaling
            )

        if prev_w_k is not None and cur_w_k is not None:
            attn_output = torch.cat([prev_attn_output, cur_attn_output], dim=-2)
        elif prev_w_k is not None:
            attn_output = prev_attn_output
        elif cur_w_k is not None:
            attn_output = cur_attn_output
        else:
            raise ValueError("There must be some bug")

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).reshape(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)

        attn_weights = None

        return attn_output, attn_weights, past_key_value