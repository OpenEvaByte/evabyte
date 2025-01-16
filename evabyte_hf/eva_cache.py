from typing import Dict, Optional, Tuple, List, Any, Union
import torch
from transformers.cache_utils import Cache

class EvaCache(Cache):
    """
    A cache that grows dynamically as more tokens are generated. This is the default for generative models.

    It stores the Key and Value states as a list of tensors, one for each layer. The expected shape for each tensor is
    `[batch_size, num_heads, seq_len, head_dim]`.
    """

    def __init__(self) -> None:
        self.w_k: List[torch.Tensor] = []
        self.w_v: List[torch.Tensor] = []

        self.rf_q: List[torch.Tensor] = []
        self.rf_k: List[torch.Tensor] = []
        self.rf_v: List[torch.Tensor] = []

        self.softmax_phi_k_v: List[torch.Tensor] = []
        self.log_sum_phi_k: List[torch.Tensor] = []
        self.rf_k_bar: List[torch.Tensor] = []
        self._seen_tokens = 0  # Used in `generate` to keep tally of how many tokens the cache has seen

        # attention masks temporary buffer
        self.rf_mask: List[Optional[torch.Tensor]] = []
        self.s_mask: List[torch.Tensor] = []
        self.chunk_mask: List[torch.Tensor] = []

    def __len__(self):
        """
        Support for backwards-compatible `past_key_value` length, e.g. `len(past_key_value)`. This value corresponds
        to the number of layers in the model.
        """
        return len(self.w_k)

    def get_usable_length(self, new_seq_length: int, layer_idx: Optional[int] = 0) -> int:
        """Given the sequence length of the new inputs, returns the usable length of the cache."""
        # Cache without size limit -> all cache is usable
        # Cache with size limit -> if the length cache plus the length of the new inputs is larger the maximum cache
        #   length, we will need to evict part of the cache (and thus not all cache is usable)
        max_length = self.get_max_length()
        previous_seq_length = self.get_seq_length(layer_idx)
        if max_length is not None and previous_seq_length + new_seq_length > max_length:
            return max_length - new_seq_length
        return previous_seq_length

    def reorder_cache(self, beam_idx: torch.LongTensor):
        """Reorders the cache for beam search, given the selected beam indices."""
        for layer_idx in range(len(self.w_k)):
            device = self.w_k[layer_idx].device
            self.w_k[layer_idx] = self.w_k[layer_idx].index_select(0, beam_idx.to(device))

            device = self.w_v[layer_idx].device
            self.w_v[layer_idx] = self.w_v[layer_idx].index_select(0, beam_idx.to(device))

            device = self.rf_q[layer_idx].device
            self.rf_q[layer_idx] = self.rf_q[layer_idx].index_select(0, beam_idx.to(device))

            device = self.rf_k[layer_idx].device
            self.rf_k[layer_idx] = self.rf_k[layer_idx].index_select(0, beam_idx.to(device))

            device = self.rf_v[layer_idx].device
            self.rf_v[layer_idx] = self.rf_v[layer_idx].index_select(0, beam_idx.to(device))

            device = self.softmax_phi_k_v[layer_idx].device
            self.softmax_phi_k_v[layer_idx] = self.softmax_phi_k_v[layer_idx].index_select(0, beam_idx.to(device))

            device = self.log_sum_phi_k[layer_idx].device
            self.log_sum_phi_k[layer_idx] = self.log_sum_phi_k[layer_idx].index_select(0, beam_idx.to(device))

            device = self.rf_k_bar[layer_idx].device
            self.rf_k_bar[layer_idx] = self.rf_k_bar[layer_idx].index_select(0, beam_idx.to(device))

            device = self.rf_mask[layer_idx].device
            self.rf_mask[layer_idx] = self.rf_mask[layer_idx].index_select(0, beam_idx.to(device))

            device = self.s_mask[layer_idx].device
            self.s_mask[layer_idx] = self.s_mask[layer_idx].index_select(0, beam_idx.to(device))

            device = self.chunk_mask[layer_idx].device
            self.chunk_mask[layer_idx] = self.chunk_mask[layer_idx].index_select(0, beam_idx.to(device))
    @property
    def seen_tokens(self):
        if hasattr(self, "_seen_tokens"):
            return self._seen_tokens
        else:
            return None

    def update_past_len(
        self,
        cur_q_len: int,
        layer_idx: int
    ):
        # Update the number of seen tokens
        if layer_idx == 0:
            self._seen_tokens += cur_q_len
        return self._seen_tokens

    def update_mask(
            self,
            prev_s_mask,
            cur_s_mask,
            chunk_mask,
            rf_mask,
            layer_idx,
            window_size,
            chunk_size,
    ):
        ############################################
        # compute masks for singletons
        ############################################
        q_len = None
        if len(self.s_mask) <= layer_idx:
            q_len = chunk_mask.shape[-2]
            # prefill stage
            # q is of shape [b, h, n, d]
            if q_len < window_size:
                assert prev_s_mask is None

            # w_v =  # [b, h, 1, j, d]
            # store the past window-wise key-value pairs
            self.s_mask.append(cur_s_mask[..., -1:, :] if cur_s_mask is not None else prev_s_mask[..., -1, -1:, :])
        else:
            # decoding stage
            prev_s_mask = None

            cached_s_mask = self.s_mask[layer_idx]
            assert cached_s_mask is not None
            if cached_s_mask.shape[-1] == window_size:
                cur_s_mask = cur_s_mask
            else:
                cur_s_mask = torch.cat([cached_s_mask, cur_s_mask], dim=-1)

            # store the past window-wise key-value pairs
            self.s_mask[layer_idx] = cur_s_mask

        ############################################
        # compute masks for intra-chunks
        ############################################
        dump_rf_mask = None
        if len(self.rf_mask) <= layer_idx:
            # initialize chunk stats
            # prefill stage
            if q_len < chunk_size:
                cur_rf_mask = rf_mask
            else:
                if q_len % chunk_size == 0:
                    dump_rf_mask = rf_mask
                    cur_rf_mask = None
                else:
                    remainder_tokens = q_len % chunk_size
                    if rf_mask is not None:
                        dump_rf_mask, cur_rf_mask = torch.split(rf_mask, [q_len - remainder_tokens, remainder_tokens], dim=-2)
                    else:
                        dump_rf_mask = None
                        cur_rf_mask = None
            self.rf_mask.append(cur_rf_mask)
        else:
            past_rf_mask = self.rf_mask[layer_idx]
            if past_rf_mask is not None:
                # when decoding tokens, we always assume the 
                # incoming token mask is 0 (not masked)
                cur_rf_mask = torch.cat([past_rf_mask, rf_mask], dim=-2)
            else:
                # we do not need to use rf_mask anymore after we receive generated tokens
                cur_rf_mask = None
            # We need to store rf_k_bar and RFA-results that 
            # compute the per-chunk RFA.
            
            # Dump the chunk if the len of current chunk reaches <chunk_size>.
            if cur_rf_mask is not None and cur_rf_mask.shape[-2] == chunk_size:
                dump_rf_mask = cur_rf_mask
                cur_rf_mask = None

            self.rf_mask[layer_idx] = cur_rf_mask

        ############################################
        # compute masks for inter chunks
        ############################################
        if len(self.chunk_mask) <= layer_idx:
            # prefill stage
            # q is of shape [b, h, n, d]
            if q_len < window_size:
                cur_chunk_mask = chunk_mask
                prev_chunk_mask = None
            else:
                if q_len % window_size == 0:
                    cur_chunk_mask = None
                    prev_chunk_mask = chunk_mask
                else:
                    remainder_tokens = q_len % window_size
                    # [b, h, n-r, d] [b, h, r, d]
                    prev_chunk_mask, cur_chunk_mask = torch.split(chunk_mask, [q_len - remainder_tokens, remainder_tokens], dim=-2)
                bsz, num_heads, _, head_dim = prev_chunk_mask.shape
                prev_chunk_mask = prev_chunk_mask.reshape(bsz, num_heads, -1, window_size, head_dim)

                assert prev_s_mask is not None
                if prev_s_mask.shape[-3] == 1 and prev_chunk_mask.shape[-3] > 1:
                    # need to expand
                    prev_s_mask = prev_s_mask.expand(-1, -1, prev_chunk_mask.shape[-3], -1, -1)
            # w_v =  # [b, h, 1, j, d]
            # store the past window-wise key-value pairs
            self.chunk_mask.append(cur_chunk_mask[..., -1:, :] if cur_chunk_mask is not None else prev_chunk_mask[..., -1, -1:, :])
        else:
            # decoding stage
            prev_chunk_mask = None
            cur_chunk_mask = self.chunk_mask[layer_idx]

            # if the current sequence length reaches <chunk_size>,
            # we append a new 1 to the end of chunk_mask
            seen_seq_len = self.get_seq_length(layer_idx)
            if seen_seq_len > 0 and seen_seq_len % chunk_size == 0:
                past_chunk_mask = self.chunk_mask[layer_idx]
                if past_chunk_mask is not None:
                    # when decoding tokens, we always assume the 
                    # incoming token mask is 0 (not masked)
                    cur_chunk_mask = torch.cat([past_chunk_mask, chunk_mask], dim=-1)
                else:
                    cur_chunk_mask = chunk_mask
                self.chunk_mask[layer_idx] = cur_chunk_mask

            # if the len of current sequence reaches <window_size> + 1,
            # we turn on the mask for most recent chunks
            if seen_seq_len > 0 and seen_seq_len % window_size == 1:
                cur_chunk_mask = self.chunk_mask[layer_idx]
                # we do not need to use rf_mask anymore after we receive generated tokens
                num_chunks_per_window = window_size // chunk_size
                cur_chunk_mask[..., -num_chunks_per_window:] = False
                self.chunk_mask[layer_idx] = cur_chunk_mask

        return (prev_s_mask, cur_s_mask, prev_chunk_mask, cur_chunk_mask, dump_rf_mask)

    def update_singletons(
            self,
            q,
            k,
            v,
            layer_idx,
            window_size,
    ):
        if len(self.w_k) <= layer_idx:
            # prefill stage
            # q is of shape [b, h, n, d]
            q_len = q.shape[-2]
            if q_len < window_size:
                w_q = q
                w_k = k
                w_v = v
                past_w_q = past_w_k = past_w_v = None
            else:
                if q_len % window_size == 0:
                    w_q = None
                    w_k = None
                    w_v = None
                    past_w_q = q
                    past_w_k = k
                    past_w_v = v
                else:
                    remainder_tokens = q_len % window_size
                    # [b, h, n-r, d] [b, h, r, d]
                    past_w_q, w_q = torch.split(q, [q_len - remainder_tokens, remainder_tokens], dim=-2) 
                    past_w_k, w_k = torch.split(k, [q_len - remainder_tokens, remainder_tokens], dim=-2)
                    past_w_v, w_v = torch.split(v, [q_len - remainder_tokens, remainder_tokens], dim=-2)
                bsz, num_heads, _, head_dim = past_w_q.shape
                past_w_q = past_w_q.reshape(bsz, num_heads, -1, window_size, head_dim)
                past_w_k = past_w_k.reshape(bsz, num_heads, -1, window_size, head_dim)
                past_w_v = past_w_v.reshape(bsz, num_heads, -1, window_size, head_dim)
            # w_q = q[..., None, -window_size:, :] # [b, h, 1, j, d]
            # w_k =  # [b, h, 1, j, d]
            # w_v =  # [b, h, 1, j, d]
            # store the past window-wise key-value pairs
            # if w_k is None, it means we happen to pass in a sqeuence that is divisible by window_size
            # we leave the cache with window_size-sized kv pairs to be cleared next iteration
            self.w_k.append(w_k if w_k is not None else past_w_k[..., -1, :, :])
            self.w_v.append(w_v if w_v is not None else past_w_v[..., -1, :, :])
        else:
            # decoding stage
            past_w_q = past_w_k = past_w_v = None
            # this is implemented as either a sliding window or fixed window
            w_q = q # [b, h, 1, d]
            w_k = k # [b, h, 1, d]
            w_v = v # [b, h, 1, d]
            
            cached_w_k = self.w_k[layer_idx]
            assert cached_w_k is not None # [b, h, j, d]
            if cached_w_k.shape[-2] == window_size:
                w_k = w_k
            else:
                w_k = torch.cat([cached_w_k, w_k], dim=-2)
            
            cached_w_v = self.w_v[layer_idx]
            assert cached_w_v is not None
            if cached_w_v.shape[-2] == window_size:
                w_v = w_v
            else:
                w_v = torch.cat([cached_w_v, w_v], dim=-2)

            # store the past window-wise key-value pairs
            self.w_k[layer_idx] = w_k
            self.w_v[layer_idx] = w_v
        return (past_w_q, past_w_k, past_w_v), (w_q, w_k, w_v)

    def update_chunks(
            self,
            q,
            k,
            v,
            layer_idx,
            chunk_size
    ):
        q_len = q.shape[-2]
        dump_q = None
        dump_k = None
        dump_v = None
        if len(self.rf_q) <= layer_idx:
            # initialize chunk stats
            # prefill stage
            if q_len < chunk_size:
                rf_q = q
                rf_k = k
                rf_v = v
            else:
                if q_len % chunk_size == 0:
                    rf_q = None
                    rf_k = None
                    rf_v = None
                    dump_q = q
                    dump_k = k
                    dump_v = v
                else:
                    remainder_tokens = q_len % chunk_size
                    # [b, h, n-r, d] [b, h, r, d]
                    dump_q, rf_q = torch.split(q, [q_len - remainder_tokens, remainder_tokens], dim=-2) 
                    dump_k, rf_k = torch.split(k, [q_len - remainder_tokens, remainder_tokens], dim=-2)
                    dump_v, rf_v = torch.split(v, [q_len - remainder_tokens, remainder_tokens], dim=-2)
            self.rf_q.append(rf_q)
            self.rf_k.append(rf_k)
            self.rf_v.append(rf_v)
        else:
            # decode tokens
            # add query, key & value to the current chunk.
            past_rf_q = self.rf_q[layer_idx]
            if past_rf_q is not None:
                rf_q = torch.cat([past_rf_q, q], dim=-2)
            else:
                rf_q = q
        
            past_rf_k = self.rf_k[layer_idx]
            if past_rf_k is not None:
                rf_k = torch.cat([past_rf_k, k], dim=-2)
            else:
                rf_k = k
        
            past_rf_v = self.rf_v[layer_idx]
            if past_rf_v is not None:
                rf_v = torch.cat([past_rf_v, v], dim=-2)
            else:
                rf_v = v

            # We need to store rf_k_bar and RFA-results that 
            # compute the per-chunk RFA.
            
            # Dump the chunk if the len of current chunk reaches <chunk_size>.
            if rf_q.shape[-2] == chunk_size:
                dump_q = rf_q
                dump_k = rf_k
                dump_v = rf_v
                # clear the chunk
                rf_q = None
                rf_k = None
                rf_v = None
        
            self.rf_q[layer_idx] = rf_q
            self.rf_k[layer_idx] = rf_k
            self.rf_v[layer_idx] = rf_v

        return dump_q, dump_k, dump_v

    def update_chunk_rfas(
        self,
        softmax_phi_k_v,
        log_sum_phi_k,
        rf_k_bar,
        layer_idx,
        random_feature_dim
    ):
        if len(self.softmax_phi_k_v) <= layer_idx:
            # prefill stage
            self.softmax_phi_k_v.append(softmax_phi_k_v)
            self.log_sum_phi_k.append(log_sum_phi_k)
            self.rf_k_bar.append(rf_k_bar)
        else:
            # token decoding
            past_softmax_phi_k_v = self.softmax_phi_k_v[layer_idx]
            past_log_sum_phi_k = self.log_sum_phi_k[layer_idx]
            past_rf_k_bar = self.rf_k_bar[layer_idx]

            if past_softmax_phi_k_v is not None:
                if random_feature_dim == 1:
                    dim = -2
                else:
                    dim = -3
                softmax_phi_k_v = torch.cat([past_softmax_phi_k_v, softmax_phi_k_v], dim=dim)
            
            if past_log_sum_phi_k is not None:
                if random_feature_dim == 1:
                    dim = -2
                else:
                    dim = -3
                log_sum_phi_k = torch.cat([past_log_sum_phi_k, log_sum_phi_k], dim=dim)
            
            if past_rf_k_bar is not None:
                rf_k_bar = torch.cat([past_rf_k_bar, rf_k_bar], dim=-2)

            self.softmax_phi_k_v[layer_idx] = softmax_phi_k_v
            self.log_sum_phi_k[layer_idx] = log_sum_phi_k
            self.rf_k_bar[layer_idx] = rf_k_bar

        return softmax_phi_k_v, log_sum_phi_k, rf_k_bar

    def get_chunk_rfas(self, layer_idx):
        if len(self.softmax_phi_k_v) <= layer_idx:
            return (
                None, 
                None, 
                None
            )
        else:
            return (
                self.softmax_phi_k_v[layer_idx],
                self.log_sum_phi_k[layer_idx],
                self.rf_k_bar[layer_idx]
            ) 

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        """Returns the sequence length of the cached states. A layer index can be optionally passed."""
        if len(self.w_k) <= layer_idx:
            return 0
        return self._seen_tokens

    def get_max_length(self) -> Optional[int]:
        """Returns the maximum sequence length of the cached states. DynamicCache does not have a maximum length."""
        return None

    def update(
        self,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError("`update` is not used in Eva Cache.")

class EvaStaticCacheForTriton(Cache):
    """
    A variant of EvaCache for eva's triton kernels
    """

    def __init__(
        self, 
        batch_size,
        num_key_value_heads,
        window_size,
        head_dim,
        num_layers,
        dtype,
        device
    ) -> None:
        self.past_window_k: List[torch.Tensor] = []
        self.past_window_v: List[torch.Tensor] = []

        cache_shape = (batch_size, num_key_value_heads, window_size, head_dim)
        for idx in range(num_layers):
            new_window_k = torch.zeros(cache_shape, dtype=dtype, device=device)
            new_window_v = torch.zeros(cache_shape, dtype=dtype, device=device)
            self.past_window_k.append(new_window_k)
            self.past_window_v.append(new_window_v)

        self.past_window_pos: List[int] = []

        self.rfa_k: List[torch.Tensor] = []
        self.rfa_v: List[torch.Tensor] = []
        # self.rfa_mask: List[torch.Tensor] = []

        self._seen_tokens = 0  # Used in `generate` to keep tally of how many tokens the cache has seen

        # attention masks temporary buffer
        self.rf_mask: List[Optional[torch.Tensor]] = []
        self.s_mask: List[torch.Tensor] = []

    def __len__(self):
        """
        Support for backwards-compatible `past_key_value` length, e.g. `len(past_key_value)`. This value corresponds
        to the number of layers in the model.
        """
        return len(self.past_window_pos)

    def get_usable_length(self, new_seq_length: int, layer_idx: Optional[int] = 0) -> int:
        """Given the sequence length of the new inputs, returns the usable length of the cache."""
        # Cache without size limit -> all cache is usable
        # Cache with size limit -> if the length cache plus the length of the new inputs is larger the maximum cache
        #   length, we will need to evict part of the cache (and thus not all cache is usable)
        max_length = self.get_max_length()
        previous_seq_length = self.get_seq_length(layer_idx)
        if max_length is not None and previous_seq_length + new_seq_length > max_length:
            return max_length - new_seq_length
        return previous_seq_length

    def reorder_cache(self, beam_idx: torch.LongTensor):
        """Reorders the cache for beam search, given the selected beam indices."""
        for layer_idx in range(len(self.past_window_k)):
            device = self.past_window_k[layer_idx].device
            self.past_window_k[layer_idx] = self.past_window_k[layer_idx].index_select(0, beam_idx.to(device))

            device = self.past_window_v[layer_idx].device
            self.past_window_v[layer_idx] = self.past_window_v[layer_idx].index_select(0, beam_idx.to(device))

            device = self.rfa_k[layer_idx].device
            self.rfa_k[layer_idx] = self.rfa_k[layer_idx].index_select(0, beam_idx.to(device))

            device = self.rfa_v[layer_idx].device
            self.rfa_v[layer_idx] = self.rfa_v[layer_idx].index_select(0, beam_idx.to(device))

            # device = self.rfa_mask[layer_idx].device
            # self.rfa_mask[layer_idx] = self.rfa_mask[layer_idx].index_select(0, beam_idx.to(device))

            device = self.rf_mask[layer_idx].device
            self.rf_mask[layer_idx] = self.rf_mask[layer_idx].index_select(0, beam_idx.to(device))

            device = self.s_mask[layer_idx].device
            self.s_mask[layer_idx] = self.s_mask[layer_idx].index_select(0, beam_idx.to(device))

    @property
    def seen_tokens(self):
        if hasattr(self, "_seen_tokens"):
            return self._seen_tokens
        else:
            return None

    def update_past_len(
        self,
        cur_q_len: int,
        layer_idx: int
    ):
        # Update the number of seen tokens
        if layer_idx == 0:
            self._seen_tokens += cur_q_len
        return self._seen_tokens

    def update_mask(
            self,
            s_mask,
            rf_mask,
            layer_idx,
            window_size,
    ):
        ############################################
        # compute masks for singletons
        ############################################
        if len(self.s_mask) <= layer_idx:
            # prefill stage
            # q is of shape [b, h, n, d]
            # s_v =  # [b, h, 1, j, d]
            # store the past window-wise key-value pairs
            if s_mask is None:
                cur_s_mask = None
            else:
                q_len = s_mask.shape[-2]
                # s_mask is of shape [b, h, n, w]
                # let r = q_len % window_size
                # if r == 0, the mask to be appended is of shape [b, h, 1, w]
                # otherwise, r < w, the mask to be appended is of shape [b, h, 1, r]
                remainder_tokens = q_len % window_size
                if remainder_tokens == 0:
                    cur_s_mask = None
                else:
                    cur_s_mask = s_mask[..., -1:, :remainder_tokens]
            self.s_mask.append(cur_s_mask)
            # we use the passed s_mask for subsequent computations
            dump_s_mask = s_mask
        else:
            # decoding stage
            past_s_mask = self.s_mask[layer_idx]
            if past_s_mask is None:
                assert s_mask is None
                cur_s_mask = None
            else:
                assert s_mask is not None
                cur_s_mask = torch.cat([past_s_mask, s_mask], dim=-1)
                
            dump_s_mask = cur_s_mask
            if cur_s_mask is not None and cur_s_mask.shape[-1] == window_size:
                cur_s_mask = None
                # store the past window-wise key-value pairs
            self.s_mask[layer_idx] = cur_s_mask

        ############################################
        # compute masks for intra-chunks
        ############################################
        dump_rf_mask = None
        if len(self.rf_mask) <= layer_idx:
            # initialize chunk stats
            # prefill stage
            if rf_mask is None:
                cur_rf_mask = None
            else:
                q_len = rf_mask.shape[-2]
                if q_len < window_size:
                    dump_rf_mask = None
                    cur_rf_mask = rf_mask
                else:
                    if q_len % window_size == 0:
                        dump_rf_mask = rf_mask
                        cur_rf_mask = None
                    else:
                        remainder_tokens = q_len % window_size
                        dump_rf_mask, cur_rf_mask = torch.split(rf_mask, [q_len - remainder_tokens, remainder_tokens], dim=-2)
            self.rf_mask.append(cur_rf_mask)
        else:
            past_rf_mask = self.rf_mask[layer_idx]
            if past_rf_mask is not None:
                # when decoding tokens, we always assume the 
                # incoming token mask is 0 (not masked)
                cur_rf_mask = torch.cat([past_rf_mask, rf_mask], dim=-2)
            else:
                cur_rf_mask = None
            
            if cur_rf_mask is not None and cur_rf_mask.shape[-2] == window_size:
                dump_rf_mask = cur_rf_mask
                cur_rf_mask = None

            self.rf_mask[layer_idx] = cur_rf_mask

        return dump_s_mask, dump_rf_mask

    def update_singletons_and_chunks(
            self,
            k,
            v,
            layer_idx,
            window_size,
    ):
        if len(self.past_window_pos) <= layer_idx:
            # prefill stage
            s_k = k
            s_v = v
            input_len = k.shape[-2]
            window_pos = 0
            if input_len <= window_size:
                new_window_pos = window_pos + input_len

                cached_window_k = k
                cached_window_v = v
                dump_k = None
                dump_v = None
            else:
                remainder_tokens = input_len % window_size
                if remainder_tokens == 0:
                    remainder_tokens = window_size
                new_window_pos = window_pos + remainder_tokens

                # [b, h, n-r, d] [b, h, r, d]
                cached_window_k = k[..., -remainder_tokens:, :]
                cached_window_v = v[..., -remainder_tokens:, :]
                dump_k = k[..., :-remainder_tokens, :]
                dump_v = v[..., :-remainder_tokens, :]
            # store the past window-wise key-value pairs
            self.past_window_k[layer_idx][:, :, window_pos : new_window_pos, :] = cached_window_k
            self.past_window_v[layer_idx][:, :, window_pos : new_window_pos, :] = cached_window_v
            self.past_window_pos.append(new_window_pos)
        else:
            # decoding stage
            # if the previous cache has full tokens,
            # roll back to the first elements
            if self.past_window_pos[layer_idx] == window_size:
                self.past_window_pos[layer_idx] = 0
                dump_k = self.past_window_k[layer_idx].clone()
                dump_v = self.past_window_v[layer_idx].clone()
            else:
                dump_k = None
                dump_v = None

            input_len = k.shape[-2]
            window_pos = self.past_window_pos[layer_idx]
            new_window_pos = window_pos + input_len

            self.past_window_k[layer_idx][:, :, window_pos : new_window_pos, :] = k
            self.past_window_v[layer_idx][:, :, window_pos : new_window_pos, :] = v

            s_k = self.past_window_k[layer_idx][:, :, : new_window_pos, :]
            s_v = self.past_window_v[layer_idx][:, :, : new_window_pos, :]

            self.past_window_pos[layer_idx] = new_window_pos

        return s_k, s_v, dump_k, dump_v

    def update_chunk_rfas(
        self,
        rfa_k,
        rfa_v,
        layer_idx,
    ):
        if len(self.rfa_k) <= layer_idx:
            # prefill stage
            self.rfa_k.append(rfa_k)
            self.rfa_v.append(rfa_v)
        else:
            # token decoding
            past_rfa_k = self.rfa_k[layer_idx]
            past_rfa_v = self.rfa_v[layer_idx]

            if past_rfa_k is not None:
                rfa_k = torch.cat([past_rfa_k, rfa_k], dim=-2)
            
            if past_rfa_v is not None:
                rfa_v = torch.cat([past_rfa_v, rfa_v], dim=-2)
            
            self.rfa_k[layer_idx] = rfa_k
            self.rfa_v[layer_idx] = rfa_v

        return rfa_k, rfa_v

    def get_past_window_pos(self, layer_idx):
        if len(self.past_window_pos) <= layer_idx:
            return None
        else:
            return self.past_window_pos[layer_idx]

    def get_past_window_kv(self, layer_idx):
        if len(self.past_window_pos) <= layer_idx:
            return None, None
        else:
            return (
                self.past_window_k[layer_idx][:, :, : self.past_window_pos[layer_idx], :], 
                self.past_window_v[layer_idx][:, :, : self.past_window_pos[layer_idx], :]
            )

    def get_chunk_rfas(self, layer_idx):
        if len(self.rfa_k) <= layer_idx:
            return None, None
        else:
            return self.rfa_k[layer_idx], self.rfa_v[layer_idx]

    def get_seq_length(self, layer_idx = 0) -> int:
        """Returns the sequence length of the cached states. A layer index can be optionally passed."""
        # layer_idx must be provided since otherwise
        # any layer > 0 can only get the updated _seen_tokens
        if len(self.past_window_pos) <= layer_idx:
            return 0
        return self._seen_tokens

    def get_max_length(self) -> Optional[int]:
        """Returns the maximum sequence length of the cached states. DynamicCache does not have a maximum length."""
        return None

    def update(
        self,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError("`update` is not used in Eva Cache.")
