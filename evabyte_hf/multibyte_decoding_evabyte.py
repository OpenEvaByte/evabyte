
# The implementation of multibyte deocidng is largely adapted from
# Medusa decoding: https://github.com/FasterDecoding/Medusa
import torch
import torch.nn.functional as F
from transformers.generation.stopping_criteria import (
    MaxLengthCriteria,
    StoppingCriteriaList,
)
from typing import Union, List
from .eva_cache import EvaStaticCacheForTriton
from .eva_prep_kv_kernel import triton_eva_prep_kv_fwd

class MultibyteEosTokenCriteria:
    """
    This class implements a simple stopping criteria to stop generation whenever
    the "end-of-sequence" token is generated in the last `new_tokens` tokens.

    Adapted from 
    https://github.com/huggingface/transformers/blob/main/src/transformers/generation/stopping_criteria.py#L446
    By default, it uses the `model.generation_config.eos_token_id`.

    Args:
        eos_token_id (`Union[int, List[int]]`):
            The id(s) of the *end-of-sequence* token.
    """

    def __init__(self, eos_token_ids: Union[int, List[int]]):
        if isinstance(eos_token_ids, int):
            eos_token_ids = [eos_token_ids]
        self.eos_token_ids = eos_token_ids
    
    def __call__(self, input_ids: torch.LongTensor, new_tokens: int) -> bool:
        current_input_len = input_ids.shape[-1]
        new_token_ids = input_ids[:, current_input_len - new_tokens:]
        for eos_token_id in self.eos_token_ids:
            if torch.any(new_token_ids == eos_token_id):
                return True
        return False

def build_tree(spec):
    nodes_at_depth = []
    nodes_at_depth.append([()])  # Root at depth 1

    for d in range(1, len(spec) + 1):
        prev_nodes = nodes_at_depth[d - 1]
        spec_list = spec[d - 1]
        current_nodes = []
        for node_idx, node in enumerate(prev_nodes):
            if node_idx < len(spec_list):
                num_children = spec_list[node_idx]
            else:
                num_children = 0
            for child_idx in range(num_children):
                new_node = node + (child_idx,)
                current_nodes.append(new_node)
        nodes_at_depth.append(current_nodes)

    # Flatten the list of nodes, excluding the root node if desired
    all_nodes = [node for depth_nodes in nodes_at_depth for node in depth_nodes if node]
    return all_nodes

evabyte_7b_95 = build_tree(
    [
        [10], 
        [10, 8, 2, 2, 1, 1], 
        [10, 4, 2, 1, 0, 0, 0, 0, 0, 0, 2, 1, 1, 0, 0, 0, 0, 0, 1],
        [8, 2, 2, 1, 0, 0, 0, 0, 0, 0, 1],
        [6, 2, 1, 1],
        [4, 2, 1, 1],
        [4, 2, 1],
    ]
)
evabyte_7b_31 = build_tree(
    [
        [4], 
        [3, 2, 1, 1], 
        [3, 2, 1, 1],
        [2, 1, 1],
        [2, 1],
        [2, 1],
        [2, 1],
    ]
)
TOPK = 10 # topk for sparse tree (10 is a placeholder and it is sufficient)

def pad_path(path, length, pad_value=-2):
    """
    Pad the given path list with a specific value up to a specified length.
    
    Parameters:
    - path (list): The original list that needs padding.
    - length (int): The desired length of the padded list.
    - pad_value (optional, default=-2): The value to use for padding.
    
    Returns:
    - list: A new list based on the original path but padded to the desired length.
    
    Example:
    >>> pad_path([1,2,3], 5)
    [1, 2, 3, -2, -2]
    
    Note:
    If the given path is already longer than the specified length, 
    then no padding occurs, and the original path is returned.
    """
    return path + [pad_value] * (length - len(path))

def reset_past_key_values(passed_key_values):
    """
    Resets the current lengths in the passed key-values to zero.

    This function is designed to be used during the evaluation of a baseline model.
    It iterates through each layer's key-values and sets their current lengths to zero,
    effectively resetting their state.

    Args:
    - passed_key_values (list of torch.Tensor): Contains past hidden states and past attention values for each layer.

    Returns:
    - passed_key_values (list of torch.Tensor): Updated past hidden states and past attention values with reset lengths.
    """
    for i in range(len(passed_key_values)):
        for j in range(2):
            passed_key_values[i][j].current_length.fill_(0)
    return passed_key_values

def get_nucleus_one_token(logit, temperature, top_p):
    """
    Performs token sampling based on the nucleus (top-p) sampling method.

    This function selects a token from a given logit distribution using the nucleus sampling strategy.
    It allows for more controlled and diverse generation compared to traditional top-k sampling.

    Args:
        logit (torch.Tensor): The logits from a language model output, expected to be a 2D tensor (BxC).
        temperature (float): A temperature parameter to control the randomness in sampling.
                             Higher values increase diversity, lower values make selections more deterministic.
        top_p (float): The cumulative probability threshold for nucleus sampling.
                       It controls the size of the set of high-probability tokens to consider for sampling.

    Returns:
        torch.Tensor: A tensor containing the indices of the sampled tokens.
    """
    if top_p >= 1:
        return torch.multinomial(F.softmax(logit / temperature, dim=-1), 1)
    logit = logit / temperature
    probs = torch.softmax(logit, dim=-1)
    sorted_logits, sorted_indices = torch.sort(probs, descending=True)
    cum_probs = torch.cumsum(sorted_logits, dim=-1)
    sorted_indices_to_remove = cum_probs > top_p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0
    indices_to_remove = sorted_indices_to_remove.scatter(dim=1, index=sorted_indices, src=sorted_indices_to_remove)
    logit[indices_to_remove] = float('-inf')
    sampled_tokens = torch.multinomial(F.softmax(logit, dim=-1), 1)
    return sampled_tokens

def get_typical_one_token(logit, temperature, posterior_threshold, posterior_alpha):
    """
    Implements token sampling based on the typical sampling method.

    This function selects a token from a given logit distribution using the typical sampling strategy,
    aiming to balance between diversity and likelihood in a more nuanced way compared to traditional methods.

    Args:
        logit (torch.Tensor): The logits from a language model output, expected to be a 2D tensor.
        temperature (float): A parameter to control the randomness in sampling.
                              Higher values increase diversity, lower values make selections more deterministic.
        posterior_threshold (float): A threshold to decide the lower bound of probabilities to be considered for sampling.
        posterior_alpha (float): A scaling factor applied to the entropy-based adaptive threshold.

    Returns:
        torch.Tensor: A tensor containing the indices of the sampled tokens.
    """
    logit = logit / temperature
    probs = torch.softmax(logit, dim=-1)
    entropy = -torch.sum(
            probs * torch.log(probs + 1e-5), dim=-1
        )
    threshold = torch.minimum(
            torch.ones_like(entropy) * posterior_threshold,
            torch.exp(-entropy) * posterior_alpha,
        )
    indices_to_remove = probs < threshold.unsqueeze(-1)
    logit[indices_to_remove] = float('-inf')
    sampled_tokens = torch.multinomial(F.softmax(logit, dim=-1), 1)
    return sampled_tokens



def generate_medusa_buffers(medusa_choices, device="cuda"):
    """
    Generate buffers for the Medusa structure based on the provided choices.
    
    Parameters:
    - medusa_choices (list): A nested list representing tree in the Medusa structure.
    - device (str): Device to which the tensors should be moved. Default is "cuda".
    
    Returns:
    - dict: A dictionary containing buffers related to the Medusa structure.
    """

    # Sort the medusa_choices based on their lengths and then their values
    sorted_medusa_choices = sorted(medusa_choices, key=lambda x: (len(x), x))
    medusa_len = len(sorted_medusa_choices) + 1

    # Initialize depth_counts to keep track of how many choices have a particular depth
    depth_counts = [0] * max([len(path) for path in sorted_medusa_choices])
    for path in sorted_medusa_choices:
        depth_counts[len(path) - 1] += 1
    
    # Create the attention mask for Medusa
    medusa_attn_mask = torch.eye(medusa_len, medusa_len)
    medusa_attn_mask[:, 0] = 1
    start = 0
    for i in range(len(depth_counts)):
        for j in range(depth_counts[i]):
            cur_medusa_choice = sorted_medusa_choices[start + j]
            # retrieve ancestor position
            if len(cur_medusa_choice) == 1:
                continue
            ancestor_idx = []
            for c in range(len(cur_medusa_choice) - 1):
                ancestor_idx.append(sorted_medusa_choices.index(cur_medusa_choice[:c+1]) + 1)
            medusa_attn_mask[j + start + 1, ancestor_idx] = 1
        start += depth_counts[i]

    # Generate tree indices for the Medusa structure
    medusa_tree_indices = torch.zeros(medusa_len, dtype=torch.long)
    medusa_tree_indices[0] = 0
    start = 0
    for i in range(len(depth_counts)):
        for j in range(depth_counts[i]):
            cur_medusa_choice = sorted_medusa_choices[start + j]
            medusa_tree_indices[start + j + 1] = cur_medusa_choice[-1] + TOPK * i + 1
        start += depth_counts[i]

    # Generate position IDs for the Medusa structure
    medusa_position_ids = torch.zeros(medusa_len, dtype=torch.long)
    start = 0
    for i in range(len(depth_counts)):
        medusa_position_ids[start + 1: start + depth_counts[i] + 1] = i + 1
        start += depth_counts[i]

    # Generate retrieval indices for Medusa structure verification
    retrieve_indices_nest = []
    retrieve_paths = []
    for i in range(len(sorted_medusa_choices)):
        cur_medusa_choice = sorted_medusa_choices[-i-1]
        retrieve_indice = []
        if cur_medusa_choice in retrieve_paths:
            continue
        else:
            for c in range(len(cur_medusa_choice)):
                retrieve_indice.append(sorted_medusa_choices.index(cur_medusa_choice[:c+1]))
                retrieve_paths.append(cur_medusa_choice[:c+1])
        retrieve_indices_nest.append(retrieve_indice)
    max_length = max([len(x) for x in retrieve_indices_nest])
    retrieve_indices = [pad_path(path, max_length) for path in retrieve_indices_nest]
    retrieve_indices = torch.tensor(retrieve_indices, dtype=torch.long)
    retrieve_indices = retrieve_indices + 1
    retrieve_indices = torch.cat([torch.zeros((retrieve_indices.shape[0], 1), dtype=torch.long), retrieve_indices], dim=1)

    # Aggregate the generated buffers into a dictionary
    medusa_buffers = {
        "medusa_attn_mask": medusa_attn_mask.unsqueeze(0).unsqueeze(0),
        "tree_indices": medusa_tree_indices,
        "medusa_position_ids": medusa_position_ids.unsqueeze(0),
        "retrieve_indices": retrieve_indices,
    }
    
    # Move the tensors in the dictionary to the specified device
    medusa_buffers = {
        k: v.clone().to(device)
        if isinstance(v, torch.Tensor)
        else torch.tensor(v, device=device)
        for k, v in medusa_buffers.items()
    }
    return medusa_buffers

def generate_candidates(
        medusa_logits, 
        logits, 
        tree_indices, 
        retrieve_indices, 
        temperature = 0, 
        posterior_threshold=0.3, 
        posterior_alpha = 0.09, 
        top_p=0.8, 
        sampling = 'typical', 
        fast = False
    ):
    # Say we have 3 heads, and the top-4 for each head are:
    # [10, 3, 8, 4]
    # [9, 5, 1, 6]
    # [7, 16, 3, 2]

    # candidates_id = 10
    if temperature == 0 or fast:
        candidates_ids = torch.argmax(logits[:, -1]).unsqueeze(0)
    else:
        if sampling == 'typical':
            candidates_ids = get_typical_one_token(logits[:, -1], temperature, posterior_threshold, posterior_alpha).squeeze(0)
        elif sampling == 'nucleus':
            candidates_ids = get_nucleus_one_token(logits[:, -1], temperature, top_p).squeeze(0)
        else:
            raise NotImplementedError

    # this calculates the top-k medusa logits
    # candidates_medusa_id = [
    #   [9, 5, 1, 6]
    #   [7, 16, 3, 2]
    # ]
    candidates_medusa_ids = torch.topk(medusa_logits[:, 0, -1], TOPK, dim=-1).indices

    # [10, 9, 5, 1, 6, 7, 16, 3, 2]
    candidate_ids = torch.cat([candidates_ids, candidates_medusa_ids.view(-1)], dim=-1)

    # based on the pre-defined tree_indices, select the corresponding candidates
    # if we select top-2 and top-3 for the two heads (we select top-1 for the first head):
    # tree_candidates = [10, 9, 5, 7, 16, 3, 7, 16, 3]
    tree_candidate_ids = candidate_ids[tree_indices]

    # tree_candidate_ids = [10, 9, 5, 7, 16, 3, 7, 16, 3, 0]
    # Sometimes the tree_indices are padded, so we append a zero here
    # so that all padded indices select the appended zero.
    tree_candidate_ids_ext = torch.cat(
        [
            tree_candidate_ids, 
            torch.zeros((1), dtype=torch.long, device=tree_candidate_ids.device)
        ], 
        dim=0
    )
    # [[10, 9, 7], [10, 9, 16], [10, 9, 3], [10, 5, 7], [10, 5, 16], [10, 5, 3]]
    unflattened_candidate_ids = tree_candidate_ids_ext[retrieve_indices]

    tree_candidate_ids = tree_candidate_ids.unsqueeze(0)
            
    return tree_candidate_ids, unflattened_candidate_ids

def get_nucleus_posterior_mask(logits, candidates, temperature, top_p):
    """
    Generates a posterior mask for token candidates using nucleus (top-p) sampling.

    This function applies nucleus sampling to a set of logits, and then generates a mask indicating 
    which candidate tokens are selected. It adapts the sampling strategy to accommodate for 
    temperature scaling and cumulative probability thresholding.

    Args:
        logits (torch.Tensor): A tensor of logits from a language model output.
        candidates (torch.Tensor): A tensor of candidate tokens to compare against sampled tokens.
        temperature (float): A parameter to scale the logits, controlling randomness in sampling.
        top_p (float): The cumulative probability threshold for nucleus sampling.

    Returns:
        torch.Tensor: A posterior mask indicating which candidate tokens match the sampled tokens.
    """
    # adapted from https://github.com/huggingface/transformers/blob/18a879f47576822aa1a5c49aecb27d89bfa5fa69/examples/run_generation.py#L79

    # Apply temperature
    logits = logits[:, :-1] / temperature
    n_samples, n_tokens = logits.shape[0], logits.shape[1]
    logits = logits.view(n_samples*n_tokens, -1)
    if top_p >= 1:
        sampled_tokens = torch.multinomial(F.softmax(logits, dim=-1), 1)
        sampled_tokens = sampled_tokens.view(n_samples, n_tokens)
        posterior_mask = (candidates[:, 1:] == sampled_tokens).int()
        return posterior_mask
    # Convert to probabilities (softmax)
    probs = F.softmax(logits, dim=-1)
    # Sort the probabilities
    sorted_logits, sorted_indices = torch.sort(probs, descending=True)

    # Compute cumulative probabilities
    cum_probs = torch.cumsum(sorted_logits, dim=-1)

    # Create mask for the top-p nucleus
    sorted_indices_to_remove = cum_probs > top_p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0

    indices_to_remove = sorted_indices_to_remove.scatter(dim=1, index=sorted_indices, src=sorted_indices_to_remove)

    
    # Remove low-probability tokens
    logits[indices_to_remove] = float('-inf')
    # Sample from the remaining tokens
    sampled_tokens = torch.multinomial(F.softmax(logits, dim=-1), 1)
    sampled_tokens = sampled_tokens.view(n_samples, n_tokens)
    # Create a mask for selected tokens
    posterior_mask = (candidates[:, 1:] == sampled_tokens).int()

    return posterior_mask

def get_typical_posterior_mask(logits, candidates, temperature, posterior_threshold, posterior_alpha):
    """
    Args:
        logits (torch.Tensor): A tensor of logits from a language model output.
        candidates (torch.Tensor): A tensor of candidate tokens to compare against sampled tokens.
        temperature (float): A parameter to scale the logits, controlling randomness in sampling.
        posterior_threshold (float): The minimum threshold for probabilities to be considered in sampling.
        posterior_alpha (float): A scaling factor applied to the entropy-based adaptive threshold.

    Returns:
        torch.Tensor: A posterior mask indicating which candidate tokens match the sampled tokens.
    """
    logits = logits[:, :-1] / temperature
    n_samples, n_tokens = logits.shape[0], logits.shape[1]
    logits = logits.view(n_samples*n_tokens, -1)
    probs = F.softmax(logits, dim=-1)
    entropy = -torch.sum(
            probs * torch.log(probs + 1e-5), dim=-1
        )
    threshold = torch.minimum(
            torch.ones_like(entropy) * posterior_threshold,
            torch.exp(-entropy) * posterior_alpha,
        )
    indices_to_remove = probs < threshold.unsqueeze(-1)
    logits[indices_to_remove] = float('-inf')
    sampled_tokens = torch.multinomial(F.softmax(logits, dim=-1), 1)
    sampled_tokens = sampled_tokens.view(n_samples, n_tokens)
    posterior_mask = (candidates[:, 1:] == sampled_tokens).int()
    return posterior_mask
    
    

def evaluate_posterior(
    logits, 
    candidates, 
    temperature, 
    posterior_threshold=0.3, 
    posterior_alpha = 0.09, 
    top_p=0.8, 
    sampling = 'typical', 
    fast = True
):
    if logits.shape[1] <= 1:
        return torch.tensor(0, dtype=torch.long, device=candidates.device), 0
    # Greedy decoding based on temperature value
    if temperature == 0:
        # Find the tokens that match the maximum logits for each position in the sequence
        posterior_mask = (
            candidates[:, 1:] == torch.argmax(logits[:, :-1], dim=-1)
        ).int()
        candidates_accept_length = (torch.cumprod(posterior_mask, dim=1)).sum(dim=1)
        accept_length = candidates_accept_length.max().item()
        # Choose the best candidate
        if accept_length == 0:
            # Default to the first candidate if none are accepted
            best_candidate = torch.tensor(0, dtype=torch.long, device=candidates.device)
        else:
            best_candidate = torch.argmax(candidates_accept_length).to(torch.long)
        return best_candidate, accept_length
    elif sampling == 'typical':
        if fast:
            posterior_prob = torch.softmax(logits[:, :-1] / temperature, dim=-1)
            candidates_prob = torch.gather(
                posterior_prob, dim=-1, index=candidates[:, 1:].unsqueeze(-1)
            ).squeeze(-1)
            posterior_entropy = -torch.sum(
                posterior_prob * torch.log(posterior_prob + 1e-5), dim=-1
            )  # torch.sum(torch.log(*)) is faster than torch.prod
            threshold = torch.minimum(
                torch.ones_like(posterior_entropy) * posterior_threshold,
                torch.exp(-posterior_entropy) * posterior_alpha,
            )
            posterior_mask = candidates_prob > threshold
            candidates_accept_length = (torch.cumprod(posterior_mask, dim=1)).sum(dim=1)

            # Choose the best candidate based on the evaluated posterior probabilities
            accept_length = candidates_accept_length.max().item()
            if accept_length == 0:
                # If no candidates are accepted, just choose the first one
                best_candidate = torch.tensor(0, dtype=torch.long, device=candidates.device)
            else:
                best_candidates = torch.where(candidates_accept_length == accept_length)[0]
                # Accept the best one according to likelihood
                likelihood = torch.sum(
                    torch.log(candidates_prob[best_candidates, :accept_length]), dim=-1
                )
                best_candidate = best_candidates[torch.argmax(likelihood)]
            return best_candidate, accept_length
        # Calculate posterior probabilities and thresholds for candidate selection
        posterior_mask = get_typical_posterior_mask(logits, candidates, temperature, posterior_threshold, posterior_alpha)
        candidates_accept_length = (torch.cumprod(posterior_mask, dim=1)).sum(dim=1)
        # Choose the best candidate based on the evaluated posterior probabilities
        accept_length = candidates_accept_length.max().item()
        
        if accept_length == 0:
            # If no candidates are accepted, just choose the first one
            best_candidate = torch.tensor(0, dtype=torch.long, device=candidates.device)
        else:
            best_candidate = torch.argmax(candidates_accept_length).to(torch.long)
            # Accept the best one according to likelihood
        return best_candidate, accept_length
    elif sampling == 'nucleus':
        assert top_p < 1.0 + 1e-6, "top_p should between 0 and 1"
        posterior_mask = get_nucleus_posterior_mask(logits, candidates, temperature, top_p)
        candidates_accept_length = (torch.cumprod(posterior_mask, dim=1)).sum(dim=1)
        accept_length = candidates_accept_length.max().item()
        # Choose the best candidate
        if accept_length == 0:
            # Default to the first candidate if none are accepted
            best_candidate = torch.tensor(0, dtype=torch.long, device=candidates.device)
        else:
            best_candidate = torch.argmax(candidates_accept_length).to(torch.long)
        return best_candidate, accept_length
    else:
        raise NotImplementedError

def update_inference_inputs(
    input_ids,
    medusa_logits,
    logits,
    candidate_ids,
    best_candidate,
    accept_length,
):
    input_ids = torch.cat(
        [
            input_ids, 
            candidate_ids[None, best_candidate, : accept_length + 1]
        ], 
        dim=-1
    )
    logits = logits[
        None, best_candidate, accept_length : accept_length + 1
    ]
    medusa_logits = medusa_logits[
        :, None, best_candidate, accept_length : accept_length + 1
    ]
    # Update the new token counter
    new_token = accept_length + 1
    return input_ids, medusa_logits, logits, new_token

def split_logits(full_logits):
    # logits has shape [b, n, heads, vocab_size]
    logits = full_logits[..., 0, :]
    medusa_logits = full_logits[..., 1:, :].permute(2, 0, 1, 3)
    return medusa_logits, logits

class MultiByteDecodingMixin:
    def multi_byte_pred_update_cache(
        self,
        past_key_values,
        retrieve_indices,
        best_candidate,
        new_tokens,
    ):
        prev_window_len = past_key_values.get_past_window_pos(0)
        select_indices = (
            retrieve_indices[best_candidate, : new_tokens] + prev_window_len
        )
        for layer_idx in range(self.config.num_hidden_layers):

            past_key_values.update_past_len(new_tokens, layer_idx)

            past_window_k = past_key_values.past_window_k[layer_idx]
            past_window_v = past_key_values.past_window_v[layer_idx]

            tgt_window_k = past_window_k[..., select_indices, :]
            tgt_window_v = past_window_v[..., select_indices, :]

            dst_window_k = past_window_k[..., prev_window_len : prev_window_len + new_tokens, :]
            dst_window_v = past_window_v[..., prev_window_len : prev_window_len + new_tokens, :]

            dst_window_k.copy_(tgt_window_k, non_blocking=True)
            dst_window_v.copy_(tgt_window_v, non_blocking=True)

            new_window_len = prev_window_len + new_tokens
            if new_window_len >= self.config.window_size:
                assert new_window_len < 2 * self.config.window_size

                dump_k = past_window_k[..., :self.config.window_size, :].clone()
                dump_v = past_window_v[..., :self.config.window_size, :].clone()

                _window_len = new_window_len - self.config.window_size
                
                if _window_len > 0:
                    new_window_k = past_window_k[..., self.config.window_size : new_window_len, :]
                    new_window_v = past_window_v[..., self.config.window_size : new_window_len, :]

                    _dst_window_k = past_window_k[..., : _window_len, :]
                    _dst_window_v = past_window_v[..., : _window_len, :]

                    _dst_window_k.copy_(new_window_k, non_blocking=True)
                    _dst_window_v.copy_(new_window_v, non_blocking=True)

                past_key_values.past_window_pos[layer_idx] = _window_len
            else:
                dump_k = None
                dump_v = None
                past_key_values.past_window_pos[layer_idx] = new_window_len

            if dump_k is not None and dump_v is not None:
                rfa_k, rfa_v = triton_eva_prep_kv_fwd(
                    dump_k, dump_v, 
                    self.model.layers[layer_idx].self_attn.adaptive_mu_k, 
                    self.model.layers[layer_idx].self_attn.adaptive_phi, 
                    None, 
                    self.model.layers[layer_idx].self_attn.head_dim_scaling, 
                    self.model.layers[layer_idx].self_attn.chunk_size
                )
                rfa_k, rfa_v = past_key_values.update_chunk_rfas(
                    rfa_k, rfa_v, layer_idx
                )
        return past_key_values

    def _multi_byte_pred_update_cache_when_prefil_len_eq_window_size(
        self,
        past_key_values,
    ):
        prev_window_len = past_key_values.get_past_window_pos(0)
        for layer_idx in range(self.config.num_hidden_layers):

            past_window_k = past_key_values.past_window_k[layer_idx]
            past_window_v = past_key_values.past_window_v[layer_idx]

            new_window_len = prev_window_len
            if new_window_len == self.config.window_size:
                dump_k = past_window_k[..., :self.config.window_size, :].clone()
                dump_v = past_window_v[..., :self.config.window_size, :].clone()
                past_key_values.past_window_pos[layer_idx] = 0

                if dump_k is not None and dump_v is not None:
                    rfa_k, rfa_v = triton_eva_prep_kv_fwd(
                        dump_k, dump_v, 
                        self.model.layers[layer_idx].self_attn.adaptive_mu_k, 
                        self.model.layers[layer_idx].self_attn.adaptive_phi, 
                        None, 
                        self.model.layers[layer_idx].self_attn.head_dim_scaling, 
                        self.model.layers[layer_idx].self_attn.chunk_size
                    )
                    rfa_k, rfa_v = past_key_values.update_chunk_rfas(
                        rfa_k, rfa_v, layer_idx
                    )
        return past_key_values

    def multi_byte_pred_update_attn_mask(
        self,
        last_iter_new_tokens,
        tree_candidate_ids,
        past_attn_mask,
        medusa_attn_mask,
        past_key_values,
    ):
        batch_size, tree_candidate_len = tree_candidate_ids.shape
        seen_tokens = past_key_values.get_seq_length()
        # NOTE: past_key_values has been updated so now 
        # seen_tokens incldues new tokens from the last tree iteration
        assert seen_tokens > 0
        # so one iteration would not cross two windows
        assert last_iter_new_tokens < self.config.window_size
        
        if past_attn_mask is not None and seen_tokens < self.config.window_size:
            past_attn_mask = torch.cat(
                [
                    past_attn_mask, 
                    torch.ones(
                        [batch_size, 1, tree_candidate_len, last_iter_new_tokens],
                        dtype=torch.bool,
                        device=self.device
                    )
                ], 
                dim=-1
            )
        else:
            # we initialize attn mask each time when
            # 1. the model crosses the window bounary, or
            # 2. after prefilling
            chunks_per_window = int(self.config.window_size // self.config.chunk_size)

            window_tokens = seen_tokens % self.config.window_size
            num_windows_seen_so_far = seen_tokens // self.config.window_size
            attn_mask_len = num_windows_seen_so_far * chunks_per_window + window_tokens
            past_attn_mask = torch.ones(
                (batch_size, 1, tree_candidate_len, attn_mask_len),
                dtype=torch.bool,
                device=self.device
            )

        # note that 1 indicates the position is not masked
        tree_attn_mask = torch.cat(
            [
                past_attn_mask,
                medusa_attn_mask.to(torch.bool)
            ],
            dim=-1
        )
        return tree_attn_mask, past_attn_mask

    @torch.no_grad()
    def multi_byte_generate(
        self,
        input_ids,
        attention_mask=None,
        temperature=0.0,
        max_length=None,
        max_new_tokens=None,
        stopping_criteria=None,
        posterior_threshold=0.09,
        posterior_alpha=0.3,
        top_p=0.8,
        sampling='typical', 
        fast=True,
        do_sample=False,
        medusa_choices=None,
        return_acc_lengths=False
    ):
        if do_sample or temperature > 0.0:
            fast = False

        ### Prepare `max_length` depending on other stopping criteria.
        if max_new_tokens is not None:
            max_length = max_new_tokens + input_ids.shape[-1]
        elif max_new_tokens is None and max_length is None:
            max_length = getattr(self.config, "max_position_embeddings", 32768)

        ### Set up stopping criteria
        eos_stop_criteria = MultibyteEosTokenCriteria(self.generation_config.eos_token_id)
        stop_criteria = StoppingCriteriaList()
        if max_length is not None:
            max_position_embeddings = getattr(self.config, "max_position_embeddings", None)
            stop_criteria.append(
                MaxLengthCriteria(
                    max_length=max_length,
                    max_position_embeddings=max_position_embeddings,
                )
            )
        if stopping_criteria is not None and len(stopping_criteria) > 0:
            stop_criteria.extend(stopping_criteria)

        assert input_ids.shape[0] == 1, "Only support batch size 1 for now"
        assert attention_mask is None, "Only support attention mask None for now"
        # Avoid modifying the input_ids in-place
        input_ids = input_ids.clone()
        position_ids = torch.arange(0, input_ids.shape[1], device=self.device, dtype=int).reshape(1, -1)

        ####################################################
        # 0. initialize the medusa buffers
        ####################################################
        if medusa_choices is None:
            medusa_choices = evabyte_7b_95
        medusa_buffers = generate_medusa_buffers(
            medusa_choices, device=self.device
        )

        past_key_values = EvaStaticCacheForTriton(
            input_ids.shape[0],
            self.config.num_attention_heads,
            # we add 256 to allow tree ids
            self.config.window_size + 256,
            self.config.hidden_size // self.config.num_attention_heads,
            self.config.num_hidden_layers,
            self.lm_head.weight.dtype,
            self.lm_head.weight.device,
        )
        # prefill to get medusa logits and logits
        full_logits, past_key_values = self.forward(
            input_ids, 
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=True,
            past_key_values=past_key_values,
            return_all_pred_logits=True,
            multibyte_decoding=False,
        )
        # handles an edge case where the prefill length == window_size
        # we force the previous window to be dumped into RFA chunks
        past_key_values = self._multi_byte_pred_update_cache_when_prefil_len_eq_window_size(
            past_key_values
        )
        medusa_logits, logits = split_logits(full_logits)

        past_attn_mask = None
        last_iter_new_tokens = 0
        max_iters = 32768
        if return_acc_lengths:
            acc_lengths = []
        for _ in range(max_iters):
            ####################################################
            # 1. generate candidate_ids with topk predictions from Medusa heads
            ####################################################
            tree_candidate_ids, unflattened_candidate_ids = generate_candidates(
                medusa_logits,
                logits,
                medusa_buffers["tree_indices"],
                medusa_buffers["retrieve_indices"],
                temperature=temperature,
                posterior_alpha=posterior_alpha,
                posterior_threshold=posterior_threshold,
                top_p=top_p,
                sampling=sampling,
                fast=fast,
            )

            ####################################################
            # 2. Build the medusa attention mask and position ids
            ####################################################
            # NOTE: 1 indicates the position is not masked
            medusa_attn_mask, past_attn_mask = self.multi_byte_pred_update_attn_mask(
                last_iter_new_tokens,
                tree_candidate_ids,
                past_attn_mask,
                medusa_buffers["medusa_attn_mask"],
                past_key_values,
            )
            medusa_position_ids = medusa_buffers["medusa_position_ids"] + input_ids.shape[1]

            ####################################################
            # 3. tree decoding
            ####################################################
            tree_full_logits, past_key_values = self.forward(
                tree_candidate_ids,
                past_key_values=past_key_values,
                attention_mask=medusa_attn_mask,
                position_ids=medusa_position_ids,
                return_all_pred_logits=True,
                multibyte_decoding=True,
            )
            _medusa_logits, _logits = split_logits(tree_full_logits)
            medusa_logits = _medusa_logits[..., 0, medusa_buffers["retrieve_indices"], :]
            logits = _logits[..., 0, medusa_buffers["retrieve_indices"], :]

            ####################################################
            # 4. candidate selection
            ####################################################
            # if the current iteration, with tree tokens, crosses window
            # boundaries, trim the condidate_ids to be within the window
            # so that those exceeded tokens (which will be inaccurate)
            # will not be considered
            tree_depth = unflattened_candidate_ids.shape[-1]
            if tree_depth + past_key_values.get_past_window_pos(0) > self.config.window_size:
                max_acc_len = self.config.window_size - past_key_values.get_past_window_pos(0)
                _trimmed_unflattened_candidate_ids = unflattened_candidate_ids[:, :max_acc_len]
                _trimmed_logits = logits[:, :max_acc_len]
            else:
                _trimmed_unflattened_candidate_ids = unflattened_candidate_ids
                _trimmed_logits = logits
            best_candidate, accept_length = evaluate_posterior(
                _trimmed_logits, 
                _trimmed_unflattened_candidate_ids, 
                temperature, 
                posterior_threshold, 
                posterior_alpha, 
                top_p=top_p, 
                sampling=sampling, 
                fast=fast
            )

            ####################################################
            # 5. update model inputs and caches
            ####################################################
            input_ids, medusa_logits, logits, last_iter_new_tokens = update_inference_inputs(
                input_ids,
                medusa_logits,
                logits,
                unflattened_candidate_ids,
                best_candidate,
                accept_length,
            )

            past_key_values = self.multi_byte_pred_update_cache(
                past_key_values,
                medusa_buffers["retrieve_indices"],
                best_candidate,
                last_iter_new_tokens,
            )

            if return_acc_lengths:
                acc_lengths.append(last_iter_new_tokens)
            if stop_criteria(input_ids, None) or eos_stop_criteria(input_ids, last_iter_new_tokens):
                if return_acc_lengths:
                    return input_ids, acc_lengths
                else:
                    return input_ids
        if return_acc_lengths:
            return input_ids, acc_lengths
        else:
            return input_ids
