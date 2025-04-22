from typing import List, Optional, Tuple, Union
import math
import torch

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

def prepare_eva_training_mask(
    target_token_type_ids,
    use_doc_boundary_attention, 
    chunk_size,
    window_size,
    EOS_TOKEN_TYPE_ID=None,
    PAD_TOKEN_TYPE_ID=None,
):
    '''
    This function prepares the attention mask for training EvaByte.
        target_token_type_ids:
            Tensor of shape (batch_size, seq_len), marking the token type ids 
            for the target sequence. In particular, this function expects
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
    batch_size, num_tokens = target_token_type_ids.shape

    chunk_causal_mask, window_causal_mask = prepare_eva_attention_mask(
        num_tokens, 
        target_token_type_ids.device, 
        chunk_size=chunk_size, 
        window_size=window_size,
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

        assert num_tokens % chunk_size == 0, "Number of tokens must be divisible by chunk size"
        assert num_tokens % window_size == 0, "Number of tokens must be divisible by window size"
        num_chunks = num_tokens // chunk_size
        num_windows = num_tokens // window_size

        article_separator = 0

        #### step 2: generate attention masks for each window
        #### NOTE: we perform exact attention within each window, 
        ####       so we only need to mask out different documents
        ####       for each window.
        token_types_windows = token_types.reshape(batch_size, num_windows, window_size, 1)
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
        token_types_chunks = token_types.reshape(batch_size, num_chunks, chunk_size)
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
        intra_chunk_mask = intra_chunk_mask

        attention_mask = (
            window_mask.reshape(batch_size, 1, num_tokens, window_size), 
            inter_chunk_mask, 
            intra_chunk_mask.reshape(batch_size, 1, num_tokens, 1)
        )
    else:
        attention_mask = None
    return attention_mask

def prepare_doc_mask_position_ids(
    input_ids: torch.LongTensor,
    chunk_size: int,
    window_size: int,
    eos_token_id: int,
):
    attention_mask = prepare_eva_training_mask(
        input_ids,
        True,
        chunk_size,
        window_size,
        EOS_TOKEN_TYPE_ID=eos_token_id,
        PAD_TOKEN_TYPE_ID=None,
    )

    bs, seq_len = input_ids.shape

    position_ids = []

    for b in range(bs):
        position_id = torch.arange(0, seq_len, dtype=torch.long)
        # Find indecies where EOD token is.
        eos_ind = position_id[input_ids[b] == eos_token_id]

        # Loop through EOD indecies:
        prev_index = 0
        for j in range(eos_ind.shape[0]):
            i = eos_ind[j]
            # Reset positions.
            position_id[(i + 1):] -= (i + 1 - prev_index)
            prev_index = i + 1
        
        position_ids.append(position_id)
    position_ids = torch.stack(position_ids, dim=0)
    return attention_mask, position_ids


## Example usage
# attention_mask, position_ids = prepare_doc_mask_position_ids(
#     input_ids,
#     args.chunk_size,
#     args.window_size,
#     eos_token_id=args.eos_id,
# )
# attention_mask = tuple(mask.cuda() for mask in attention_mask)
# position_ids = position_ids.cuda()
# input_ids = input_ids.cuda()
# labels = labels.cuda()
# losses = model(
#     input_ids=input_ids,
#     attention_mask=attention_mask,
#     position_ids=position_ids,
#     labels=labels,
# )
