
import math
import torch
import triton
import triton.language as tl

@triton.heuristics(
    {
        "EVEN_N": lambda args: args["seqlen"] % args["BLOCK_N"] == 0,
        "EVEN_HEADDIM": lambda args: args["headdim"] == args["BLOCK_HEADDIM"],
    }
)
@triton.jit
def _fwd_eva_prep_kv_kernel(
    K, # [b, h, n, d]
    V, # [b, h, n, d]
    PARAM_MU, # [1, h, 1, 1, d]
    PARAM_PHI,  # [1, h, 1, 1, d]
    ChunkMask, # [b, h, n, 1]
    Out_RFA_K, # [b, h, c, d]
    Out_RFA_V, # [b, h, c, d]
    softmax_scale,
    stride_kb, stride_kh, stride_kn,
    stride_vb, stride_vh, stride_vn,
    stride_mu_h,
    stride_phi_h,
    stride_mb, stride_mn,
    stride_ok_b, stride_ok_h, stride_ok_c,
    stride_ov_b, stride_ov_h, stride_ov_c,
    nheads,
    seqlen,
    nchunks,
    headdim,
    CACHE_KEY_SEQLEN, # TODO: why keeping this
    CACHE_KEY_NCHUNKS, # TODO: why keeping this
    CHUNKS_PER_BLOCK: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
    MASK_TYPE: tl.constexpr,
    BLOCK_HEADDIM: tl.constexpr,
    EVEN_N: tl.constexpr,
    EVEN_HEADDIM: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    start_n = tl.program_id(0)
    offs_bh = tl.program_id(1)
    offs_h = offs_bh % nheads
    offs_b = offs_bh // nheads
    # initialize offsets
    # we load BLOCK_N keys and values each time, and
    # reshape it to [CHUNKS_PER_BLOCK, CHUNK_SIZE]
    offs_c = tl.arange(0, CHUNKS_PER_BLOCK)
    offs_m = tl.arange(0, CHUNK_SIZE)
    offs_d = tl.arange(0, BLOCK_HEADDIM)

    k_ptrs = (
        K +
        offs_b * stride_kb +
        offs_h * stride_kh +
        (
            (
                start_n * BLOCK_N + 
                offs_c[:, None, None] * CHUNK_SIZE + 
                offs_m[None, :, None]
            ) * stride_kn + 
            offs_d[None, None, :]
        )
    )
    v_ptrs = (
        V +
        offs_b * stride_vb +
        offs_h * stride_vh +
        (
            (
                start_n * BLOCK_N +
                offs_c[:, None, None] * CHUNK_SIZE + 
                offs_m[None, :, None]
            ) * stride_vn + 
            offs_d[None, None, :]
        )
    )
    param_mu_ptrs = (
        PARAM_MU +
        offs_h * stride_mu_h +
        offs_d[None, None, :]
    )
    param_phi_ptrs = (
        PARAM_PHI +
        offs_h * stride_phi_h +
        offs_d[None, None, :]
    )
    log2e = 1.4426950408889634
    if MASK_TYPE == 1:
        m_ptrs = (
            ChunkMask +
            offs_b * stride_mb +
            (
                (
                    start_n * BLOCK_N +
                    offs_c[:, None] * CHUNK_SIZE + 
                    offs_m[None, :]
                ) * stride_mn
            )
        )
    if EVEN_N:
        if EVEN_HEADDIM:
            k = tl.load(
                k_ptrs
            )
        else:
            k = tl.load(
                k_ptrs,
                mask=offs_d[None, None, :] < headdim,
                other=0.0
            )
    else:
        if EVEN_HEADDIM:
            k = tl.load(
                k_ptrs,
                mask=(
                        start_n * BLOCK_N +
                        offs_c[:, None, None] * CHUNK_SIZE + 
                        offs_m[None, :, None]
                    ) < seqlen,
                other=0.0
            )
        else:
            k = tl.load(
                k_ptrs,
                mask=(
                        (
                            start_n * BLOCK_N +
                            offs_c[:, None, None] * CHUNK_SIZE + 
                            offs_m[None, :, None]
                        ) < seqlen
                    ) & (offs_d[None, None, :] < headdim),
                other=0.0
            )
    
    param_mu = tl.load(param_mu_ptrs).to(k.dtype)
    rfa_k_c_w = tl.zeros([CHUNKS_PER_BLOCK, CHUNK_SIZE], dtype=tl.float32)
    rfa_k_c_w += tl.sum(k * param_mu, axis=-1)
    rfa_k_c_w *= log2e
    if MASK_TYPE == 1:
        if EVEN_N:
            mask = tl.load(
                m_ptrs
            ).to(tl.float32)
        else:
            mask = tl.load(
                m_ptrs,
                mask=(
                        start_n * BLOCK_N +
                        offs_c[:, None] * CHUNK_SIZE + 
                        offs_m[None, :]
                    ) < seqlen,
                other=0.0,
            ).to(tl.float32)
        rfa_k_c_w = rfa_k_c_w + mask

    rfa_k_c_w = tl.exp2(rfa_k_c_w - tl.max(rfa_k_c_w, axis=-1)[:, None])
    rfa_k_c_w = rfa_k_c_w / tl.sum(rfa_k_c_w, axis=-1)[:, None]
    rfa_k_c = tl.sum(k * rfa_k_c_w[:, :, None].to(k.dtype), axis=-2)
    # TODO: understand why rematerialize offsets to save registers?
    offs_out_c = start_n * CHUNKS_PER_BLOCK + tl.arange(0, CHUNKS_PER_BLOCK)
    out_rfa_k_ptrs = (
        Out_RFA_K +
        offs_b * stride_ok_b +
        offs_h * stride_ok_h +
        (offs_out_c[:, None] * stride_ok_c + offs_d[None, :])
    )

    if EVEN_N:
        if EVEN_HEADDIM:
            tl.store(
                out_rfa_k_ptrs, rfa_k_c
            )
        else:
            tl.store(
                out_rfa_k_ptrs, rfa_k_c,
                mask=offs_d[None, :] < headdim
            )
    else:
        if EVEN_HEADDIM:
            tl.store(
                out_rfa_k_ptrs, rfa_k_c,
                mask=offs_out_c[:, None] < nchunks
            )
        else:
            tl.store(
                out_rfa_k_ptrs, rfa_k_c,
                mask=(offs_out_c[:, None] < nchunks) & (offs_d[None, :] < headdim)
            )


    param_phi = tl.load(param_phi_ptrs).to(k.dtype)
    rfa_v_c_w = tl.zeros([CHUNKS_PER_BLOCK, CHUNK_SIZE], dtype=tl.float32)
    rfa_v_c_w += tl.sum(k * param_phi, axis=-1)
    rfa_v_c_w -= (0.5 * tl.sum(k * k, axis=-1))
    rfa_v_c_w *= log2e * softmax_scale
    if not EVEN_N:  # Need to mask out otherwise the softmax is wrong
        rfa_v_c_w += tl.where(
            (
                start_n * BLOCK_N +
                offs_c[:, None] * CHUNK_SIZE + 
                offs_m[None, :]
            ) < seqlen, 
            0, 
            float("-inf")
        )

    if MASK_TYPE == 1:
        rfa_v_c_w = rfa_v_c_w + mask

    if EVEN_N:
        if EVEN_HEADDIM:
            v = tl.load(
                v_ptrs
            )
        else:
            v = tl.load(
                v_ptrs,
                mask=offs_d[None, None, :] < headdim,
                other=0.0
            )
    else:
        if EVEN_HEADDIM:
            v = tl.load(
                v_ptrs,
                mask=(
                        start_n * BLOCK_N +
                        offs_c[:, None, None] * CHUNK_SIZE + 
                        offs_m[None, :, None]
                    ) < seqlen,
                other=0.0
            )
        else:
            v = tl.load(
                v_ptrs,
                mask=(
                        (
                            start_n * BLOCK_N +
                            offs_c[:, None, None] * CHUNK_SIZE + 
                            offs_m[None, :, None]
                        ) < seqlen
                    ) & (offs_d[None, None, :] < headdim),
                other=0.0
            )
    
    rfa_v_c_w = tl.exp2(rfa_v_c_w - tl.max(rfa_v_c_w, axis=-1)[:, None])
    rfa_v_c_w = rfa_v_c_w / tl.sum(rfa_v_c_w, axis=-1)[:, None]
    rfa_v_c = tl.sum(v * rfa_v_c_w[:, :, None].to(v.dtype), axis=-2)

    offs_out_c = start_n * CHUNKS_PER_BLOCK + tl.arange(0, CHUNKS_PER_BLOCK)
    out_rfa_v_ptrs = (
        Out_RFA_V +
        offs_b * stride_ov_b +
        offs_h * stride_ov_h +
        (offs_out_c[:, None] * stride_ov_c + offs_d[None, :])
    )
    if EVEN_N:
        if EVEN_HEADDIM:
            tl.store(
                out_rfa_v_ptrs, rfa_v_c
            )
        else:
            tl.store(
                out_rfa_v_ptrs, rfa_v_c,
                mask=offs_d[None, :] < headdim
            )
    else:
        if EVEN_HEADDIM:
            tl.store(
                out_rfa_v_ptrs, rfa_v_c,
                mask=offs_out_c[:, None] < nchunks
            )
        else:
            tl.store(
                out_rfa_v_ptrs, rfa_v_c,
                mask=(offs_out_c[:, None] < nchunks) & (offs_d[None, :] < headdim)
            )

def triton_eva_prep_kv_fwd(k, v, param_mu, param_phi, chunk_mask, softmax_scale, chunksize):
    k, v, param_mu, param_phi = [
        x if x.stride(-1) == 1 else x.contiguous() 
        for x in [k, v, param_mu, param_phi]
    ]

    # shape constraints
    batch, nheads, seqlen, head_dim = k.shape
    assert seqlen % chunksize == 0, "seqlen must be divisible by chunksize"
    nchunks = seqlen // chunksize
    assert k.shape == (batch, nheads, seqlen, head_dim)
    assert v.shape == (batch, nheads, seqlen, head_dim)
    assert param_mu.shape == (1, nheads, 1, 1, head_dim)
    assert param_phi.shape == (1, nheads, 1, 1, head_dim)
    assert head_dim <= 128, "We only test head dimensions up to 128"
    assert k.dtype == v.dtype == param_mu.dtype == param_phi.dtype, "All tensors must have the same type"
    assert k.dtype in [torch.bfloat16, torch.float], "Only support bf16 and fp32 for now"
    assert k.is_cuda and v.is_cuda
    softmax_scale = softmax_scale or 1.0 / math.sqrt(head_dim)

    mask_type = 0
    if chunk_mask is not None:
        mask_type = 1
        assert chunk_mask.dtype == k.dtype
        assert chunk_mask.is_cuda
        assert chunk_mask.dim() == 4
        assert chunk_mask.shape == (batch, 1, seqlen, 1)
        if chunk_mask.stride(-1) != 1:
            chunk_mask = chunk_mask.contiguous()
    mask_strides = (
        (chunk_mask.stride(0), chunk_mask.stride(2)) 
        if mask_type == 1 else 
        (0, 0)
    )
    out_rfa_k = torch.empty((batch, nheads, nchunks, head_dim), dtype=k.dtype, device=k.device)
    out_rfa_v = torch.empty((batch, nheads, nchunks, head_dim), dtype=v.dtype, device=v.device)

    BLOCK_HEADDIM = max(triton.next_power_of_2(head_dim), 16)
    BLOCK = 128
    num_warps = 4 if head_dim <= 64 else 8
    
    assert (BLOCK > chunksize) & (BLOCK % chunksize) == 0, "BLOCK must be divisible by chunksize"
    chunks_per_block = BLOCK // chunksize

    grid = lambda META: (triton.cdiv(seqlen, META["BLOCK_N"]), batch * nheads)
    _fwd_eva_prep_kv_kernel[grid](
        k,
        v,
        param_mu,
        param_phi,
        chunk_mask,
        out_rfa_k,
        out_rfa_v,
        softmax_scale,
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        param_mu.stride(1),
        param_phi.stride(1),
        mask_strides[0], mask_strides[1],
        out_rfa_k.stride(0), out_rfa_k.stride(1), out_rfa_k.stride(2),
        out_rfa_v.stride(0), out_rfa_v.stride(1), out_rfa_v.stride(2),
        nheads,
        seqlen,
        nchunks,
        head_dim,
        seqlen // 32,
        nchunks // 32,
        chunks_per_block,
        chunksize,
        mask_type,
        BLOCK_HEADDIM,
        BLOCK_N=BLOCK,
        num_warps=num_warps,
        num_stages=1,
    )
    return out_rfa_k, out_rfa_v
