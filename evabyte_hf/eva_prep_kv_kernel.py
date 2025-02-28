
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
    Mask, # [b, h, n, 1]
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
            Mask +
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
            )
        else:
            mask = tl.load(
                m_ptrs,
                mask=(
                        start_n * BLOCK_N +
                        offs_c[:, None] * CHUNK_SIZE + 
                        offs_m[None, :]
                    ) < seqlen,
                other=1,
            )
        rfa_k_c_w = tl.where(mask, float("-inf"), rfa_k_c_w)
        
    m_rfa_k_c_w = tl.max(rfa_k_c_w, axis=-1)
    masked_out_rows_rfa_k = (m_rfa_k_c_w == float("-inf"))
    m_rfa_k_c_w_masked = tl.where(masked_out_rows_rfa_k, 0, m_rfa_k_c_w)
    rfa_k_c_w = tl.exp2(rfa_k_c_w - m_rfa_k_c_w_masked[:, None])
    denom_k = tl.sum(rfa_k_c_w, axis=-1)
    denom_k = tl.where(denom_k == 0.0, 1.0, denom_k)
    rfa_k_c_w = rfa_k_c_w / denom_k[:, None]
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
        rfa_v_c_w = tl.where(mask, float("-inf"), rfa_v_c_w)

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
    

    m_rfa_v_c_w = tl.max(rfa_v_c_w, axis=-1)
    masked_out_rows_rfa_v = (m_rfa_v_c_w == float("-inf"))
    m_rfa_v_c_w_masked = tl.where(masked_out_rows_rfa_v, 0, m_rfa_v_c_w)
    rfa_v_c_w = tl.exp2(rfa_v_c_w - m_rfa_v_c_w_masked[:, None])
    denom_v = tl.sum(rfa_v_c_w, axis=-1)
    denom_v = tl.where(denom_v == 0.0, 1.0, denom_v)
    rfa_v_c_w = rfa_v_c_w / denom_v[:, None]
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



@triton.heuristics(
    {
        "EVEN_N": lambda args: args["seqlen"] % args["BLOCK_N"] == 0,
        "EVEN_HEADDIM": lambda args: args["headdim"] == args["BLOCK_HEADDIM"],
    }
)
@triton.jit
def _bwd_eva_prep_kv_kernel(
    RFA_K, # [b, h, c, d]
    RFA_V, # [b, h, c, d]
    K, # [b, h, n, d]
    V, # [b, h, n, d]
    PARAM_MU, # [1, h, 1, 1, d]
    PARAM_PHI,  # [1, h, 1, 1, d]
    Mask, # [b, h, n, 1]
    D_RFA_K, # [b, h, c, d]
    D_RFA_V, # [b, h, c, d]
    D_K, # [b, h, n, d]
    D_V, # [b, h, n, d]
    D_PARAM_MU_PARTIAL, # [b, h, g, d]
    D_PARAM_PHI_PARTIAL, # [b, h, g, d]
    softmax_scale,
    stride_rfa_k_b, stride_rfa_k_h, stride_rfa_k_c,
    stride_rfa_v_b, stride_rfa_v_h, stride_rfa_v_c,
    stride_kb, stride_kh, stride_kn,
    stride_vb, stride_vh, stride_vn,
    stride_mu_h,
    stride_phi_h,
    stride_mb, stride_mn,
    stride_d_rfa_k_b, stride_d_rfa_k_h, stride_d_rfa_k_c,
    stride_d_rfa_v_b, stride_d_rfa_v_h, stride_d_rfa_v_c,
    stride_d_k_b, stride_d_k_h, stride_d_k_n,
    stride_d_v_b, stride_d_v_h, stride_d_v_n,
    stride_d_mu_b, stride_d_mu_h, stride_d_mu_g,
    stride_d_phi_b, stride_d_phi_h, stride_d_phi_g,
    nheads,
    seqlen,
    nchunks,
    headdim,
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

    offs_rfa_c = start_n * CHUNKS_PER_BLOCK + offs_c

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
    rfa_k_ptrs = (
        RFA_K +
        offs_b * stride_rfa_k_b +
        offs_h * stride_rfa_k_h +
        (offs_rfa_c[:, None] * stride_rfa_k_c + offs_d[None, :])
    )
    rfa_v_ptrs = (
        RFA_V +
        offs_b * stride_rfa_v_b +
        offs_h * stride_rfa_v_h +
        (offs_rfa_c[:, None] * stride_rfa_v_c + offs_d[None, :])
    )

    d_rfa_k_ptrs = (
        D_RFA_K +
        offs_b * stride_d_rfa_k_b +
        offs_h * stride_d_rfa_k_h +
        (offs_rfa_c[:, None] * stride_d_rfa_k_c + offs_d[None, :])
    )
    d_rfa_v_ptrs = (
        D_RFA_V +
        offs_b * stride_d_rfa_v_b +
        offs_h * stride_d_rfa_v_h +
        (offs_rfa_c[:, None] * stride_d_rfa_v_c + offs_d[None, :])
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
            Mask +
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

    if EVEN_N:
        if EVEN_HEADDIM:
            rfa_k = tl.load(
                rfa_k_ptrs
            )
        else:
            rfa_k = tl.load(
                rfa_k_ptrs,
                mask=offs_d[None, :] < headdim,
                other=0.0
            )
    else:
        if EVEN_HEADDIM:
            rfa_k = tl.load(
                rfa_k_ptrs,
                mask=offs_rfa_c[:, None] < nchunks,
                other=0.0
            )
        else:
            rfa_k = tl.load(
                rfa_k_ptrs,
                mask=(offs_rfa_c[:, None] < nchunks) & (offs_d[None, :] < headdim),
                other=0.0
            )
    
    if EVEN_N:
        if EVEN_HEADDIM:
            d_rfa_k = tl.load(
                d_rfa_k_ptrs
            )
        else:
            d_rfa_k = tl.load(
                d_rfa_k_ptrs,
                mask=offs_d[None, :] < headdim,
                other=0.0
            )
    else:
        if EVEN_HEADDIM:
            d_rfa_k = tl.load(
                d_rfa_k_ptrs,
                mask=offs_rfa_c[:, None] < nchunks,
                other=0.0
            )
        else:
            d_rfa_k = tl.load(
                d_rfa_k_ptrs,
                mask=(offs_rfa_c[:, None] < nchunks) & (offs_d[None, :] < headdim),
                other=0.0
            )
    
    param_mu = tl.load(param_mu_ptrs).to(k.dtype)
    mu_c_w = tl.zeros([CHUNKS_PER_BLOCK, CHUNK_SIZE], dtype=tl.float32)
    mu_c_w += tl.sum(k * param_mu, axis=-1)
    mu_c_w *= log2e

    if not EVEN_N:  # Need to mask out otherwise the softmax is wrong
        mu_c_w += tl.where(
            (
                start_n * BLOCK_N +
                offs_c[:, None] * CHUNK_SIZE + 
                offs_m[None, :]
            ) < seqlen, 
            0, 
            float("-inf")
        )

    if MASK_TYPE == 1:
        if EVEN_N:
            mask = tl.load(
                m_ptrs
            )
        else:
            mask = tl.load(
                m_ptrs,
                mask=(
                        start_n * BLOCK_N +
                        offs_c[:, None] * CHUNK_SIZE + 
                        offs_m[None, :]
                    ) < seqlen,
                other=1,
            )
        mu_c_w = tl.where(mask, float("-inf"), mu_c_w)

    # [c, w]
    m_mu_c_w = tl.max(mu_c_w, axis=-1)
    masked_out_rows_mu = (m_mu_c_w == float("-inf"))
    m_mu_c_w_masked = tl.where(masked_out_rows_mu, 0, m_mu_c_w)
    mu_c_w = tl.exp2(mu_c_w - m_mu_c_w_masked[:, None])
    denom_mu = tl.sum(mu_c_w, axis=-1)
    denom_mu = tl.where(denom_mu == 0.0, 1.0, denom_mu)
    mu_tilde_c_w = mu_c_w / denom_mu[:, None]
    mu_tilde_c_w = mu_tilde_c_w.to(k.dtype)
    # [c, d] [c, w, d] -> [c, w]
    d_mu_tilde_c_w = tl.sum(d_rfa_k[:, None, :] * k, axis=-1)
    # [c, d] [c, d] -> [c]
    d_out_rfa_k_t_rfa_k = tl.sum(d_rfa_k * rfa_k, axis=-1)[:, None]
    d_mu_c_w = (d_mu_tilde_c_w - d_out_rfa_k_t_rfa_k) * mu_tilde_c_w

    # [c, w] [c, w, d] -> [d]
    d_param_mu = tl.sum(tl.sum(d_mu_c_w[:, :, None] * k, axis=0), axis=0)
    # [c, w] [c, d] + [c, w] [1, 1, d] -> [c, w, d]
    d_k = mu_tilde_c_w[:, :, None] * d_rfa_k[:, None, :] + d_mu_c_w[:, :, None] * param_mu

    d_param_mu_partial_ptrs = (
        D_PARAM_MU_PARTIAL +
        offs_b * stride_d_mu_b +
        offs_h * stride_d_mu_h +
        start_n * stride_d_mu_g +
        offs_d
    )
    if EVEN_HEADDIM:
        tl.store(
            d_param_mu_partial_ptrs, d_param_mu
        )
    else:
        tl.store(
            d_param_mu_partial_ptrs, d_param_mu,
            mask=offs_d < headdim
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


    if EVEN_N:
        if EVEN_HEADDIM:
            rfa_v = tl.load(
                rfa_v_ptrs
            )
        else:
            rfa_v = tl.load(
                rfa_v_ptrs,
                mask=offs_d[None, :] < headdim,
                other=0.0
            )
    else:
        if EVEN_HEADDIM:
            rfa_v = tl.load(
                rfa_v_ptrs,
                mask=offs_rfa_c[:, None] < nchunks,
                other=0.0
            )
        else:
            rfa_v = tl.load(
                rfa_v_ptrs,
                mask=(offs_rfa_c[:, None] < nchunks) & (offs_d[None, :] < headdim),
                other=0.0
            )
    
    if EVEN_N:
        if EVEN_HEADDIM:
            d_rfa_v = tl.load(
                d_rfa_v_ptrs
            )
        else:
            d_rfa_v = tl.load(
                d_rfa_v_ptrs,
                mask=offs_d[None, :] < headdim,
                other=0.0
            )
    else:
        if EVEN_HEADDIM:
            d_rfa_v = tl.load(
                d_rfa_v_ptrs,
                mask=offs_rfa_c[:, None] < nchunks,
                other=0.0
            )
        else:
            d_rfa_v = tl.load(
                d_rfa_v_ptrs,
                mask=(offs_rfa_c[:, None] < nchunks) & (offs_d[None, :] < headdim),
                other=0.0
            )
    
    param_phi = tl.load(param_phi_ptrs).to(k.dtype)
    phi_c_w = tl.zeros([CHUNKS_PER_BLOCK, CHUNK_SIZE], dtype=tl.float32)
    phi_c_w += tl.sum(k * param_phi, axis=-1)
    phi_c_w -= (0.5 * tl.sum(k * k, axis=-1))
    phi_c_w *= log2e * softmax_scale
    if not EVEN_N:  # Need to mask out otherwise the softmax is wrong
        phi_c_w += tl.where(
            (
                start_n * BLOCK_N +
                offs_c[:, None] * CHUNK_SIZE + 
                offs_m[None, :]
            ) < seqlen, 
            0, 
            float("-inf")
        )

    if MASK_TYPE == 1:
        phi_c_w = tl.where(mask, float("-inf"), phi_c_w)


    m_phi_c_w = tl.max(phi_c_w, axis=-1)
    masked_out_rows_phi = (m_phi_c_w == float("-inf"))
    m_phi_c_w_masked = tl.where(masked_out_rows_phi, 0, m_phi_c_w)
    phi_c_w = tl.exp2(phi_c_w - m_phi_c_w_masked[:, None])
    denom_phi = tl.sum(phi_c_w, axis=-1)
    denom_phi = tl.where(denom_phi == 0.0, 1.0, denom_phi)
    phi_tilde_c_w = phi_c_w / denom_phi[:, None]
    # phi_c_w = tl.exp2(phi_c_w - tl.max(phi_c_w, axis=-1)[:, None])
    # phi_tilde_c_w = phi_c_w / tl.sum(phi_c_w, axis=-1)[:, None]
    phi_tilde_c_w = phi_tilde_c_w.to(k.dtype)
    d_phi_tilde_c_w = tl.sum(d_rfa_v[:, None, :] * v, axis=-1)
    d_out_rfa_v_t_rfa_v = tl.sum(d_rfa_v * rfa_v, axis=-1)[:, None]
    d_phi_c_w = (d_phi_tilde_c_w.to(tl.float32) - d_out_rfa_v_t_rfa_v.to(tl.float32)) * phi_tilde_c_w

    d_param_phi = tl.sum(tl.sum(d_phi_c_w[:, :, None] * k * softmax_scale, axis=0), axis=0)
    d_v = phi_tilde_c_w[:, :, None] * d_rfa_v[:, None, :]
    # [c, w, d] + [c, w] * [1, 1, d] - [c, w, d]
    d_k = d_k + softmax_scale * d_phi_c_w[:, :, None] * (param_phi - k)

    d_k_ptrs = (
        D_K +
        offs_b * stride_d_k_b +
        offs_h * stride_d_k_h +
        (
            (
                start_n * BLOCK_N +
                offs_c[:, None, None] * CHUNK_SIZE + 
                offs_m[None, :, None]
            ) * stride_d_k_n + 
            offs_d[None, None, :]
        )
    )
    d_v_ptrs = (
        D_V +
        offs_b * stride_d_v_b +
        offs_h * stride_d_v_h +
        (
            (
                start_n * BLOCK_N +
                offs_c[:, None, None] * CHUNK_SIZE + 
                offs_m[None, :, None]
            ) * stride_d_v_n + 
            offs_d[None, None, :]
        )
    )
    if EVEN_N:
        if EVEN_HEADDIM:
            tl.store(
                d_k_ptrs, d_k
            )
            tl.store(
                d_v_ptrs, d_v
            )
        else:
            tl.store(
                d_k_ptrs, d_k,
                mask=offs_d[None, None, :] < headdim
            )
            tl.store(
                d_v_ptrs, d_v,
                mask=offs_d[None, None, :] < headdim
            )
    else:
        if EVEN_HEADDIM:
            tl.store(
                d_k_ptrs, d_k,
                mask=(
                        (
                            start_n * BLOCK_N +
                            offs_c[:, None, None] * CHUNK_SIZE + 
                            offs_m[None, :, None]
                        ) < seqlen
                    ),
            )
            tl.store(
                d_v_ptrs, d_v,
                mask=(
                        (
                            start_n * BLOCK_N +
                            offs_c[:, None, None] * CHUNK_SIZE + 
                            offs_m[None, :, None]
                        ) < seqlen
                    ),
            )
        else:
            tl.store(
                d_k_ptrs, d_k,
                mask=(
                        (
                            start_n * BLOCK_N +
                            offs_c[:, None, None] * CHUNK_SIZE + 
                            offs_m[None, :, None]
                        ) < seqlen
                    ) & (offs_d[None, None, :] < headdim),
            )
            tl.store(
                d_v_ptrs, d_v,
                mask=(
                        (
                            start_n * BLOCK_N +
                            offs_c[:, None, None] * CHUNK_SIZE + 
                            offs_m[None, :, None]
                        ) < seqlen
                    ) & (offs_d[None, None, :] < headdim),
            )
    d_param_phi_partial_ptrs = (
        D_PARAM_PHI_PARTIAL +
        offs_b * stride_d_phi_b +
        offs_h * stride_d_phi_h +
        start_n * stride_d_phi_g +
        offs_d
    )
    if EVEN_HEADDIM:
        tl.store(
            d_param_phi_partial_ptrs, d_param_phi
        )
    else:
        tl.store(
            d_param_phi_partial_ptrs, d_param_phi,
            mask=offs_d < headdim
        )

def triton_eva_prep_kv_fwd(k, v, param_mu, param_phi, mask, softmax_scale, chunksize):
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
    if mask is not None:
        mask_type = 1
        assert mask.dtype == torch.bool
        assert mask.is_cuda
        assert mask.dim() == 4
        assert mask.shape == (batch, 1, seqlen, 1)
        if mask.stride(-1) != 1:
            mask = mask.contiguous()
    mask_strides = (
        (mask.stride(0), mask.stride(2)) 
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
        mask,
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
        chunks_per_block,
        chunksize,
        mask_type,
        BLOCK_HEADDIM,
        BLOCK_N=BLOCK,
        num_warps=num_warps,
        num_stages=1,
    )
    return out_rfa_k, out_rfa_v

def triton_eva_prep_kv_bwd(
        d_rfa_k, d_rfa_v,
        k, v, param_mu, param_phi, 
        mask, 
        rfa_k, rfa_v,
        d_k, d_v, d_param_mu, d_param_phi,
        softmax_scale, 
        mask_type,
        chunksize
    ):
    d_rfa_k, d_rfa_v = [
        x if x.stride(-1) == 1 else x.contiguous() 
        for x in [d_rfa_k, d_rfa_v]
    ]

    # shape constraints
    batch, nheads, seqlen, head_dim = k.shape
    assert seqlen % chunksize == 0, "seqlen must be divisible by chunksize"
    nchunks = seqlen // chunksize
    softmax_scale = softmax_scale or 1.0 / math.sqrt(head_dim)

    mask_strides = (
        (mask.stride(0), mask.stride(2))
        if mask_type == 1 else 
        (0, 0)
    )

    BLOCK_HEADDIM = max(triton.next_power_of_2(head_dim), 16)
    BLOCK = 128
    num_warps = 4 if head_dim <= 64 else 8
    
    assert (BLOCK > chunksize) & (BLOCK % chunksize) == 0, "BLOCK must be divisible by chunksize"
    chunks_per_block = BLOCK // chunksize

    partial_groups = triton.cdiv(seqlen, BLOCK)
    d_param_mu_partial = torch.zeros((batch, nheads, partial_groups, head_dim), dtype=torch.float32, device=d_rfa_k.device)
    d_param_phi_partial = torch.zeros((batch, nheads, partial_groups, head_dim), dtype=torch.float32, device=d_rfa_k.device)
    grid = lambda META: (partial_groups, batch * nheads)
    _bwd_eva_prep_kv_kernel[grid](
        rfa_k, # [b, h, c, d]
        rfa_v, # [b, h, c, d]
        k, # [b, h, n, d]
        v, # [b, h, n, d]
        param_mu, # [1, h, 1, 1, d]
        param_phi,  # [1, h, 1, 1, d]
        mask, # [b, h, n, 1]
        d_rfa_k, # [b, h, c, d]
        d_rfa_v, # [b, h, c, d]
        d_k, # [b, h, n, d]
        d_v, # [b, h, n, d]
        d_param_mu_partial, # [b, h, g, d]
        d_param_phi_partial, # [b, h, g, d]
        softmax_scale,
        rfa_k.stride(0), rfa_k.stride(1), rfa_k.stride(2),
        rfa_v.stride(0), rfa_v.stride(1), rfa_v.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        param_mu.stride(1),
        param_phi.stride(1),
        mask_strides[0], mask_strides[1],
        d_rfa_k.stride(0), d_rfa_k.stride(1), d_rfa_k.stride(2),
        d_rfa_v.stride(0), d_rfa_v.stride(1), d_rfa_v.stride(2),
        d_k.stride(0), d_k.stride(1), d_k.stride(2),
        d_v.stride(0), d_v.stride(1), d_v.stride(2),
        d_param_mu_partial.stride(0), d_param_mu_partial.stride(1), d_param_mu_partial.stride(2),
        d_param_phi_partial.stride(0), d_param_phi_partial.stride(1), d_param_phi_partial.stride(2),
        nheads,
        seqlen,
        nchunks,
        head_dim,
        chunks_per_block,
        chunksize,
        mask_type,
        BLOCK_HEADDIM,
        BLOCK_N=BLOCK,
        num_warps=num_warps,
        num_stages=1,
    )
    d_param_mu.copy_(d_param_mu_partial.sum(dim=(0, -2), keepdim=True).unsqueeze(-2).to(d_param_mu.dtype))
    d_param_phi.copy_(d_param_phi_partial.sum(dim=(0, -2), keepdim=True).unsqueeze(-2).to(d_param_phi.dtype))



class EvaPrepKVFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, k, v, param_mu, param_phi, mask, softmax_scale=None, chunksize=None):
        if mask is not None:
            mask_type = 1
        else:
            mask_type = 0
        rfa_k, rfa_v = triton_eva_prep_kv_fwd(
            k, v, param_mu, param_phi, mask, softmax_scale, chunksize
        )
        ctx.save_for_backward(k, v, param_mu, param_phi, mask, rfa_k, rfa_v)
        ctx.softmax_scale = softmax_scale
        ctx.chunksize = chunksize
        ctx.mask_type = mask_type
        return rfa_k, rfa_v

    @staticmethod
    def backward(ctx, d_rfa_k, d_rfa_v):
        k, v, param_mu, param_phi, mask, rfa_k, rfa_v = ctx.saved_tensors
        d_k = torch.empty_like(k)
        d_v = torch.empty_like(v)
        d_param_mu = torch.empty_like(param_mu)
        d_param_phi = torch.empty_like(param_phi)
        triton_eva_prep_kv_bwd(
            d_rfa_k, d_rfa_v,
            k, v, param_mu, param_phi, 
            mask, 
            rfa_k, rfa_v,
            d_k, d_v, d_param_mu, d_param_phi,
            ctx.softmax_scale, 
            ctx.mask_type,
            ctx.chunksize
        )
        return d_k, d_v, d_param_mu, d_param_phi, None, None, None

def eva_prep_kv_func_triton(
        k, v, 
        param_mu, param_phi,
        mask, 
        softmax_scale=None, chunksize=None
    ):
    return EvaPrepKVFunc.apply(
        k, v, 
        param_mu, param_phi,
        mask, 
        softmax_scale, chunksize
    )
