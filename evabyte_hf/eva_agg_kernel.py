
import math
import torch
import triton
import triton.language as tl

# Disabling autotune for now, set num_warps=4 if headdim=64 and num_warps=8 if headdim=128
# @triton.autotune(
#     configs=[
#         triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=4, num_stages=1),
#         # This config has a race condition when EVEN_M == False, disabling it for now.
#         # triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=4, num_stages=1),
#     ],
#     key=['CACHE_KEY_SEQLEN_Q', 'CACHE_KEY_SEQLEN_K', 'BIAS_TYPE', 'IS_CAUSAL', 'BLOCK_HEADDIM']
# )
@triton.heuristics(
    {
        "EVEN_M": lambda args: args["seqlen_q"] % args["BLOCK_M"] == 0,
        "EVEN_N": lambda args: args["seqlen_k"] % args["BLOCK_N"] == 0,
        "EVEN_C": lambda args: args["nchunks"] % args["BLOCK_N"] == 0,
        "EVEN_W": lambda args: args["WINDOW_SIZE"] % args["BLOCK_N"] == 0,
        "EVEN_HEADDIM": lambda args: args["headdim"] == args["BLOCK_HEADDIM"],
    }
)
@triton.jit
def _fwd_eva_agg_kernel(
    Q,
    K,
    V,
    RFA_K,
    RFA_V,
    WindowMask,
    Out,
    softmax_scale,
    stride_qb, stride_qh, stride_qm,
    stride_kb, stride_kh, stride_kn,
    stride_vb, stride_vh, stride_vn,
    stride_rfa_kb, stride_rfa_kh, stride_rfa_kc,
    stride_rfa_vb, stride_rfa_vh, stride_rfa_vc,
    stride_mb, stride_mm,
    stride_ob, stride_oh, stride_om,
    nheads,
    seqlen_q,
    seqlen_k,
    nchunks,
    headdim,
    CACHE_KEY_SEQLEN_Q, # TODO: why keeping this
    CACHE_KEY_SEQLEN_K, # TODO: why keeping this
    CACHE_KEY_NCHUNKS, # TODO: why keeping this
    CHUNKS_PER_WINDOW: tl.constexpr,
    WINDOW_SIZE: tl.constexpr,
    MASK_TYPE: tl.constexpr,
    EMPTY_RFA_KV: tl.constexpr,
    BLOCK_HEADDIM: tl.constexpr,
    EVEN_M: tl.constexpr,
    EVEN_N: tl.constexpr,
    EVEN_W: tl.constexpr,
    EVEN_C: tl.constexpr,
    EVEN_HEADDIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_bh = tl.program_id(1)
    off_h = off_bh % nheads
    off_b = off_bh // nheads
    # initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_w = (start_m * BLOCK_M) // WINDOW_SIZE
    offs_n = tl.arange(0, BLOCK_N)
    offs_c = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_HEADDIM)
    # TODO: add paratheses or not
    q_ptrs = (
        Q +
        off_b * stride_qb +
        off_h * stride_qh +
        (offs_m[:, None] * stride_qm + offs_d[None, :])
    )
    k_ptrs = (
        K +
        off_b * stride_kb +
        off_h * stride_kh +
        (offs_n[:, None] * stride_kn + offs_d[None, :])
    )
    v_ptrs = (
        V +
        off_b * stride_vb +
        off_h * stride_vh +
        (offs_n[:, None] * stride_vn + offs_d[None, :])
    )
    if EMPTY_RFA_KV == 0:
        rfa_k_ptrs = (
            RFA_K +
            off_b * stride_rfa_kb +
            off_h * stride_rfa_kh +
            (offs_c[:, None] * stride_rfa_kc + offs_d[None, :])
        )
        rfa_v_ptrs = (
            RFA_V +
            off_b * stride_rfa_vb +
            off_h * stride_rfa_vh +
            (offs_c[:, None] * stride_rfa_vc + offs_d[None, :])
        )

    qk_scale = softmax_scale
    qk_scale *= 1.4426950408889634  # log2(e)
    if MASK_TYPE == 1:
        m_ptrs = (
            WindowMask +
            off_b * stride_mb +
            (offs_m[:, None] * stride_mm + offs_n[None, :])
        )
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    d_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc_o = tl.zeros([BLOCK_M, BLOCK_HEADDIM], dtype=tl.float32)
    # load q: it will stay in SRAM throughout
    # [2022-10-30] TD: Triton bug - in the case of EVEN_M=True and EVEN_N=False, if we just call
    # tl.load(q_ptrs), we get the wrong output!
    if EVEN_M & EVEN_N:
        if EVEN_HEADDIM:
            q = tl.load(
                q_ptrs
            )
        else:
            q = tl.load(
                q_ptrs,
                mask=offs_d[None, :] < headdim,
                other=0.0
            )
    else:
        if EVEN_HEADDIM:
            q = tl.load(
                q_ptrs,
                mask=offs_m[:, None] < seqlen_q,
                other=0.0
            )
        else:
            q = tl.load(
                q_ptrs,
                mask=(offs_m[:, None] < seqlen_q) & (offs_d[None, :] < headdim),
                other=0.0
            )
    # loop over k, v and update accumulator
    # Iterate over local singletons;
    # so we only iterate over blocks within the current window
    start_idx_n = offs_w * WINDOW_SIZE
    end_idx_n = tl.minimum((start_m + 1) * BLOCK_M, seqlen_k)
    for start_n in range(start_idx_n, end_idx_n, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        # -- compute qk ----
        if EVEN_N & EVEN_M:
            if EVEN_HEADDIM:
                k = tl.load(
                    k_ptrs + start_n * stride_kn
                )
            else:
                k = tl.load(
                    k_ptrs + start_n * stride_kn,
                    mask=offs_d[None, :] < headdim,
                    other=0.0
                )
        else:
            if EVEN_HEADDIM:
                k = tl.load(
                    k_ptrs + start_n * stride_kn,
                    mask=(start_n + offs_n)[:, None] < seqlen_k,
                    other=0.0,
                )
            else:
                k = tl.load(
                    k_ptrs + start_n * stride_kn,
                    mask=((start_n + offs_n)[:, None] < seqlen_k) & (offs_d[None, :] < headdim),
                    other=0.0,
                )
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, tl.trans(k))
        # Trying to combine the two masks seem to make the result wrong
        if not EVEN_N:  # Need to mask out otherwise the softmax is wrong
            qk += tl.where((start_n + offs_n)[None, :] < seqlen_k, 0, float("-inf"))

        if MASK_TYPE == 1:
            if EVEN_M & EVEN_W:
                mask = tl.load(
                    m_ptrs + start_n - start_idx_n
                ).to(tl.float32)
            else:
                mask = tl.load(
                    m_ptrs + start_n - start_idx_n,
                    mask=(offs_m[:, None] < seqlen_q)
                    & ((start_n - start_idx_n + offs_n)[None, :] < WINDOW_SIZE),
                    other=0.0,
                ).to(tl.float32)
            # Slightly faster to multiply the softmax_scale in the tl.exp below since the compiler
            # can then fuse the mult and add into an fma instruction. But if we have bias we need to
            # to multiply with softmax_scale here.
            # we assume mask already implies the causal masking
            qk = qk * qk_scale + mask
            m_ij = tl.maximum(tl.max(qk, 1), m_i)
            p = tl.exp2(qk - m_ij[:, None])
        else:
            qk += tl.where(offs_m[:, None] >= (start_n + offs_n)[None, :], 0, float("-inf"))
            m_ij = tl.maximum(tl.max(qk, 1) * qk_scale, m_i)
            p = tl.exp2(qk * qk_scale - m_ij[:, None])

        d_ij = tl.sum(p, 1)

        # scale acc_o
        prev_scale = tl.exp2(m_i - m_ij)
        # # -- update output accumulator --
        acc_o = acc_o * prev_scale[:, None]
        # update acc_o
        if EVEN_N & EVEN_M:  # If we just do "if EVEN_N", there seems to be some race condition
            if EVEN_HEADDIM:
                v = tl.load(
                    v_ptrs + start_n * stride_vn
                )
            else:
                v = tl.load(
                    v_ptrs + start_n * stride_vn,
                    mask=offs_d[None, :] < headdim,
                    other=0.0
                )
        else:
            if EVEN_HEADDIM:
                v = tl.load(
                    v_ptrs + start_n * stride_vn,
                    mask=(start_n + offs_n)[:, None] < seqlen_k,
                    other=0.0,
                )
            else:
                v = tl.load(
                    v_ptrs + start_n * stride_vn,
                    mask=((start_n + offs_n)[:, None] < seqlen_k) & (offs_d[None, :] < headdim),
                    other=0.0,
                )
        p = p.to(v.dtype)
        acc_o = tl.dot(p, v, acc_o)

        # -- update statistics
        d_i = d_i * prev_scale + d_ij
        m_i = m_ij

    if EMPTY_RFA_KV == 0:
        # Iterate over RFA chunks
        # we only iterate over chunks before the current local singleton window
        end_idx_c = tl.minimum(offs_w * CHUNKS_PER_WINDOW, nchunks)
        for start_c in range(0, end_idx_c, BLOCK_N):
            start_c = tl.multiple_of(start_c, BLOCK_N)
            # -- compute qk ----
            if EVEN_C & EVEN_M:
                if EVEN_HEADDIM:
                    rfa_k = tl.load(
                        rfa_k_ptrs + start_c * stride_rfa_kc
                    )
                else:
                    rfa_k = tl.load(
                        rfa_k_ptrs + start_c * stride_rfa_kc,
                        mask=offs_d[None, :] < headdim,
                        other=0.0
                    )
            else:
                if EVEN_HEADDIM:
                    rfa_k = tl.load(
                        rfa_k_ptrs + start_c * stride_rfa_kc,
                        mask=(start_c + offs_c)[:, None] < nchunks,
                        other=0.0,
                    )
                else:
                    rfa_k = tl.load(
                        rfa_k_ptrs + start_c * stride_rfa_kc,
                        mask=((start_c + offs_c)[:, None] < nchunks) & (offs_d[None, :] < headdim),
                        other=0.0,
                    )
            qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
            qk += tl.dot(q, tl.trans(rfa_k))
            # Trying to combine the two masks seem to make the result wrong
            if not EVEN_C:  # Need to mask out otherwise the softmax is wrong
                qk += tl.where((start_c + offs_c)[None, :] < nchunks, 0, float("-inf"))

            m_ij = tl.maximum(tl.max(qk, 1) * qk_scale, m_i)
            p = tl.exp2(qk * qk_scale - m_ij[:, None])

            d_ij = tl.sum(p, 1)

            # scale acc_o
            prev_scale = tl.exp2(m_i - m_ij)
            # # -- update output accumulator --
            acc_o = acc_o * prev_scale[:, None]
            # update acc_o
            # TODO: If we just do "if EVEN_N", there seems to be some race condition ?
            if EVEN_C & EVEN_M:  
                if EVEN_HEADDIM:
                    rfa_v = tl.load(
                        rfa_v_ptrs + start_c * stride_rfa_vc
                    )
                else:
                    rfa_v = tl.load(
                        rfa_v_ptrs + start_c * stride_rfa_vc,
                        mask=offs_d[None, :] < headdim,
                        other=0.0
                    )
            else:
                if EVEN_HEADDIM:
                    rfa_v = tl.load(
                        rfa_v_ptrs + start_c * stride_rfa_vc,
                        mask=(start_c + offs_n)[:, None] < nchunks,
                        other=0.0,
                    )
                else:
                    rfa_v = tl.load(
                        rfa_v_ptrs + start_c * stride_rfa_vc,
                        mask=((start_c + offs_n)[:, None] < nchunks) & (offs_d[None, :] < headdim),
                        other=0.0,
                    )
            p = p.to(rfa_v.dtype)
            acc_o = tl.dot(p, rfa_v, acc_o)

            # -- update statistics
            d_i = d_i * prev_scale + d_ij
            m_i = m_ij

    # BUG: have to store and immediately load
    acc_o = acc_o / d_i[:, None]
    # TODO: understand why rematerialize offsets to save registers?
    start_m = tl.program_id(0)
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_HEADDIM)
    out_ptrs = (
        Out +
        off_b * stride_ob +
        off_h * stride_oh +
        (offs_m[:, None] * stride_om + offs_d[None, :])
    )
    if EVEN_M:
        if EVEN_HEADDIM:
            tl.store(
                out_ptrs, acc_o
            )
        else:
            tl.store(
                out_ptrs, acc_o,
                mask=offs_d[None, :] < headdim
            )
    else:
        if EVEN_HEADDIM:
            tl.store(
                out_ptrs, acc_o,
                mask=offs_m[:, None] < seqlen_q
            )
        else:
            tl.store(
                out_ptrs, acc_o,
                mask=(offs_m[:, None] < seqlen_q) & (offs_d[None, :] < headdim)
            )

def triton_eva_agg_fwd(q, k, v, rfa_k, rfa_v, window_mask, softmax_scale, window_size, chunks_per_window):
    if rfa_k is None and rfa_v is None:
        empty_rfa_kv = 1

        q, k, v = [
            x if x.stride(-1) == 1 else x.contiguous() 
            for x in [q, k, v]
        ]
    else:
        assert rfa_k is not None and rfa_v is not None, "Both rfa_k and rfa_v must either be None or have values at the same time."
        empty_rfa_kv = 0

        q, k, v, rfa_k, rfa_v = [
            x if x.stride(-1) == 1 else x.contiguous() 
            for x in [q, k, v, rfa_k, rfa_v]
        ]

    # shape constraints
    batch, nheads, seqlen_q, head_dim = q.shape
    _,     _,      seqlen_k, _        = k.shape
    if empty_rfa_kv == 0:
        nchunks = rfa_k.shape[-2]
        assert rfa_k.shape == (batch, nheads, nchunks, head_dim)
        assert rfa_v.shape == (batch, nheads, nchunks, head_dim)
        assert q.dtype == k.dtype == v.dtype == rfa_k.dtype == rfa_v.dtype
    else:
        nchunks = 0
        assert q.dtype == k.dtype == v.dtype, "All tensors must have the same type"
    assert k.shape == (batch, nheads, seqlen_k, head_dim)
    assert v.shape == (batch, nheads, seqlen_k, head_dim)

    assert head_dim <= 128, "We only test head dimensions up to 128"
    # assert q.dtype in [torch.float16, torch.bfloat16], "Only support fp16 and bf16"
    assert q.dtype in [torch.bfloat16, torch.float], "Only support bf16 and fp32 for now"
    assert q.is_cuda and k.is_cuda and v.is_cuda
    softmax_scale = softmax_scale or 1.0 / math.sqrt(head_dim)

    mask_type = 0
    if window_mask is not None:
        mask_type = 1
        assert window_mask.dtype == q.dtype, torch.float
        assert window_mask.is_cuda
        assert window_mask.dim() == 4
        assert window_mask.shape == (batch, 1, seqlen_q, window_size)
        if window_mask.stride(-1) != 1:
            window_mask = window_mask.contiguous()
    mask_strides = (
        (window_mask.stride(0), window_mask.stride(2)) 
        if mask_type == 1 else 
        (0, 0)
    )

    rfa_k_strides = (
        (rfa_k.stride(0), rfa_k.stride(1), rfa_k.stride(2))
        if empty_rfa_kv == 0 else
        (0, 0, 0)
    )
    rfa_v_strides = (
        (rfa_v.stride(0), rfa_v.stride(1), rfa_v.stride(2))
        if empty_rfa_kv == 0 else
        (0, 0, 0)
    )
    assert chunks_per_window > 0, "chunks_per_window must be greater than 0"

    o = torch.empty_like(q)

    BLOCK_HEADDIM = max(triton.next_power_of_2(head_dim), 16)
    if q.dtype == torch.float:
        BLOCK = 64
    else:
        BLOCK = 128
    num_warps = 4 if head_dim <= 64 else 8
    assert chunks_per_window >= BLOCK, "chunks_per_window must be greater than BLOCK" 
    # WINDOW_MASK_TYPE:
    # - 0: regular causal mask, simply None
    # - 1: the shape must be B, 1, W, I, J

    grid = lambda META: (triton.cdiv(seqlen_q, META["BLOCK_M"]), batch * nheads)
    _fwd_eva_agg_kernel[grid](
        q,
        k,
        v,
        rfa_k,
        rfa_v,
        window_mask,
        o,
        softmax_scale,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        rfa_k_strides[0], rfa_k_strides[1], rfa_k_strides[2],
        rfa_v_strides[0], rfa_v_strides[1], rfa_v_strides[2],
        mask_strides[0], mask_strides[1],
        o.stride(0), o.stride(1), o.stride(2),
        nheads,
        seqlen_q,
        seqlen_k,
        nchunks,
        head_dim,
        seqlen_q // 32,
        seqlen_k // 32,
        nchunks // 32,
        chunks_per_window,
        window_size,
        mask_type,
        empty_rfa_kv,
        BLOCK_HEADDIM,
        BLOCK_M=BLOCK,
        BLOCK_N=BLOCK,
        num_warps=num_warps,
        num_stages=1,
    )
    return o
