
import math
import torch
import triton
import triton.language as tl

@triton.heuristics(
    {
        "EVEN_M": lambda args: args["seqlen_q"] % args["BLOCK_M"] == 0,
        "EVEN_N": lambda args: args["seqlen_k"] % args["BLOCK_N"] == 0,
        "EVEN_W": lambda args: args["WINDOW_SIZE"] % args["BLOCK_N"] == 0,
        "EVEN_HEADDIM": lambda args: args["headdim"] == args["BLOCK_HEADDIM"],
    }
)
@triton.jit
def _bwd_eva_agg_kernel_dkdv(
    Q,
    K,
    V,
    WindowMask,
    DO,
    LSE,
    DO_T_O,
    DK,
    DV,
    softmax_scale,
    stride_qb, stride_qh, stride_qm,
    stride_kb, stride_kh, stride_kn,
    stride_vb, stride_vh, stride_vn,
    stride_window_mask_b, stride_window_mask_m,
    stride_do_b, stride_do_h, stride_do_m,
    stride_lse_b, stride_lse_h,
    stride_do_t_o_b, stride_do_t_o_h,
    stride_dk_b, stride_dk_h, stride_dk_n,
    stride_dv_b, stride_dv_h, stride_dv_n,
    nheads,
    seqlen_q,
    seqlen_k,
    headdim,
    WINDOW_SIZE: tl.constexpr,
    MASK_TYPE: tl.constexpr,
    BLOCK_HEADDIM: tl.constexpr,
    EVEN_M: tl.constexpr,
    EVEN_N: tl.constexpr,
    EVEN_W: tl.constexpr,
    EVEN_HEADDIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    off_bh = tl.program_id(1)
    off_h = off_bh % nheads
    off_b = off_bh // nheads

    start_n = tl.program_id(0)
    # determine which window the current KV block belongs to
    offs_w = (start_n * BLOCK_N) // WINDOW_SIZE
    offs_n = start_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_m = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_HEADDIM)

    # initialize pointers
    q_ptrs = (
        Q + 
        off_b * stride_qb + 
        off_h * stride_qh +
        offs_m[:, None] * stride_qm + offs_d[None, :]
    )
    k_ptrs = (
        K + 
        off_b * stride_kb + 
        off_h * stride_kh +
        offs_n[:, None] * stride_kn + offs_d[None, :]
    )
    v_ptrs = (
        V + 
        off_b * stride_vb + 
        off_h * stride_vh +
        offs_n[:, None] * stride_vn + offs_d[None, :]
    )
    do_ptrs = (
        DO + 
        off_b * stride_do_b + 
        off_h * stride_do_h +
        offs_m[:, None] * stride_do_m + offs_d[None, :]
    )
    do_t_o_ptrs = (
        DO_T_O + 
        off_b * stride_do_t_o_b + 
        off_h * stride_do_t_o_h +
        offs_m[:, None]
    )
    lse_ptrs = (
        LSE + 
        off_b * stride_lse_b + 
        off_h * stride_lse_h +
        offs_m[:, None]
    )
    if MASK_TYPE == 1:
        m_ptrs = (
            WindowMask +
            off_b * stride_window_mask_b +
            (offs_m[:, None] * stride_window_mask_m + offs_n[None, :])
        )
    dk_ptrs = (
        DK + 
        off_b * stride_dk_b + 
        off_h * stride_dk_h +
        offs_n[:, None] * stride_dk_n + offs_d[None, :]
    )
    dv_ptrs = (
        DV + 
        off_b * stride_dv_b + 
        off_h * stride_dv_h +
        offs_n[:, None] * stride_dv_n + offs_d[None, :]
    )

    # 1. for singletons
    # determine start and end of query block
    begin_m = ((start_n * BLOCK_N) // BLOCK_M) * BLOCK_M
    end_m = tl.minimum((offs_w + 1) * WINDOW_SIZE, seqlen_q)

    dk = tl.zeros([BLOCK_N, BLOCK_HEADDIM], dtype=tl.float32)
    dv = tl.zeros([BLOCK_N, BLOCK_HEADDIM], dtype=tl.float32)
    if EVEN_N & EVEN_M:
        if EVEN_HEADDIM:
            k = tl.load(k_ptrs)
            v = tl.load(v_ptrs)
        else:
            k = tl.load(k_ptrs, mask=offs_d[None, :] < headdim, other=0.0)
            v = tl.load(v_ptrs, mask=offs_d[None, :] < headdim, other=0.0)
    else:
        if EVEN_HEADDIM:
            k = tl.load(k_ptrs, mask=offs_n[:, None] < seqlen_k, other=0.0)
            v = tl.load(v_ptrs, mask=offs_n[:, None] < seqlen_k, other=0.0)
        else:
            k = tl.load(
                k_ptrs, mask=(offs_n[:, None] < seqlen_k) & (offs_d[None, :] < headdim), other=0.0
            )
            v = tl.load(
                v_ptrs, mask=(offs_n[:, None] < seqlen_k) & (offs_d[None, :] < headdim), other=0.0
            )
    for start_m in range(begin_m, end_m, BLOCK_M):
        start_m = tl.multiple_of(start_m, BLOCK_M)
        # load q, do, and lse
        if EVEN_M & EVEN_N:
            if EVEN_HEADDIM:
                q = tl.load(
                    q_ptrs + start_m * stride_qm
                )
                do = tl.load(
                    do_ptrs + start_m * stride_do_m
                )
            else:
                q = tl.load(
                    q_ptrs + start_m * stride_qm, 
                    mask=offs_d[None, :] < headdim, 
                    other=0.0
                )
                do = tl.load(
                    do_ptrs + start_m * stride_do_m, 
                    mask=offs_d[None, :] < headdim, 
                    other=0.0
                )
            do_t_o = tl.load(
                do_t_o_ptrs + start_m
            )
            lse = tl.load(
                lse_ptrs + start_m
            )
        else:
            if EVEN_HEADDIM:
                q = tl.load(
                    q_ptrs + start_m * stride_qm, 
                    mask=(start_m + offs_m)[:, None] < seqlen_q, 
                    other=0.0
                )
                do = tl.load(
                    do_ptrs + start_m * stride_do_m, 
                    mask=(start_m + offs_m)[:, None] < seqlen_q, 
                    other=0.0
                )
            else:
                q = tl.load(
                    q_ptrs + start_m * stride_qm, 
                    mask=((start_m + offs_m)[:, None] < seqlen_q) & (offs_d[None, :] < headdim), 
                    other=0.0
                )
                do = tl.load(
                    do_ptrs + start_m * stride_do_m, 
                    mask=((start_m + offs_m)[:, None] < seqlen_q) & (offs_d[None, :] < headdim), 
                    other=0.0
                )
            do_t_o = tl.load(
                do_t_o_ptrs + start_m, 
                mask=(start_m + offs_m)[:, None] < seqlen_q, 
                other=0.0
            )
            lse = tl.load(
                lse_ptrs + start_m, 
                mask=(start_m + offs_m)[:, None] < seqlen_q, 
                other=0.0
            )
        lse = tl.where(lse == float("-inf"), 0.0, lse)
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, tl.trans(k))
        if not EVEN_M:
            qk += tl.where((start_m + offs_m)[:, None] < seqlen_q, 0, float("-inf"))

        if MASK_TYPE == 1:
            if EVEN_M & EVEN_W:
                mask = tl.load(
                    m_ptrs + (start_m * stride_window_mask_m) - (offs_w * WINDOW_SIZE)
                )
            else:
                mask = tl.load(
                    m_ptrs + (start_m * stride_window_mask_m) - (offs_w * WINDOW_SIZE),
                    mask=((start_m + offs_m)[:, None] < seqlen_q)
                    & (((start_m * stride_window_mask_m) - (offs_w * WINDOW_SIZE) + offs_n)[None, :] < WINDOW_SIZE),
                    other=1,
                )
            # Slightly faster to multiply the softmax_scale in the tl.exp below since the compiler
            # can then fuse the mult and add into an fma instruction. But if we have bias we need to
            # to multiply with softmax_scale here.
            # we assume mask already implies the causal masking
            qk = qk * softmax_scale
            qk = tl.where(mask, float("-inf"), qk)
            p = tl.exp(qk - lse)
        else:
            qk += tl.where((start_m + offs_m)[:, None] >= offs_n[None, :], 0, float("-inf"))
            p = tl.exp(qk * softmax_scale - lse)

        # dp [M, N]
        dp = tl.dot(do, tl.trans(v))
        # p [M, N],  dp [M, N], do_t_o [M, 1] -> ds [M, N]
        ds = (p * (dp - do_t_o) * softmax_scale).to(q.dtype)
        # p is fp32 and [M, N], convert to q.dtype
        # do [M, D] -> dv [N, D]
        dv += tl.dot(tl.trans(p.to(do.dtype)), do)
        # dk [N, D]
        dk += tl.dot(tl.trans(ds), q) 
    if EVEN_N & EVEN_M:
        if EVEN_HEADDIM:
            tl.store(dv_ptrs, dv)
            tl.store(dk_ptrs, dk)
        else:
            tl.store(dv_ptrs, dv, mask=offs_d[None, :] < headdim)
            tl.store(dk_ptrs, dk, mask=offs_d[None, :] < headdim)
    else:
        if EVEN_HEADDIM:
            tl.store(dv_ptrs, dv, mask=offs_n[:, None] < seqlen_k)
            tl.store(dk_ptrs, dk, mask=offs_n[:, None] < seqlen_k)
        else:
            tl.store(dv_ptrs, dv, mask=(offs_n[:, None] < seqlen_k) & (offs_d[None, :] < headdim))
            tl.store(dk_ptrs, dk, mask=(offs_n[:, None] < seqlen_k) & (offs_d[None, :] < headdim))

@triton.heuristics(
    {
        "EVEN_M": lambda args: args["seqlen_q"] % args["BLOCK_M"] == 0,
        "EVEN_C": lambda args: args["nchunks"] % args["BLOCK_N"] == 0,
        "EVEN_HEADDIM": lambda args: args["headdim"] == args["BLOCK_HEADDIM"],
    }
)
@triton.jit
def _bwd_eva_agg_kernel_drfa_kv(
    Q,
    RFA_K,
    RFA_V,
    ChunkMask,
    DO,
    LSE,
    DO_T_O,
    D_RFA_K,
    D_RFA_V,
    softmax_scale,
    stride_qb, stride_qh, stride_qm,
    stride_rfa_kb, stride_rfa_kh, stride_rfa_kc,
    stride_rfa_vb, stride_rfa_vh, stride_rfa_vc,
    stride_chunk_mask_b, stride_chunk_mask_m,
    stride_do_b, stride_do_h, stride_do_m,
    stride_lse_b, stride_lse_h,
    stride_do_t_o_b, stride_do_t_o_h,
    stride_d_rfa_k_b, stride_d_rfa_k_h, stride_d_rfa_k_c,
    stride_d_rfa_v_b, stride_d_rfa_v_h, stride_d_rfa_v_c,
    nheads,
    seqlen_q,
    nchunks,
    headdim,
    CHUNKS_PER_WINDOW: tl.constexpr,
    WINDOW_SIZE: tl.constexpr,
    MASK_TYPE: tl.constexpr,
    BLOCK_HEADDIM: tl.constexpr,
    EVEN_M: tl.constexpr,
    EVEN_C: tl.constexpr,
    EVEN_HEADDIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    off_bh = tl.program_id(1)
    off_h = off_bh % nheads
    off_b = off_bh // nheads
    start_c = tl.program_id(0)
    # there are 128 chunks per window
    offs_c = start_c * BLOCK_N + tl.arange(0, BLOCK_N)
    # determine which window the current KV block belongs to
    offs_w = (start_c * BLOCK_N) // CHUNKS_PER_WINDOW
    offs_m = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_HEADDIM)

    # initialize pointers
    q_ptrs = (
        Q + 
        off_b * stride_qb + 
        off_h * stride_qh +
        (offs_m[:, None] * stride_qm + offs_d[None, :])
    )
    do_ptrs = (
        DO + 
        off_b * stride_do_b + 
        off_h * stride_do_h +
        (offs_m[:, None] * stride_do_m + offs_d[None, :])
    )
    do_t_o_ptrs = (
        DO_T_O + 
        off_b * stride_do_t_o_b + 
        off_h * stride_do_t_o_h +
        (offs_m[:, None])
    )
    lse_ptrs = (
        LSE + 
        off_b * stride_lse_b + 
        off_h * stride_lse_h +
        (offs_m[:, None])
    )
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
    if MASK_TYPE == 1:
        rfa_m_ptrs = (
            ChunkMask +
            off_b * stride_chunk_mask_b +
            (offs_m[:, None] * stride_chunk_mask_m + offs_c[None, :])
        )
    d_rfa_k_ptrs = (
        D_RFA_K + 
        off_b * stride_d_rfa_k_b + 
        off_h * stride_d_rfa_k_h +
        (offs_c[:, None] * stride_d_rfa_k_c + offs_d[None, :])
    )
    d_rfa_v_ptrs = (
        D_RFA_V + 
        off_b * stride_d_rfa_v_b + 
        off_h * stride_d_rfa_v_h +
        (offs_c[:, None] * stride_d_rfa_v_c + offs_d[None, :])
    )

    d_rfa_k = tl.zeros([BLOCK_N, BLOCK_HEADDIM], dtype=tl.float32)
    d_rfa_v = tl.zeros([BLOCK_N, BLOCK_HEADDIM], dtype=tl.float32)
    if EVEN_C & EVEN_M:
        if EVEN_HEADDIM:
            rfa_k = tl.load(rfa_k_ptrs)
            rfa_v = tl.load(rfa_v_ptrs)
        else:
            rfa_k = tl.load(rfa_k_ptrs, mask=offs_d[None, :] < headdim, other=0.0)
            rfa_v = tl.load(rfa_v_ptrs, mask=offs_d[None, :] < headdim, other=0.0)
    else:
        if EVEN_HEADDIM:
            rfa_k = tl.load(rfa_k_ptrs, mask=offs_c[:, None] < nchunks, other=0.0)
            rfa_v = tl.load(rfa_v_ptrs, mask=offs_c[:, None] < nchunks, other=0.0)
        else:
            rfa_k = tl.load(
                rfa_k_ptrs, mask=(offs_c[:, None] < nchunks) & (offs_d[None, :] < headdim), other=0.0
            )
            rfa_v = tl.load(
                rfa_v_ptrs, mask=(offs_c[:, None] < nchunks) & (offs_d[None, :] < headdim), other=0.0
            )
    begin_m = tl.minimum((offs_w + 1) * WINDOW_SIZE, seqlen_q)
    end_m = seqlen_q
    for start_m in range(begin_m, end_m, BLOCK_M):
        start_m = tl.multiple_of(start_m, BLOCK_M)
        # load q, do, and lse
        if EVEN_M:
            if EVEN_HEADDIM:
                q = tl.load(
                    q_ptrs + start_m * stride_qm
                )
                do = tl.load(
                    do_ptrs + start_m * stride_do_m
                )
            else:
                q = tl.load(
                    q_ptrs + start_m * stride_qm, 
                    mask=offs_d[None, :] < headdim, 
                    other=0.0
                )
                do = tl.load(
                    do_ptrs + start_m * stride_do_m, 
                    mask=offs_d[None, :] < headdim, 
                    other=0.0
                )
            do_t_o = tl.load(
                do_t_o_ptrs + start_m
            )
            lse = tl.load(
                lse_ptrs + start_m
            )
        else:
            if EVEN_HEADDIM:
                q = tl.load(
                    q_ptrs + start_m * stride_qm, 
                    mask=(start_m + offs_m)[:, None] < seqlen_q, 
                    other=0.0
                )
                do = tl.load(
                    do_ptrs + start_m * stride_do_m, 
                    mask=(start_m + offs_m)[:, None] < seqlen_q, 
                    other=0.0
                )
            else:
                q = tl.load(
                    q_ptrs + start_m * stride_qm, 
                    mask=((start_m + offs_m)[:, None] < seqlen_q) & (offs_d[None, :] < headdim), 
                    other=0.0
                )
                do = tl.load(
                    do_ptrs + start_m * stride_do_m, 
                    mask=((start_m + offs_m)[:, None] < seqlen_q) & (offs_d[None, :] < headdim), 
                    other=0.0
                )
            do_t_o = tl.load(
                do_t_o_ptrs + start_m, 
                mask=(start_m + offs_m)[:, None] < seqlen_q, 
                other=0.0
            )
            lse = tl.load(
                lse_ptrs + start_m, 
                mask=(start_m + offs_m)[:, None] < seqlen_q, 
                other=0.0
            )
        lse = tl.where(lse == float("-inf"), 0.0, lse)
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, tl.trans(rfa_k))
        if not EVEN_M:
            qk += tl.where((start_m + offs_m)[:, None] < seqlen_q, 0, float("-inf"))

        if MASK_TYPE == 1:
            if EVEN_M & EVEN_C:
                mask = tl.load(
                    rfa_m_ptrs + (start_m * stride_chunk_mask_m)
                )
            else:
                mask = tl.load(
                    rfa_m_ptrs + (start_m * stride_chunk_mask_m),
                    mask=((start_m + offs_m)[:, None] < seqlen_q)
                    & (offs_c[None, :] < nchunks),
                    other=1,
                )
            # Slightly faster to multiply the softmax_scale in the tl.exp below since the compiler
            # can then fuse the mult and add into an fma instruction. But if we have bias we need to
            # to multiply with softmax_scale here.
            # we assume mask already implies the causal masking
            qk = qk * softmax_scale
            qk = tl.where(mask, float("-inf"), qk)
            p = tl.exp(qk - lse)
        else:
            p = tl.exp(qk * softmax_scale - lse)

        dp = tl.dot(do, tl.trans(rfa_v))
        ds = (p * (dp - do_t_o) * softmax_scale).to(q.dtype)
        # p is fp32, convert to q.dtype
        d_rfa_v += tl.dot(tl.trans(p.to(do.dtype)), do)
        # move softmax_scale to ds to save computation
        d_rfa_k += tl.dot(tl.trans(ds), q) 
    if EVEN_C & EVEN_M:
        if EVEN_HEADDIM:
            tl.store(d_rfa_v_ptrs, d_rfa_v)
            tl.store(d_rfa_k_ptrs, d_rfa_k)
        else:
            tl.store(d_rfa_v_ptrs, d_rfa_v, mask=offs_d[None, :] < headdim)
            tl.store(d_rfa_k_ptrs, d_rfa_k, mask=offs_d[None, :] < headdim)
    else:
        if EVEN_HEADDIM:
            tl.store(d_rfa_v_ptrs, d_rfa_v, mask=offs_c[:, None] < nchunks)
            tl.store(d_rfa_k_ptrs, d_rfa_k, mask=offs_c[:, None] < nchunks)
        else:
            tl.store(d_rfa_v_ptrs, d_rfa_v, mask=(offs_c[:, None] < nchunks) & (offs_d[None, :] < headdim))
            tl.store(d_rfa_k_ptrs, d_rfa_k, mask=(offs_c[:, None] < nchunks) & (offs_d[None, :] < headdim))

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
def _bwd_eva_agg_kernel_dq(
    Q,
    K,
    V,
    RFA_K,
    RFA_V,
    WindowMask,
    ChunkMask,
    DO,
    LSE,
    DO_T_O,
    DQ,
    softmax_scale,
    stride_qb, stride_qh, stride_qm,
    stride_kb, stride_kh, stride_kn,
    stride_vb, stride_vh, stride_vn,
    stride_rfa_kb, stride_rfa_kh, stride_rfa_kc,
    stride_rfa_vb, stride_rfa_vh, stride_rfa_vc,
    stride_window_mask_b, stride_window_mask_m,
    stride_chunk_mask_b, stride_chunk_mask_m,
    stride_do_b, stride_do_h, stride_do_m,
    stride_lse_b, stride_lse_h,
    stride_do_t_o_b, stride_do_t_o_h,
    stride_dq_b, stride_dq_h, stride_dq_m,
    nheads,
    seqlen_q,
    seqlen_k,
    nchunks,
    headdim,
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
    dq_ptrs = (
        DQ + 
        off_b * stride_dq_b +
        off_h * stride_dq_h +
        (offs_m[:, None] * stride_dq_m + offs_d[None, :])
    )
    do_ptrs = (
        DO + 
        off_b * stride_do_b +
        off_h * stride_do_h +
        (offs_m[:, None] * stride_do_m + offs_d[None, :])
    )
    do_t_o_ptrs = (
        DO_T_O + 
        off_b * stride_do_t_o_b +
        off_h * stride_do_t_o_h +
        offs_m[:, None]
    )
    lse_ptrs = (
        LSE + 
        off_b * stride_lse_b +
        off_h * stride_lse_h +
        offs_m[:, None]
    )
    ### load q, do, do_t_o, lse ####
    if EVEN_M:
        if EVEN_HEADDIM:
            q = tl.load(
                q_ptrs
            )
            do = tl.load(
                do_ptrs
            )
        else:
            q = tl.load(
                q_ptrs, 
                mask=offs_d[None, :] < headdim, 
                other=0.0
            )
            do = tl.load(
                do_ptrs, 
                mask=offs_d[None, :] < headdim, 
                other=0.0
            )
        do_t_o = tl.load(
            do_t_o_ptrs
        )
        lse = tl.load(
            lse_ptrs
        )
    else:
        if EVEN_HEADDIM:
            q = tl.load(
                q_ptrs, 
                mask=offs_m[:, None] < seqlen_q, 
                other=0.0
            )
            do = tl.load(
                do_ptrs, 
                mask=offs_m[:, None] < seqlen_q, 
                other=0.0
            )
        else:
            q = tl.load(
                q_ptrs, 
                mask=(offs_m[:, None] < seqlen_q) & (offs_d[None, :] < headdim), 
                other=0.0
            )
            do = tl.load(
                do_ptrs, 
                mask=(offs_m[:, None] < seqlen_q) & (offs_d[None, :] < headdim), 
                other=0.0
            )
        do_t_o = tl.load(
            do_t_o_ptrs, 
            mask=offs_m[:, None] < seqlen_q, 
            other=0.0
        )
        lse = tl.load(
            lse_ptrs, 
            mask=offs_m[:, None] < seqlen_q, 
            other=0.0
        )
    lse = tl.where(lse == float("-inf"), 0.0, lse)
    lse *= 1.4426950408889634  # log2(e)
    qk_scale = softmax_scale
    qk_scale *= 1.4426950408889634  # log2(e)
    if MASK_TYPE == 1:
        window_mask_ptrs = (
            WindowMask +
            off_b * stride_window_mask_b +
            (offs_m[:, None] * stride_window_mask_m + offs_n[None, :])
        )
        if EMPTY_RFA_KV == 0:
            chunk_mask_ptrs = (
                ChunkMask +
                off_b * stride_chunk_mask_b +
                (offs_m[:, None] * stride_chunk_mask_m + offs_c[None, :])
            )

    dq = tl.zeros([BLOCK_M, BLOCK_HEADDIM], dtype=tl.float32)
    # loop over k, v and update accumulator
    # Iterate over local singletons;
    # so we only iterate over blocks within the current window
    start_idx_n = offs_w * WINDOW_SIZE
    end_idx_n = tl.minimum((start_m + 1) * BLOCK_M, seqlen_k)
    for start_n in range(start_idx_n, end_idx_n, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
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
                window_mask = tl.load(
                    window_mask_ptrs + start_n - start_idx_n
                )
            else:
                window_mask = tl.load(
                    window_mask_ptrs + start_n - start_idx_n,
                    mask=(offs_m[:, None] < seqlen_q)
                    & ((start_n - start_idx_n + offs_n)[None, :] < WINDOW_SIZE),
                    other=1,
                )
            # Slightly faster to multiply the softmax_scale in the tl.exp below since the compiler
            # can then fuse the mult and add into an fma instruction. But if we have bias we need to
            # to multiply with softmax_scale here.
            # we assume mask already implies the causal masking
            qk = qk * qk_scale
            qk = tl.where(window_mask, float("-inf"), qk)
            p = tl.exp2(qk - lse)
        else:
            qk += tl.where(offs_m[:, None] >= (start_n + offs_n)[None, :], 0, float("-inf"))
            p = tl.exp2(qk * qk_scale - lse)

        if EVEN_N & EVEN_M:
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
        dp = tl.dot(do, tl.trans(v))
        ds = (p * (dp - do_t_o) * softmax_scale).to(q.dtype)
        dq += tl.dot(ds, k)

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

            if MASK_TYPE == 1:
                if EVEN_C & EVEN_M:
                    chunk_mask = tl.load(
                        chunk_mask_ptrs + start_c
                    )
                else:
                    chunk_mask = tl.load(
                        chunk_mask_ptrs + start_c,
                        mask=(offs_m[:, None] < seqlen_q) & ((start_c + offs_c)[None, :] < nchunks),
                        other=1,
                    )
                # Slightly faster to multiply the softmax_scale in the tl.exp below since the compiler
                # can then fuse the mult and add into an fma instruction. But if we have bias we need to
                # to multiply with softmax_scale here.
                # we assume mask already implies the causal masking
                qk = qk * qk_scale
                qk = tl.where(chunk_mask, float("-inf"), qk)
                p = tl.exp2(qk - lse)
            else:
                p = tl.exp2(qk * qk_scale - lse)

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
            dp = tl.dot(do, tl.trans(rfa_v))
            ds = (p * (dp - do_t_o) * softmax_scale).to(q.dtype)
            dq += tl.dot(ds, rfa_k)

    start_m = tl.program_id(0)
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_HEADDIM)
    dq_ptrs = (
        DQ +
        off_b * stride_dq_b +
        off_h * stride_dq_h +
        (offs_m[:, None] * stride_dq_m + offs_d[None, :])
    )
    if EVEN_M:
        if EVEN_HEADDIM:
            tl.store(
                dq_ptrs, dq
            )
        else:
            tl.store(
                dq_ptrs, dq,
                mask=offs_d[None, :] < headdim
            )
    else:
        if EVEN_HEADDIM:
            tl.store(
                dq_ptrs, dq,
                mask=offs_m[:, None] < seqlen_q
            )
        else:
            tl.store(
                dq_ptrs, dq,
                mask=(offs_m[:, None] < seqlen_q) & (offs_d[None, :] < headdim)
            )

_capability_90_config = {
    "fwd": {
        (torch.bfloat16, 64): (128, 128, 4, 3), 
        (torch.bfloat16, 128): (128, 128, 8, 3),
        (torch.float32, 64): (128, 64, 8, 3),
        (torch.float32, 128): (64, 32, 4, 3),
    },
    "bwd_dq": {
        (torch.bfloat16, 64): (128, 64, 4, 3),
        (torch.bfloat16, 128): (128, 64, 8, 3),
        (torch.float32, 64): (128, 64, 8, 2),
        (torch.float32, 128): (32, 32, 4, 2),
    },
    "bwd_dkdv": {
        (torch.bfloat16, 64): (128, 64, 4, 2),
        (torch.bfloat16, 128): (128, 64, 8, 2),
        (torch.float32, 64): (128, 64, 8, 2),
        (torch.float32, 128): (32, 32, 4, 1),
    },
    "bwd_drfa_kv": {
        (torch.bfloat16, 64): (128, 64, 4, 2),
        (torch.bfloat16, 128): (128, 64, 8, 2),
        (torch.float32, 64): (128, 64, 8, 2),
        (torch.float32, 128): (32, 32, 4, 1),
    }
}

_capability_80_config = {
    "fwd": {
        (torch.bfloat16, 64): (64, 64, 4, 3),
        (torch.bfloat16, 128): (64, 64, 8, 3),
        (torch.float32, 64): (64, 32, 4, 2),
        (torch.float32, 128): (64, 32, 8, 1),
    },
    "bwd_dq": {
        (torch.bfloat16, 64): (64, 64, 4, 3),
        (torch.bfloat16, 128): (64, 32, 4, 2),
        (torch.float32, 64): (32, 32, 4, 2),
        (torch.float32, 128): (32, 32, 4, 2),
    },
    "bwd_dkdv": {
        (torch.bfloat16, 64): (64, 64, 4, 3),
        (torch.bfloat16, 128): (32, 32, 4, 2),
        (torch.float32, 64): (32, 32, 4, 1),
        (torch.float32, 128): (16, 64, 8, 1),
    },
    "bwd_drfa_kv": {
        (torch.bfloat16, 64): (64, 64, 4, 3),
        (torch.bfloat16, 128): (64, 32, 4, 3),
        (torch.float32, 64): (32, 32, 4, 1),
        (torch.float32, 128): (32, 32, 4, 1),
    }
}

def _get_config(dtype, head_dim, mode) -> tuple[int, int, int, int]:
    capability = torch.cuda.get_device_capability()
    if capability >= (9, 0):
        kernel_config = _capability_90_config[mode].get((dtype, head_dim), (32, 32, 4, 1))
    elif capability >= (8, 0):
        kernel_config = _capability_80_config[mode].get((dtype, head_dim), (16, 16, 4, 1))
    else:
        if mode == "fwd":
            if dtype == torch.float32:
                kernel_config = (32, 16, 4, 2)
            else:
                kernel_config = (64, 32, 4, 2)
        else:
            if dtype == torch.float32:
                kernel_config = (16, 16, 4, 1)
            else:
                kernel_config = (32, 32, 4, 1)
    return kernel_config

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
    ChunkMask,
    Out,
    LSE,
    softmax_scale,
    stride_qb, stride_qh, stride_qm,
    stride_kb, stride_kh, stride_kn,
    stride_vb, stride_vh, stride_vn,
    stride_rfa_kb, stride_rfa_kh, stride_rfa_kc,
    stride_rfa_vb, stride_rfa_vh, stride_rfa_vc,
    stride_window_mask_b, stride_window_mask_m,
    stride_chunk_mask_b, stride_chunk_mask_m,
    stride_ob, stride_oh, stride_om,
    stride_lse_b, stride_lse_h,
    nheads,
    seqlen_q,
    seqlen_k,
    nchunks,
    headdim,
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
        window_mask_ptrs = (
            WindowMask +
            off_b * stride_window_mask_b +
            (offs_m[:, None] * stride_window_mask_m + offs_n[None, :])
        )
        if EMPTY_RFA_KV == 0:
            chunk_mask_ptrs = (
                ChunkMask +
                off_b * stride_chunk_mask_b +
                (offs_m[:, None] * stride_chunk_mask_m + offs_c[None, :])
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
                window_mask = tl.load(
                    window_mask_ptrs + start_n - start_idx_n
                )
            else:
                window_mask = tl.load(
                    window_mask_ptrs + start_n - start_idx_n,
                    mask=(offs_m[:, None] < seqlen_q)
                    & ((start_n - start_idx_n + offs_n)[None, :] < WINDOW_SIZE),
                    other=1,
                )
            # Slightly faster to multiply the softmax_scale in the tl.exp below since the compiler
            # can then fuse the mult and add into an fma instruction. But if we have bias we need to
            # to multiply with softmax_scale here.
            # we assume mask already implies the causal masking
            qk = qk * qk_scale
            qk = tl.where(window_mask, float("-inf"), qk)
            m_ij = tl.maximum(tl.max(qk, 1), m_i)
            masked_out_rows = (m_ij == float("-inf"))
            m_ij_masked = tl.where(masked_out_rows, 0, m_ij)
            p = tl.exp2(qk - m_ij_masked[:, None])
        else:
            qk += tl.where(offs_m[:, None] >= (start_n + offs_n)[None, :], 0, float("-inf"))
            m_ij = tl.maximum(tl.max(qk, 1) * qk_scale, m_i)
            masked_out_rows = (m_ij == float("-inf"))
            m_ij_masked = tl.where(masked_out_rows, 0, m_ij)
            p = tl.exp2(qk * qk_scale - m_ij_masked[:, None])

        d_ij = tl.sum(p, 1)

        # scale acc_o
        prev_scale = tl.exp2(m_i - m_ij_masked)
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

            if MASK_TYPE == 1:
                if EVEN_C & EVEN_M:
                    chunk_mask = tl.load(
                        chunk_mask_ptrs + start_c
                    )
                else:
                    chunk_mask = tl.load(
                        chunk_mask_ptrs + start_c,
                        mask=(offs_m[:, None] < seqlen_q) & ((start_c + offs_c)[None, :] < nchunks),
                        other=1,
                    )
                # Slightly faster to multiply the softmax_scale in the tl.exp below since the compiler
                # can then fuse the mult and add into an fma instruction. But if we have bias we need to
                # to multiply with softmax_scale here.
                # we assume mask already implies the causal masking
                qk = qk * qk_scale
                qk = tl.where(chunk_mask, float("-inf"), qk)
                m_ij = tl.maximum(tl.max(qk, 1), m_i)
                masked_out_rows = (m_ij == float("-inf"))
                m_ij_masked = tl.where(masked_out_rows, 0, m_ij)
                p = tl.exp2(qk - m_ij_masked[:, None])
            else:
                m_ij = tl.maximum(tl.max(qk, 1) * qk_scale, m_i)
                masked_out_rows = (m_ij == float("-inf"))
                m_ij_masked = tl.where(masked_out_rows, 0, m_ij)
                p = tl.exp2(qk * qk_scale - m_ij_masked[:, None])

            d_ij = tl.sum(p, 1)

            # scale acc_o
            prev_scale = tl.exp2(m_i - m_ij_masked)
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

    # for rows that are all -inf, set d_i to 1.0
    d_i = tl.where(d_i == 0.0, 1.0, d_i)
    # multiply by log(2)
    lse_m = (m_i + tl.math.log2(d_i)) * 0.6931471805599453
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
    lse_ptrs = (
        LSE +
        off_b * stride_lse_b +
        off_h * stride_lse_h +
        offs_m
    )
    if EVEN_M:
        tl.store(
            lse_ptrs, lse_m,
        )
    else:
        tl.store(
            lse_ptrs, lse_m,
            mask=offs_m < seqlen_q
        )

def triton_eva_agg_fwd(
    q, k, v, rfa_k, rfa_v, 
    window_mask, 
    chunk_mask,
    softmax_scale, 
    window_size, 
    chunks_per_window
):
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
        assert window_mask.dtype == torch.bool
        assert window_mask.is_cuda
        assert window_mask.dim() == 4
        assert window_mask.shape == (batch, 1, seqlen_q, window_size)
        if window_mask.stride(-1) != 1:
            window_mask = window_mask.contiguous()

        assert chunk_mask is not None
        assert chunk_mask.dtype == torch.bool
        assert chunk_mask.is_cuda
        assert chunk_mask.dim() == 4
        assert chunk_mask.shape == (batch, 1, seqlen_q, nchunks)
        if chunk_mask.stride(-1) != 1:
            chunk_mask = chunk_mask.contiguous()

    chunk_mask_strides = (
        (chunk_mask.stride(0), chunk_mask.stride(2)) 
        if mask_type == 1 else 
        (0, 0)
    )
    window_mask_strides = (
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

    o = torch.empty_like(q)
    lse = torch.empty((q.shape[0], q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32)

    BLOCK_HEADDIM = max(triton.next_power_of_2(head_dim), 16)
    
    BLOCK_M, BLOCK_N, num_warps, num_stages = _get_config(q.dtype, head_dim, "fwd")

    assert chunks_per_window >= BLOCK_N, "chunks_per_window must be greater than BLOCK" 
    assert chunks_per_window % BLOCK_N == 0, "chunks_per_window must be a multiple of BLOCK_N"

    grid = lambda META: (triton.cdiv(seqlen_q, META["BLOCK_M"]), batch * nheads)
    _fwd_eva_agg_kernel[grid](
        q,
        k,
        v,
        rfa_k,
        rfa_v,
        window_mask,
        chunk_mask,
        o,
        lse,
        softmax_scale,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        rfa_k_strides[0], rfa_k_strides[1], rfa_k_strides[2],
        rfa_v_strides[0], rfa_v_strides[1], rfa_v_strides[2],
        window_mask_strides[0], window_mask_strides[1],
        chunk_mask_strides[0], chunk_mask_strides[1],
        o.stride(0), o.stride(1), o.stride(2),
        lse.stride(0), lse.stride(1),
        nheads,
        seqlen_q,
        seqlen_k,
        nchunks,
        head_dim,
        chunks_per_window,
        window_size,
        mask_type,
        empty_rfa_kv,
        BLOCK_HEADDIM,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return o, lse

def triton_eva_agg_bwd(
    do, 
    q, k, v, rfa_k, rfa_v, 
    window_mask, chunk_mask,
    o, lse, 
    dq, dk, dv, d_rfa_k, d_rfa_v, 
    softmax_scale, 
    window_size, 
    chunks_per_window,
    empty_rfa_kv,
    mask_type,
):
    if do.stride(-1) != 1:
        do = do.contiguous()

    # shape constraints
    batch, nheads, seqlen_q, head_dim = q.shape
    _,     _,      seqlen_k, _        = k.shape
    if empty_rfa_kv == 0:
        nchunks = rfa_k.shape[-2]
        assert rfa_k.shape == (batch, nheads, nchunks, head_dim)
        assert rfa_v.shape == (batch, nheads, nchunks, head_dim)
        assert d_rfa_k.stride(-1) == d_rfa_v.stride(-1) == 1
        assert q.dtype == k.dtype == v.dtype == rfa_k.dtype == rfa_v.dtype
    else:
        nchunks = 0
        assert q.dtype == k.dtype == v.dtype, "All tensors must have the same type"

    assert lse.shape == (batch, nheads, seqlen_q)
    assert q.stride(-1) == k.stride(-1) == v.stride(-1) == o.stride(-1) == rfa_k.stride(-1) == rfa_v.stride(-1) == 1
    assert dq.stride(-1) == dk.stride(-1) == dv.stride(-1) == 1
    softmax_scale = softmax_scale or 1.0 / math.sqrt(head_dim)

    assert head_dim <= 128, "We only test head dimensions up to 128"

    window_mask_strides = (
        (window_mask.stride(0), window_mask.stride(2)) 
        if mask_type == 1 else 
        (0, 0)
    )
    chunk_mask_strides = (
        (chunk_mask.stride(0), chunk_mask.stride(2)) 
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

    d_rfa_k_strides = (
        (d_rfa_k.stride(0), d_rfa_k.stride(1), d_rfa_k.stride(2))
        if empty_rfa_kv == 0 else
        (0, 0, 0)
    )
    d_rfa_v_strides = (
        (d_rfa_v.stride(0), d_rfa_v.stride(1), d_rfa_v.stride(2))
        if empty_rfa_kv == 0 else
        (0, 0, 0)
    )

    BLOCK_HEADDIM = max(triton.next_power_of_2(head_dim), 16)

    do_t_o = torch.sum(do.to(torch.float32) * o.to(torch.float32), dim=-1).to(do.dtype)

    BLOCK_M, BLOCK_N, num_warps, num_stages = _get_config(q.dtype, head_dim, "bwd_dq")

    assert chunks_per_window >= BLOCK_N, "chunks_per_window must be greater than BLOCK" 
    assert chunks_per_window % BLOCK_N == 0, "chunks_per_window must be a multiple of BLOCK"
    grid = lambda META: (
        triton.cdiv(seqlen_q, META["BLOCK_M"]),
        batch * nheads,
    )
    _bwd_eva_agg_kernel_dq[grid](
        q,
        k,
        v,
        rfa_k,
        rfa_v,
        window_mask,
        chunk_mask,
        do,
        lse,
        do_t_o,
        dq,
        softmax_scale,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        rfa_k_strides[0], rfa_k_strides[1], rfa_k_strides[2],
        rfa_v_strides[0], rfa_v_strides[1], rfa_v_strides[2],
        window_mask_strides[0], window_mask_strides[1],
        chunk_mask_strides[0], chunk_mask_strides[1],
        do.stride(0), do.stride(1), do.stride(2),
        lse.stride(0), lse.stride(1),
        do_t_o.stride(0), do_t_o.stride(1),
        dq.stride(0), dq.stride(1), dq.stride(2),
        nheads,
        seqlen_q,
        seqlen_k,
        nchunks,
        head_dim,
        chunks_per_window,
        window_size,
        mask_type,
        empty_rfa_kv,
        BLOCK_HEADDIM,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    BLOCK_M, BLOCK_N, num_warps, num_stages = _get_config(q.dtype, head_dim, "bwd_dkdv")
    grid = lambda META: (
        triton.cdiv(seqlen_k, META["BLOCK_N"]),
        batch * nheads,
    )
    _bwd_eva_agg_kernel_dkdv[grid](
        q,
        k,
        v,
        window_mask,
        do,
        lse,
        do_t_o,
        dk,
        dv,
        softmax_scale,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        window_mask_strides[0], window_mask_strides[1],
        do.stride(0), do.stride(1), do.stride(2),
        lse.stride(0), lse.stride(1),
        do_t_o.stride(0), do_t_o.stride(1),
        dk.stride(0), dk.stride(1), dk.stride(2),
        dv.stride(0), dv.stride(1), dv.stride(2),
        nheads,
        seqlen_q,
        seqlen_k,
        head_dim,
        window_size,
        mask_type,
        BLOCK_HEADDIM,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    if empty_rfa_kv == 0:
        BLOCK_M, BLOCK_N, num_warps, num_stages = _get_config(q.dtype, head_dim, "bwd_drfa_kv")
        grid = lambda META: (
            triton.cdiv(nchunks, META["BLOCK_N"]),
            batch * nheads,
        )
        _bwd_eva_agg_kernel_drfa_kv[grid](
            q,
            rfa_k,
            rfa_v,
            chunk_mask,
            do,
            lse,
            do_t_o,
            d_rfa_k,
            d_rfa_v,
            softmax_scale,
            q.stride(0), q.stride(1), q.stride(2),
            rfa_k_strides[0], rfa_k_strides[1], rfa_k_strides[2],
            rfa_v_strides[0], rfa_v_strides[1], rfa_v_strides[2],
            chunk_mask_strides[0], chunk_mask_strides[1],
            do.stride(0), do.stride(1), do.stride(2),
            lse.stride(0), lse.stride(1),
            do_t_o.stride(0), do_t_o.stride(1),
            d_rfa_k_strides[0], d_rfa_k_strides[1], d_rfa_k_strides[2],
            d_rfa_v_strides[0], d_rfa_v_strides[1], d_rfa_v_strides[2],
            nheads,
            seqlen_q,
            nchunks,
            head_dim,
            chunks_per_window,
            window_size,
            mask_type,
            BLOCK_HEADDIM,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            num_warps=num_warps,
            num_stages=num_stages,
        )


class EvaAggFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, rfa_k, rfa_v, window_mask, chunk_mask, softmax_scale=None, window_size=None, chunks_per_window=None):
        if rfa_k is None and rfa_v is None:
            empty_rfa_kv = 1
        else:
            assert rfa_k is not None and rfa_v is not None, "Both rfa_k and rfa_v must either be None or have values at the same time."
            empty_rfa_kv = 0
        
        if window_mask is not None:
            mask_type = 1
        else:
            mask_type = 0
        o, lse = triton_eva_agg_fwd(
            q, k, v, rfa_k, rfa_v, window_mask, chunk_mask, softmax_scale, window_size, chunks_per_window
        )
        ctx.save_for_backward(q, k, v, o, lse, rfa_k, rfa_v, window_mask, chunk_mask)
        ctx.softmax_scale = softmax_scale
        ctx.window_size = window_size
        ctx.chunks_per_window = chunks_per_window
        ctx.empty_rfa_kv = empty_rfa_kv
        ctx.mask_type = mask_type
        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, o, lse, rfa_k, rfa_v, window_mask, chunk_mask = ctx.saved_tensors
        dq = torch.empty_like(q)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)
        if ctx.empty_rfa_kv == 0:
            d_rfa_k = torch.empty_like(rfa_k)
            d_rfa_v = torch.empty_like(rfa_v)
        else:
            d_rfa_k = None
            d_rfa_v = None
        triton_eva_agg_bwd(
            do,
            q,
            k,
            v,
            rfa_k,
            rfa_v,
            window_mask,
            chunk_mask,
            o,
            lse,
            dq,
            dk,
            dv,
            d_rfa_k,
            d_rfa_v,
            softmax_scale=ctx.softmax_scale,
            window_size=ctx.window_size,
            chunks_per_window=ctx.chunks_per_window,
            empty_rfa_kv=ctx.empty_rfa_kv,
            mask_type=ctx.mask_type,
        )
        return dq, dk, dv, d_rfa_k, d_rfa_v, None, None, None, None, None


def eva_agg_func_triton(
        q, k, v, rfa_k, rfa_v, 
        window_mask, chunk_mask, 
        softmax_scale=None, window_size=None, chunks_per_window=None,
    ):
    return EvaAggFunc.apply(
        q, k, v, rfa_k, rfa_v, 
        window_mask, chunk_mask, 
        softmax_scale, window_size, chunks_per_window,
    )
