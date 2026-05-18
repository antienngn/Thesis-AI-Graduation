import torch
from xformers.ops import memory_efficient_attention
from xformers.ops.fmha.attn_bias import BlockDiagonalCausalFromBottomRightMask


# ---------------------------------------------------------------------------
# Vectorised KV-cache gather
# ---------------------------------------------------------------------------

def _gather_kv_from_cache_batched(
    k_cache: torch.Tensor,   # [num_blocks, num_kv_heads, head_size/x, block_size, x]
    v_cache: torch.Tensor,   # [num_blocks, num_kv_heads, head_size,   block_size  ]
    b_loc: torch.Tensor,     # [batch, max_blocks_per_seq]
    b_ctx_len: torch.Tensor, # [batch]  int
    max_ctx_len: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Gather prefix KV tokens for ALL batch items at once.

    Returns:
        k_ctx_list: list of length batch, each [ctx_len_i, num_kv_heads, head_size]
        v_ctx_list: list of length batch, each [ctx_len_i, num_kv_heads, head_size]

    We gather per-batch because each item has a different ctx_len, so we
    cannot pack them into a single rectangular tensor without padding.
    The gather itself is fully vectorised over positions within each item.
    """
    num_kv_heads = k_cache.shape[1]
    head_size    = k_cache.shape[2] * k_cache.shape[4]   # (head_size/x) * x
    block_size   = k_cache.shape[3]
    device       = k_cache.device
    batch        = b_ctx_len.shape[0]

    k_ctx_list = []
    v_ctx_list = []

    for i in range(batch):
        ctx_len = b_ctx_len[i].item()
        if ctx_len == 0:
            k_ctx_list.append(None)
            v_ctx_list.append(None)
            continue

        # All position indices for this item — vectorised, no inner loop
        pos           = torch.arange(ctx_len, device=device)      # [ctx_len]
        block_ids     = b_loc[i, pos // block_size]               # [ctx_len]
        block_offsets = pos % block_size                           # [ctx_len]
        seq_idx       = torch.arange(ctx_len, device=device)      # [ctx_len]

        # k_cache[block_ids]: [ctx_len, num_kv_heads, head_size/x, block_size, x]
        k_gathered  = k_cache[block_ids]                           # [ctx_len, nh, d/x, bs, x]
        k_gathered  = k_gathered[seq_idx, :, :, block_offsets, :] # [ctx_len, nh, d/x, x]
        k_ctx       = k_gathered.reshape(ctx_len, num_kv_heads, head_size)

        # v_cache[block_ids]: [ctx_len, num_kv_heads, head_size, block_size]
        v_gathered  = v_cache[block_ids]                           # [ctx_len, nh, head_size, bs]
        v_ctx       = v_gathered[seq_idx, :, :, block_offsets]    # [ctx_len, nh, head_size]

        k_ctx_list.append(k_ctx)
        v_ctx_list.append(v_ctx)

    return k_ctx_list, v_ctx_list


# ---------------------------------------------------------------------------
# Main entry point — drop-in replacement for context_attention_fwd (Triton)
# ---------------------------------------------------------------------------

@torch.inference_mode()
def context_attention_fwd(
    q: torch.Tensor,           # [total_query_tokens, num_heads,    head_size]
    k: torch.Tensor,           # [total_query_tokens, num_kv_heads, head_size]
    v: torch.Tensor,           # [total_query_tokens, num_kv_heads, head_size]
    o: torch.Tensor,           # [total_query_tokens, num_heads,    head_size]  written in-place
    k_cache: torch.Tensor,     # [num_blocks, num_kv_heads, head_size/x, block_size, x]
    v_cache: torch.Tensor,     # [num_blocks, num_kv_heads, head_size,   block_size  ]
    b_loc: torch.Tensor,       # [batch, max_blocks_per_seq]
    b_start_loc: torch.Tensor, # [batch]  cumulative start in flat token dim
    b_seq_len: torch.Tensor,   # [batch]  total len = ctx + query
    b_ctx_len: torch.Tensor,   # [batch]  prefix (cached) len
    max_input_len: int,        # kept for API compatibility
    alibi_slopes: torch.Tensor = None,  # [num_heads] float32 or None
):
    """
    Drop-in replacement for the Triton context_attention_fwd.

    Attention pattern (identical to Triton kernel):
      - Each query token attends to ALL ctx (prefix) tokens  [no causal mask]
      - Each query token attends causally to new tokens 0..i [causal mask]
    This is exactly BlockDiagonalCausalFromBottomRightMask semantics.

    Performance:
      - KV gather: fully vectorised per-item (no position-level Python loop)
      - Attention: single xFormers call over the whole batch (no batch loop)
        → GPU processes all (batch × heads × query_tiles) in parallel,
          matching Triton's grid = (batch, head, cdiv(max_input_len, BLOCK))
    """
    batch              = b_seq_len.shape[0]
    num_heads          = q.shape[1]
    num_kv_heads       = k.shape[1]
    head_size          = q.shape[2]
    num_queries_per_kv = num_heads // num_kv_heads
    sm_scale           = 1.0 / (head_size ** 0.5)
    block_size         = k_cache.shape[3]

    # ---------------------------------------------------------------------- #
    # Step 1 — gather prefix KV from paged cache (vectorised per item)
    # ---------------------------------------------------------------------- #
    max_ctx_len  = int(b_ctx_len.max().item())
    k_ctx_list, v_ctx_list = _gather_kv_from_cache_batched(
        k_cache, v_cache, b_loc, b_ctx_len, max_ctx_len
    )

    # ---------------------------------------------------------------------- #
    # Step 2 — build flat Q, K_full, V_full across all batch items
    #          and collect subquery_lens / seq_lens for the mask
    # ---------------------------------------------------------------------- #
    q_parts  = []
    k_parts  = []
    v_parts  = []
    subquery_lens = []   # new-token lengths per item
    seq_lens_full = []   # total lengths (ctx + new) per item

    for i in range(batch):
        ctx_len   = b_ctx_len[i].item()
        seq_len   = b_seq_len[i].item()
        query_len = seq_len - ctx_len
        start_idx = b_start_loc[i].item()

        if query_len <= 0:
            continue

        q_i   = q[start_idx: start_idx + query_len]   # [query_len, num_heads,    head_size]
        k_new = k[start_idx: start_idx + query_len]   # [query_len, num_kv_heads, head_size]
        v_new = v[start_idx: start_idx + query_len]   # [query_len, num_kv_heads, head_size]

        if ctx_len > 0 and k_ctx_list[i] is not None:
            k_full_i = torch.cat([k_ctx_list[i], k_new], dim=0)  # [ctx+q, num_kv_heads, head_size]
            v_full_i = torch.cat([v_ctx_list[i], v_new], dim=0)
        else:
            k_full_i = k_new
            v_full_i = v_new

        q_parts.append(q_i)
        k_parts.append(k_full_i)
        v_parts.append(v_full_i)
        subquery_lens.append(query_len)
        seq_lens_full.append(ctx_len + query_len)

    if not q_parts:
        return  # nothing to do

    # Concatenate all items into single flat tensors
    # q_flat:  [total_query_tokens, num_heads,    head_size]
    # kv_flat: [total_kv_tokens,    num_kv_heads, head_size]
    q_flat = torch.cat(q_parts, dim=0)
    k_flat = torch.cat(k_parts, dim=0)
    v_flat = torch.cat(v_parts, dim=0)

    # ---------------------------------------------------------------------- #
    # Step 3 — GQA / MQA expansion
    #
    # xFormers memory_efficient_attention supports the grouped layout
    # [B, M, G, H, K] natively (same as _run_memory_efficient_xformers_forward).
    # We reshape rather than expand to avoid materialising a large tensor.
    # ---------------------------------------------------------------------- #
    if num_queries_per_kv > 1:
        # q: [T_q, num_kv_heads, num_queries_per_kv, head_size]
        q_flat = q_flat.view(q_flat.shape[0], num_kv_heads, num_queries_per_kv, head_size)
        # k/v: [T_kv, num_kv_heads, 1, head_size] — will broadcast over G
        k_flat = k_flat[:, :, None, :].expand(-1, num_kv_heads, num_queries_per_kv, head_size)
        v_flat = v_flat[:, :, None, :].expand(-1, num_kv_heads, num_queries_per_kv, head_size)

    # Add batch dimension: [1, T, ...]
    xq = q_flat.unsqueeze(0)
    xk = k_flat.unsqueeze(0)
    xv = v_flat.unsqueeze(0)

    # ---------------------------------------------------------------------- #
    # Step 4 — build BlockDiagonalCausalFromBottomRightMask
    #
    # This mask encodes exactly the same attention pattern as the Triton kernel:
    #   - query token i (absolute pos ctx_len + i) attends to positions 0..ctx_len+i
    #   - i.e. full prefix visibility + causal over new tokens
    # No tensor bias needed → cutlassF runs with no stride constraint.
    # ---------------------------------------------------------------------- #
    if alibi_slopes is None:
        attn_bias = BlockDiagonalCausalFromBottomRightMask.from_seqlens(
            subquery_lens, seq_lens_full
        )
    else:
        # ALiBi: xFormers does not have a built-in ALiBi + prefix mask combo,
        # so fall back to per-item additive bias (same quality, slightly slower).
        # Build [1, num_heads, total_q, total_kv] additive bias with padding trick.
        total_q  = sum(subquery_lens)
        total_kv = sum(seq_lens_full)
        total_kv_padded = (total_kv + 7) // 8 * 8

        bias_padded = torch.zeros(
            num_heads, total_q, total_kv_padded,
            device=q.device, dtype=q.dtype,
        )

        q_offset  = 0
        kv_offset = 0
        for idx, (sq, sl) in enumerate(zip(subquery_lens, seq_lens_full)):
            ctx_l = sl - sq
            q_pos  = torch.arange(ctx_l, ctx_l + sq,
                                   device=q.device, dtype=torch.float32)
            k_pos  = torch.arange(0, sl,
                                   device=q.device, dtype=torch.float32)
            dist   = k_pos[None, :] - q_pos[:, None]                   # [sq, sl]

            slopes = alibi_slopes.to(dtype=torch.float32, device=q.device)
            alibi  = slopes[:, None, None] * dist[None, :, :]           # [nh, sq, sl]

            # causal mask for new tokens
            qi_idx = torch.arange(sq,  device=q.device)
            ki_idx = torch.arange(sl, device=q.device)
            causal = torch.where(
                ki_idx[None, :] > (ctx_l + qi_idx[:, None]),
                torch.full((), float("-inf"), device=q.device, dtype=torch.float32),
                torch.zeros((),              device=q.device, dtype=torch.float32),
            )  # [sq, sl]

            combined = (alibi + causal[None, :, :]).to(dtype=q.dtype)  # [nh, sq, sl]
            bias_padded[:, q_offset: q_offset + sq,
                           kv_offset: kv_offset + sl] = combined

            q_offset  += sq
            kv_offset += sl

        attn_bias = bias_padded[:, :, :total_kv].unsqueeze(0)  # [1, nh, total_q, total_kv]

    # ---------------------------------------------------------------------- #
    # Step 5 — single xFormers call (all batch × heads in parallel)
    # ---------------------------------------------------------------------- #
    out_flat = memory_efficient_attention(
        xq, xk, xv,
        attn_bias=attn_bias,
        scale=sm_scale,
    )  # [1, total_query_tokens, num_heads (or groups), head_size]

    out_flat = out_flat.squeeze(0)   # [total_query_tokens, ...]

    if num_queries_per_kv > 1:
        # reshape back from grouped [T_q, num_kv_heads, G, head_size]
        # to flat     [T_q, num_heads, head_size]
        out_flat = out_flat.reshape(out_flat.shape[0], num_heads, head_size)

    # ---------------------------------------------------------------------- #
    # Step 6 — scatter output back into o (in-place, matching Triton tl.store)
    # ---------------------------------------------------------------------- #
    q_offset = 0
    for i in range(batch):
        ctx_len   = b_ctx_len[i].item()
        seq_len   = b_seq_len[i].item()
        query_len = seq_len - ctx_len
        start_idx = b_start_loc[i].item()

        if query_len <= 0:
            continue

        o[start_idx: start_idx + query_len] = out_flat[q_offset: q_offset + query_len]
        q_offset += query_len



