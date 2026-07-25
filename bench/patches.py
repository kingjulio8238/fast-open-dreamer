"""Toggleable, semantics-preserving inference optimisations.

Each patch here targets one bottleneck found in the released model code and is
applied by monkeypatching, so a single benchmark run can attribute a speedup to
a specific change without maintaining a forked copy of `models.py`.

Patches:
  bf16_weights   store parameters in bfloat16 instead of float32.
                 The configs set param_dtype=float32 while dtype=bfloat16, so
                 every Linear reads 4 bytes/param and casts down at use. At
                 batch 1 the frame loop is weight-bandwidth bound, so this is
                 close to a straight 2x on the dominant term. NOT bit-exact:
                 verify against a real checkpoint before shipping.

  no_roll_kv     read the KV ring buffer in place instead of `jnp.roll`-ing it.
                 EXACT. RoPE is applied with absolute positions before the
                 write, and attention scores depend only on position
                 *differences*, so the physical order of the buffer is
                 irrelevant — only the mask is. The released
                 `get_ordered_kv` rolls both K and V (a full read + full write
                 of the cache) on every time layer of every denoising pass:
                 8 layers x 5 passes = 40 full cache round-trips per frame.

  fast_kv_write  drop the `lax.cond` in `KVCache.update` when T == 1.
                 EXACT. With T == 1, write_idx is in [0, window-1] so
                 write_idx + T <= window always holds and the wrapping branch
                 is dead — but it is still traced, emitting a scatter and
                 forcing both branches to materialise a full buffer copy.

  block_attn     split the tokenizer's masked space attention into two
                 unmasked calls so it stops materialising the (S x S) score
                 matrix. EXACT -- both encoder and decoder masks are 2-block
                 structures around the latent/patch boundary, and each block is
                 a plain rectangular attention. See the long note by
                 `_detect_block_split`.

  fp8_weights    e4m3 weights + dynamically scaled e4m3 activations. MEASURED
                 AND REJECTED: XLA emits no fused fp8 GEMM, so the dequantize
                 multiply materialises the bf16 weight matrix anyway and the
                 frame gets 1.37x SLOWER at B=1. Kept for reproducibility and
                 as the starting point if a supported fp8 path is tried; do not
                 put it in a serving stack. See the note above `_linear_fp8_call`.

  no_remat       disable gradient checkpointing in the forward path. EXACT.
                 `SpaceSelfAttention`, `TimeSelfAttention` and `BlockCausalLayer`
                 wrap their bodies in `nnx.remat` unconditionally, including at
                 inference where there is no backward pass to save memory for.
                 remat is an optimisation barrier, so it also blocks fusion.

Usage:
    from bench import patches
    patches.apply(["no_roll_kv", "fast_kv_write", "no_remat"], pkg="pipeline")
    model = patches.cast_params(model, "bfloat16")   # bf16_weights
"""
from __future__ import annotations

import importlib
import json

import jax
import jax.numpy as jnp
from einops import rearrange
from flax import nnx

_APPLIED: list[str] = []
_ORIGINALS: dict[str, object] = {}


# ---------------------------------------------------------------------------
# no_roll_kv  +  fast_kv_write
# ---------------------------------------------------------------------------

def _get_ordered_kv_inplace(self, query_len):
    """Drop-in for `KVCache.get_ordered_kv` with no roll.

    Returns the buffer untouched plus a mask over its physical slots.

    Slot j holds the most recent absolute position p with p % window == j.
    After `update`, `self.index` (call it I) is the number of positions ever
    written, so the newest is I-1 and slot j's occupant has

        age_j = (I - 1 - j) mod window        # 0 = newest
        p_j   = I - 1 - age_j

    Query i (0-indexed in the current block of `query_len`) sits at absolute
    position I - query_len + i, so:

        causal : p_j <= I - query_len + i   <=>  age_j >= query_len - 1 - i
        written: p_j >= 0                   <=>  age_j <= I - 1

    which reproduces the released mask exactly, without moving any data.

    LATENT HAZARD (dormant on every current inference path, but a real trap):
    the released `get_ordered_kv` returns a mask indexed in ORDERED key space,
    whereas this one is indexed in PHYSICAL slot space. `GroupedQueryAttention`
    combines it with any caller-supplied mask via
    `jnp.logical_and(mask, cache_mask)`. A `time_mask` is authored in ordered
    key space, so combining the two would be WRONG under this patch. Every
    inference path traced here passes `time_mask=None`, so nothing is broken
    today — but anyone who starts passing one must either drop this patch or
    permute their mask into physical slot order first.
    """
    W = self.window_size
    idx = self.index                                       # I, traced i32
    j = jnp.arange(W)[None, None, None, :]                 # (1,1,1,W)
    age = jnp.mod(idx - 1 - j, W)                          # (1,1,1,W)

    i = jnp.arange(query_len)[None, None, :, None]         # (1,1,q,1)
    causal = age >= (query_len - 1 - i)
    written = age <= (idx - 1)

    return self.k, self.v, jnp.logical_and(causal, written)


def _update_fast(self, k_new, v_new):
    """Drop-in for `KVCache.update`; skips the `lax.cond` when T == 1."""
    cls = type(self)
    T = k_new.shape[1]                                     # static
    write_idx = self.index % self.window_size

    if T == 1:
        # write_idx in [0, W-1] so write_idx + 1 <= W always: contiguous.
        k = jax.lax.dynamic_update_slice(self.k, k_new, (0, write_idx, 0, 0))
        v = jax.lax.dynamic_update_slice(self.v, v_new, (0, write_idx, 0, 0))
        return cls(k=k, v=v, index=self.index + T, window_size=self.window_size)

    return _ORIGINALS["KVCache.update"](self, k_new, v_new)


# ---------------------------------------------------------------------------
# no_remat
# ---------------------------------------------------------------------------

def _remat_passthrough(fn, *args, **kwargs):
    """`nnx.remat(f, static_argnums=...)` -> `f`. Signature-compatible."""
    return fn


# ---------------------------------------------------------------------------
# bf16_weights
# ---------------------------------------------------------------------------

def cast_params(model, dtype: str = "bfloat16"):
    """Cast every `nnx.Param` in `model` to `dtype`, in place.

    Reduces resident weight bytes and, more importantly, the bytes read per
    forward pass. Norm scales are cast too; if that ever matters numerically,
    filter them out here.
    """
    target = {"bfloat16": jnp.bfloat16, "float16": jnp.float16,
              "float32": jnp.float32}[dtype]

    def _cast(p):
        if not jnp.issubdtype(p.dtype, jnp.floating):
            return p
        # Never widen an already-quantized weight: with both fp8_weights and
        # bf16_weights active this would silently undo the fp8 quantization and
        # the benchmark would measure bf16 while claiming fp8.
        if p.dtype.itemsize <= jnp.dtype(target).itemsize and p.dtype != target:
            return p
        return p.astype(target)

    nnx.update(model, jax.tree.map(_cast, nnx.state(model, nnx.Param)))
    return model


def param_bytes(model) -> int:
    _, state, _ = nnx.split(model, nnx.Param, ...)
    return sum(l.size * l.dtype.itemsize for l in jax.tree.leaves(state))


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------

ALL = ("no_roll_kv", "fast_kv_write", "no_remat", "block_attn", "fp8_weights",
       "ragged_kv")


def apply(names, pkg: str = "dreamer"):
    """Apply the named patches to `<pkg>.models`. Idempotent per name.

    `bf16_weights` is not applied here — it acts on a built model; call
    `cast_params` after construction.
    """
    models = importlib.import_module(f"{pkg}.models")
    KVCache = models.KVCache

    for name in names:
        if name in _APPLIED:
            continue
        if name == "no_roll_kv":
            _ORIGINALS.setdefault("KVCache.get_ordered_kv", KVCache.get_ordered_kv)
            KVCache.get_ordered_kv = _get_ordered_kv_inplace
        elif name == "fast_kv_write":
            _ORIGINALS.setdefault("KVCache.update", KVCache.update)
            KVCache.update = _update_fast
        elif name == "no_remat":
            _ORIGINALS.setdefault("nnx.remat", models.nnx.remat)
            models.nnx.remat = _remat_passthrough
        elif name == "block_attn":
            _ORIGINALS.setdefault("SpaceSelfAttention.__call__",
                                  models.SpaceSelfAttention.__call__)
            models.SpaceSelfAttention.__call__ = _space_attn_call
        elif name == "ragged_kv":
            _ORIGINALS.setdefault("KVCache.update", KVCache.update)
            _ORIGINALS.setdefault("KVCache.get_ordered_kv", KVCache.get_ordered_kv)
            _ORIGINALS.setdefault("RoPE.__call__",
                                  models.RotaryEmbedding1D.__call__)
            KVCache.update = _ragged_update
            KVCache.get_ordered_kv = _ragged_get_ordered_kv
            models.RotaryEmbedding1D.__call__ = _ragged_rope_call
        elif name == "fp8_weights":
            _ORIGINALS.setdefault("nnx.Linear.__call__", nnx.Linear.__call__)
            nnx.Linear.__call__ = _linear_fp8_call
        elif name == "bf16_weights":
            pass  # handled by cast_params on the built model
        else:
            raise ValueError(f"unknown patch {name!r}; known: {ALL + ('bf16_weights',)}")
        _APPLIED.append(name)
    return _APPLIED


def revert(pkg: str = "dreamer"):
    """Undo everything `apply` did, so one process can A/B configurations.

    EXCEPTION: raises if `fp8_weights` quantized any kernel. That patch is
    destructive -- the original values are gone -- so there is nothing to
    revert to, and silently restoring the stock `Linear.__call__` would run it
    over an fp8 kernel with no scale applied. A/B against an fp8 config
    therefore requires a fresh process, not a revert. The raise is intentional,
    not a bug.
    """
    models = importlib.import_module(f"{pkg}.models")
    if "KVCache.get_ordered_kv" in _ORIGINALS:
        models.KVCache.get_ordered_kv = _ORIGINALS.pop("KVCache.get_ordered_kv")
    if "KVCache.update" in _ORIGINALS:
        models.KVCache.update = _ORIGINALS.pop("KVCache.update")
    if "nnx.remat" in _ORIGINALS:
        models.nnx.remat = _ORIGINALS.pop("nnx.remat")
    if "SpaceSelfAttention.__call__" in _ORIGINALS:
        models.SpaceSelfAttention.__call__ = _ORIGINALS.pop("SpaceSelfAttention.__call__")
    if "RoPE.__call__" in _ORIGINALS:
        models.RotaryEmbedding1D.__call__ = _ORIGINALS.pop("RoPE.__call__")
    if "nnx.Linear.__call__" in _ORIGINALS:
        nnx.Linear.__call__ = _ORIGINALS.pop("nnx.Linear.__call__")
    # Counters are process-global; leaving them set would let a second config's
    # firing check pass on the first config's counts.
    _BLOCK_ATTN_STATS.update(split=0, dense=0, shape_mismatch=0,
                             degenerate_skipped=0)
    _BLOCK_ATTN_STATS["unmatched"].clear()
    _FP8_STATS.update(quantized=0, skipped=0, params_before=0, params_after=0)
    if _FP8_QUANTIZED:
        raise RuntimeError(
            "revert(): fp8_weights quantized "
            f"{len(_FP8_QUANTIZED)} Linear kernels in place and this cannot be "
            "undone -- the original fp32 values are gone. Rebuild the model "
            "instead of reverting. (Restoring nnx.Linear.__call__ without "
            "undoing the quantization would run the stock forward on an fp8 "
            "kernel with no scale applied.)")
    _APPLIED.clear()


# ---------------------------------------------------------------------------
# correctness gate
# ---------------------------------------------------------------------------

def check_kv_equivalence(pkg: str = "dreamer", window: int = 16, batch: int = 2,
                         kv_heads: int = 3, head_dim: int = 8, steps: int = 40,
                         tol: float = 0.0):
    """Assert the patched cache produces the same attention result as the
    released one, across the ring-buffer wrap.

    Covers BOTH patches, and both against genuinely unpatched references:

      no_roll_kv    -- compares `_get_ordered_kv_inplace` to the original
                       `get_ordered_kv` reading the same buffer.
      fast_kv_write -- maintains TWO caches, one written by the original
                       `update` and one by `_update_fast`, and compares the
                       buffers directly.

    The second check exists because an earlier version of this function drove
    both arms from a single cache that had already been written by the patched
    `update`: a wrong write index corrupted both arms identically and the
    assertion passed. `fast_kv_write` had no working gate at all.
    """
    models = importlib.import_module(f"{pkg}.models")
    KVCache = models.KVCache
    orig_get = _ORIGINALS.get("KVCache.get_ordered_kv", KVCache.get_ordered_kv)
    orig_upd = _ORIGINALS.get("KVCache.update", KVCache.update)
    cur_upd = KVCache.update

    key = jax.random.PRNGKey(0)
    c_ref = KVCache.init(batch, window, kv_heads, head_dim, dtype=jnp.float32)
    c_pat = KVCache.init(batch, window, kv_heads, head_dim, dtype=jnp.float32)
    max_read_err = max_write_err = 0.0

    for t in range(steps):
        key, k1, k2, k3 = jax.random.split(key, 4)
        k_new = jax.random.normal(k1, (batch, 1, kv_heads, head_dim))
        v_new = jax.random.normal(k2, (batch, 1, kv_heads, head_dim))

        # --- fast_kv_write: original writer vs current writer, side by side ---
        c_ref = orig_upd(c_ref, k_new, v_new)
        c_pat = cur_upd(c_pat, k_new, v_new)
        w = max(float(jnp.max(jnp.abs(c_ref.k - c_pat.k))),
                float(jnp.max(jnp.abs(c_ref.v - c_pat.v))))
        max_write_err = max(max_write_err, w)
        assert w <= tol + 1e-6, f"KV WRITE mismatch at step {t}: max|diff| = {w}"
        assert int(c_ref.index) == int(c_pat.index), \
            f"KV index diverged at step {t}: {c_ref.index} vs {c_pat.index}"

        # --- no_roll_kv: original reader vs patched reader on the same buffer ---
        q = jax.random.normal(k3, (batch, 1, kv_heads, head_dim))

        def attend(get_fn, cache):
            k, v, m = get_fn(cache, query_len=1)
            logits = jnp.einsum("bqnh,bknh->bnqk", q, k) * head_dim ** -0.5
            logits = jnp.where(m, logits, jnp.finfo(logits.dtype).min)
            return jnp.einsum("bnqk,bknh->bqnh", jax.nn.softmax(logits, -1), v)

        r = float(jnp.max(jnp.abs(attend(orig_get, c_ref)
                                  - attend(_get_ordered_kv_inplace, c_pat))))
        max_read_err = max(max_read_err, r)
        assert r <= tol + 1e-6, f"KV READ mismatch at step {t}: max|diff| = {r}"

    print(f"check_kv_equivalence: PASS over {steps} writes "
          f"(window {window}, wrapped {steps // window}x)  "
          f"write max|diff| = {max_write_err:.2e}, read max|diff| = {max_read_err:.2e}")
    return max(max_read_err, max_write_err)


# ---------------------------------------------------------------------------
# block_attn  --  split the tokenizer's masked space attention into two
#                 UNMASKED calls, so it stops materialising (S x S) scores
# ---------------------------------------------------------------------------
#
# The encoder and decoder both pass an explicit boolean space mask into
# `jax.nn.dot_product_attention`. An arbitrary mask forces the XLA reference
# path, which materialises the full (B*T, heads, S, S) score matrix in fp32.
# With S = 1432 that is 197 MB per (batch, frame) slice in the encoder, and it
# is what made encoding 8 windows x 32 frames request 46.9 GiB and OOM an 80 GB
# H100:  8*32 * 24 * 1432^2 * 4 = 50,396,135,424 bytes.
#
# But neither mask is arbitrary. Both are 2-block structures around the
# latent/patch split point p (= n_latents):
#
#   decoder:  latents attend to latents only;  patches attend to everything
#   encoder:  latents attend to everything;    patches attend to patches only
#
# Each block is a plain rectangular attention with NO mask, so the pair can go
# down the fused/flash path, O(1) in S rather than O(S^2). It also skips the
# always-masked-out quadrant: dense is S^2 = 2,050,624 score entries, the split
# is p^2 + (S-p)*S = 1,579,584, a 23% FLOP saving on top.
#
# EXACT. The two blocks compute softmax over exactly the key sets the mask
# permitted, and RoPE is applied to the full q/k BEFORE splitting, so absolute
# positions are preserved. `check_block_attn_equivalence` asserts it against
# the real unpatched GroupedQueryAttention.

# jax.nn.dot_product_attention(implementation=None) does NOT auto-select a
# fused kernel -- it uses the XLA reference path, which materialises the score
# matrix whether or not a mask is present. Removing the mask is therefore
# necessary but not sufficient: cuDNN has to be requested explicitly to get
# flash attention and its O(1) memory. Set via `set_block_attn_impl`.
_BLOCK_ATTN_IMPL = None

# Minimum size of the smaller block. Originally checked only in
# tag_block_attention, leaving the mask-detection fallback able to take a
# degenerate split (the dynamics wm_agent mask is a valid 2-block at p=1).
# Empirically the dynamics stayed dense anyway, but that was luck, not design.
_BLOCK_ATTN_MIN = 8

_BLOCK_ATTN_STATS = {"split": 0, "dense": 0, "unmatched": [],
                     "shape_mismatch": 0, "degenerate_skipped": 0}


def _detect_block_split(mask):
    """Return (p, kind) for a 2-block square mask, else None.

    kind "decoder": rows <p see cols <p;   rows >=p see everything.
    kind "encoder": rows <p see everything; rows >=p see cols >=p.

    Verified by reconstructing the mask and requiring an exact match, so a mask
    that is merely close to block-structured falls back to the dense path
    instead of silently changing the model.
    """
    import numpy as np
    try:
        m = np.asarray(jax.lax.stop_gradient(mask))
    except Exception:
        return None                      # a tracer: values unknown at trace time
    while m.ndim > 2:
        if m.shape[0] != 1:
            return None                  # per-batch/head mask: not a static block
        m = m[0]
    if m.ndim != 2 or m.shape[0] != m.shape[1] or m.dtype != np.bool_:
        return None
    S = m.shape[0]
    rows_all = m.all(axis=1)             # queries that see every key
    if rows_all.all() or not rows_all.any():
        return None                      # no mask at all, or no full rows
    idx = np.arange(S)
    if rows_all[-1]:                     # trailing block sees everything
        p = int(np.argmax(rows_all))     # first full row
        kind = "decoder"
        ref = (idx[:, None] < p) & (idx[None, :] < p) | (idx[:, None] >= p)
    else:                                # leading block sees everything
        p = int(np.argmin(rows_all))     # first non-full row
        kind = "encoder"
        ref = (idx[:, None] < p) | ((idx[:, None] >= p) & (idx[None, :] >= p))
    if p <= 0 or p >= S or not np.array_equal(m, ref):
        return None
    return p, kind


def _attn_core_split(self, x, split, deterministic=True, rngs=None):
    """`GroupedQueryAttention` forward with the space mask replaced by two
    unmasked block attentions. Non-causal, no KV cache -- space layers only."""
    p, kind = split
    q = self.to_q(x)
    q = rearrange(q, "B T (N H) -> B T N H", N=self.num_heads)
    kv = self.to_kv(x)
    k, v = rearrange(kv, "B S (C K H) -> C B S K H", C=2, K=self.num_kv_heads)

    scale = q.shape[-1] ** -0.5
    if self.qk_norm_type == 'qknorm':
        q = self.q_ln(q).astype(self.dtype)
        k = self.k_ln(k).astype(self.dtype)
    elif self.qk_norm_type == 'quest':
        k = k / (jnp.linalg.norm(k, axis=-1, keepdims=True) + 1e-6)
        scale = 1.0

    # RoPE on the FULL sequence before any split, so the second block's queries
    # keep absolute positions p..S-1 instead of restarting at 0.
    q, k = self.rope(q, k, start_pos=0)

    def dpa(qq, kk, vv):
        return jax.nn.dot_product_attention(
            qq, kk, vv, mask=None, scale=scale, is_causal=False,
            implementation=_BLOCK_ATTN_IMPL)

    if kind == "decoder":          # latents -> latents;  patches -> everything
        head, tail = dpa(q[:, :p], k[:, :p], v[:, :p]), dpa(q[:, p:], k, v)
    else:                          # encoder: latents -> everything; patches -> patches
        head, tail = dpa(q[:, :p], k, v), dpa(q[:, p:], k[:, p:], v[:, p:])

    attn = rearrange(jnp.concatenate([head, tail], axis=1), "B T N H -> B T (N H)")
    out = self.to_out(attn)
    return self.dropout(out, deterministic=deterministic, rngs=rngs)


def tag_block_attention(model, n_latents: int | None = None,
                        min_block: int = 8, verbose: bool = True) -> int:
    """Stamp each space-attention layer with its (split_point, kind).

    Value-based detection of the mask inside the model is not reliable:
    `build_space_mask` runs inside the jitted function, and whether its result
    arrives as a concrete array or a tracer depends on the trace context.
    (Sniffing for `__array__` does not distinguish them -- tracers define it too
    and raise only when called.) So the mask is rebuilt EAGERLY here, outside
    any trace, where its values are always readable, and the resulting split is
    recorded on the layer.

    `_block_spec` is a plain tuple, so nnx keeps it as static graphdef metadata
    and it survives the split/merge that jit performs on the model.

    Handles all three maskers:
      encoder  latents -> everything;  patches -> patches         split at n_latents
      decoder  latents -> latents;     patches -> everything      split at n_latents
      wm_agent action  -> action;      the rest -> everything     split at 1
    A dynamics model built WITH agent tokens is a 3-block mask; the detector
    rejects it and those layers stay on the dense path.

    Returns the number of layers tagged. 0 means the walk missed and the patch
    would silently no-op.
    """
    enc, dec = getattr(model, "encoder", None), getattr(model, "decoder", None)
    if enc is not None and dec is not None:
        targets = [(enc, "encoder"), (dec, "decoder")]            # Tokenizer
    elif hasattr(model, "get_token_layout") and hasattr(model, "transformer"):
        targets = [(model, "wm_agent")]                           # Dynamics
    else:
        targets = []

    n = 0
    for mod, mode in targets:
        try:
            if mode == "encoder":
                # Encoder derives n_patches from the frame size; the decoder
                # already knows both, so borrow them rather than hardcoding.
                layout = mod.get_token_layout(dec.H, dec.W)
            elif mode == "decoder":
                layout = mod.get_token_layout()
            else:
                if n_latents is None:
                    raise ValueError("dynamics tagging needs n_latents "
                                     "(the UNPACKED tokenizer latent count)")
                layout = mod.get_token_layout(n_latents=n_latents, n_agent=0)
            mask = layout.build_space_mask(mode)
        except Exception as exc:
            if verbose:
                print(f"  tag_block_attention: could not build {mode} mask: {exc!r}")
            continue

        split = _detect_block_split(mask)
        if split is None:
            if verbose:
                print(f"  tag_block_attention: {mode} mask is not 2-block "
                      f"(shape {getattr(mask, 'shape', '?')}) -- staying dense")
            continue

        # The split only pays when both blocks are substantial. The dynamics
        # wm_agent mask is technically 2-block but splits at p=1 (the action
        # token attends only to itself), so it peels off a single-query
        # attention -- an extra kernel launch plus a concatenate to save 1/290
        # of the score entries. Measured: ladder B=1 went 14.81 -> 15.53 ms
        # with it on. Skip degenerate splits by default.
        p_split, _kind = split
        S = layout.S
        if min(p_split, S - p_split) < min_block:
            if verbose:
                print(f"  tag_block_attention: {mode} split at {p_split}/{S} is "
                      f"degenerate (min block < {min_block}) -- staying dense")
            continue
        for layer in mod.transformer.layers:
            if not layer.is_time_layer:
                # Carry S so the call site can confirm the runtime token count
                # matches the layout this split was derived from. Without it, a
                # model built with agent tokens (n_agent > 0) would present a
                # 3-block mask at a different S while still carrying a 2-block
                # tag, and the split would silently compute the wrong answer.
                layer.attn._block_spec = (split[0], split[1], layout.S)
                n += 1
        if verbose:
            print(f"  tag_block_attention: {mode} split at {split[0]}/{layout.S} "
                  f"as {split[1]}")
    return n


def _space_attn_call(self, x, *, mask=None, local_window_size=None,
                     deterministic=True, cache=None, rngs=None,
                     return_weights=False):
    """Drop-in for `SpaceSelfAttention.__call__`.

    Patched HERE rather than inside `GroupedQueryAttention` because this is the
    last point where `mask` is a concrete array. One line down it is passed as
    positional arg 1 into `nnx.remat(self.attn, static_argnums=(2,3,4,6))`,
    which does not mark it static -- so inside remat it is a tracer and its
    values cannot be inspected at trace time. An earlier version of this patch
    sat at the GQA level and silently no-opped on all 34 attention calls for
    exactly that reason.
    """
    B, T, S, D = x.shape
    xf = rearrange(x, "B T S D -> (B T) S D")

    split = None
    if not return_weights:
        # Architectural tag first (always available), mask inspection as a
        # fallback for models that were not tagged.
        spec = getattr(self, "_block_spec", None)
        if spec is not None and mask is not None:
            p_spec, kind_spec, s_spec = spec
            if s_spec == S:
                split = (p_spec, kind_spec)
            else:
                _BLOCK_ATTN_STATS["shape_mismatch"] = \
                    _BLOCK_ATTN_STATS.get("shape_mismatch", 0) + 1
        if split is None and mask is not None:
            cand = _detect_block_split(mask)
            if cand is not None and min(cand[0], S - cand[0]) >= _BLOCK_ATTN_MIN:
                split = cand
            elif cand is not None:
                _BLOCK_ATTN_STATS["degenerate_skipped"] = \
                    _BLOCK_ATTN_STATS.get("degenerate_skipped", 0) + 1

    if split is not None:
        _BLOCK_ATTN_STATS["split"] += 1
        # NOTE: the split path deliberately does not wrap in nnx.remat. That is
        # correct for inference (no backward pass, so remat only costs fusion
        # opportunities) but means this patch must NOT be used for training --
        # activation memory would grow by the checkpointed amount.
        out = _attn_core_split(self.attn, xf, split, deterministic, rngs)
        attn_weights = None
    else:
        if mask is not None and not return_weights:
            _BLOCK_ATTN_STATS["dense"] += 1
            shp = getattr(mask, "shape", None)
            import jax.core as _jc
            kindstr = "tracer" if isinstance(mask, _jc.Tracer) else "concrete"
            entry = f"{shp} {kindstr}"
            if entry not in _BLOCK_ATTN_STATS["unmatched"]:
                _BLOCK_ATTN_STATS["unmatched"].append(entry)
        out, _, attn_weights = nnx.remat(self.attn, static_argnums=(2, 3, 4, 6))(
            xf, mask, None, deterministic, None, rngs, return_weights)

    out = rearrange(out, "(B T) S D -> B T S D", B=B, T=T)
    if attn_weights is not None:
        attn_weights = rearrange(attn_weights, "(B T) N S1 S2 -> B T N S1 S2", B=B, T=T)
    return out, None, attn_weights


def set_block_attn_impl(impl):
    """'cudnn' for flash attention, 'xla' for the reference path, None = default.

    cuDNN needs bf16/fp16 and head_dim a multiple of 8 up to 128; the tokenizer
    is bf16 with head_dim 64, so it qualifies. If it does not, JAX raises at
    trace time rather than silently falling back, which is what we want.
    """
    global _BLOCK_ATTN_IMPL
    _BLOCK_ATTN_IMPL = impl
    return _BLOCK_ATTN_IMPL


def block_attn_stats() -> dict:
    """How many attention calls took the split path. If `split` is 0 the patch
    silently did nothing, which is the failure mode worth checking for."""
    d = dict(_BLOCK_ATTN_STATS)
    d["implementation"] = _BLOCK_ATTN_IMPL
    return d


def check_block_attn_equivalence(pkg: str = "dreamer", n_latents: int = 512,
                                 n_patches: int = 920, heads: int = 16,
                                 kv_heads: int = 2, head_dim: int = 64,
                                 batch: int = 2, tol: float | None = None,
                                 dtype=None):
    """Assert `_attn_core_split` reproduces the real dense masked attention.

    This builds an actual `GroupedQueryAttention` and runs the PATCHED code
    path against the unpatched one. An earlier version inlined its own split
    and never called `_attn_core_split`, so every bug it could plausibly catch
    -- wrong `start_pos`, a dropped qk_norm branch, wrong `scale`, RoPE applied
    after the slice, a mismatched `to_out`/dropout -- was invisible to it. It
    tested only the mask algebra, which `_detect_block_split` already proves by
    reconstruction.

    `GroupedQueryAttention.__call__` is never patched (the patch replaces
    `SpaceSelfAttention.__call__`), so calling it directly gives the true
    reference. Both qk_norm variants are exercised.

    The dtype follows `_BLOCK_ATTN_IMPL`. cuDNN raises
    `NotImplementedError: Q must be fp16/bf16/fp8_...` on float32, so a float32
    check structurally *cannot* validate the flash kernel that is actually
    shipped — it would silently only ever test the XLA arm. When cuDNN is
    selected this runs in bfloat16 with a tolerance appropriate to 8 mantissa
    bits, so the kernel under test is the one being benchmarked.
    """
    if dtype is None:
        dtype = jnp.bfloat16 if _BLOCK_ATTN_IMPL == "cudnn" else jnp.float32
    if tol is None:
        # bf16 has 8 mantissa bits: ~4e-3 relative per op. The dense and split
        # paths take different reduction orders, so allow for that rather than
        # holding bf16 to an fp32 bar.
        tol = 3e-2 if dtype == jnp.bfloat16 else 1e-3
    models = importlib.import_module(f"{pkg}.models")
    utils = importlib.import_module(f"{pkg}.utils")
    parallel = importlib.import_module(f"{pkg}.parallel")
    Modality, TokenLayout = utils.Modality, utils.TokenLayout

    S = n_latents + n_patches
    segs = ((Modality.LATENT, n_latents), (Modality.IMAGE, n_patches))
    worst = 0.0

    for qk in ("qknorm", "quest"):
        attn = models.GroupedQueryAttention(
            dim=heads * head_dim, num_heads=heads, num_kv_heads=kv_heads,
            dropout_rate=0.0, qk_norm_type=qk, is_causal=False,
            dtype=dtype, param_dtype=dtype,
            mesh_rules=parallel.MeshRules(), rngs=nnx.Rngs(0))
        x = jax.random.normal(jax.random.PRNGKey(1),
                              (batch, S, heads * head_dim), dtype)

        for mode in ("encoder", "decoder"):
            mask = TokenLayout(segs).build_space_mask(mode)
            det = _detect_block_split(mask)
            assert det is not None, f"{mode}: mask not recognised as 2-block"
            assert det == (n_latents, mode), f"{mode}: got {det}"

            # Reference: the unpatched GQA forward with the dense mask.
            ref, _, _ = attn(x, mask=mask, deterministic=True)
            # Under test: the actual patched split implementation.
            got = _attn_core_split(attn, x, det, deterministic=True, rngs=None)

            err = float(jnp.max(jnp.abs(ref - got)))
            rel = err / float(jnp.max(jnp.abs(ref)))
            worst = max(worst, rel)
            assert rel <= tol, (f"{mode}/{qk}: _attn_core_split differs from the "
                                f"dense path by {rel} relative")
            print(f"check_block_attn_equivalence[{mode}/{qk}]: PASS  "
                  f"split {det[0]}/{S}, max|diff| {err:.2e}, relative {rel:.2e}  "
                  f"[{jnp.dtype(dtype).name}, sdpa="
                  f"{_BLOCK_ATTN_IMPL or 'xla-default'}, tol {tol:.0e}]")

    # --- dispatch layer: exercise _space_attn_call itself ---------------
    # _attn_core_split is now covered above, but the DISPATCH around it was
    # not: the tag lookup, the s_spec == S guard, the `mask is not None`
    # condition, the return_weights bypass, and the (B T) rearrange round-trip.
    # One call through a real SpaceSelfAttention against the saved original
    # covers all of it.
    orig_space = _ORIGINALS.get("SpaceSelfAttention.__call__")
    if orig_space is not None:
        # Snapshot: this test deliberately provokes a stale tag and extra
        # splits. Leaving them in the global counters would inflate the no-op
        # check and, worse, make `shape_mismatch != 0` for the whole run, which
        # silently flips dec_flops back to dense FLOPs.
        _saved = {k: (list(v) if isinstance(v, list) else v)
                  for k, v in _BLOCK_ATTN_STATS.items()}
        sa = models.SpaceSelfAttention(
            dim=heads * head_dim, num_heads=heads, num_kv_heads=kv_heads,
            dropout_rate=0.0, qk_norm_type="qknorm",
            dtype=dtype, param_dtype=dtype,
            mesh_rules=parallel.MeshRules(), rngs=nnx.Rngs(0))
        xt = jax.random.normal(jax.random.PRNGKey(2), (1, 2, S, heads * head_dim), dtype)
        mask = TokenLayout(segs).build_space_mask("decoder")

        ref_out, ref_cache, ref_w = orig_space(
            sa, xt, mask=mask, deterministic=True)
        sa._block_spec = (n_latents, "decoder", S)               # tagged path
        got_out, got_cache, got_w = _space_attn_call(
            sa, xt, mask=mask, deterministic=True)
        d = float(jnp.max(jnp.abs(ref_out - got_out))) / float(jnp.max(jnp.abs(ref_out)))
        assert got_out.shape == ref_out.shape, "dispatch changed the output shape"
        assert ref_cache is None and got_cache is None, "cache contract broken"
        assert (ref_w is None) == (got_w is None), "attn_weights contract broken"
        assert d <= tol, f"_space_attn_call (tagged) differs by {d} relative"
        worst = max(worst, d)

        # A stale tag whose S does not match must be REJECTED and fall back.
        # Output equality alone cannot prove this -- the fallback detector finds
        # the same split from the same mask, so a stale tag that was wrongly
        # trusted would produce an identical result. The counter is the only
        # thing that distinguishes the two paths, so assert on it.
        before = _BLOCK_ATTN_STATS.get("shape_mismatch", 0)
        sa._block_spec = (n_latents, "decoder", S + 4)
        stale, _, _ = _space_attn_call(sa, xt, mask=mask, deterministic=True)
        after = _BLOCK_ATTN_STATS.get("shape_mismatch", 0)
        ds = float(jnp.max(jnp.abs(ref_out - stale))) / float(jnp.max(jnp.abs(ref_out)))
        assert after == before + 1, (
            f"stale tag was NOT rejected: shape_mismatch stayed at {after}. "
            f"The s_spec == S guard is not firing.")
        assert ds <= tol, f"stale-tag fallback differs by {ds} relative"

        # And a stale tag with NO usable mask must not split at all.
        b2 = _BLOCK_ATTN_STATS["split"]
        _space_attn_call(sa, xt, mask=None, deterministic=True)
        assert _BLOCK_ATTN_STATS["split"] == b2, \
            "split taken with mask=None -- the `mask is not None` guard is not firing"

        del sa._block_spec
        print(f"check_block_attn_equivalence[dispatch]: PASS  tagged {d:.2e} relative; "
              f"stale tag rejected (shape_mismatch {before}->{after}) and fell back "
              f"to {ds:.2e}; mask=None did not split")
        _BLOCK_ATTN_STATS.clear()
        _BLOCK_ATTN_STATS.update(_saved)      # restore; the test is not the run

    # The residual is float32 GEMM precision (JAX permits TF32 on NVIDIA by
    # default) plus reduction order -- not algebra. The two paths sum the same
    # terms: masked logits go to finfo.min and underflow to exactly 0 after
    # exp, so the dense sum over S keys and the split sum over its subset are
    # term-for-term equal. A float64 replay of the same geometry agrees to 0.0.
    return worst


FP8_MAX = 448.0          # max finite magnitude of float8_e4m3fn
_FP8_STATS = {"quantized": 0, "skipped": 0, "params_before": 0, "params_after": 0}
_FP8_QUANTIZED: list = []   # modules whose kernels were destructively quantized


def _fp8_dtype():
    return jnp.float8_e4m3fn


def _iter_linears(root):
    """Yield every `nnx.Linear` under `root`.

    `nnx.iter_graph` exists but its yield shape has moved between flax
    versions, and the released inference repo pins an older one; an explicit
    walk is version-proof and this runs once at startup.
    """
    seen, out, stack = set(), [], [root]
    while stack:
        node = stack.pop()
        if id(node) in seen:
            continue
        seen.add(id(node))
        if isinstance(node, nnx.Linear):
            out.append(node)
            continue                      # a Linear has no nested Linears
        children = []
        if isinstance(node, (list, tuple)):
            children = list(node)
        elif isinstance(node, dict):
            children = list(node.values())
        elif hasattr(node, "__dict__"):
            children = list(vars(node).values())
        if hasattr(node, "__len__") and hasattr(node, "__getitem__") \
                and not isinstance(node, (str, bytes, list, tuple, dict)):
            try:                          # nnx.List and friends
                children += [node[i] for i in range(len(node))]
            except Exception:
                pass
        stack.extend(c for c in children if not isinstance(c, (str, bytes, int, float, bool, type(None))))
    return out


def quantize_linear_fp8(model, min_elems: int = 1 << 20, pkg: str = "dreamer"):
    """Convert large `nnx.Linear` kernels to e4m3 with a per-tensor scale.

    Only large kernels are touched. Embeddings, norms, the timestep MLPs and
    the zero-initialised flow head are left alone: they are a rounding error in
    bytes, and the flow head in particular is the model's output, where 3
    mantissa bits would be felt directly.
    """
    n_q = n_skip = 0
    before = after = 0
    for node in _iter_linears(model):
        w = node.kernel.value
        before += w.size * w.dtype.itemsize
        if w.size < min_elems or getattr(node, "_fp8_scale", None) is not None:
            n_skip += 1
            after += w.size * w.dtype.itemsize
            continue
        w32 = w.astype(jnp.float32)
        amax = float(jnp.max(jnp.abs(w32)))
        if not (amax > 0):
            n_skip += 1
            after += w.size * w.dtype.itemsize
            continue
        scale = amax / FP8_MAX
        wq = jnp.clip(w32 / scale, -FP8_MAX, FP8_MAX).astype(_fp8_dtype())
        node.kernel = nnx.Param(wq)
        # Python float, so it is static in the graphdef and constant-folded
        # into the GEMM epilogue rather than becoming another array to read.
        node._fp8_scale = float(scale)
        _FP8_QUANTIZED.append(id(node))
        n_q += 1
        after += wq.size * wq.dtype.itemsize
    _FP8_STATS.update(quantized=n_q, skipped=n_skip,
                      params_before=before, params_after=after)
    return model


def _linear_fp8_call(self, inputs):
    """Drop-in for `nnx.Linear.__call__` on fp8-quantized layers."""
    s = getattr(self, "_fp8_scale", None)
    if s is None:
        return _ORIGINALS["nnx.Linear.__call__"](self, inputs)

    ct = self.dtype if self.dtype is not None else inputs.dtype
    x32 = inputs.astype(jnp.float32)
    # Dynamic per-tensor activation scale. Static calibration would avoid the
    # amax reduction, but it needs a calibration pass and risks clipping on
    # out-of-distribution activations; the reduction is ~1 MB here.
    a_amax = jnp.maximum(jnp.max(jnp.abs(x32)), 1e-6)
    a_scale = a_amax / FP8_MAX
    xq = jnp.clip(x32 / a_scale, -FP8_MAX, FP8_MAX).astype(_fp8_dtype())

    # This exact shape -- convert-then-scale on both operands feeding a dot --
    # is what XLA's GemmRewriter folds into one cublasLt fp8 GEMM.
    y = jax.lax.dot_general(
        xq.astype(ct) * a_scale.astype(ct),
        self.kernel.value.astype(ct) * jnp.asarray(s, ct),
        (((inputs.ndim - 1,), (0,)), ((), ())),
    )
    if self.use_bias and getattr(self, "bias", None) is not None:
        y = y + self.bias.value.astype(ct)
    return y


def fp8_stats() -> dict:
    d = dict(_FP8_STATS)
    if d["params_before"]:
        d["compression"] = round(d["params_before"] / max(d["params_after"], 1), 3)
    return d


def check_fp8_accuracy(shape=(290, 1920), out=7680, dtype=jnp.bfloat16, seed=0):
    """Relative error e4m3 induces on one representative GEMM.

    Uses the real dynamics shapes (S=290 tokens, d_model=1920, SwiGLU hidden
    7680) and lecun-normal-scaled weights, so the number is comparable to what
    the model actually does rather than to a toy matmul.
    """
    k1, k2 = jax.random.split(jax.random.PRNGKey(seed))
    x = jax.random.normal(k1, shape, jnp.float32)
    w = jax.random.normal(k2, (shape[-1], out), jnp.float32) * (shape[-1] ** -0.5)

    ref = (x.astype(jnp.float32) @ w.astype(jnp.float32))

    bf = (x.astype(dtype) @ w.astype(dtype)).astype(jnp.float32)

    ws = float(jnp.max(jnp.abs(w))) / FP8_MAX
    xs = float(jnp.max(jnp.abs(x))) / FP8_MAX
    wq = jnp.clip(w / ws, -FP8_MAX, FP8_MAX).astype(_fp8_dtype())
    xq = jnp.clip(x / xs, -FP8_MAX, FP8_MAX).astype(_fp8_dtype())
    f8 = ((xq.astype(dtype) * xs) @ (wq.astype(dtype) * ws)).astype(jnp.float32)

    def rel(a):
        return float(jnp.linalg.norm(a - ref) / jnp.linalg.norm(ref))

    r_bf, r_f8 = rel(bf), rel(f8)
    print(f"check_fp8_accuracy on {shape} @ ({shape[-1]},{out}):")
    print(f"  bf16 vs fp32 : {r_bf:.5f} relative error")
    print(f"  fp8  vs fp32 : {r_f8:.5f} relative error   ({r_f8 / max(r_bf, 1e-12):.1f}x bf16)")
    return {"bf16_rel_err": r_bf, "fp8_rel_err": r_f8,
            "fp8_over_bf16": r_f8 / max(r_bf, 1e-12)}


def fp8_gemm_kernels_in_trace(trace_dir: str) -> dict:
    """Did XLA actually emit fused fp8 GEMMs, or just dequantize and fall back?

    The whole patch rests on cublasLt fp8 kernels appearing. If they did not,
    fp8 is pure overhead and the benchmark number is measuring a regression.
    """
    import gzip
    from collections import defaultdict
    from pathlib import Path
    cands = list(Path(trace_dir).rglob("*.trace.json.gz"))
    if not cands:
        return {"error": f"no trace under {trace_dir}"}
    ev = json.loads(gzip.open(max(cands, key=lambda p: p.parent.name), "rt").read())
    tot = defaultdict(float)
    for e in ev.get("traceEvents", []):
        if e.get("ph") == "X" and "dur" in e:
            tot[e["name"]] += e["dur"]
    fp8 = {k: round(v / 1e3, 3) for k, v in tot.items()
           if "fp8" in k.lower() or "e4m3" in k.lower()}
    conv = {k: round(v / 1e3, 3) for k, v in tot.items()
            if "convert" in k.lower()}
    return {"fp8_kernels": fp8, "fp8_total_ms": round(sum(fp8.values()), 3),
            "convert_kernels": dict(sorted(conv.items(), key=lambda kv: -kv[1])[:5]),
            "convert_total_ms": round(sum(conv.values()), 3),
            "verdict": ("fused fp8 GEMM present" if fp8
                        else "NO fp8 kernels -- XLA fell back; patch is overhead")}


# ---------------------------------------------------------------------------
# ragged_kv  --  per-sequence KV positions, the prerequisite for continuous
#                batching
# ---------------------------------------------------------------------------
#
# `KVCache.index` is a SCALAR shared by the whole batch, so every sequence in a
# batch must sit at the same rollout step. That is static batching: you can
# serve N streams only if they all start together and none joins or leaves.
#
# The measured reason to care: bandwidth efficiency is set by weight-matrix
# size, and every GEMM climbs steeply with M. `to_q` runs at 33 TFLOP/s at
# M=256 and 334 at M=4640 (bench/GEMM_DIAGNOSTIC.md). Batching concurrent
# sessions is the only lever that fixes small GEMMs without touching numerics --
# but real sessions join and leave at arbitrary times, which a scalar index
# cannot express.
#
# This makes `index` shape (B,), which requires three changes:
#   1. update()           per-row writes        (vmap of dynamic_update_slice)
#   2. get_ordered_kv()   per-row mask          (age computed from index[:,...])
#   3. RoPE start_pos     per-row rotation      (t = arange(T) + start_pos[:,None])
#
# EXACT when all rows share an index: `check_ragged_kv_equivalence` asserts the
# ragged path reproduces the scalar path, and separately that a genuinely
# ragged batch matches running each sequence on its own.

_RAGGED_STATS = {"ragged_calls": 0}


def _ragged_update(self, k_new, v_new):
    """`KVCache.update` with a per-sequence write position."""
    cls = type(self)
    T = k_new.shape[1]
    idx = jnp.asarray(self.index)
    if idx.ndim == 0:                       # scalar cache: defer to the fast path
        return _update_fast(self, k_new, v_new)

    write_idx = idx % self.window_size      # (B,)

    def _row(buf, new, pos):
        # buf (W,K,H), new (T,K,H), pos scalar. dynamic_update_slice clamps, and
        # T==1 can never wrap since pos <= W-1.
        return jax.lax.dynamic_update_slice(buf, new, (pos, 0, 0))

    k = jax.vmap(_row)(self.k, k_new, write_idx)
    v = jax.vmap(_row)(self.v, v_new, write_idx)
    return cls(k=k, v=v, index=idx + T, window_size=self.window_size)


def _ragged_get_ordered_kv(self, query_len):
    """`get_ordered_kv` with a per-sequence mask. Same age algebra as
    `_get_ordered_kv_inplace`, lifted over the batch axis."""
    W = self.window_size
    idx = jnp.asarray(self.index)
    if idx.ndim == 0:
        return _get_ordered_kv_inplace(self, query_len)

    idx_b = idx.reshape(-1, 1, 1, 1)                        # (B,1,1,1)
    j = jnp.arange(W)[None, None, None, :]                  # (1,1,1,W)
    age = jnp.mod(idx_b - 1 - j, W)                         # (B,1,1,W)
    i = jnp.arange(query_len)[None, None, :, None]          # (1,1,q,1)
    causal = age >= (query_len - 1 - i)                     # (B,1,q,W)
    written = age <= (idx_b - 1)                            # (B,1,1,W)
    _RAGGED_STATS["ragged_calls"] += 1
    return self.k, self.v, jnp.logical_and(causal, written)


def _ragged_rope_call(self, q, k, start_pos=0):
    """`RotaryEmbedding1D.__call__` accepting a per-sequence `start_pos`.

    The released version does `jnp.outer(arange(T) + start_pos, inv_freq)`,
    which silently flattens if `start_pos` is a vector. With ragged positions
    each sequence needs its own rotation.
    """
    T = q.shape[1]
    sp = jnp.asarray(start_pos)
    inv = self.inv_freq.value

    if sp.ndim == 0:
        t = (jnp.arange(T, dtype=self.dtype) + sp)[None, :]       # (1,T)
    else:
        t = jnp.arange(T, dtype=self.dtype)[None, :] + sp[:, None]  # (B,T)

    freqs = t[..., None].astype(jnp.float32) * inv[None, None, :]   # (B,T,D/2)
    cos = jnp.cos(freqs).astype(self.dtype)[:, :, None, :]          # (B,T,1,D/2)
    sin = jnp.sin(freqs).astype(self.dtype)[:, :, None, :]
    return self._apply(q, k, cos, sin)


def ragged_kv_init(cls, batch_size, window_size, num_kv_heads, head_dim,
                   dtype=jnp.float32):
    """`KVCache.init` variant whose index is per-sequence, shape (B,)."""
    dt = jnp.bfloat16 if dtype == "bfloat16" else dtype
    return cls(
        k=jnp.zeros((batch_size, window_size, num_kv_heads, head_dim), dtype=dt),
        v=jnp.zeros((batch_size, window_size, num_kv_heads, head_dim), dtype=dt),
        index=jnp.zeros((batch_size,), dtype=jnp.int32),
        window_size=window_size,
    )


def ragged_stats() -> dict:
    return dict(_RAGGED_STATS)


def check_ragged_kv_equivalence(pkg: str | None = None, window: int = 16,
                                batch: int = 4, kv_heads: int = 3,
                                head_dim: int = 8, steps: int = 40,
                                tol: float = 1e-6):
    """Two assertions, both against genuinely independent references.

    ALIGNED  a ragged cache whose rows all hold the same index must reproduce
             the scalar-index path exactly.
    RAGGED   a batch at genuinely different positions must match running each
             sequence separately -- the property continuous batching needs and
             the one a scalar index cannot provide.
    """
    # "pipeline" in the released inference repo, "dreamer" in this fork. Both
    # sit on PYTHONPATH in the bench container and importing the wrong one dies
    # on an unrelated JAX version skew, so take the same answer the profiler
    # takes instead of hardcoding either name.
    if pkg is None:
        try:
            from bench.prod_config import MODEL_PKG as pkg
        except ImportError:
            pkg = "dreamer"
    models = importlib.import_module(f"{pkg}.models")
    KVCache = models.KVCache
    orig_get = _ORIGINALS.get("KVCache.get_ordered_kv", KVCache.get_ordered_kv)
    orig_upd = _ORIGINALS.get("KVCache.update", KVCache.update)

    def attend(k, v, m, q):
        logits = jnp.einsum("bqnh,bknh->bnqk", q, k) * head_dim ** -0.5
        logits = jnp.where(m, logits, jnp.finfo(logits.dtype).min)
        return jnp.einsum("bnqk,bknh->bqnh", jax.nn.softmax(logits, -1), v)

    key = jax.random.PRNGKey(0)

    # --- 1. aligned: ragged must equal scalar ---------------------------------
    c_scalar = KVCache.init(batch, window, kv_heads, head_dim, dtype=jnp.float32)
    c_ragged = ragged_kv_init(KVCache, batch, window, kv_heads, head_dim,
                              dtype=jnp.float32)
    worst_aligned = 0.0
    for t in range(steps):
        key, k1, k2, k3 = jax.random.split(key, 4)
        kn = jax.random.normal(k1, (batch, 1, kv_heads, head_dim))
        vn = jax.random.normal(k2, (batch, 1, kv_heads, head_dim))
        q = jax.random.normal(k3, (batch, 1, kv_heads, head_dim))
        c_scalar = orig_upd(c_scalar, kn, vn)
        c_ragged = _ragged_update(c_ragged, kn, vn)
        a = attend(*orig_get(c_scalar, query_len=1), q)
        b = attend(*_ragged_get_ordered_kv(c_ragged, query_len=1), q)
        e = float(jnp.max(jnp.abs(a - b)))
        worst_aligned = max(worst_aligned, e)
        assert e <= tol, f"aligned mismatch at step {t}: {e}"

    # --- 2. ragged: batch must equal per-sequence-alone -----------------------
    # Give each row a different number of writes, then compare against caches
    # advanced individually. This is the case a scalar index cannot represent.
    offsets = [0, 3, window, window + 7][:batch]
    c_multi = ragged_kv_init(KVCache, batch, window, kv_heads, head_dim,
                             dtype=jnp.float32)
    singles = [ragged_kv_init(KVCache, 1, window, kv_heads, head_dim,
                              dtype=jnp.float32) for _ in offsets]
    key = jax.random.PRNGKey(7)
    n_steps = max(offsets) + 5
    for t in range(n_steps):
        key, k1, k2 = jax.random.split(key, 3)
        kn = jax.random.normal(k1, (batch, 1, kv_heads, head_dim))
        vn = jax.random.normal(k2, (batch, 1, kv_heads, head_dim))
        active = jnp.array([t >= o for o in offsets])
        # Rows advance only once their session has "joined": emulate by writing
        # the same values but holding the index for not-yet-active rows.
        prev = c_multi.index
        c_multi = _ragged_update(c_multi, kn, vn)
        c_multi = type(c_multi)(k=c_multi.k, v=c_multi.v,
                                index=jnp.where(active, c_multi.index, prev),
                                window_size=window)
        for s, o in enumerate(offsets):
            if t >= o:
                singles[s] = _ragged_update(singles[s], kn[s:s + 1], vn[s:s + 1])

    key, kq = jax.random.split(key)
    q = jax.random.normal(kq, (batch, 1, kv_heads, head_dim))
    got = attend(*_ragged_get_ordered_kv(c_multi, query_len=1), q)
    worst_ragged = 0.0
    for s in range(batch):
        ref = attend(*_ragged_get_ordered_kv(singles[s], query_len=1), q[s:s + 1])
        e = float(jnp.max(jnp.abs(ref - got[s:s + 1])))
        worst_ragged = max(worst_ragged, e)
        assert e <= tol, f"ragged row {s} (offset {offsets[s]}) mismatch: {e}"

    print(f"check_ragged_kv_equivalence: PASS  aligned {worst_aligned:.2e} over "
          f"{steps} writes; ragged {worst_ragged:.2e} across offsets {offsets} "
          f"(window {window})")
    return max(worst_aligned, worst_ragged)
