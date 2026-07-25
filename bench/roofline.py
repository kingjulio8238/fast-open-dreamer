"""Analytical param / FLOP / bytes model for Open Dreamer inference.

Pure python, mirrors dreamer/models.py exactly. No JAX needed.
"""

def transformer_layer_params(d, n_heads, n_kv, mlp_ratio, qknorm=True, swiglu=True):
    hd = d // n_heads
    kv_dim = n_kv * hd
    attn = d * d + d * 2 * kv_dim + d * d          # to_q, to_kv, to_out
    if qknorm:
        attn += 2 * hd                              # q_ln, k_ln scales
    attn += d                                       # pre-attn RMSNorm (BlockCausalLayer.norm)
    h = int(d * mlp_ratio)
    mlp = d * (2 * h if swiglu else h) + h * d
    mlp += d                                        # MLP RMSNorm
    return attn + mlp, attn, mlp


def transformer_matmul_params(d, n_heads, n_kv, mlp_ratio, swiglu=True):
    """Only the GEMM weights (what dominates FLOPs / weight traffic)."""
    hd = d // n_heads
    kv_dim = n_kv * hd
    attn = 2 * d * d + d * 2 * kv_dim
    h = int(d * mlp_ratio)
    mlp = d * (2 * h if swiglu else h) + h * d
    return attn + mlp


# ---------------- configs (from configs/*.yaml) ----------------
class Dyn:
    depth = 30
    d = 64 * depth              # 1920
    n_heads = d // 64           # 30
    n_kv = max(1, n_heads // 8) # 3
    hd = d // n_heads           # 64
    mlp_ratio = 4
    d_bottleneck = 16
    packing = 2
    n_register = 32
    n_latents_tok = 512
    n_spatial = n_latents_tok // packing        # 256
    S = 1 + 1 + n_spatial + n_register          # 290
    time_every = 4
    time_off = 0
    ctx = 192
    k_max = 256
    n_bin, n_cat = 27, 121


class Enc:
    depth = 12
    d = 128 * depth             # 1536
    n_heads = d // 64           # 24
    n_kv = max(1, n_heads // 8) # 3
    hd = 64
    mlp_ratio = 4
    n_latents = 512
    patch = 16
    H, W = 368, 640
    n_patches = (H // patch) * (W // patch)     # 920
    S = n_latents + n_patches                   # 1432
    d_bottleneck = 16
    time_every, time_off = 4, 3
    ctx = 16


class Dec:
    depth = 8
    d = 128 * depth             # 1024
    n_heads = d // 64           # 16
    n_kv = max(1, n_heads // 8) # 2
    hd = 64
    mlp_ratio = 4
    n_latents = 512
    patch = 16
    H, W = 368, 640
    n_patches = (H // patch) * (W // patch)
    S = n_latents + n_patches
    d_patch = patch * patch * 3                 # 768
    d_bottleneck = 16
    time_every, time_off = 4, 3
    ctx = 16


def n_time_layers(depth, every, off):
    return sum(1 for i in range(depth) if (i + off) % every == 0)


def report():
    out = {}

    # ---- Dynamics params ----
    per_layer, attn_p, mlp_p = transformer_layer_params(Dyn.d, Dyn.n_heads, Dyn.n_kv, Dyn.mlp_ratio)
    body = per_layer * Dyn.depth
    extras = (
        Dyn.d_bottleneck * Dyn.packing * Dyn.d          # spatial_proj
        + Dyn.n_register * Dyn.d                        # register tokens
        + Dyn.d                                         # base action emb
        + Dyn.n_bin * 2 * Dyn.d                         # binary embeds
        + Dyn.n_cat * Dyn.d                             # categorical embed
        + 2 * (256 * (Dyn.d // 2) + (Dyn.d // 2) + (Dyn.d // 2) ** 2 + (Dyn.d // 2))  # step+signal embedders
        + Dyn.d * Dyn.d_bottleneck * Dyn.packing        # flow_x_head
    )
    dyn_params = body + extras
    dyn_mm = transformer_matmul_params(Dyn.d, Dyn.n_heads, Dyn.n_kv, Dyn.mlp_ratio) * Dyn.depth

    # ---- Encoder / Decoder params ----
    enc_body = transformer_layer_params(Enc.d, Enc.n_heads, Enc.n_kv, Enc.mlp_ratio)[0] * Enc.depth
    enc_extras = (Enc.patch**2 * 3) * Enc.d + Enc.d * Enc.d_bottleneck + Enc.n_latents * Enc.d + Enc.d
    enc_params = enc_body + enc_extras

    dec_body = transformer_layer_params(Dec.d, Dec.n_heads, Dec.n_kv, Dec.mlp_ratio)[0] * Dec.depth
    dec_extras = Dec.d_bottleneck * Dec.d + Dec.d * Dec.d_patch + Dec.n_patches * Dec.d
    dec_params = dec_body + dec_extras
    dec_mm = transformer_matmul_params(Dec.d, Dec.n_heads, Dec.n_kv, Dec.mlp_ratio) * Dec.depth \
        + Dec.d_bottleneck * Dec.d + Dec.d * Dec.d_patch

    out['params'] = dict(dynamics=dyn_params, encoder=enc_params, decoder=dec_params,
                         dyn_per_layer=per_layer, dyn_attn=attn_p, dyn_mlp=mlp_p)

    # ---- Per-frame inference cost ----
    dyn_time = n_time_layers(Dyn.depth, Dyn.time_every, Dyn.time_off)
    dyn_space = Dyn.depth - dyn_time
    dec_time = n_time_layers(Dec.depth, Dec.time_every, Dec.time_off)
    dec_space = Dec.depth - dec_time

    def dyn_forward(B):
        """One dynamics forward at T=1, batch B. Returns (flops, weight_bytes_f32, kv_bytes)."""
        tokens = B * Dyn.S
        gemm = 2 * tokens * dyn_mm
        # space attn: per (B) group over S tokens, QK^T + AV
        sp = dyn_space * 4 * (Dyn.S ** 2) * Dyn.d * B
        # time attn: q_len=1 over window, batch B*S
        tm = dyn_time * 4 * 1 * Dyn.ctx * Dyn.d * B * Dyn.S
        return gemm + sp + tm, sp, tm

    def kv_bytes_dyn(B, dtype_bytes=2):
        return dyn_time * B * Dyn.S * Dyn.ctx * 2 * Dyn.n_kv * Dyn.hd * dtype_bytes

    def dec_forward(B):
        tokens = B * Dec.S
        gemm = 2 * tokens * dec_mm
        sp = dec_space * 4 * (Dec.S ** 2) * Dec.d * B
        tm = dec_time * 4 * 1 * Dec.ctx * Dec.d * B * Dec.S
        return gemm + sp + tm, sp, tm

    out['layers'] = dict(dyn_time=dyn_time, dyn_space=dyn_space, dec_time=dec_time, dec_space=dec_space)
    out['fwd'] = dict(dyn=dyn_forward, dec=dec_forward, kv=kv_bytes_dyn)
    out['mm'] = dict(dyn=dyn_mm, dec=dec_mm)
    return out


if __name__ == '__main__':
    r = report()
    p = r['params']
    G = 1e9
    print("=" * 78)
    print("PARAMETERS")
    print("=" * 78)
    print(f"  dynamics : {p['dynamics']/1e6:9.1f} M   ({p['dynamics']/G:.3f} B)")
    print(f"    per layer {p['dyn_per_layer']/1e6:.2f} M  = attn {p['dyn_attn']/1e6:.2f} M + mlp {p['dyn_mlp']/1e6:.2f} M")
    print(f"  tok enc  : {p['encoder']/1e6:9.1f} M")
    print(f"  tok dec  : {p['decoder']/1e6:9.1f} M")
    print(f"  TOTAL serve (dyn+dec) : {(p['dynamics']+p['decoder'])/1e6:.1f} M")
    print(f"  layer split: {r['layers']}")

    print()
    print("=" * 78)
    print("PER-GENERATED-FRAME COST  (shortcut: 4 denoise steps + 1 cache-commit pass)")
    print("=" * 78)
    N_PASS = 5
    for B in (1, 4, 16, 64):
        dyn_f, dyn_sp, dyn_tm = r['fwd']['dyn'](B)
        dec_f, dec_sp, dec_tm = r['fwd']['dec'](B)
        total_f = N_PASS * dyn_f + dec_f
        # weight traffic: params read once per pass (f32 storage as configured)
        w32 = N_PASS * p['dynamics'] * 4 + p['decoder'] * 4
        w16 = N_PASS * p['dynamics'] * 2 + p['decoder'] * 2
        kv = r['fwd']['kv'](B, 2)
        # jnp.roll = read full + write full; then attention reads rolled copy -> 3x
        kv_traffic = N_PASS * 3 * kv
        print(f"\n  batch {B}:")
        print(f"    FLOPs/frame        : {total_f/1e12:8.3f} TFLOP  "
              f"(dyn {N_PASS*dyn_f/1e12:.3f}, dec {dec_f/1e12:.3f})")
        print(f"      dyn attn share   : space {N_PASS*dyn_sp/dyn_f/N_PASS*100:5.1f}%  time {N_PASS*dyn_tm/dyn_f/N_PASS*100:5.1f}%  of one dyn pass")
        print(f"      dec attn share   : space {dec_sp/dec_f*100:5.1f}%  time {dec_tm/dec_f*100:5.1f}%")
        print(f"    weight bytes f32   : {w32/1e9:8.3f} GB   (as-configured param_dtype=float32)")
        print(f"    weight bytes bf16  : {w16/1e9:8.3f} GB")
        print(f"    KV cache resident  : {kv/1e9:8.3f} GB  (dynamics, bf16, window {Dyn.ctx})")
        print(f"    KV traffic/frame   : {kv_traffic/1e9:8.3f} GB  (3x per pass from jnp.roll)")
        print(f"    arith intensity f32: {total_f/(w32+kv_traffic):8.1f} FLOP/byte")
        print(f"    arith intensity bf16(no-roll): {total_f/(w16+N_PASS*kv):8.1f} FLOP/byte")

    print()
    print("=" * 78)
    print("ROOFLINE  (H100 SXM: 990 TF bf16 dense, 3.35 TB/s | B200: 2250 TF, 8 TB/s)")
    print("=" * 78)
    for name, tf, bw in (("H100", 990e12, 3.35e12), ("B200", 2250e12, 8.0e12)):
        print(f"\n  {name}  ridge point = {tf/bw:.0f} FLOP/byte")
        for B in (1, 4, 16, 64):
            dyn_f, _, _ = r['fwd']['dyn'](B)
            dec_f, _, _ = r['fwd']['dec'](B)
            total_f = N_PASS * dyn_f + dec_f
            w32 = N_PASS * p['dynamics'] * 4 + p['decoder'] * 4
            w16 = N_PASS * p['dynamics'] * 2 + p['decoder'] * 2
            kv = r['fwd']['kv'](B, 2)
            b_now = w32 + N_PASS * 3 * kv
            b_best = w16 + N_PASS * kv
            t_now = max(total_f / tf, b_now / bw) * 1e3
            t_best = max(total_f / tf, b_best / bw) * 1e3
            print(f"    B={B:3d}  as-configured {t_now:7.2f} ms/frame ({1000/t_now:7.1f} fps, "
                  f"{1000/t_now*B:8.1f} fps aggregate)   |  bf16+no-roll {t_best:6.2f} ms "
                  f"({1000/t_best*B:8.1f} fps agg)")
