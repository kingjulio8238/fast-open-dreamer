"""Does fusing to_q + to_kv into one GEMM help, and is the 290->296 pad real?

The sweep showed small weight matrices (7.4 MB to_q) reach 0.13 TB/s while the
59 MB fc_in reaches 1.02 TB/s -- an 8x efficiency gap from matrix size alone.
to_q and to_kv share the same input, so they can be a single
(1920, 1920+384) GEMM. That is exact and halves the small-GEMM launches.

Also checks whether feeding M=290 vs M=296 matters once XLA is free to pad
either way, since the HLO shows it already emits bf16[296,...].
"""
import time, jax, jax.numpy as jnp

def best(fn,*a,w=20,n=30):
    for _ in range(w): jax.block_until_ready(fn(*a))
    b=1e9
    for _ in range(n):
        t=time.perf_counter(); jax.block_until_ready(fn(*a)); b=min(b,time.perf_counter()-t)
    return b*1e3

k=jax.random.PRNGKey(0)
for M in (290, 296):
    x  = jax.random.normal(k,(M,1920),jnp.bfloat16)
    wq = jax.random.normal(k,(1920,1920),jnp.bfloat16)
    wkv= jax.random.normal(k,(1920,384),jnp.bfloat16)
    wqkv=jax.random.normal(k,(1920,2304),jnp.bfloat16)

    sep = jax.jit(lambda x,a,b:(x@a, x@b))
    fus = jax.jit(lambda x,a:(lambda o:(o[:,:1920],o[:,1920:]))(x@a))
    t_s, t_f = best(sep,x,wq,wkv), best(fus,x,wqkv)
    print(f"M={M}: separate q+kv {t_s*1e3:7.1f} us | fused qkv {t_f*1e3:7.1f} us "
          f"| {(1-t_f/t_s)*100:+5.1f}%  -> x30 layers x5 fwd = "
          f"{(t_s-t_f)*30*5:+.2f} ms/frame")

# whole-layer stand-in: attn projections + MLP, 290 vs 296
for M in (290, 296):
    x=jax.random.normal(k,(M,1920),jnp.bfloat16)
    wq=jax.random.normal(k,(1920,1920),jnp.bfloat16)
    wkv=jax.random.normal(k,(1920,384),jnp.bfloat16)
    wo=jax.random.normal(k,(1920,1920),jnp.bfloat16)
    wi=jax.random.normal(k,(1920,15360),jnp.bfloat16)
    wf=jax.random.normal(k,(7680,1920),jnp.bfloat16)
    @jax.jit
    def layer(x,wq,wkv,wo,wi,wf):
        q=x@wq; kv=x@wkv; a=(q[:, :1920])@wo
        pre=a@wi; u,v=jnp.split(pre,2,-1)
        return (u*jax.nn.silu(v))@wf + kv.sum()
    t=best(layer,x,wq,wkv,wo,wi,wf)
    print(f"M={M}: one layer {t*1e3:7.1f} us -> x30 x5 = {t*150:6.2f} ms/frame")
