"""Production model configs, resolved from configs/dynamics.yaml + configs/tokenizer.yaml.

Kept as plain Python so benchmarks can build the real 1.6B dynamics model and the
tokenizer without Hydra, a dataset, or a checkpoint. Values here mirror the YAML
resolvers exactly; see the comments for the expression each field comes from.
"""
import importlib
import os

# Which package holds the model code. "dreamer" = this training fork;
# "pipeline" = the released inference repo (reactor-team/open-dreamer), whose
# models.py is identical apart from the extra policy/task-embedder classes.
# Set OD_MODEL_PKG=pipeline to benchmark the released inference path.
MODEL_PKG = os.environ.get("OD_MODEL_PKG", "dreamer")

_cfg = importlib.import_module(f"{MODEL_PKG}.configs")
DynamicsModelConfig = _cfg.DynamicsModelConfig
TokenizerModelConfig = _cfg.TokenizerModelConfig
EncoderModelConfig = _cfg.EncoderModelConfig
DecoderModelConfig = _cfg.DecoderModelConfig

# configs/dataset/minecraft_vpt.yaml
RAW_H, RAW_W = 360, 640
PAD_H, PAD_W = (4, 4), (0, 0)
H = RAW_H + PAD_H[0] + PAD_H[1]  # 368
W = RAW_W + PAD_W[0] + PAD_W[1]  # 640
PATCH = 16
NUM_BINARY_ACTIONS = 27
CATEGORICAL_ACTION_DIM = 121
CONTINUOUS_ACTION_DIM = 0
DATASET_MEAN = (0.2241, 0.2348, 0.2086)
DATASET_STD = (0.1809, 0.1874, 0.2282)

# configs/dataset/minecraft_vpt_latent.yaml (16 channels)
LATENT_MEAN = (
    -0.01275550201535225, -0.04425295069813728, 0.08248031884431839, 0.042714960873126984,
    0.008957703597843647, -0.0018820130499079823, -0.012893370352685452, -0.08244539052248001,
    0.011176219210028648, -0.09440866857767105, -0.05825792998075485, -0.0497550331056118,
    -0.0025538108311593533, 0.04231492802500725, -0.06914764642715454, 0.07559845596551895,
)
LATENT_STD = (
    0.0848616436123848, 0.09066474437713623, 0.10068009793758392, 0.09407731145620346,
    0.1937398761510849, 0.08929944783449173, 0.0907244011759758, 0.10503409057855606,
    0.1614910215139389, 0.1081596165895462, 0.15006859600543976, 0.12637652456760406,
    0.0923914909362793, 0.10154268890619278, 0.10797301679849625, 0.10930383950471878,
)

# configs/tokenizer.yaml
TOK_SHORT_T = 16  # dataset.dataloader_cfg.short_T -> encoder/decoder context_length
N_LATENTS = 512


def dynamics_config(dtype: str = "bfloat16", param_dtype: str = "float32") -> DynamicsModelConfig:
    """configs/dynamics.yaml -> `dynamics:` block, resolvers expanded."""
    depth = 30
    d_model = 64 * depth                      # ${mul:64,${dynamics.depth}} = 1920
    n_heads = d_model // 64                   # 30
    n_kv_heads = max(1, n_heads // 8)         # 3
    return DynamicsModelConfig(
        d_bottleneck=16,
        depth=depth,
        d_model=d_model,
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        packing_factor=2,
        n_register=32,
        qk_norm_type="qknorm",
        rope_theta=10000.0,
        time_every=4,
        time_layer_offset=0,
        mlp_ratio=4.0,
        dropout_rate=0.0,
        use_residual_lambdas=False,
        use_bias=False,
        use_rmsnorm_scale=True,
        dtype=dtype,
        param_dtype=param_dtype,
        k_max=256,
        context_length=192,                   # ${min:192,${...long_T}} with long_T=256
        num_binary_actions=NUM_BINARY_ACTIONS,
        categorical_action_dim=CATEGORICAL_ACTION_DIM,
        continuous_action_dim=CONTINUOUS_ACTION_DIM,
        latent_mean=LATENT_MEAN,
        latent_std=LATENT_STD,
    )


def tokenizer_config(dtype: str = "bfloat16", param_dtype: str = "float32") -> TokenizerModelConfig:
    """configs/tokenizer.yaml -> `tokenizer:` block, resolvers expanded."""
    enc_depth = 12
    enc_d = 128 * enc_depth                   # 1536
    dec_depth = 8
    dec_d = 128 * dec_depth                   # 1024
    encoder = EncoderModelConfig(
        n_latents=N_LATENTS,
        d_bottleneck=16,
        depth=enc_depth,
        d_model=enc_d,
        n_heads=enc_d // 64,                  # 24
        n_kv_heads=max(1, (enc_d // 64) // 8),  # 3
        patch_size=PATCH,
        dropout_rate=0.0,
        qk_norm_type="qknorm",
        rope_theta=10000.0,
        time_every=4,
        time_layer_offset=3,
        mae_p_min=0.0,
        mae_p_max=0.9,
        use_residual_lambdas=False,
        use_bias=False,
        use_rmsnorm_scale=True,
        dtype=dtype,
        param_dtype=param_dtype,
        context_length=TOK_SHORT_T,
        dataset_mean=DATASET_MEAN,
        dataset_std=DATASET_STD,
    )
    decoder = DecoderModelConfig(
        n_latents=N_LATENTS,
        d_bottleneck=16,
        depth=dec_depth,
        d_model=dec_d,
        n_heads=dec_d // 64,                  # 16
        n_kv_heads=max(1, (dec_d // 64) // 8),  # 2
        patch_size=PATCH,
        d_patch=PATCH * PATCH * 3,            # 768
        dropout_rate=0.0,
        qk_norm_type="qknorm",
        rope_theta=10000.0,
        time_every=4,
        time_layer_offset=3,
        use_residual_lambdas=False,
        use_bias=False,
        use_rmsnorm_scale=True,
        dtype=dtype,
        param_dtype=param_dtype,
        context_length=TOK_SHORT_T,
        H=H,
        W=W,
        dataset_mean=DATASET_MEAN,
        dataset_std=DATASET_STD,
    )
    return TokenizerModelConfig(encoder=encoder, decoder=decoder)
