import haiku as hk
import jax
import jax.numpy as jnp
import pytest

from alphagenome_ft.fp8_lora import BackboneLoRAConfig, patch_haiku_linear
from alphagenome_ft.lora import get_lora_parameter_paths
from alphagenome_ft.custom_model import _cast_runtime_backbone_params


def test_patch_haiku_linear_adds_lora_only_to_named_targets():
    config = BackboneLoRAConfig(rank=2, alpha=2.0, target_names=("q_layer",))

    def forward(x):
        with patch_haiku_linear(config):
            adapted = hk.Linear(4, with_bias=False, name="q_layer")(x)
            plain = hk.Linear(4, with_bias=False, name="plain_layer")(x)
        return adapted + plain

    transformed = hk.transform(forward)
    params = transformed.init(jax.random.PRNGKey(0), jnp.ones((1, 3)))

    assert set(params["q_layer"]) == {"w", "lora_a", "lora_b"}
    assert set(params["plain_layer"]) == {"w"}
    assert sorted(get_lora_parameter_paths(params)) == [
        "q_layer/lora_a",
        "q_layer/lora_b",
    ]


def test_lora_precision_config_controls_parameter_storage_dtype():
    config = BackboneLoRAConfig(
        rank=2,
        alpha=2.0,
        target_names=("q_layer",),
        base_param_dtype="float32",
        lora_param_dtype="float16",
        activation_dtype="float16",
        base_compute_dtype="float16",
        lora_compute_dtype="float16",
    )

    def forward(x):
        with patch_haiku_linear(config):
            return hk.Linear(4, with_bias=False, name="q_layer")(x)

    transformed = hk.transform(forward)
    params = transformed.init(jax.random.PRNGKey(0), jnp.ones((1, 3), dtype=jnp.bfloat16))
    y = transformed.apply(params, jax.random.PRNGKey(1), jnp.ones((1, 3), dtype=jnp.bfloat16))

    assert params["q_layer"]["w"].dtype == jnp.float32
    assert params["q_layer"]["lora_a"].dtype == jnp.float16
    assert params["q_layer"]["lora_b"].dtype == jnp.float16
    assert y.dtype == jnp.float16


def test_base_param_dtype_fp8_stores_patched_base_weight_as_float8():
    fp8_dtype = getattr(jnp, "float8_e4m3fn", None)
    if fp8_dtype is None:
        pytest.skip("Installed JAX does not expose jnp.float8_e4m3fn")

    config = BackboneLoRAConfig(
        rank=2,
        alpha=2.0,
        target_names=("q_layer",),
        base_param_dtype="fp8",
        lora_param_dtype="float16",
        activation_dtype="bfloat16",
        base_compute_dtype="bfloat16",
        lora_compute_dtype="bfloat16",
    )

    def forward(x):
        with patch_haiku_linear(config):
            return hk.Linear(4, with_bias=True, name="q_layer")(x)

    transformed = hk.transform(forward)
    params = transformed.init(jax.random.PRNGKey(0), jnp.ones((1, 3), dtype=jnp.bfloat16))
    y = transformed.apply(params, jax.random.PRNGKey(1), jnp.ones((1, 3), dtype=jnp.bfloat16))

    assert params["q_layer"]["w"].dtype == fp8_dtype
    assert params["q_layer"]["b"].dtype == jnp.bfloat16
    assert params["q_layer"]["lora_a"].dtype == jnp.float16
    assert params["q_layer"]["lora_b"].dtype == jnp.float16
    assert y.dtype == jnp.bfloat16


def test_base_param_dtype_fp4_stores_patched_base_weight_as_float4():
    fp4_dtype = getattr(jnp, "float4_e2m1fn", None)
    if fp4_dtype is None:
        pytest.skip("Installed JAX does not expose jnp.float4_e2m1fn")

    config = BackboneLoRAConfig(
        rank=2,
        alpha=2.0,
        target_names=("q_layer",),
        base_param_dtype="fp4",
        lora_param_dtype="float16",
        activation_dtype="bfloat16",
        base_compute_dtype="bfloat16",
        lora_compute_dtype="bfloat16",
    )

    def forward(x):
        with patch_haiku_linear(config):
            return hk.Linear(4, with_bias=True, name="q_layer")(x)

    transformed = hk.transform(forward)
    params = transformed.init(jax.random.PRNGKey(0), jnp.ones((1, 3), dtype=jnp.bfloat16))
    y = transformed.apply(params, jax.random.PRNGKey(1), jnp.ones((1, 3), dtype=jnp.bfloat16))

    assert params["q_layer"]["w"].dtype == fp4_dtype
    assert params["q_layer"]["b"].dtype == jnp.bfloat16
    assert params["q_layer"]["lora_a"].dtype == jnp.float16
    assert params["q_layer"]["lora_b"].dtype == jnp.float16
    assert y.dtype == jnp.bfloat16


def test_runtime_backbone_param_cast_preserves_trainable_leaves():
    params = {
        "alphagenome": {
            "sequence_encoder": {
                "conv": {"w": jnp.ones((2, 2), dtype=jnp.float32)}
            },
            "head": {
                "custom": {"w": jnp.ones((2, 1), dtype=jnp.float32)},
                "unused_standard": {"w": jnp.ones((2, 1), dtype=jnp.float32)},
            },
            "transformer_tower": {
                "q_layer": {
                    "w": jnp.ones((2, 2), dtype=jnp.float32),
                    "lora_a": jnp.ones((2, 1), dtype=jnp.float32),
                    "lora_b": jnp.ones((1, 2), dtype=jnp.float32),
                }
            },
        },
        "head": {
            "custom": {"w": jnp.ones((2, 1), dtype=jnp.float32)}
        },
    }

    cast = _cast_runtime_backbone_params(
        params,
        "bf16",
        trainable_head_names=("custom",),
    )

    assert cast["alphagenome"]["sequence_encoder"]["conv"]["w"].dtype == jnp.bfloat16
    assert cast["alphagenome"]["head"]["custom"]["w"].dtype == jnp.float32
    assert cast["alphagenome"]["head"]["unused_standard"]["w"].dtype == jnp.bfloat16
    assert cast["alphagenome"]["transformer_tower"]["q_layer"]["w"].dtype == jnp.bfloat16
    assert cast["alphagenome"]["transformer_tower"]["q_layer"]["lora_a"].dtype == jnp.float32
    assert cast["alphagenome"]["transformer_tower"]["q_layer"]["lora_b"].dtype == jnp.float32
    assert cast["head"]["custom"]["w"].dtype == jnp.float32


def test_runtime_fp8_cast_only_quantizes_lora_target_base_weights():
    fp8_dtype = getattr(jnp, "float8_e4m3fn", None)
    if fp8_dtype is None:
        pytest.skip("Installed JAX does not expose jnp.float8_e4m3fn")

    params = {
        "alphagenome": {
            "sequence_encoder": {
                "conv": {"w": jnp.ones((2, 2), dtype=jnp.float32)}
            },
            "transformer_tower": {
                "q_layer": {
                    "w": jnp.ones((2, 2), dtype=jnp.float32),
                    "b": jnp.ones((2,), dtype=jnp.float32),
                    "lora_a": jnp.ones((2, 1), dtype=jnp.float32),
                    "lora_b": jnp.ones((1, 2), dtype=jnp.float32),
                },
                "plain_layer": {"w": jnp.ones((2, 2), dtype=jnp.float32)},
            },
        },
        "head": {
            "custom": {"w": jnp.ones((2, 1), dtype=jnp.float32)}
        },
    }

    cast = _cast_runtime_backbone_params(
        params,
        "fp8",
        trainable_head_names=("custom",),
        fp8_base_target_names=("q_layer",),
    )

    assert cast["alphagenome"]["sequence_encoder"]["conv"]["w"].dtype == jnp.bfloat16
    assert cast["alphagenome"]["transformer_tower"]["q_layer"]["w"].dtype == fp8_dtype
    assert cast["alphagenome"]["transformer_tower"]["q_layer"]["b"].dtype == jnp.bfloat16
    assert cast["alphagenome"]["transformer_tower"]["q_layer"]["lora_a"].dtype == jnp.float32
    assert cast["alphagenome"]["transformer_tower"]["q_layer"]["lora_b"].dtype == jnp.float32
    assert cast["alphagenome"]["transformer_tower"]["plain_layer"]["w"].dtype == jnp.bfloat16
    assert cast["head"]["custom"]["w"].dtype == jnp.float32


def test_runtime_cast_can_match_trainable_heads_to_lora_param_dtype():
    params = {
        "alphagenome": {
            "head": {
                "custom": {"w": jnp.ones((2, 1), dtype=jnp.float32)},
            },
            "transformer_tower": {
                "q_layer": {
                    "w": jnp.ones((2, 2), dtype=jnp.float32),
                    "lora_a": jnp.ones((2, 1), dtype=jnp.float32),
                    "lora_b": jnp.ones((1, 2), dtype=jnp.float32),
                }
            },
        },
        "head": {
            "custom": {"w": jnp.ones((2, 1), dtype=jnp.float32)}
        },
    }

    cast = _cast_runtime_backbone_params(
        params,
        "bf16",
        trainable_head_names=("custom",),
        trainable_param_dtype="fp16",
    )

    assert cast["alphagenome"]["head"]["custom"]["w"].dtype == jnp.float16
    assert cast["alphagenome"]["transformer_tower"]["q_layer"]["lora_a"].dtype == jnp.float16
    assert cast["alphagenome"]["transformer_tower"]["q_layer"]["lora_b"].dtype == jnp.float16
    assert cast["head"]["custom"]["w"].dtype == jnp.float16
    assert cast["alphagenome"]["transformer_tower"]["q_layer"]["w"].dtype == jnp.bfloat16


def test_fp8_enabled_requires_transformer_engine():
    config = BackboneLoRAConfig(rank=16, alpha=16.0, fp8_enabled=True, target_names=("q_layer",))

    def forward(x):
        with patch_haiku_linear(config):
            return hk.Linear(16, with_bias=False, name="q_layer")(x)

    transformed = hk.transform(forward)
    try:
        transformed.init(jax.random.PRNGKey(0), jnp.ones((1, 16)))
    except ImportError as exc:
        assert "transformer_engine[jax]" in str(exc)
    except RuntimeError as exc:
        if "CUDA" in str(exc) or "device" in str(exc):
            pytest.skip(f"Transformer Engine FP8 is installed but unavailable here: {exc}")
        raise
    else:
        pytest.skip("transformer_engine.jax is installed in this environment")


def test_fp8_rank_must_be_multiple_of_16():
    config = BackboneLoRAConfig(rank=8, alpha=8.0, fp8_enabled=True)
    with pytest.raises(ValueError, match="multiple of 16"):
        config.validate()


def test_fp4_enabled_resolves_to_nvfp4_compute():
    config = BackboneLoRAConfig(rank=32, alpha=32.0, fp4_enabled=True)

    assert config.resolved_lora_compute_dtype() == "fp4"
    assert config.uses_fp4()
    assert config.uses_transformer_engine()


def test_fp8_and_fp4_flags_are_mutually_exclusive():
    config = BackboneLoRAConfig(rank=16, fp8_enabled=True, fp4_enabled=True)
    with pytest.raises(ValueError, match="mutually exclusive"):
        config.validate()


def test_fp4_rank_must_be_multiple_of_32():
    config = BackboneLoRAConfig(rank=16, alpha=16.0, fp4_enabled=True)
    with pytest.raises(ValueError, match="multiple of 32"):
        config.validate()


def test_fp4_storage_is_supported_only_for_base_parameters():
    config = BackboneLoRAConfig(lora_param_dtype="fp4")
    with pytest.raises(ValueError, match="only base projection weights"):
        config.validate()


def test_runtime_fp4_cast_only_quantizes_lora_target_base_weights():
    fp4_dtype = getattr(jnp, "float4_e2m1fn", None)
    if fp4_dtype is None:
        pytest.skip("Installed JAX does not expose jnp.float4_e2m1fn")

    params = {
        "alphagenome": {
            "sequence_encoder": {"conv": {"w": jnp.ones((2, 2), dtype=jnp.float32)}},
            "transformer_tower": {
                "q_layer": {
                    "w": jnp.ones((2, 2), dtype=jnp.float32),
                    "b": jnp.ones((2,), dtype=jnp.float32),
                    "lora_a": jnp.ones((2, 1), dtype=jnp.float32),
                    "lora_b": jnp.ones((1, 2), dtype=jnp.float32),
                },
                "plain_layer": {"w": jnp.ones((2, 2), dtype=jnp.float32)},
            },
        },
        "head": {"custom": {"w": jnp.ones((2, 1), dtype=jnp.float32)}},
    }

    cast = _cast_runtime_backbone_params(
        params,
        "fp4",
        trainable_head_names=("custom",),
        fp8_base_target_names=("q_layer",),
    )

    assert cast["alphagenome"]["sequence_encoder"]["conv"]["w"].dtype == jnp.bfloat16
    assert cast["alphagenome"]["transformer_tower"]["q_layer"]["w"].dtype == fp4_dtype
    assert cast["alphagenome"]["transformer_tower"]["q_layer"]["b"].dtype == jnp.bfloat16
    assert cast["alphagenome"]["transformer_tower"]["q_layer"]["lora_a"].dtype == jnp.float32
    assert cast["alphagenome"]["transformer_tower"]["q_layer"]["lora_b"].dtype == jnp.float32
    assert cast["alphagenome"]["transformer_tower"]["plain_layer"]["w"].dtype == jnp.bfloat16
    assert cast["head"]["custom"]["w"].dtype == jnp.float32


def test_fp4_enabled_requires_transformer_engine():
    config = BackboneLoRAConfig(rank=32, alpha=32.0, fp4_enabled=True, target_names=("q_layer",))

    def forward(x):
        with patch_haiku_linear(config):
            return hk.Linear(16, with_bias=False, name="q_layer")(x)

    transformed = hk.transform(forward)
    try:
        transformed.init(jax.random.PRNGKey(0), jnp.ones((1, 16), dtype=jnp.bfloat16))
    except ImportError as exc:
        assert "FP4 LoRA requires `transformer_engine[jax]`" in str(exc)
    except (AssertionError, RuntimeError) as exc:
        if "NVFP4" in str(exc) or "CUDA" in str(exc) or "device" in str(exc):
            pytest.skip(f"Transformer Engine NVFP4 is installed but unavailable here: {exc}")
        raise
    else:
        pytest.skip("transformer_engine.jax NVFP4 is installed in this environment")


def test_fp8_activation_requires_fp8_compute_path():
    config = BackboneLoRAConfig(
        activation_dtype="fp8",
        base_compute_dtype="bfloat16",
        lora_compute_dtype="bfloat16",
    )
    with pytest.raises(ValueError, match="activation_dtype='fp8'"):
        config.validate()
