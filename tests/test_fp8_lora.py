from contextlib import nullcontext

import haiku as hk
import jax
import jax.numpy as jnp
import pytest

from alphagenome_ft.fp8_lora import (
    BackboneLoConConfig,
    BackboneLoRAConfig,
    expand_adapter_parameter_tree,
    patch_haiku_linear,
    patch_haiku_locon,
)
from alphagenome_ft.lora import get_lora_parameter_paths
from alphagenome_ft.custom_model import (
    _backbone_attention_dot_compatibility,
    _backbone_mixed_precision_policy,
    _cast_runtime_backbone_params,
    _resolve_backbone_compute_dtype,
)
from alphagenome_ft.custom_heads import _FactorizedMultiOrganismLinear


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        (None, jnp.bfloat16),
        ("bf16", jnp.bfloat16),
        ("float16", jnp.float16),
        ("fp32", jnp.float32),
    ],
)
def test_resolve_backbone_compute_dtype(name, expected):
    _, dtype = _resolve_backbone_compute_dtype(name)
    assert dtype == expected


def test_backbone_mixed_precision_policy_uses_requested_compute_and_output_dtype():
    policy, dtype = _backbone_mixed_precision_policy("float32")

    assert dtype == jnp.float32
    assert policy.compute_dtype == jnp.float32
    assert policy.output_dtype == jnp.float32


def test_backbone_compute_dtype_rejects_quantized_storage_names():
    with pytest.raises(ValueError, match="runtime_backbone_compute_dtype"):
        _resolve_backbone_compute_dtype("fp8")


def test_non_bfloat16_backbone_replaces_explicit_attention_bfloat16_dot_preset():
    from alphagenome_research.model import attention

    lhs = jnp.ones((2, 3), dtype=jnp.float32)
    rhs = jnp.ones((3, 4), dtype=jnp.float32)
    original_jnp = attention.jnp

    with _backbone_attention_dot_compatibility("float32"):
        result = attention.jnp.einsum(
            "ij,jk->ik",
            lhs,
            rhs,
            precision=jax.lax.DotAlgorithmPreset.BF16_BF16_F32,
        )

    assert result.dtype == jnp.float32
    assert attention.jnp is original_jnp


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


def test_patch_haiku_locon_adds_locon_only_to_matching_conv_paths():
    from alphagenome_research.model import convolutions

    config = BackboneLoConConfig(
        rank=2,
        alpha=1.0,
        target_names=("target_block",),
    )

    def forward(x):
        with patch_haiku_locon(config):
            adapted = convolutions.ConvBlock(
                num_channels=4,
                width=3,
                name="target_block",
            )(x, is_training=True)
            plain = convolutions.ConvBlock(
                num_channels=4,
                width=3,
                name="plain_block",
            )(x, is_training=True)
        return adapted + plain

    transformed = hk.transform_with_state(forward)
    params, state = transformed.init(
        jax.random.PRNGKey(0),
        jnp.ones((1, 8, 3), dtype=jnp.bfloat16),
    )
    del state

    adapted_key = "target_block/standardized_conv1_d"
    plain_key = "plain_block/standardized_conv1_d"
    assert {"w", "scale", "bias", "locon_down_w", "locon_up_w"} <= set(
        params[adapted_key]
    )
    assert set(params[plain_key]) == {"w", "scale", "bias"}
    assert sorted(get_lora_parameter_paths(params)) == [
        f"{adapted_key}/locon_down_w",
        f"{adapted_key}/locon_up_w",
    ]


def test_expand_adapter_parameter_tree_preserves_existing_residuals():
    source = {
        "linear": {
            "w": jnp.ones((3, 2)),
            "lora_a": jnp.arange(6, dtype=jnp.float32).reshape(3, 2),
            "lora_b": jnp.arange(8, dtype=jnp.float32).reshape(2, 4),
        },
        "conv": {
            "w": jnp.ones((3, 2, 4)),
            "locon_down_w": jnp.arange(12, dtype=jnp.float32).reshape(3, 2, 2),
            "locon_up_w": jnp.arange(8, dtype=jnp.float32).reshape(1, 2, 4),
        },
    }
    target = {
        "linear": {
            "w": jnp.zeros((3, 2)),
            "lora_a": jnp.full((3, 4), 7.0),
            "lora_b": jnp.zeros((4, 4)),
        },
        "conv": {
            "w": jnp.zeros((3, 2, 4)),
            "locon_down_w": jnp.full((3, 2, 4), 5.0),
            "locon_up_w": jnp.zeros((1, 4, 4)),
        },
        "new_conv": {
            "locon_down_w": jnp.ones((3, 4, 4)),
            "locon_up_w": jnp.zeros((1, 4, 4)),
        },
    }
    source_lora = BackboneLoRAConfig(rank=2, alpha=2.0)
    target_lora = BackboneLoRAConfig(rank=4, alpha=2.0)
    source_locon = BackboneLoConConfig(rank=2, alpha=1.0)
    target_locon = BackboneLoConConfig(rank=4, alpha=1.0)

    expanded, stats = expand_adapter_parameter_tree(
        source,
        target,
        source_lora_config=source_lora,
        target_lora_config=target_lora,
        source_locon_config=source_locon,
        target_locon_config=target_locon,
    )

    source_linear = source["linear"]["lora_a"] @ source["linear"]["lora_b"]
    target_linear = expanded["linear"]["lora_a"] @ expanded["linear"]["lora_b"]
    assert jnp.allclose(
        source_linear * (source_lora.alpha / source_lora.rank),
        target_linear * (target_lora.alpha / target_lora.rank),
    )
    source_conv = jnp.einsum(
        "wir,uro->wio",
        source["conv"]["locon_down_w"],
        source["conv"]["locon_up_w"],
    ) * (source_locon.alpha / source_locon.rank)
    target_conv = jnp.einsum(
        "wir,uro->wio",
        expanded["conv"]["locon_down_w"],
        expanded["conv"]["locon_up_w"],
    ) * (target_locon.alpha / target_locon.rank)
    assert jnp.allclose(source_conv, target_conv)
    assert jnp.all(expanded["new_conv"]["locon_up_w"] == 0)
    assert stats == {
        "copied_leaves": 2,
        "expanded_leaves": 4,
        "initialized_adapter_leaves": 2,
    }


def test_expand_adapter_parameter_tree_rejects_rank_reduction():
    source = {"linear": {"lora_a": jnp.ones((3, 4))}}
    target = {"linear": {"lora_a": jnp.ones((3, 2))}}

    with pytest.raises(ValueError, match="Cannot expand"):
        expand_adapter_parameter_tree(
            source,
            target,
            source_lora_config=BackboneLoRAConfig(rank=4),
            target_lora_config=BackboneLoRAConfig(rank=2),
            source_locon_config=None,
            target_locon_config=None,
        )


def test_expand_adapter_parameter_tree_adds_zero_residuals_to_adapter_free_source():
    source = {
        "linear": {"w": jnp.arange(6, dtype=jnp.float32).reshape(3, 2)},
        "conv": {"w": jnp.arange(24, dtype=jnp.float32).reshape(3, 2, 4)},
    }
    target = {
        "linear": {
            "w": jnp.zeros((3, 2)),
            "lora_a": jnp.ones((3, 4)),
            "lora_b": jnp.zeros((4, 2)),
        },
        "conv": {
            "w": jnp.zeros((3, 2, 4)),
            "locon_down_w": jnp.ones((3, 2, 2)),
            "locon_up_w": jnp.zeros((1, 2, 4)),
        },
    }

    expanded, stats = expand_adapter_parameter_tree(
        source,
        target,
        source_lora_config=None,
        target_lora_config=BackboneLoRAConfig(rank=4, alpha=4.0),
        source_locon_config=None,
        target_locon_config=BackboneLoConConfig(rank=2, alpha=1.0),
    )

    assert jnp.array_equal(expanded["linear"]["w"], source["linear"]["w"])
    assert jnp.array_equal(expanded["conv"]["w"], source["conv"]["w"])
    assert jnp.all(expanded["linear"]["lora_b"] == 0)
    assert jnp.all(expanded["conv"]["locon_up_w"] == 0)
    assert stats == {
        "copied_leaves": 2,
        "expanded_leaves": 0,
        "initialized_adapter_leaves": 4,
    }


def test_locon_does_not_change_shared_lora_or_reserved_head_initialization():
    from alphagenome_research.model import convolutions

    lora_config = BackboneLoRAConfig(rank=2, alpha=2.0, target_names=("q_layer",))
    locon_config = BackboneLoConConfig(
        rank=2,
        alpha=1.0,
        target_names=("target_block",),
    )

    def initialize_and_apply(include_locon):
        def forward(x):
            head_rng_context = (
                hk.with_rng(hk.next_rng_key()) if hk.running_init() else nullcontext()
            )
            with patch_haiku_linear(lora_config):
                locon_context = (
                    patch_haiku_locon(locon_config) if include_locon else nullcontext()
                )
                with locon_context:
                    trunk = convolutions.ConvBlock(
                        num_channels=4,
                        width=3,
                        name="target_block",
                    )(x, is_training=True)
                    trunk = hk.Linear(3, with_bias=False, name="q_layer")(trunk)
            with head_rng_context, hk.name_scope("head"):
                full_output = hk.Linear(2, with_bias=False, name="output")(trunk)
                factorized_output = _FactorizedMultiOrganismLinear(
                    output_size=2,
                    num_organisms=1,
                    rank=1,
                )(trunk, jnp.zeros((trunk.shape[0],), dtype=jnp.int32))
                return full_output + factorized_output

        transformed = hk.transform_with_state(forward)
        x = jnp.ones((1, 8, 3), dtype=jnp.bfloat16)
        params, state = transformed.init(
            jax.random.PRNGKey(42),
            x,
        )
        output, _ = transformed.apply(params, state, None, x)
        return params, output

    lora_params, lora_output = initialize_and_apply(False)
    combo_params, combo_output = initialize_and_apply(True)
    assert jnp.array_equal(
        lora_params["q_layer"]["lora_a"], combo_params["q_layer"]["lora_a"]
    )
    assert jnp.array_equal(
        lora_params["q_layer"]["lora_b"], combo_params["q_layer"]["lora_b"]
    )
    assert jnp.array_equal(
        lora_params["head/output"]["w"], combo_params["head/output"]["w"]
    )
    for name in ("factor_in_w", "factor_out_w", "b"):
        assert jnp.array_equal(
            lora_params["head/factorized_multi_organism_linear"][name],
            combo_params["head/factorized_multi_organism_linear"][name],
        )
    assert jnp.array_equal(lora_output, combo_output)


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
