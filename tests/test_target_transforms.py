import json

import numpy as np

from alphagenome_ft.finetune.target_transforms import (
    PiecewiseLinearTargetTransform,
    SpatialRebinTargetTransform,
    load_target_transform,
)


def test_piecewise_transform_round_trip(tmp_path):
    path = tmp_path / "transform.json"
    path.write_text(
        json.dumps(
            {
                "kind": "piecewise_linear",
                "source_knots": [[0.0, 1.0, 3.0], [0.0, 2.0, 4.0]],
                "transformed_knots": [[0.0, 2.0, 8.0], [0.0, 1.0, 3.0]],
            }
        )
    )
    transform = PiecewiseLinearTargetTransform.from_path(path)
    values = np.asarray([[[0.5, 1.0], [2.0, 3.0]]], dtype=np.float32)

    transformed = transform.forward_numpy(values)
    np.testing.assert_allclose(np.asarray(transform.forward_jax(values)), transformed)
    recovered = np.asarray(transform.inverse_jax(transformed))

    np.testing.assert_allclose(recovered, values, rtol=1e-6, atol=1e-6)


def test_piecewise_transform_rejects_nonmonotone_knots(tmp_path):
    path = tmp_path / "transform.json"
    path.write_text(
        json.dumps(
            {
                "kind": "piecewise_linear",
                "source_knots": [[0.0, 1.0, 1.0]],
                "transformed_knots": [[0.0, 1.0, 2.0]],
            }
        )
    )

    with np.testing.assert_raises_regex(ValueError, "strictly increasing"):
        PiecewiseLinearTargetTransform.from_path(path)


def test_spatial_rebin_matches_numpy_and_restores_units(tmp_path):
    path = tmp_path / "transform.json"
    path.write_text(json.dumps({"kind": "spatial_rebin", "width": 4, "output_scale": 2.0}))
    transform = load_target_transform(path)
    assert isinstance(transform, SpatialRebinTargetTransform)
    values = np.arange(10, dtype=np.float32).reshape(1, 10, 1)

    expected = np.asarray([1.5] * 4 + [5.5] * 4 + [8.5] * 2, dtype=np.float32)
    transformed = np.asarray(transform.forward_jax(values))[0, :, 0]

    np.testing.assert_allclose(transformed, expected * 2.0)
    np.testing.assert_allclose(transform.forward_numpy(values)[0, :, 0], transformed)
    np.testing.assert_allclose(np.asarray(transform.inverse_jax(transformed)), expected)
