import math

import pytest

from alphagenome_ft.finetune.train import create_learning_rate_schedule


def test_constant_learning_rate_is_unchanged() -> None:
    rate = create_learning_rate_schedule(
        1e-3,
        schedule="constant",
        total_train_steps=100,
        warmup_steps=0,
        minimum_ratio=0.1,
    )

    assert rate == 1e-3


def test_warmup_cosine_reaches_peak_and_final_rate() -> None:
    schedule = create_learning_rate_schedule(
        1e-3,
        schedule="warmup_cosine",
        total_train_steps=100,
        warmup_steps=10,
        minimum_ratio=0.1,
    )

    assert float(schedule(0)) == 0.0
    assert math.isclose(float(schedule(10)), 1e-3, rel_tol=1e-6)
    assert math.isclose(float(schedule(100)), 1e-4, rel_tol=1e-5)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"schedule": "unknown"}, "Unknown learning-rate schedule"),
        ({"warmup_steps": 100}, "smaller than total_train_steps"),
        ({"minimum_ratio": 1.1}, "between zero and one"),
    ],
)
def test_invalid_learning_rate_schedule_is_rejected(kwargs, message: str) -> None:
    arguments = {
        "schedule": "warmup_cosine",
        "total_train_steps": 100,
        "warmup_steps": 10,
        "minimum_ratio": 0.1,
        **kwargs,
    }
    with pytest.raises(ValueError, match=message):
        create_learning_rate_schedule(1e-3, **arguments)
