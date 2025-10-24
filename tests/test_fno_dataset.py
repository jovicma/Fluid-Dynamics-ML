"""Tests for FNO dataset."""

from __future__ import annotations

from pathlib import Path

from riemann_ml.ml.fno.dataset import RiemannH5Dataset


def test_fno_dataset_shapes(tmp_path: Path):
    # Use existing dataset if available; otherwise skip
    dataset_path = Path("data/processed/sod_like.h5")
    if not dataset_path.exists():
        import pytest

        pytest.skip("dataset file not generated")

    dataset = RiemannH5Dataset(dataset_path)
    x, y = dataset[0]
    assert x.ndim == 2
    assert y.ndim == 2
    assert x.shape[1] == y.shape[1]
    assert x.shape[0] in (3, 4)
    assert y.shape[0] == 3
