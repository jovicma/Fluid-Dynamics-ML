"""Tests for dataset generation utilities."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

from riemann_ml.data.generate import sample_riemann_ic, solve_and_store


def test_generate_dataset_shapes_and_finiteness(tmp_path: Path) -> None:
    ranges = {
        "rho_left": (0.9, 1.1),
        "p_left": (0.9, 1.1),
        "rho_right": (0.2, 0.4),
        "p_right": (0.05, 0.3),
    }
    samples = sample_riemann_ic(3, ranges, seed=123)
    out_file = tmp_path / "dataset.h5"

    solve_and_store(
        ic_batch=samples,
        num_cells=64,
        cfl=0.45,
        final_time=0.05,
        out_path=out_file,
        gamma=1.4,
    )

    assert out_file.exists()

    with h5py.File(out_file, "r") as h5f:
        assert h5f.attrs["num_samples"] == len(samples)
        assert h5f.attrs["num_cells"] == 64
        x = h5f["x"][:]
        rho = h5f["rho"][:]
        vel = h5f["velocity"][:]
        pres = h5f["pressure"][:]

        assert x.shape == (64,)
        assert rho.shape == (len(samples), 64)
        assert vel.shape == rho.shape
        assert pres.shape == rho.shape

        assert np.isfinite(rho).all()
        assert np.isfinite(vel).all()
        assert np.isfinite(pres).all()

        ic_group = h5f["initial_conditions"]
        for key in ("rho_left", "p_left", "rho_right", "p_right"):
            data = ic_group[key][:]
            assert data.shape == (len(samples),)
            assert np.isfinite(data).all()
