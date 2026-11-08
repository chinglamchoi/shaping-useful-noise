from __future__ import annotations

import json
import sys

import numpy as np
import pytest

import peld
from peld._io import load_scores
from peld.cli import main


class TensorLike:
    def __init__(self, values):
        self.values = np.asarray(values)

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.values


def test_collect_scores_accepts_tuple_batches_and_tensor_like_outputs() -> None:
    batches = [(np.array([[1.0], [2.0]]), [0, 1]), (np.array([[3.0]]), [2])]
    scores = peld.collect_scores(lambda images: TensorLike(images * 2), batches)
    np.testing.assert_allclose(scores, [2.0, 4.0, 6.0])


def test_dataset_eld_scores_both_iterables() -> None:
    scorer = lambda values: TensorLike(values)
    result = peld.dataset_eld(
        scorer,
        [np.arange(10, dtype=float)],
        [np.arange(20, 30, dtype=float)],
    )
    assert result.percentile == pytest.approx(0.5)
    assert result.raw == pytest.approx(20.0)


def test_load_csv_and_npz(tmp_path) -> None:
    csv_path = tmp_path / "scores.csv"
    csv_path.write_text("energy,other\n1.0,9\n2.0,8\n")
    np.testing.assert_allclose(load_scores(csv_path, column="energy"), [1.0, 2.0])

    npz_path = tmp_path / "scores.npz"
    np.savez(npz_path, energy=np.array([3.0, 4.0]))
    np.testing.assert_allclose(load_scores(npz_path), [3.0, 4.0])


def test_cli_json_output(tmp_path, monkeypatch, capsys) -> None:
    reference = tmp_path / "reference.npy"
    candidate = tmp_path / "candidate.npy"
    np.save(reference, np.arange(10, dtype=float))
    np.save(candidate, np.arange(20, 30, dtype=float))
    monkeypatch.setattr(sys, "argv", ["peld", str(reference), str(candidate), "--metric", "both", "--json"])
    main()
    payload = json.loads(capsys.readouterr().out)
    assert payload["percentile_eld"] == pytest.approx(0.5)
    assert payload["raw_eld"] == pytest.approx(20.0)
