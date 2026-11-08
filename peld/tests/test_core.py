from __future__ import annotations

import numpy as np
import pytest

import peld


def test_identical_unique_reference_has_zero_peld() -> None:
    reference = np.array([4.0, 1.0, 3.0, 2.0])
    assert peld.percentile_eld(reference, reference) == 0.0


def test_candidate_above_reference_has_maximal_peld() -> None:
    reference = np.arange(10, dtype=float)
    candidate = np.arange(20, 30, dtype=float)
    assert peld.percentile_eld(reference, candidate) == pytest.approx(0.5)


def test_raw_eld_matches_simple_shift() -> None:
    reference = np.array([0.0, 2.0])
    candidate = np.array([1.0, 3.0])
    assert peld.raw_eld(reference, candidate) == pytest.approx(1.0)


def test_peld_is_invariant_to_strictly_increasing_calibration() -> None:
    reference = np.array([-2.0, -1.0, 0.0, 0.5, 2.0])
    candidate = np.array([-1.5, -0.2, 0.1, 1.0])
    original = peld.peld(reference, candidate)
    transformed = peld.peld(np.exp(reference), np.exp(candidate))
    assert transformed == pytest.approx(original)


def test_logp_and_energy_conventions_agree() -> None:
    reference_logp = np.array([-1.0, -2.0, -4.0, -8.0])
    candidate_logp = np.array([-1.5, -3.0, -6.0])
    assert peld.peld(reference_logp, candidate_logp, score_kind="logp") == pytest.approx(
        peld.peld(-reference_logp, -candidate_logp, score_kind="energy")
    )
    assert peld.raw_eld(reference_logp, candidate_logp, score_kind="logp") == pytest.approx(
        peld.raw_eld(-reference_logp, -candidate_logp, score_kind="energy")
    )


def test_midrank_tie_handling() -> None:
    ranks = peld.percentile_ranks([0.0, 0.0, 1.0, 1.0], [0.0, 1.0])
    np.testing.assert_allclose(ranks, [0.25, 0.75])


def test_reusable_metric_matches_functional_api() -> None:
    reference = np.linspace(-3.0, 3.0, 101)
    candidate = np.linspace(-2.0, 4.0, 73)
    metric = peld.PELD(reference)
    assert metric(candidate) == pytest.approx(peld.peld(reference, candidate))
    result = metric.both(candidate)
    assert result.percentile == pytest.approx(metric(candidate))
    assert result.raw == pytest.approx(peld.raw_eld(reference, candidate))
    assert result.n_reference == 101
    assert result.n_candidate == 73


def test_metric_round_trip(tmp_path) -> None:
    metric = peld.PELD([-1.0, -2.0, -4.0], score_kind="logp")
    path = metric.save(tmp_path / "reference.cache")
    loaded = peld.PELD.load(path)
    assert loaded.score_kind == "logp"
    assert loaded([-1.5, -3.0]) == pytest.approx(metric([-1.5, -3.0]))


def test_score_many() -> None:
    metric = peld.PELD(np.arange(20))
    output = metric.score_many({"same": np.arange(20), "high": np.arange(30, 40)})
    assert output["same"] == 0.0
    assert output["high"] == pytest.approx(0.5)


@pytest.mark.parametrize("values", [[], [0.0, np.nan], [[1.0, 2.0], [3.0, 4.0]]])
def test_invalid_scores_fail(values) -> None:
    with pytest.raises(ValueError):
        peld.PELD(values)


def test_invalid_percentiles_fail() -> None:
    with pytest.raises(ValueError):
        peld.uniform_wasserstein([-0.1, 0.5])
