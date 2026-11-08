import numpy as np

from density_audit.eld import empirical_percentile_ranks, logp_to_energy, uniform_w1_from_ranks


def test_log_probability_converts_to_energy() -> None:
    values = np.array([-3.0, 0.5, 2.0])
    np.testing.assert_array_equal(logp_to_energy(values), np.array([3.0, -0.5, -2.0]))


def test_reference_sample_has_zero_percentile_eld() -> None:
    reference = np.array([4.0, 1.0, 3.0, 2.0])
    ranks = empirical_percentile_ranks(reference, reference)
    assert uniform_w1_from_ranks(ranks) == 0.0


def test_shifted_sample_has_nonzero_percentile_eld() -> None:
    reference = np.linspace(-1.0, 1.0, 101)
    shifted = reference + 10.0
    assert uniform_w1_from_ranks(empirical_percentile_ranks(reference, shifted)) > 0.45
