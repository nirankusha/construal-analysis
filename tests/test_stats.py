import sys
import pathlib
import numpy as np
from math import isclose
from scipy import stats

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from construal.common.stats import chi2_2x2, holm_correction, proportion_tests
from statsmodels.stats.proportion import proportions_ztest, binom_test as sm_binom_test


def test_chi2_2x2_no_correction():
    res = chi2_2x2(1, 2, 3, 4)
    assert res["ha_correction"] is False
    table = np.array([[1, 2], [3, 4]], dtype=float)
    chi2, p, dof, _ = stats.chi2_contingency(table, correction=False)
    assert isclose(res["chi2"], chi2)
    assert isclose(res["p"], p)
    assert res["dof"] == dof


def test_chi2_2x2_haldane_anscombe():
    res = chi2_2x2(1, 0, 3, 4)
    assert res["ha_correction"] is True
    table = np.array([[1, 0], [3, 4]], dtype=float)
    table += 0.5
    chi2, p, dof, _ = stats.chi2_contingency(table, correction=False)
    assert isclose(res["chi2"], chi2)
    expected_or = (table[0, 0] * table[1, 1]) / (table[0, 1] * table[1, 0])
    assert isclose(res["or"], expected_or)


def test_holm_correction_matches_statsmodels():
    pvals = [0.01, 0.04, 0.03]
    adj = holm_correction(pvals)
    from statsmodels.stats.multitest import multipletests
    _, expected, _, _ = multipletests(pvals, method="holm")
    assert np.allclose(adj, expected)


def test_proportion_tests_one_tailed():
    res = proportion_tests(7, 10, p0=0.5)
    stat, pz = proportions_ztest(7, 10, value=0.5, alternative="larger")
    p_exact = sm_binom_test(7, 10, prop=0.5, alternative="larger")
    assert isclose(res["z"], stat)
    assert isclose(res["p_z"], pz)
    assert isclose(res["p_exact"], p_exact)