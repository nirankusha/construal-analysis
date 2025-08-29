import numpy as np
from scipy import stats
from statsmodels.stats.proportion import proportions_ztest, binom_test as sm_binom_test
from statsmodels.stats.contingency_tables import StratifiedTable
from statsmodels.stats.multitest import multipletests
from math import sqrt

def holm_correction(pvals):
    import numpy as np
    p = np.asarray(list(pvals), dtype=float)
    _, p_adj, _, _ = multipletests(p, method="holm")
    return p_adj

def proportion_tests(k: int, n: int, p0: float=0.5) -> dict:
    stat, pz = proportions_ztest(k, n, value=p0, aletrnative="larger")
    try:
        p_exact = sm_binom_test(k, n, prop=p0, alternative="larger")
    except TypeError:
        from scipy.stats import binomtest
        p_exact = binomtest(k, n, p=p0, alternative="larger").pvalue
    return {"n": n, "k": k, "prop": k/n if n else np.nan, "z": float(stat), "p_z": float(pz), "p_exact": float(p_exact)}

def chi2_2x2(a11, a12, a21, a22) -> dict:
    """Chi-square test for a 2x2 contingency table.

    Applies the Haldane–Anscombe correction (adds 0.5 to all cells) if any
    cell in the table is zero. The chi-square test, odds ratio, and confidence
    interval are computed from the (possibly corrected) table.

    Returns a dictionary with the chi-square statistic, p-value, degrees of
    freedom, odds ratio, confidence interval bounds, effect size (V), the
    table total, and whether the correction was applied.
    """

    table = np.array([[a11, a12], [a21, a22]], dtype=float)
    ha_correction = False
    if (table <= 0).any():
        table += 0.5
        ha_correction = True

    n = table.sum()
    
    try:
        chi2, p, dof, _ = stats.chi2_contingency(table, correction=False)
        or_est = (table[0, 0] * table[1, 1]) / (table[0, 1] * table[1, 0])
        se = sqrt(
            1 / table[0, 0]
            + 1 / table[0, 1]
            + 1 / table[1, 0]
            + 1 / table[1, 1]
        )
        ci_low = np.exp(np.log(or_est) - 1.96 * se)
        ci_hi = np.exp(np.log(or_est) + 1.96 * se)
        V = sqrt(chi2 / n) if n > 0 else np.nan
    except ValueError:
        chi2 = p = or_est = ci_low = ci_hi = V = np.nan
        dof = 1

    return {
        "chi2": float(chi2),
        "p": float(p),
        "dof": int(dof),
        "or": float(or_est),
        "ci_low": float(ci_low),
        "ci_hi": float(ci_hi),
        "V": float(V),
        "n": int(n),
        "ha_correction": ha_correction,
    }
def chi2_2x3(counts_2x3) -> dict:
    counts_2x3 = np.asarray(counts_2x3, dtype=float)
    chi2, p, dof, _ = stats.chi2_contingency(counts_2x3, correction=False)
    n = counts_2x3.sum()
    k = 2
    V = np.sqrt(chi2/(n*(k-1))) if n>0 else np.nan
    return {"chi2":float(chi2), "p":float(p), "dof":int(dof), "V":float(V), "n":int(n)}

def cmh_from_2x2_list(tables: list) -> dict:
    st = StratifiedTable(tables)
    cmh_p = float(st.test_null_odds().pvalue)
    bd_p = float(st.test_equal_odds().pvalue)
    return {"p_cmh": cmh_p, "p_breslowday": bd_p, "k": len(tables)}
