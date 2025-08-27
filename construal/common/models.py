import statsmodels.formula.api as smf

def ols_cluster(formula: str, data, cluster):
    return smf.ols(formula, data=data).fit(cov_type="cluster", cov_kwds={"groups": cluster})

def ols_hc3(formula: str, data):
    return smf.ols(formula, data=data).fit(cov_type="HC3")

def mixedlm_random_intercept(formula: str, data, group):
    md = smf.mixedlm(formula, data, groups=group)
    return md.fit(method="lbfgs", maxiter=500, disp=False)

def gee_logit(formula: str, data, group):
    import statsmodels.api as sm
    fam = sm.families.Binomial()
    ind = sm.cov_struct.Exchangeable()
    model = sm.GEE.from_formula(formula, groups=group, data=data, family=fam, cov_struct=ind)
    return model.fit()
