import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm
from ..common.config import Config
from ..common.io import write_table, write_text
from ..common.models import mixedlm_random_intercept, ols_cluster

def run(df: pd.DataFrame, cfg: Config):
    outs = []
    outdir = cfg.out_dir

    # τ by order: ANOVA + MW
    if cfg.tau_col in df.columns and cfg.order_col in df.columns:
        sub = df[[cfg.tau_col, cfg.order_col]].dropna().copy()
        sub[cfg.order_col] = sub[cfg.order_col].astype("category")
        if len(sub)>=3 and sub[cfg.order_col].nunique()>=2:
            model = smf.ols(f"{cfg.tau_col} ~ C({cfg.order_col})", data=sub).fit()
            aov = anova_lm(model, typ=2).reset_index().rename(columns={"index":"term"})
            write_table(aov, f"{outdir}/step04_tau_anova.csv")
            outs.append({"kind":"csv","path":f"{outdir}/step04_tau_anova.csv"})
            sv = sub[sub[cfg.order_col]=="SV"][cfg.tau_col]
            vs = sub[sub[cfg.order_col]=="VS"][cfg.tau_col]
            if len(sv)>0 and len(vs)>0:
                U,p = stats.mannwhitneyu(sv, vs, alternative="two-sided")
                write_table(pd.DataFrame([{"test":"MW","U":float(U),"p":float(p),"n_SV":len(sv),"n_VS":len(vs)}]),
                            f"{outdir}/step04_tau_mw.csv")
                outs.append({"kind":"csv","path":f"{outdir}/step04_tau_mw.csv"})

    # Mixed model τ with controls
    if all(c in df.columns for c in [cfg.tau_col, cfg.order_col, cfg.item_col]):
        cols = [cfg.tau_col, cfg.order_col, cfg.item_col]
        for opt in [cfg.family_col, cfg.gen2_col]:
            if opt in df.columns:
                cols.append(opt)
        sub = df[cols].dropna().copy()
        for c in [cfg.order_col, cfg.family_col, cfg.gen2_col]:
            if c in sub.columns:
                sub[c] = sub[c].astype("category")
        try:
            formula = f"{cfg.tau_col} ~ C({cfg.order_col})"
            if cfg.family_col in sub.columns:
                formula += f" + C({cfg.family_col})"
            if cfg.gen2_col in sub.columns:
                formula += f" + C({cfg.gen2_col})"
            mdf = mixedlm_random_intercept(
                formula,
                sub,
                sub[cfg.item_col].astype(str)
                )                                                                                                                                                                          
            write_text(mdf.summary().as_text(), f"{outdir}/step04_tau_mixed.txt")
            outs.append({"kind":"txt","path":f"{outdir}/step04_tau_mixed.txt"})
        except Exception as e:
            write_text(f"MixedLM failed: {e}", f"{outdir}/step04_tau_mixed.txt")
            outs.append({"kind":"txt","path":f"{outdir}/step04_tau_mixed.txt"})

    # Metrics by order: OLS (clustered by item) + MW
    if cfg.order_col in df.columns:
        for metric in getattr(cfg, "metrics", []):
            if metric not in df.columns: 
                continue
            sub = df[[metric, cfg.order_col, cfg.item_col]].dropna().copy()
            if len(sub) < 10 or sub[cfg.order_col].nunique()<2:
                continue
            model = ols_cluster(f"{metric} ~ C({cfg.order_col})", sub, sub[cfg.item_col].astype(str))
            txt = model.summary().as_text()
            write_text(txt, f"{outdir}/step04_metric_{metric}_ols_cluster.txt")
            outs.append({"kind":"txt","path":f"{outdir}/step04_metric_{metric}_ols_cluster.txt"})
            sv = sub[sub[cfg.order_col]=="SV"][metric]
            vs = sub[sub[cfg.order_col]=="VS"][metric]
            if len(sv)>0 and len(vs)>0:
                U,p = stats.mannwhitneyu(sv, vs, alternative="two-sided")
                write_table(pd.DataFrame([{"metric":metric,"test":"MW","U":float(U),"p":float(p),"n_SV":len(sv),"n_VS":len(vs)}]),
                            f"{outdir}/step04_metric_{metric}_mw.csv")
                outs.append({"kind":"csv","path":f"{outdir}/step04_metric_{metric}_mw.csv"})

    return {"step":"04_alignment_quality","outputs":outs}
