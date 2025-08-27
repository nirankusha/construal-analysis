import pandas as pd
from scipy.stats import kendalltau, kruskal
import statsmodels.formula.api as smf
from ..common.config import Config
from ..common.io import write_table, write_text

def run(df: pd.DataFrame, cfg: Config):
    outs = []
    outdir = cfg.out_dir

    # τ ↔ success
    if all(c in df.columns for c in [cfg.tau_col, cfg.success_col]):
        sub = df[[cfg.tau_col, cfg.success_col]].dropna().copy()
        if len(sub)>=5:
            r = sub[[cfg.tau_col, cfg.success_col]].corr(method="pearson").iloc[0,1]
            write_table(pd.DataFrame([{"test":"pointbiserial(Pearson)", "r":float(r), "n":len(sub)}]),
                        f"{outdir}/step05_tau_success_corr.csv")
            try:
                m = smf.logit(f"{cfg.success_col} ~ {cfg.tau_col}", data=sub).fit(disp=False)
                write_text(m.summary().as_text(), f"{outdir}/step05_tau_success_logit.txt")
                outs += [{"kind":"csv","path":f"{outdir}/step05_tau_success_corr.csv"},
                         {"kind":"txt","path":f"{outdir}/step05_tau_success_logit.txt"}]
            except Exception as e:
                write_text(f"logit failed: {e}", f"{outdir}/step05_tau_success_logit.txt")
                outs += [{"kind":"csv","path":f"{outdir}/step05_tau_success_corr.csv"},
                         {"kind":"txt","path":f"{outdir}/step05_tau_success_logit.txt"}]

    # τ ↔ determiner (ordinal)
    if all(c in df.columns for c in [cfg.tau_col, cfg.det_col]):
        sub = df[[cfg.tau_col, cfg.det_col]].dropna().copy()
        if len(sub)>=5 and sub[cfg.det_col].nunique()>=2:
            ord_map = {"none":0,"ind":1,"def":2}
            y = sub[cfg.det_col].astype(str).str.lower().map(ord_map)
            tau, p = kendalltau(sub[cfg.tau_col], y)
            write_table(pd.DataFrame([{"test":"Kendall_tau_b (tau vs det_ord)","tau":float(tau),"p":float(p),"n":len(sub)}]),
                        f"{outdir}/step05_tau_det_kendall.csv")
            groups = [sub.loc[sub[cfg.det_col]==k, cfg.tau_col].values for k in ["none","ind","def"]]
            if all(len(g)>0 for g in groups):
                H, pk = kruskal(*groups)
                write_table(pd.DataFrame([{"test":"KruskalWallis(tau by det)","H":float(H),"p":float(pk)}]),
                            f"{outdir}/step05_tau_det_kw.csv")
            try:
                sub = sub.assign(det_ord=y)
                m = smf.mnlogit(f"det_ord ~ {cfg.tau_col}", data=sub).fit(disp=False)
                write_text(m.summary().as_text(), f"{outdir}/step05_tau_det_mnlogit.txt")
                outs += [{"kind":"csv","path":f"{outdir}/step05_tau_det_kendall.csv"},
                         {"kind":"csv","path":f"{outdir}/step05_tau_det_kw.csv"},
                         {"kind":"txt","path":f"{outdir}/step05_tau_det_mnlogit.txt"}]
            except Exception as e:
                write_text(f"Multinomial logit failed (try ordinal logit in future): {e}", f"{outdir}/step05_tau_det_mnlogit.txt")
                outs += [{"kind":"csv","path":f"{outdir}/step05_tau_det_kendall.csv"}]

    if not outs:
        write_text("Step05 skipped: missing columns.", f"{outdir}/step05_readme.txt")
        outs.append({"kind":"txt","path":f"{outdir}/step05_readme.txt"})
    return {"step":"05_tau_vs_discrete","outputs":outs}
