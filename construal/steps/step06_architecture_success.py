import pandas as pd
import statsmodels.formula.api as smf
from ..common.config import Config
from ..common.io import write_text

def run(df: pd.DataFrame, cfg: Config):
    outs = []
    outdir = cfg.out_dir
    need = [cfg.success_col, cfg.order_col, cfg.family_col, cfg.item_col, cfg.gen2_col]
    if not all(c in df.columns for c in need):
        write_text("Step06 skipped: required columns missing.", f"{outdir}/step06_readme.txt")
        return {"step":"06_architecture_success","outputs":[{"kind":"txt","path":f"{outdir}/step06_readme.txt"}]}

    sub = df[need].dropna().copy()
    for c in [cfg.order_col, cfg.family_col, cfg.gen2_col]:
        sub[c] = sub[c].astype("category")
    try:
        m = smf.logit(f"{cfg.success_col} ~ C({cfg.order_col})*C({cfg.family_col}) + C({cfg.gen2_col})",
                      data=sub).fit(disp=False, cov_type="cluster", cov_kwds={"groups": sub[cfg.item_col].astype(str)})
        write_text(m.summary().as_text(), f"{outdir}/step06_architecture_logit.txt")
        outs.append({"kind":"txt","path":f"{outdir}/step06_architecture_logit.txt"})
    except Exception as e:
        write_text(f"Architecture logit failed: {e}", f"{outdir}/step06_architecture_logit.txt")
        outs.append({"kind":"txt","path":f"{outdir}/step06_architecture_logit.txt"})
    return {"step":"06_architecture_success","outputs":outs}
