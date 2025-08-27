import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from ..common.config import Config
from ..common.io import write_text

def derive_strategy(df: pd.DataFrame) -> pd.Series:
    strat = pd.Series("greedy", index=df.index, dtype="string")
    if "objective" in df.columns:
        strat = strat.mask(df["objective"].str.contains("mbr", case=False, na=False), "mbrish")
    if "top_p_used" in df.columns:
        strat = strat.mask(df["top_p_used"].notna(), "nucleus")
    if "beam_used" in df.columns:
        strat = strat.mask(df["beam_used"].notna(), "beam")
    return strat

def run(df: pd.DataFrame, cfg: Config):
    outs = []
    outdir = cfg.out_dir
    if "strategy" not in df.columns:
        df = df.copy()
        df["strategy"] = derive_strategy(df)

    need = [cfg.success_col, cfg.order_col, "strategy", cfg.family_col, cfg.item_col]
    if not all(c in df.columns for c in need):
        write_text("Step07 skipped: required columns missing.", f"{outdir}/step07_readme.txt")
        return {"step":"07_strategy_success","outputs":[{"kind":"txt","path":f"{outdir}/step07_readme.txt"}]}

    sub = df[need].dropna().copy()
    for c in [cfg.order_col, "strategy", cfg.family_col]:
        sub[c] = sub[c].astype("category")
    try:
        m = smf.logit(f"{cfg.success_col} ~ C({cfg.order_col})*C(strategy) + C({cfg.family_col})",
                      data=sub).fit(disp=False, cov_type="cluster", cov_kwds={"groups": sub[cfg.item_col].astype(str)})
        write_text(m.summary().as_text(), f"{outdir}/step07_strategy_logit.txt")
        outs.append({"kind":"txt","path":f"{outdir}/step07_strategy_logit.txt"})
    except Exception as e:
        write_text(f"Strategy logit failed: {e}", f"{outdir}/step07_strategy_logit.txt")
        outs.append({"kind":"txt","path":f"{outdir}/step07_strategy_logit.txt"})
    return {"step":"07_strategy_success","outputs":outs}
