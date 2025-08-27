import pandas as pd
import numpy as np
from ..common.config import Config
from ..common.io import write_table, write_text
from ..common.stats import chi2_2x3, cmh_from_2x2_list
from ..common.tables import per_model_2x3

def run(df: pd.DataFrame, cfg: Config):
    outs = []
    outdir = cfg.out_dir
    if cfg.det_col not in df.columns:
        write_text("Step03 skipped: det_cat column missing.", f"{outdir}/step03_readme.txt")
        return {"step":"03_determiner_dist","outputs":[{"kind":"txt","path":f"{outdir}/step03_readme.txt"}]}

    # Global 2×3
    ct = pd.crosstab(df[cfg.order_col], df[cfg.det_col]).reindex(index=["SV","VS"], columns=["def","ind","none"], fill_value=0)
    res = chi2_2x3(ct.values.astype(float))
    write_table(ct.reset_index(), f"{outdir}/step03_global_2x3_counts.csv")
    write_table(pd.DataFrame([res]), f"{outdir}/step03_global_2x3_stats.csv")
    outs += [{"kind":"csv","path":f"{outdir}/step03_global_2x3_counts.csv"},
             {"kind":"csv","path":f"{outdir}/step03_global_2x3_stats.csv"}]

    # Per-model 2×3
    rows = per_model_2x3(df, cfg.order_col, cfg.det_col, cfg.model_col)
    if rows:
        write_table(pd.DataFrame(rows), f"{outdir}/step03_per_model_2x3.csv")
        outs.append({"kind":"csv","path":f"{outdir}/step03_per_model_2x3.csv"})

    # CMH contrasts (Def vs Others, Ind vs Others)
    tables_def, tables_ind = [], []
    for m, g in df.groupby(cfg.model_col):
        g = g.copy()
        g["def_bin"] = (g[cfg.det_col] == "def").astype(int)
        ct2 = pd.crosstab(g[cfg.order_col], g["def_bin"]).reindex(index=["SV","VS"], columns=[0,1], fill_value=0)
        tables_def.append(ct2.values.astype(float))

        g["ind_bin"] = (g[cfg.det_col] == "ind").astype(int)
        ct2i = pd.crosstab(g[cfg.order_col], g["ind_bin"]).reindex(index=["SV","VS"], columns=[0,1], fill_value=0)
        tables_ind.append(ct2i.values.astype(float))

    if tables_def:
        try:
            cmh_def = cmh_from_2x2_list(tables_def)
            write_table(pd.DataFrame([cmh_def]), f"{outdir}/step03_cmh_def_vs_others.csv")
            outs.append({"kind":"csv","path":f"{outdir}/step03_cmh_def_vs_others.csv"})
        except Exception as e:
            write_text(f"CMH (def vs others) failed: {e}", f"{outdir}/step03_cmh_def_vs_others.txt")
            outs.append({"kind":"txt","path":f"{outdir}/step03_cmh_def_vs_others.txt"})
    if tables_ind:
        try:
            cmh_ind = cmh_from_2x2_list(tables_ind)
            write_table(pd.DataFrame([cmh_ind]), f"{outdir}/step03_cmh_ind_vs_others.csv")
            outs.append({"kind":"csv","path":f"{outdir}/step03_cmh_ind_vs_others.csv"})
        except Exception as e:
            write_text(f"CMH (ind vs others) failed: {e}", f"{outdir}/step03_cmh_ind_vs_others.txt")
            outs.append({"kind":"txt","path":f"{outdir}/step03_cmh_ind_vs_others.txt"})

    write_text("Step03: determiner distribution finished.", f"{outdir}/step03_readme.txt")
    outs.append({"kind":"txt","path":f"{outdir}/step03_readme.txt"})
    return {"step":"03_determiner_dist","outputs":outs}
