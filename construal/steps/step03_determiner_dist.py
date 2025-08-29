import pandas as pd
import numpy as np
from ..common.config import Config
from ..common.io import write_table, write_text
from ..common.stats import chi2_2x3, cmh_from_2x2_list, chi2_2x2, holm_correction
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
        df_counts = pd.DataFrame(rows)
        write_table(df_counts, f"{outdir}/step03_per_model_2x3.csv")
        outs.append({"kind":"csv","path":f"{outdir}/step03_per_model_2x3.csv"})

        stats_rows = []
        for r in rows:
            table = [[r["SV_def"], r["SV_ind"], r["SV_none"]],
                     [r["VS_def"], r["VS_ind"], r["VS_none"]]]
            stats_rows.append({"model": r["model"], **chi2_2x3(table)})
        df_stats = pd.DataFrame(stats_rows)
        df_stats["p_holm"] = holm_correction(df_stats["p"].values)
        write_table(df_stats, f"{outdir}/step03_per_model_2x3_stats.csv")
        outs.append({"kind":"csv","path":f"{outdir}/step03_per_model_2x3_stats.csv"})

    # CMH contrasts (Def vs Others, Ind vs Others)
    tables_def, tables_ind = [], []
    def_rows, ind_rows = [], []
    for m, g in df.groupby(cfg.model_col):
        g = g.copy()
        g["def_bin"] = (g[cfg.det_col] == "def").astype(int)
        ct2 = pd.crosstab(g[cfg.order_col], g["def_bin"]).reindex(index=["SV","VS"], columns=[0,1], fill_value=0)
        tables_def.append(ct2.values.astype(float))

        sv_def = int(ct2.loc["SV",1]) if "SV" in ct2.index else 0
        sv_oth = int(ct2.loc["SV",0]) if "SV" in ct2.index else 0
        vs_def = int(ct2.loc["VS",1]) if "VS" in ct2.index else 0
        vs_oth = int(ct2.loc["VS",0]) if "VS" in ct2.index else 0
        res_def = chi2_2x2(sv_def, sv_oth, vs_def, vs_oth)
        def_rows.append({"model": m, "SV_def": sv_def, "SV_others": sv_oth, "VS_def": vs_def, "VS_others": vs_oth, **res_def})

        g["ind_bin"] = (g[cfg.det_col] == "ind").astype(int)
        ct2i = pd.crosstab(g[cfg.order_col], g["ind_bin"]).reindex(index=["SV","VS"], columns=[0,1], fill_value=0)
        tables_ind.append(ct2i.values.astype(float))

        sv_ind = int(ct2i.loc["SV",1]) if "SV" in ct2i.index else 0
        sv_oth_i = int(ct2i.loc["SV",0]) if "SV" in ct2i.index else 0
        vs_ind = int(ct2i.loc["VS",1]) if "VS" in ct2i.index else 0
        vs_oth_i = int(ct2i.loc["VS",0]) if "VS" in ct2i.index else 0
        res_ind = chi2_2x2(sv_ind, sv_oth_i, vs_ind, vs_oth_i)
        ind_rows.append({"model": m, "SV_ind": sv_ind, "SV_others": sv_oth_i, "VS_ind": vs_ind, "VS_others": vs_oth_i, **res_ind})

    if def_rows:
        df_def = pd.DataFrame(def_rows)
        df_def["p_holm"] = holm_correction(df_def["p"].values)
        write_table(df_def, f"{outdir}/step03_per_model_def_vs_others.csv")
        outs.append({"kind":"csv","path":f"{outdir}/step03_per_model_def_vs_others.csv"})
    if tables_def:
        try:
            cmh_def = cmh_from_2x2_list(tables_def)
            write_table(pd.DataFrame([cmh_def]), f"{outdir}/step03_cmh_def_vs_others.csv")
            outs.append({"kind":"csv","path":f"{outdir}/step03_cmh_def_vs_others.csv"})
        except Exception as e:
            write_text(f"CMH (def vs others) failed: {e}", f"{outdir}/step03_cmh_def_vs_others.txt")
            outs.append({"kind":"txt","path":f"{outdir}/step03_cmh_def_vs_others.txt"})
    if ind_rows:
        df_ind = pd.DataFrame(ind_rows)
        df_ind["p_holm"] = holm_correction(df_ind["p"].values)
        write_table(df_ind, f"{outdir}/step03_per_model_ind_vs_others.csv")
        outs.append({"kind":"csv","path":f"{outdir}/step03_per_model_ind_vs_others.csv"})
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
