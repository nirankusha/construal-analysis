import pandas as pd
import numpy as np
from ..common.config import Config
from ..common.io import write_table, write_text
from ..common.stats import chi2_2x2, cmh_from_2x2_list
from ..common.tables import pooled_any_success, per_model_2x2

def run(df: pd.DataFrame, cfg: Config):
    outs = []
    outdir = cfg.out_dir

    # 2×2 pooled by sentence (ANY success)
    agg = pooled_any_success(df, cfg.order_col, cfg.success_col, cfg.sent_col)
    if set(agg.columns)>= {cfg.order_col,"success","n"} and len(agg)>=2:
        def get_counts(order):
            row = agg[agg[cfg.order_col]==order]
            if len(row)==0: return (0,0)
            s = int(row["success"].iloc[0]); n = int(row["n"].iloc[0])
            return (s, n - s)
        sv_s, sv_f = get_counts("SV")
        vs_s, vs_f = get_counts("VS")
        res = chi2_2x2(sv_s, sv_f, vs_s, vs_f)
        tab = pd.DataFrame([{
            "SV_success": sv_s, "SV_fail": sv_f,
            "VS_success": vs_s, "VS_fail": vs_f,
            **res
        }])
        write_table(tab, f"{outdir}/step02_pooled_any_2x2.csv")
        outs.append({"kind":"csv","path":f"{outdir}/step02_pooled_any_2x2.csv"})

    # Per-model 2×2 and CMH
    rows = per_model_2x2(df, cfg.order_col, cfg.success_col, cfg.model_col)
    if rows:
        write_table(pd.DataFrame(rows), f"{outdir}/step02_per_model_2x2_counts.csv")
        outs.append({"kind":"csv","path":f"{outdir}/step02_per_model_2x2_counts.csv"})
        tables = [np.array([[r["SV_success"], r["SV_fail"]],[r["VS_success"], r["VS_fail"]]], dtype=float) for r in rows]
        try:
            cmh = cmh_from_2x2_list(tables)
            write_table(pd.DataFrame([cmh]), f"{outdir}/step02_cmh_models.csv")
            outs.append({"kind":"csv","path":f"{outdir}/step02_cmh_models.csv"})
        except Exception as e:
            write_text(f"CMH failed: {e}", f"{outdir}/step02_cmh_models.txt")
            outs.append({"kind":"txt","path":f"{outdir}/step02_cmh_models.txt"})

    write_text("Step02: SV/VS success finished.", f"{outdir}/step02_readme.txt")
    outs.append({"kind":"txt","path":f"{outdir}/step02_readme.txt"})
    return {"step":"02_success_sv_vs","outputs":outs}
