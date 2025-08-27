import pandas as pd
import numpy as np

def pooled_any_success(df: pd.DataFrame, order_col: str, success_col: str, sent_col: str) -> pd.DataFrame:
    if sent_col not in df.columns:
        agg = df.groupby(order_col)[success_col].agg(["sum","count"]).reset_index()
        agg.rename(columns={"sum":"success","count":"n"}, inplace=True)
        return agg
    sent_any = (df.pivot_table(index=[sent_col, order_col],
                               values=success_col, aggfunc="max").reset_index())
    agg = sent_any.groupby(order_col)[success_col].agg(["sum","count"]).reset_index()
    agg.rename(columns={"sum":"success","count":"n"}, inplace=True)
    return agg

def per_model_2x2(df: pd.DataFrame, order_col: str, success_col: str, model_col: str) -> list[dict]:
    out = []
    for m, g in df.groupby(model_col):
        ct = pd.crosstab(g[order_col], g[success_col]).reindex(index=["SV","VS"], columns=[0,1], fill_value=0)
        out.append({
            "model": m,
            "SV_fail": int(ct.loc["SV",0]) if "SV" in ct.index else 0,
            "SV_success": int(ct.loc["SV",1]) if "SV" in ct.index else 0,
            "VS_fail": int(ct.loc["VS",0]) if "VS" in ct.index else 0,
            "VS_success": int(ct.loc["VS",1]) if "VS" in ct.index else 0,
        })
    return out

def per_model_2x3(df: pd.DataFrame, order_col: str, det_col: str, model_col: str) -> list[dict]:
    out = []
    for m, g in df.groupby(model_col):
        ct = pd.crosstab(g[order_col], g[det_col]).reindex(index=["SV","VS"], columns=["def","ind","none"], fill_value=0)
        out.append({
            "model": m,
            "SV_def": int(ct.loc["SV","def"]), "SV_ind": int(ct.loc["SV","ind"]), "SV_none": int(ct.loc["SV","none"]),
            "VS_def": int(ct.loc["VS","def"]), "VS_ind": int(ct.loc["VS","ind"]), "VS_none": int(ct.loc["VS","none"]),
        })
    return out
