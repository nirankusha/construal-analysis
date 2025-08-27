import json
import numpy as np
import pandas as pd
from .config import Config

# Minimal determinacy logic
# Expected mapping (from Polish order):
#  src_order_binary: 1 -> "Def", 0 -> "Indef"; else -> "None"
def expected_from_pl_order_binary(v):
    if v == 1:  return "Def"
    if v == 0:  return "Indef"
    return "None"

def expected_from_pl_order_str(s):
    if s == "SUBJ_before_ROOT":  return "Def"
    if s == "ROOT_before_SUBJ":  return "Indef"
    return "None"

def derive_design(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    df = df.copy()

    # --- (0) Fix odd header like "sent_id/...selected_summary.parquet ..." -> "sent_id"
    bad_sent_cols = [c for c in df.columns if str(c).startswith("sent_id")]
    if bad_sent_cols and cfg.sent_col not in df.columns:
        df.rename(columns={bad_sent_cols[0]: cfg.sent_col}, inplace=True)

    # --- (1) Ensure sent_id / item_id
    if cfg.sent_col not in df.columns:
        if "sentence_id" in df.columns:
            df.rename(columns={"sentence_id": cfg.sent_col}, inplace=True)
        else:
            df[cfg.sent_col] = pd.factorize(df.index)[0]
    if cfg.item_col not in df.columns:
        df[cfg.item_col] = df[cfg.sent_col]

    # --- (2) Map your schema -> expected names
    # ann_det_general -> det_cat; pool_type -> gen2
    colmap = {
        "ann_det_general": cfg.det_col,
        "pool_type": cfg.gen2_col,
    }
    for src, dst in colmap.items():
        if src in df.columns and dst not in df.columns:
            df.rename(columns={src: dst}, inplace=True)

    # --- (3) order_cond from src_order (and normalize)
    if cfg.order_col not in df.columns and "src_order" in df.columns:
        src = df["src_order"].astype("string")
        df[cfg.order_col] = src.map({
            "SUBJ_before_ROOT": "SV",
            "ROOT_before_SUBJ": "VS"
        })
    if cfg.order_col in df.columns:
        df[cfg.order_col] = (
            df[cfg.order_col]
            .astype("string")
            .str.upper()
            .map({"SV": "SV", "VS": "VS"})
        )
        df[cfg.order_col] = pd.Categorical(df[cfg.order_col], categories=["SV", "VS"])

    # --- (4) Determiner normalization to match determinacy ("Def"/"Indef"/"None")
    if cfg.det_col in df.columns:
        df[cfg.det_col] = (
            df[cfg.det_col]
            .astype("string")
            .str.strip()
            .str.title()
            .map({"Def": "Def", "Indef": "Indef", "None": "None"})
        )
        df[cfg.det_col] = pd.Categorical(
            df[cfg.det_col], categories=["None", "Indef", "Def"], ordered=True
        )

    # --- (5) Strategy derivation (from 'mode' and/or 'decode_params')
    if "strategy" not in df.columns:
        if "mode" in df.columns:
            df["strategy"] = df["mode"].astype("string").str.lower()
        else:
            df["strategy"] = pd.Series(pd.NA, index=df.index, dtype="string")

    if "decode_params" in df.columns:
        def _extract(dp, key):
            try:
                obj = json.loads(dp) if isinstance(dp, str) else (dp or {})
                return obj.get(key, None)
            except Exception:
                return None
        if "beam_used" not in df.columns:
            df["beam_used"] = df["decode_params"].apply(lambda x: _extract(x, "num_beams"))
        if "top_p_used" not in df.columns:
            df["top_p_used"] = df["decode_params"].apply(lambda x: _extract(x, "top_p"))

    # --- (6) Core dtypes
    for col in [cfg.model_col, cfg.family_col, cfg.gen2_col, "strategy"]:
        if col in df.columns:
            df[col] = df[col].astype("string")

    if cfg.tau_col in df.columns:
        df[cfg.tau_col] = pd.to_numeric(df[cfg.tau_col], errors="coerce")

    # --- (7) Handle success column creation - this is the key fix
    # First check if success column already exists and try to use it
    if cfg.success_col in df.columns:
        df[cfg.success_col] = pd.to_numeric(df[cfg.success_col], errors="coerce").astype("Int64")
    else:
        # Try different approaches to create the success column
        success_created = False
        
        # Approach 1: Direct binary order comparison (original analysis script approach)
        if "src_order_binary" in df.columns and "ann_order_binary" in df.columns:
            src_bin = pd.to_numeric(df["src_order_binary"], errors="coerce")
            ann_bin = pd.to_numeric(df["ann_order_binary"], errors="coerce")
            # Create success as 1 when they match, 0 when they don't, NA when either is NA
            df[cfg.success_col] = (src_bin == ann_bin).astype("Int64")
            success_created = True
        
        # Approach 2: Determinacy-based comparison (fallback)
        elif cfg.det_col in df.columns:
            # Expected determinacy from source order
            exp = pd.Series("None", index=df.index, dtype="string")
            
            if "src_order_binary" in df.columns:
                exp = pd.to_numeric(df["src_order_binary"], errors="coerce").map(expected_from_pl_order_binary)
            elif "src_order" in df.columns:
                exp = df["src_order"].astype("string").map(expected_from_pl_order_str)
            
            det = df[cfg.det_col].astype("string")
            # Only score success for scorable pairs (both Def/Indef)
            mask = det.isin(["Def", "Indef"]) & exp.isin(["Def", "Indef"])
            
            # Create comparison
            cmp_bool = (det == exp)
            arr = np.where(mask, cmp_bool.fillna(False), np.nan)
            df[cfg.success_col] = pd.Series(arr).map({True: 1, False: 0}).astype("Int64")
            success_created = True
        
        # If we still couldn't create the success column, create a dummy one
        if not success_created:
            print(f"Warning: Could not derive {cfg.success_col} column. Creating dummy column with all NAs.")
            df[cfg.success_col] = pd.Series(pd.NA, index=df.index, dtype="Int64")

    # --- (8) Enforce one row per (sent_id, model)
    if cfg.sent_col in df.columns and cfg.model_col in df.columns:
        df = (
            df.sort_values([cfg.sent_col, cfg.model_col])
              .drop_duplicates(subset=[cfg.sent_col, cfg.model_col], keep="first")
        )

    return df