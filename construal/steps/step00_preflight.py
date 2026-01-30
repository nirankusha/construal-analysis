import pandas as pd
from ..common.config import Config
from ..common.io import write_table, write_text


def run(df: pd.DataFrame, cfg: Config):
    outs = []
    outdir = cfg.out_dir

    required = [
        cfg.sent_col,
        cfg.item_col,
        cfg.model_col,
        cfg.order_col,
        cfg.det_col,
        cfg.success_col,
        cfg.tau_col,
    ]
    optional = [
        cfg.family_col,
        cfg.gen2_col,
        "strategy",
        "beam_used",
        "top_p_used",
        *list(getattr(cfg, "metrics", [])),
    ]

    total = len(df)
    rows = []
    for col in required + optional:
        present = col in df.columns
        non_null = int(df[col].notna().sum()) if present else 0
        coverage = float(non_null / total) if present and total else 0.0
        rows.append(
            {
                "column": col,
                "required": col in required,
                "present": present,
                "non_null": non_null if present else None,
                "total": total,
                "coverage": coverage if present else None,
                "dtype": str(df[col].dtype) if present else None,
            }
        )

    coverage = pd.DataFrame(rows)
    write_table(coverage, f"{outdir}/step00_column_coverage.csv")
    outs.append({"kind": "csv", "path": f"{outdir}/step00_column_coverage.csv"})

    missing_required = [col for col in required if col not in df.columns]
    low_coverage = coverage[
        (coverage["required"])
        & (coverage["present"])
        & (coverage["coverage"] < 0.95)
    ]["column"].tolist()

    summary_lines = [
        f"Rows: {total}",
        f"Columns: {len(df.columns)}",
        f"Missing required columns: {', '.join(missing_required) if missing_required else 'None'}",
        f"Required columns below 95% coverage: {', '.join(low_coverage) if low_coverage else 'None'}",
        "Columns present:",
        ", ".join(sorted(df.columns.astype(str))),
    ]
    write_text("\n".join(summary_lines), f"{outdir}/step00_preflight_summary.txt")
    outs.append({"kind": "txt", "path": f"{outdir}/step00_preflight_summary.txt"})

    return {"step": "00_preflight", "outputs": outs}