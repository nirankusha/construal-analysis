# Construal Analysis (selected.parquet)

This pipeline analyzes **construal mapping** assuming **one canonical output per sentence×model** (e.g., `selected.parquet`).

## Install
```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

## Run (all steps)
```bash
python -m construal.cli --in selected.parquet --out artifacts --all
```

## Run specific steps
```bash
python -m construal.cli --in selected.parquet --out artifacts --steps 1 2 3 4 5 6 7
```

## Expected columns
Required:
- `sent_id` (or `item_id`), `model`, `order_cond` ∈ {SV, VS}
- `construal_match_bin` ∈ {0,1}
- `ann_align_kendall_tau` (float)

Optional:
- `model_family`, `gen2` (plain|mbrish)
- `det_cat` ∈ {def, ind, none}
- Decoding metadata (`beam_used`, `top_p_used`, `objective`) to derive `strategy` in Step 07

Each step skips gracefully if required columns are missing and still emits a tiny README for traceability.

## Statistical utilities

The helper `chi2_2x2` automatically applies the Haldane–Anscombe correction by
adding 0.5 to all cells when zero counts occur before computing the
chi-square test and odds ratio.

The `holm_correction` helper adjusts a sequence of p-values using the
Holm–Bonferroni method, and steps that emit per-model p-values include an
additional `p_holm` column.

The `proportion_tests` helper performs one-tailed (greater) z and exact
binomial tests against a specified null proportion `p0`.
