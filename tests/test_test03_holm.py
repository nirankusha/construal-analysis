import sys
import pathlib
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from construal.steps.step03_determiner_dist import run as step03_run
from construal.common.config import Config


def test_step03_per_model_p_holm(tmp_path):
    data = [
        {"order_cond": "SV", "det_cat": "def", "model": "m1"},
        {"order_cond": "SV", "det_cat": "ind", "model": "m1"},
        {"order_cond": "VS", "det_cat": "def", "model": "m1"},
        {"order_cond": "VS", "det_cat": "none", "model": "m1"},
        {"order_cond": "SV", "det_cat": "def", "model": "m2"},
        {"order_cond": "SV", "det_cat": "none", "model": "m2"},
        {"order_cond": "VS", "det_cat": "ind", "model": "m2"},
        {"order_cond": "VS", "det_cat": "none", "model": "m2"},
    ]
    df = pd.DataFrame(data)
    cfg = Config(in_path=Path("dummy"), out_dir=tmp_path)
    step03_run(df, cfg)
    stats_path = tmp_path / "step03_per_model_2x3_stats.csv"
    assert stats_path.exists()
    df_stats = pd.read_csv(stats_path)
    assert "p_holm" in df_stats.columns
    def_path = tmp_path / "step03_per_model_def_vs_others.csv"
    ind_path = tmp_path / "step03_per_model_ind_vs_others.csv"
    assert def_path.exists() and ind_path.exists()
    df_def = pd.read_csv(def_path)
    df_ind = pd.read_csv(ind_path)
    assert "p_holm" in df_def.columns
    assert "p_holm" in df_ind.columns
