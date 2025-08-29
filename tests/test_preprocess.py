
import sys
import pathlib
import pandas as pd
import pandas.testing as pdt

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from construal.common.preprocess import derive_design
from construal.common.config import Config


def test_determinacy_comparison_normalizes_strings():
    df = pd.DataFrame({
        'src_order': ['SUBJ_before_ROOT', 'ROOT_before_SUBJ', 'ROOT_before_SUBJ', 'SUBJ_before_ROOT'],
        'det_cat': ['Def', 'indef', 'IND', '   DEF  '],
        'model': ['m', 'm', 'm', 'm'],
    })
    cfg = Config(in_path=pathlib.Path('in'), out_dir=pathlib.Path('out'))
    result = derive_design(df, cfg)

    expected = pd.Series([1, 1, pd.NA, 1], name=cfg.success_col, dtype="Int64")
    pdt.assert_series_equal(result[cfg.success_col], expected)


