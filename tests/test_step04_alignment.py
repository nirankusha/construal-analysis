import pandas as pd
from pathlib import Path
from construal.common.config import Config
from construal.steps import step04_alignment_quality as step04

def test_step04_handles_missing_family(tmp_path: Path):
    df = pd.DataFrame({
        'ann_align_kendall_tau': [0.1, 0.2, 0.3, 0.4],
        'order_cond': ['SV', 'VS', 'SV', 'VS'],
        'item_id': [1, 2, 3, 4]
    })
    cfg = Config(in_path=tmp_path/'dummy', out_dir=tmp_path)
    res = step04.run(df, cfg)
    assert res['step'] == '04_alignment_quality'