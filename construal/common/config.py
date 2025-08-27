from dataclasses import dataclass
from pathlib import Path
from typing import Sequence, Literal

PoolRule = Literal["any", "sum"]

@dataclass
class Config:
    in_path: Path
    out_dir: Path
    tau_col: str = "ann_align_kendall_tau"
    success_col: str = "construal_match_bin"
    order_col: str = "order_cond"
    det_col: str = "det_cat"             # {def, ind, none}
    model_col: str = "model"
    family_col: str = "model_family"
    gen2_col: str = "gen2"
    item_col: str = "item_id"
    sent_col: str = "sent_id"
    metrics: Sequence[str] = ("rt_chrf","rt_bleu","rt_ter")
    pooling_rule: PoolRule = "any"       # for pooled 2×2
