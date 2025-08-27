from pathlib import Path
from .common.config import Config
from .common.io import read_df, write_text
from .common.preprocess import derive_design
from .steps import (
    step01_chance, step02_success_sv_vs, step03_determiner_dist,
    step04_alignment_quality, step05_tau_vs_determiner,
    step06_architecture_success, step07_strategy_success,
)

def run_pipeline(args):
    cfg = Config(in_path=Path(args.in_path), out_dir=Path(args.out_dir))
    df = derive_design(read_df(cfg.in_path), cfg)

    step_map = {
        "1": step01_chance,
        "2": step02_success_sv_vs,
        "3": step03_determiner_dist,
        "4": step04_alignment_quality,
        "5": step05_tau_vs_determiner,
        "6": step06_architecture_success,
        "7": step07_strategy_success,
    }
    steps = list(step_map.keys()) if args.all else (args.steps or [])
    if not steps:
        steps = list(step_map.keys())
    completed = []
    for s in steps:
        arts = step_map[s].run(df, cfg)
        completed.append(arts.get("step","?"))
    write_text("Completed steps: " + ", ".join(completed), f"{cfg.out_dir}/PIPELINE_DONE.txt")
