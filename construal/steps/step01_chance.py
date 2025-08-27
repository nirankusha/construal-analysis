import pandas as pd
from ..common.config import Config
from ..common.io import write_table, write_text
from ..common.stats import proportion_tests

def run(df: pd.DataFrame, cfg: Config):
    outs = []
    outdir = cfg.out_dir

    # Global
    sub = df[[cfg.success_col]].dropna()
    if len(sub) > 0:
        k = int(sub[cfg.success_col].sum()); n = int(len(sub))
        glob = proportion_tests(k, n)
        write_table(pd.DataFrame([glob]), f"{outdir}/step01_chance_global.csv")
        outs.append({"kind":"csv","path":f"{outdir}/step01_chance_global.csv"})

    # By model
    rows = []
    if cfg.model_col in df.columns:
        for m, g in df.groupby(cfg.model_col):
            ss = g[[cfg.success_col]].dropna()
            if len(ss)==0: continue
            ks = int(ss[cfg.success_col].sum()); ns = int(len(ss))
            res = proportion_tests(ks, ns); res["model"]=m
            rows.append(res)
    if rows:
        write_table(pd.DataFrame(rows), f"{outdir}/step01_chance_by_model.csv")
        outs.append({"kind":"csv","path":f"{outdir}/step01_chance_by_model.csv"})

    # By gen2
    rows = []
    if cfg.gen2_col in df.columns:
        for g2, g in df.groupby(cfg.gen2_col):
            ss = g[[cfg.success_col]].dropna()
            if len(ss)==0: continue
            ks = int(ss[cfg.success_col].sum()); ns = int(len(ss))
            res = proportion_tests(ks, ns); res["gen2"]=g2
            rows.append(res)
    if rows:
        write_table(pd.DataFrame(rows), f"{outdir}/step01_chance_by_gen2.csv")
        outs.append({"kind":"csv","path":f"{outdir}/step01_chance_by_gen2.csv"})

    write_text("Step01: chance-level tests completed.", f"{outdir}/step01_readme.txt")
    outs.append({"kind":"txt","path":f"{outdir}/step01_readme.txt"})
    return {"step":"01_chance","outputs":outs}
