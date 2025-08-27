from pathlib import Path
import pandas as pd
import os

def ensure_dir(path: str | os.PathLike) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)

def read_df(path: str | os.PathLike) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)

def write_table(df: pd.DataFrame, path: str) -> None:
    ensure_dir(path)
    df.to_csv(path, index=False)

def write_text(text: str, path: str) -> None:
    ensure_dir(path)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
