from pathlib import Path
import pandas as pd
import yaml

def read_cfg(path="config.yaml"):
  with open(path, "r", encoding="utf-8") as f:
    return yaml.safe_load(f)

def ensure_dir(p: Path):
  p.mkdir(parents=True, exist_ok=True)

def read_csv_eu(path: str) -> pd.DataFrame:
  raw = pd.read_csv(path, sep=";", dtype=str)
  return raw

def write_csv(df, path: Path):
  df.to_csv(path, index=False)
