from pathlib import Path
import pandas as pd
from pricing_lab.io import read_cfg, ensure_dir, write_csv
from pricing_lab.elasticity import fit_elasticities

cfg = read_cfg(); out_dir = Path(cfg["paths"]["out_dir"]); ensure_dir(out_dir)
df = pd.read_csv(out_dir/"enriched_dataset.csv", parse_dates=["date"])
out = fit_elasticities(df,
        min_rows=cfg["model"]["min_rows_per_sku"],
        min_levels=cfg["model"]["min_price_levels_sku"])
if out.empty:
  print("No elasticities computed. Consider lowering thresholds in config.yaml.")
else:
  write_csv(out, out_dir/"elasticity_table.csv")
  print("Elasticities -> outputs/elasticity_table.csv")
