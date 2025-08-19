from pathlib import Path
import pandas as pd
from pricing_lab.io import read_cfg, ensure_dir, write_csv
from pricing_lab.simulate import run_scenario, build_standard_policy, summarize_portfolio

cfg = read_cfg(); out_dir = Path(cfg["paths"]["out_dir"]); ensure_dir(out_dir)
df   = pd.read_csv(out_dir/"enriched_dataset.csv", parse_dates=["date"])
elas = pd.read_csv(out_dir/"elasticity_table.csv")

# Policies
skus = df["sku_id"].unique()
pol_opt = {s: cfg["scenarios"]["optimistic_pct"]   for s in skus}
pol_con = {s: cfg["scenarios"]["conservative_pct"] for s in skus}
base    = df.groupby("sku_id")["date"].max()  # dummy to feed builder
pol_std = build_standard_policy(df, df[df["sku_id"].isin(skus)].groupby(["sku_id","category"]).tail(1),
                                cfg["scenarios"]["top_share"],
                                cfg["scenarios"]["standard_top_disc"],
                                cfg["scenarios"]["standard_tail_incr"])

round_to_99 = cfg["scenarios"]["round_to_99"]; margin_pct = cfg["scenarios"]["assumed_margin_pct"]
opt = run_scenario("optimistic",   pol_opt, df, elas, round_to_99, margin_pct)
con = run_scenario("conservative", pol_con, df, elas, round_to_99, margin_pct)
std = run_scenario("standard",     pol_std, df, elas, round_to_99, margin_pct)

write_csv(opt.sort_values(["category","sku_id"]), out_dir/"scenario_optimistic.csv")
write_csv(con.sort_values(["category","sku_id"]), out_dir/"scenario_conservative.csv")
write_csv(std.sort_values(["category","sku_id"]), out_dir/"scenario_standard.csv")

import pandas as pd
summary = pd.DataFrame([
  summarize_portfolio(opt,"optimistic"),
  summarize_portfolio(con,"conservative"),
  summarize_portfolio(std,"standard"),
])
write_csv(summary, out_dir/"scenario_portfolio_summary.csv")
print("Scenarios -> outputs/*scenario_*.csv ; Portfolio -> outputs/scenario_portfolio_summary.csv")
