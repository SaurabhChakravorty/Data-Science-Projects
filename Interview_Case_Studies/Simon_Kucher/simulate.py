import numpy as np, pandas as pd

def latest_baseline(df):
  last = df.groupby("sku_id")["date"].max().rename("last_date")
  base = df.merge(last, left_on="sku_id", right_index=True)
  base = base[base["date"]==base["last_date"]].groupby(["sku_id","category"],as_index=False)\
         .agg(price_old=("price","mean"), q_old=("sales","mean"))
  return base

def apply_price_change(p_old, pct, round_to_99=True):
  p = np.maximum(p_old*(1+pct), 0.01)
  if round_to_99:
    p = np.floor(p)+0.99; p = np.maximum(p, 0.99)
  return p

def demand_transform(q_old, p_old, p_new, e):
  ratio = np.clip(p_new/np.clip(p_old,1e-6,None), 1e-6, 1e6)
  return q_old * np.power(ratio, e)

def compute(res, margin_pct=0.25):
  cost = (1.0 - margin_pct) * res["price_old"]
  res["rev_old"]=res["price_old"]*res["q_old"]; res["rev_new"]=res["price_new"]*res["q_new"]
  res["margin_old"]=(res["price_old"]-cost)*res["q_old"]; res["margin_new"]=(res["price_new"]-cost)*res["q_new"]
  res["d_rev"]=res["rev_new"]-res["rev_old"]; res["d_margin"]=res["margin_new"]-res["margin_old"]
  res["pct_d_rev"]=np.where(res["rev_old"]>0, res["d_rev"]/res["rev_old"], np.nan)
  res["pct_d_margin"]=np.where(res["margin_old"]>0, res["d_margin"]/res["margin_old"], np.nan)
  return res

def build_standard_policy(df_full, base, top_share=0.30, top_disc=-0.05, tail_incr=0.03):
  q_by_sku = df_full.groupby("sku_id")["sales"].sum().sort_values(ascending=False)
  cum = (q_by_sku/q_by_sku.sum()).cumsum()
  top = set(cum[cum<=top_share].index)
  return {sku:(top_disc if sku in top else tail_incr) for sku in base["sku_id"]}

def run_scenario(label, pct_by_sku, df_full, elas, round_to_99=True, margin_pct=0.25):
  base = latest_baseline(df_full)
  base = base.merge(elas[["sku_id","category","elasticity"]], on=["sku_id","category"], how="left")
  base["elasticity"]=base["elasticity"].fillna(-1.0)
  pct_df = pd.DataFrame(list(pct_by_sku.items()), columns=["sku_id","pct_change"])
  res = base.merge(pct_df, on="sku_id", how="left"); res["pct_change"]=res["pct_change"].fillna(0.0)
  res["price_new"]=apply_price_change(res["price_old"], res["pct_change"], round_to_99)
  res["q_new"]=demand_transform(res["q_old"], res["price_old"], res["price_new"], res["elasticity"])
  res=compute(res, margin_pct); res["scenario"]=label
  return res

def summarize_portfolio(df_rows, label):
  S=lambda c: df_rows[c].sum()
  return {"scenario":label, "skus":df_rows["sku_id"].nunique(),
          "rev_old":S("rev_old"),"rev_new":S("rev_new"),
          "d_rev":S("d_rev"),"pct_d_rev":(S("rev_new")/S("rev_old")-1.0 if S("rev_old")>0 else np.nan),
          "margin_old":S("margin_old"),"margin_new":S("margin_new"),
          "d_margin":S("d_margin"),"pct_d_margin":(S("margin_new")/S("margin_old")-1.0 if S("margin_old")>0 else np.nan)}
