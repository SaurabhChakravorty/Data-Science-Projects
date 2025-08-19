import numpy as np, pandas as pd
from pathlib import Path
from .kb_parse import parse_de_months, parse_fr_quarters, parse_en_quarters, quarter_of_date, TREND_SCORE

EVENTS = {"bf_window":[11], "xmas_window":[12], "back_to_school":[8,9,10], "launch_window":[9,10]}

def _to_float_eu(s: pd.Series):
  return pd.to_numeric(s.astype(str).str.replace(".","",regex=False).str.replace(",",".",regex=False), errors="coerce")

def load_sales(csv_path: str) -> pd.DataFrame:
  raw = pd.read_csv(csv_path, sep=";", dtype=str)
  df = pd.DataFrame({
    "month": raw["Month"].str.strip(),
    "product_id": raw["Product ID"].str.strip(),
    "product_category": raw["Product Category"].str.strip(),
    "price": _to_float_eu(raw["Price"]),
    "competitor_price": _to_float_eu(raw.get("Competitor Price", "")),
    "inflation_index": _to_float_eu(raw.get("Inflation Index", "")),
    "sales": pd.to_numeric(raw["Sales"], errors="coerce")
  })
  df["date"] = pd.to_datetime(df["month"], format="%Y-%m", errors="coerce")
  df["sku_id"] = df["product_id"]
  df["category"] = (df["product_category"].str.lower()
                    .replace({"headphone":"headphones","smartwatch":"smartwatches","tablet":"tablets","laptop":"laptops"}))
  return df.dropna(subset=["date"])

def _build_month_feats(df: pd.DataFrame, kb_monthly: pd.DataFrame):
  cmap={}
  for _,r in kb_monthly.iterrows():
    for m in (r["months_detected"] or []):
      cmap[(r["category"], int(m))]=1
  def row_feats(r):
    m=int(r["date"].month); c=r["category"]
    return pd.Series({
      "season_index": 1 if cmap.get((c,m),0)==1 else 0,
      "bf_window": int(m in EVENTS["bf_window"]),
      "xmas_window": int(m in EVENTS["xmas_window"]),
      "back_to_school": int(m in EVENTS["back_to_school"]),
      "launch_window": int(m in EVENTS["launch_window"]) if c=="smartwatches" else 0
    })
  df2 = df.copy()
  df2[["season_index","bf_window","xmas_window","back_to_school","launch_window"]] = df2.apply(row_feats, axis=1)
  return df2

def _add_quarter_trend(df: pd.DataFrame, kb_fr: pd.DataFrame, kb_en: pd.DataFrame):
  df2=df.copy()
  df2["quarter"]=df2["date"].apply(quarter_of_date)
  df2["trend_score"]=0.0
  def merge_trend(d, kb):
    if kb is None or kb.empty: return d
    tmp=kb.copy(); tmp["trend_score_add"]=tmp["trend"].map(TREND_SCORE).astype(float)
    return d.merge(tmp[["category","quarter","trend_score_add"]], on=["category","quarter"], how="left")
  df2=merge_trend(df2, kb_fr)
  if "trend_score_add" in df2.columns:
    df2["trend_score"]+=df2["trend_score_add"].fillna(0.0); df2.drop(columns=["trend_score_add"], inplace=True)
  df2=merge_trend(df2, kb_en)
  if "trend_score_add" in df2.columns:
    df2["trend_score"]+=df2["trend_score_add"].fillna(0.0); df2.drop(columns=["trend_score_add"], inplace=True)
  df2["trend_score"]=df2["trend_score"].fillna(0.0).clip(-2,2)
  df2["log_price"]=np.log(np.clip(df2["price"],1e-6,None))
  df2["log_sales"]=np.log(np.clip(df2["sales"],1e-6,None))
  return df2

def enrich(csv_path, de_html, fr_html, en_html) -> tuple[pd.DataFrame, pd.DataFrame]:
  df = load_sales(csv_path)
  kb_de = parse_de_months(de_html)
  kb_fr = parse_fr_quarters(fr_html)
  kb_en = parse_en_quarters(en_html)
  df = _build_month_feats(df, kb_de if not kb_de.empty else pd.DataFrame(columns=["category","months_detected"]))
  df = _add_quarter_trend(df, kb_fr, kb_en)
  kb_facts = []
  for _,r in kb_de.iterrows(): kb_facts.append({"source":"DE","category":r["category"],"type":"monthly_peaks","value":",".join(map(str,r["months_detected"]))})
  for _,r in kb_fr.iterrows(): kb_facts.append({"source":"FR","category":r["category"],"type":r["quarter"],"value":r["trend"]})
  for _,r in kb_en.iterrows(): kb_facts.append({"source":"EN","category":r["category"],"type":r["quarter"],"value":r["trend"]})
  return df.sort_values(["category","product_id","date"]).reset_index(drop=True), pd.DataFrame(kb_facts)
