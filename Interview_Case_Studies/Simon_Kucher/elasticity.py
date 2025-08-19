import numpy as np, pandas as pd, statsmodels.api as sm

def _ensure_flags(df):
  for c in ["season_index","bf_window","xmas_window","back_to_school","launch_window","trend_score"]:
    if c not in df.columns: df[c]=0
  return df

def _fit(sub: pd.DataFrame):
  X = pd.DataFrame({
    "log_price": sub["log_price"],
    "season_index": sub["season_index"],
    "bf_window": sub["bf_window"],
    "xmas_window": sub["xmas_window"],
    "back_to_school": sub["back_to_school"],
    "launch_window": sub["launch_window"],
    "trend_score": sub["trend_score"],
  })
  sub2=sub.copy()
  sub2["date"]=pd.to_datetime(sub2["date"], errors="coerce")
  X = pd.concat([X, pd.get_dummies(sub2["date"].dt.month, prefix="m", drop_first=True)], axis=1)
  X = X.apply(pd.to_numeric, errors="coerce").replace([np.inf,-np.inf], np.nan)
  y = pd.to_numeric(sub2["log_sales"], errors="coerce").replace([np.inf,-np.inf], np.nan)
  data = pd.concat([y,X], axis=1).dropna()
  if len(data) < 5: raise ValueError("Too few rows after cleaning")
  yv = data.iloc[:,0].astype(float).values
  Xv = sm.add_constant(data.iloc[:,1:].astype(float), has_constant="add").values
  return sm.OLS(yv, Xv).fit()

def fit_elasticities(df, min_rows=10, min_levels=4):
  df = df.dropna(subset=["sku_id","category","price","sales","date"]).copy()
  df = df[(df["price"]>0) & (df["sales"]>0)]
  df["log_price"]=np.log(df["price"].astype(float))
  df["log_sales"]=np.log(df["sales"].astype(float))
  df=_ensure_flags(df)

  rows=[]
  for sku,g in df.groupby("sku_id"):
    if len(g)>=min_rows and g["price"].nunique()>=min_levels:
      try:
        m=_fit(g); beta=m.params[1]  # first after const is log_price
        rows.append({"sku_id":sku,"category":g["category"].iloc[0],"level":"sku",
                     "n_obs":len(g),"price_levels":g["price"].nunique(),
                     "elasticity":float(beta),"r2":float(m.rsquared)})
      except Exception:
        pass

  covered={r["sku_id"] for r in rows}
  rem=df[~df["sku_id"].isin(covered)]
  if not rem.empty:
    for cat,g in rem.groupby("category"):
      try:
        m=_fit(g); beta=m.params[1]
        for sku in g["sku_id"].unique():
          rows.append({"sku_id":sku,"category":cat,"level":"category",
                       "n_obs":int(df[df["sku_id"]==sku].shape[0]),
                       "price_levels":int(df[df["sku_id"]==sku]["price"].nunique()),
                       "elasticity":float(beta),"r2":float(m.rsquared)})
      except Exception:
        pass

  out=pd.DataFrame(rows)
  if not out.empty:
    out["elasticity"]=out["elasticity"].clip(-5,-0.05)
    return out.sort_values(["category","sku_id"]).reset_index(drop=True)
  return out
