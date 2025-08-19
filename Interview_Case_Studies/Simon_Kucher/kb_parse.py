import re
import pandas as pd
from bs4 import BeautifulSoup
from pathlib import Path

DE_MONTHS = {"januar":1,"februar":2,"märz":3,"maerz":3,"april":4,"mai":5,"juni":6,
             "juli":7,"august":8,"september":9,"oktober":10,"november":11,"dezember":12}
CAT_KEYS_DE = {"headphones":"Headphones","smartwatches":"Smartwatches","tablets":"Tablets","laptops":"Laptops"}
QTR_REGEX = re.compile(r"(Q[1-4]\s*20(2[3-5]))\s*:\s*(.+?)\.", re.IGNORECASE)
TREND_SCORE = {"down_light":-1,"stable":0,"up_light":+1,"up_moderate":+2,"mixed":0}

def soup(path: str):
  p = Path(path)
  return BeautifulSoup(p.read_text(encoding="utf-8"), "html.parser") if p.exists() else None

def _sections_by_h2(soup):
  sections, cur = {}, None
  for el in soup.find_all(["h2","p"]):
    if el.name=="h2":
      cur = el.get_text(strip=True); sections[cur] = []
    elif el.name=="p" and cur:
      sections[cur].append(el.get_text(" ", strip=True))
  return {k:" ".join(v) for k,v in sections.items()}

def _months_in_de(text:str):
  found, low = set(), text.lower()
  for name,num in DE_MONTHS.items():
    if re.search(rf"\b{name}\b", low): found.add(num)
  return sorted(found)

def parse_de_months(html_path: str) -> pd.DataFrame:
  s = soup(html_path); rows=[]
  if not s: return pd.DataFrame(columns=["category","months_detected"])
  for h2, txt in _sections_by_h2(s).items():
    for canon in CAT_KEYS_DE:
      if h2.lower().startswith(canon):
        rows.append({"category":canon, "months_detected":_months_in_de(txt)})
  return pd.DataFrame(rows)

def _label_fr(t:str)->str:
  t=t.lower()
  if "léger recul" in t or "leger recul" in t or "baisse" in t: return "down_light"
  if "croissance modérée" in t or "croissance moderee" in t: return "up_moderate"
  if "légère croissance" in t or "legere croissance" in t: return "up_light"
  if "stables" in t or "stable" in t: return "stable"
  return "mixed"

def parse_fr_quarters(html_path: str) -> pd.DataFrame:
  s = soup(html_path); rows=[]
  if not s: return pd.DataFrame(columns=["category","quarter","trend"])
  txt = s.get_text(" ", strip=True)
  for qtr,_y,desc in QTR_REGEX.findall(txt):
    for part in [p.strip().lower() for p in desc.split(";")]:
      if "montres" in part: rows.append({"category":"smartwatches","quarter":qtr,"trend":_label_fr(part)})
      if "tablettes" in part: rows.append({"category":"tablets","quarter":qtr,"trend":_label_fr(part)})
  return pd.DataFrame(rows)

def parse_en_quarters(html_path: str) -> pd.DataFrame:
  s = soup(html_path); rows=[]
  if not s: return pd.DataFrame(columns=["category","quarter","trend"])
  txt = s.get_text(" ", strip=True).lower()
  for qtr,_y,desc in QTR_REGEX.findall(txt):
    parts = re.split(r";|,|\bwhile\b", desc)
    def classify(p):
      if "slight decline" in p or "decline slightly" in p: return "down_light"
      if "slight growth" in p or "begin slight growth" in p: return "up_light"
      if "stable to slight decline" in p: return "down_light"
      if "stable to slight growth" in p: return "up_light"
      if "stable" in p: return "stable"
      return "mixed"
    tags={}
    for p in parts:
      p=p.strip()
      if "headphones" in p: tags["headphones"]=classify(p)
      if "laptops" in p: tags["laptops"]=classify(p)
    for cat,tag in tags.items():
      rows.append({"category":cat,"quarter":qtr,"trend":tag})
  return pd.DataFrame(rows)

def quarter_of_date(dt): return f"Q{((dt.month-1)//3)+1} {dt.year}"
