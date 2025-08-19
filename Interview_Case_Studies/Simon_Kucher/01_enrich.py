from pathlib import Path
from pricing_lab.io import read_cfg, ensure_dir, write_csv
from pricing_lab.enrich import enrich

cfg = read_cfg()
out_dir = Path(cfg["paths"]["out_dir"]); ensure_dir(out_dir)

df_enriched, kb = enrich(cfg["paths"]["csv"], cfg["paths"]["de_html"], cfg["paths"]["fr_html"], cfg["paths"]["en_html"])
write_csv(df_enriched, out_dir/"enriched_dataset.csv")
write_csv(kb, out_dir/"kb_facts.csv")
print("Enriched -> outputs/enriched_dataset.csv ; KB -> outputs/kb_facts.csv")
