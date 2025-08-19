import sys, os
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent))              # add project root
sys.path.append(str(Path(__file__).resolve().parent / "scripts"))  # (optional) add scripts

print("Step 1: Enrichment...")
os.system("python 01_enrich.py")
print("Step 2: Elasticity Modeling...")
os.system("python 02_fit_elasticity.py")
print("Step 3: Scenario Simulation...")
os.system("python 03_simulate.py")
print("All steps completed. Check the outputs/ folder.")
