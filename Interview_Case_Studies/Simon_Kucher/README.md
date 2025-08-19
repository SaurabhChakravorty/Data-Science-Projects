# Case Study: Pricing Elasticity & Scenario Analysis

## Project Overview
This project analyzes **pricing elasticity** and evaluates **Conservative, Standard, and Optimistic scenarios** for product categories including **Headphones, Laptops, Tablets, and Smartwatches**.  

It combines:
- Historical **sales and pricing data** (`dataset.csv`)
- Market insights from articles
- Elasticity-driven **scenario modeling**
- Visualization in **Looker Studio**

---

## Project Structure
```
pricing_lab/
├── data                    # Enriched dataset with prices, sales, categories, elasticity
├── main.py                  # Orchestrates the workflow
├── elasticity.py            # Runs elasticity regressions and outputs coefficients
├── scenario.py              # Generates conservative/standard/optimistic scenarios
├── simulate.py              # Simulate scenarios
├── README.md                # Project documentation
└── requirements.txt         # Python dependencies
└── outputs                  # output files
├── config.yaml              # yaml files with configuration
```

---

## ⚙️ Setup Instructions

### 1. Environment Setup
```bash
git clone <repo-url>
cd pricing_lab
python3 -m venv .venv
source .venv/bin/activate   # Mac/Linux
.venv\Scripts\activate      # Windows
pip install -r requirements.txt
```

### 2. Run the Scripts
**Elasticity modeling**
```bash
python elasticity.py
```
- Fits regression models (`log(sales) ~ log(price)`).
- Saves elasticity coefficients.

**Scenario generation**
```bash
python scenario.py
```
- Builds **Conservative / Standard / Optimistic** scenario outputs.
- Produces:
  - `scenario_conservative.csv`
  - `scenario_standard.csv`
  - `scenario_optimistic.csv`

**Main orchestration**
```bash
python main.py
```
- Runs the pipeline end-to-end.
- Produces enriched dataset ready for Looker Studio.

---

## Looker Studio Setup

### Step 1: Upload Data
Upload the 3 scenario CSVs into Looker Studio with BigQuery

### Step 2: Create Pages
- Refer to the file

### Step 3: Filtering
- Add **Category filter** (Headphones, Laptops, Tablets, Smartwatches).
- Add **Date filter** if needed.

---

## Deliverables
- **Enriched Dataset**: `dataset.csv`  
- **Scenario Outputs**: Conservative, Standard, Optimistic CSVs  
- **Looker Studio Dashboard**:  
  - Page 1: Business Overview  
  - Page 2: Scenario Comparison  

---
