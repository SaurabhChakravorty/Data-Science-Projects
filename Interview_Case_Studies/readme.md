#  Multi-Touch Attribution Case Study

This project aims to analyze and model the contribution of various marketing channels to customer conversion using multiple attribution techniques — including rule-based, probabilistic, and machine learning models.

---

##  Objective

To determine how different marketing channels contribute to user conversion and provide stakeholders with actionable insights to optimize marketing spend and user journey strategy.

---

## Project Structure

| File / Notebook              | Description |
|-----------------------------|-------------|
| `01_analysis.ipynb`         | Exploratory Data Analysis: touchpoints, channels, conversion time, Sankey flow |
| `02_models.ipynb`           | Attribution models: First/Last click, Markov Chains, BiLSTM with attention, Random Forest |
| `03_results_and_conclusion.ipynb` | Summary of attribution outputs, conversion rates, user share, and final recommendations |
| `main.py`                   | Core logic for Markov, LSTM, RF models, and attribution computation |

---

## Attribution Methods Implemented

| Method         | Description |
|----------------|-------------|
| First Click    | 100% credit to the first touchpoint |
| Last Click     | 100% credit to the last touchpoint |
| Markov Model   | Uses transition probabilities and removal effect to assign credit |
| BiLSTM + Attention | Learns sequential patterns and temporal impact of touchpoints |
| Random Forest  | Non-sequential model capturing feature importance at touchpoint level |

---

## Key Insights

- **Email** and **Direct** are strong closers — high credit in Markov and LSTM.
- **Social** and **Display** often start journeys but show weak direct conversion — awareness drivers.
- **Paid Search** plays a strong assisting role — important for mid-funnel engagement.

---

## Production Considerations

- Model scoring (e.g., LSTM probability) can be applied to real-time or batch journey data.
- Attribution results can be served via dashboards (Looker, Tableau).
- Monitoring via conversion probability drift, attribution shifts, or retrain triggers.

---

## Next Improvements

- Segment models by user type (e.g., new vs. returning)
- Add attribution over time (time series analysis)
- Simulate budget shifts using attribution weights

---

## Summary

This project bridges marketing strategy with data science to help businesses identify and invest in the right channels. By blending interpretability (Markov, RF) with predictive power (LSTM), we ensure stakeholders receive both **explainable** and **actionable** insights.

---

### Author

*Built by Saurabh Charavorty for SUMUP *
