# Demand Forecasting & Inventory Planning

Forecasts weekly demand for a UK online gift wholesaler and turns the forecasts into reorder decisions, then simulates those decisions to see what they would have cost.

**Result:** on the 12 held-out test weeks, ordering up to the 95th-percentile demand forecast met **93.9%** of demand straight from the shelf, against **78.5%** for the textbook safety-stock formula, at **69% lower total cost**. It had the lowest cost under every shortage-cost assumption tested.

## Data

[Online Retail II](https://archive.ics.uci.edu/dataset/502/online+retail+ii) (UCI Machine Learning Repository): every transaction of a UK online gift wholesaler from 1 Dec 2009 to 9 Dec 2011, about 1 million rows.

- Cancellations, returns, zero-price rows and non-product codes (postage, fees) are removed, leaving 1,014,945 rows.
- The two yearly sheets overlap on 1–9 Dec 2010; the overlap is kept once.
- The 75 products used are the ones that sold in the most weeks, then by revenue. They are picked using only data before the test period.

## Approach

| Step | File | What it does |
|---|---|---|
| Load | `src/load_data.py` | Clean the raw Excel file, pick products, save to SQLite |
| SQL analysis | `src/sql_analysis.py` | Revenue summaries and weekly demand per product |
| Features | `src/build_features.py` | Lags, 4-week rolling mean/median/std, same weeks last year; weeks with no sales filled with zero |
| Clustering | `src/cluster_products.py` | KMeans on volume, volatility and price (3 groups) |
| Models | `src/train_models.py` | Point and quantile forecasts, tuning, scoring |
| Inventory | `src/inventory_planning.py` | Reorder policies simulated week by week |
| Export | `run_pipeline.py` | Copy the tables the dashboard needs into a small `data/dashboard.db` |
| Dashboard | `dashboard.py` | Streamlit app for all of the above |

**Time split.** The last 12 full weeks (12 Sep – 28 Nov 2011) are the test period. The 12 weeks before are the validation period, used to tune models. Every model is retrained each week on data available at that point (walk-forward), so no forecast sees the future.

**Forecasting.**
- Models predict demand as a ratio of the recent 4-week average rather than raw units. Tree models can't predict beyond the range they trained on; a ratio still works when demand climbs before Christmas.
- Point forecasts: linear regression (Ridge) and gradient boosting (absolute-error loss, to resist bulk orders), compared against naive and rolling-average baselines.
- Quantile forecasts: gradient boosting at the 90th and 95th percentiles, for safety stock.
- Two targets: next week's demand (for accuracy), and total demand over the next 5 weeks (for reordering, see below).

**Inventory policy.** Stock is reviewed weekly and orders arrive after a 4-week lead time, so each order has to cover 5 weeks. Each week, if the inventory position (on hand + on order − backorders) is below the target, the difference is ordered. Three ways of setting the target are compared:

1. **95th-percentile forecast** of 5-week demand
2. **90th-percentile forecast** of 5-week demand
3. **Textbook formula:** 5 × recent weekly mean + 1.645 × 12-week standard deviation × √5, which assumes normally distributed demand

Unmet demand is backordered. Holding cost is 25% of stock value per year. Shortage cost has no single right value, so it is tested at 0.25×, 1× and 4× the item's price per unit not shipped on time.

## Results

### Forecast accuracy (test weeks)

| Model | MAE, next week | MAE, next 5 weeks | WAPE, next 5 weeks | Bias, next 5 weeks |
|---|---|---|---|---|
| Gradient boosting | 112.7 | 411.6 | 0.36 | −126 |
| Linear regression | 116.8 | 422.4 | 0.37 | −48 |
| 4-week median | 118.5 | 507.6 | 0.45 | −196 |
| 4-week mean | 120.5 | 499.0 | 0.44 | −91 |
| Last week | 136.7 | 648.1 | 0.57 | −64 |

The models beat the baselines, but not by much week to week: much of the remaining error comes from one-off bulk orders that past sales can't predict. Gradient boosting has the lowest error but under-forecasts, because absolute-error loss predicts the typical week rather than the average.

The quantile forecasts are well calibrated. Over 5 weeks, actual demand stayed under the 95th-percentile forecast **95.0%** of the time and under the 90th **90.7%** of the time. The spiky product group was slightly under (93.8% at the 95th).

### Inventory simulation

| Policy | Fill rate | Weeks with a stockout | Total cost at 0.25× | at 1× | at 4× |
|---|---|---|---|---|---|
| **95th-percentile forecast** | **93.9%** | **4.1%** | **£15,254** | **£28,532** | **£81,642** |
| 90th-percentile forecast | 86.8% | 7.3% | £17,530 | £47,750 | £168,631 |
| Textbook formula | 78.5% | 18.7% | £28,252 | £92,232 | £348,151 |

Test weeks. Fill rate = share of demand shipped from the shelf in the week it was ordered.

- The 95th-percentile policy holds more stock, but stockouts cost far more than holding, even at the lowest shortage cost tested.
- The same ranking holds on the validation weeks (fill rate 93.3% / 88.2% / 86.4%).
- The textbook formula underperforms substantially on this data, likely because it doesn't capture the demand patterns here (bulk orders, the pre-Christmas rise). That wasn't tested directly.

### Checks

- **Starting stock:** the simulation starts in the validation period, so the first weeks act as warm-up. Re-running with 0.5× and 1.5× the starting stock leaves the test results essentially unchanged (fill rates within 0.2 points).
- **Stock accounting:** for every product, policy and week, stock on hand minus backorders equals starting stock plus deliveries minus total demand. The pipeline stops if this fails.

## Limitations

- **The test set is not untouched.** Hyperparameters were tuned on validation weeks only, but model design choices (features, target, loss) were explored with test results visible, so test scores may be slightly optimistic. The policy ranking is confirmed on the validation weeks, which test results didn't influence. Validation results are not an independent holdout either.
- **One test period.** The test weeks are the run-up to Christmas, the hardest stretch of the year. Results may differ in calmer periods.
- **Simulated, not real, inventory.** The data has no stock levels. Lead time is fixed at 4 weeks, there are no order minimums or batch sizes, and costs are assumptions.
- **Small groups.** The spiky product group has only 6 products, so its results are noisy.

## Running it

Requires Python 3.10+.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

To view the dashboard straight away, without the raw data:

```bash
streamlit run dashboard.py    # opens at http://localhost:8501
```

It reads `data/dashboard.db`, a small copy of the results that is committed to the repo. The same file lets the app deploy to Streamlit Community Cloud as-is.

To rebuild everything from the raw data, download `online_retail_II.xlsx` from the [UCI page](https://archive.ics.uci.edu/dataset/502/online+retail+ii) into `data/`, then:

```bash
python run_pipeline.py        # about 3 minutes; rebuilds the database, models and dashboard.db
```

Settings (lead time, test length, costs, number of products) are in `src/config.py`.

## Project structure

```
├── run_pipeline.py       # runs every step in order
├── dashboard.py          # Streamlit dashboard
├── requirements.txt
├── .streamlit/           # dashboard theme (dark)
├── data/
│   └── dashboard.db      # small results database for the dashboard (in git)
│                         # raw Excel file and full database are not in git
├── models/               # trained models (not in git)
└── src/
    ├── config.py
    ├── load_data.py
    ├── sql_analysis.py
    ├── build_features.py
    ├── cluster_products.py
    ├── train_models.py
    └── inventory_planning.py
```
