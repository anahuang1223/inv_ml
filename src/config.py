from pathlib import Path

# File locations
root = Path(__file__).resolve().parent.parent
data_dir = root / "data"
raw_xlsx = data_dir / "online_retail_II.xlsx"
db_path = data_dir / "inventory.db"
# Small copy with only what the dashboard reads, committed so the app can be deployed
dashboard_db_path = data_dir / "dashboard.db"
model_dir = root / "models"

# How many top-selling products to work with (project scope: 50-100)
n_products = 75

# Supplier lead time: weeks between placing an order and receiving it
lead_time_weeks = 4

# Stock is checked and reordered once a week, so an order has to cover the
# lead time plus the week until the next chance to order
review_weeks = 1

# Yearly cost of holding stock, as a share of its value
holding_rate = 0.25

# Cost of each unit not shipped on time, as a multiple of its price. No single
# value is known, so policies are compared across a range: 0.25 is roughly the
# lost margin, higher values add lost goodwill and customers
shortage_multiples = [0.25, 1, 4]

# Hold out the last 12 weeks to test on
test_weeks = 12

# Last training weeks used to tune models, so the test weeks stay unseen
validation_weeks = 12

# Product groups by demand pattern
n_clusters = 3

# Starting stock as a share of the first target, to check results don't
# depend on the assumed starting point
start_stock_factors = [0.5, 1, 1.5]
