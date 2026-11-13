from pathlib import Path

# File locations
root = Path(__file__).resolve().parent.parent
data_dir = root / "data"
raw_xlsx = data_dir / "online_retail_II.xlsx"
db_path = data_dir / "inventory.db"
model_dir = root / "models"

# How many top-selling products to work with (project scope: 50-100)
n_products = 75

# Supplier lead time: stock has to cover total demand over these weeks
lead_time_weeks = 4

# Hold out the last 12 weeks to test on
test_weeks = 12

# Last training weeks used to tune models, so the test weeks stay unseen
validation_weeks = 12

# Product groups by demand pattern
n_clusters = 3
