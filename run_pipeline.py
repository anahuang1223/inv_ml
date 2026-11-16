import sqlite3
import time

import pandas as pd

from src import cluster_products, config, inventory_planning, load_data, sql_analysis, train_models

# Tables the dashboard reads; the million-row sales table is left out
dashboard_tables = ["products", "product_clusters", "forecasts", "model_scores", "quantile_scores",
                    "inventory_sim", "inventory_summary"]


def export_dashboard_data():
    # Under 1 MB instead of ~80 MB, small enough to commit and deploy
    config.dashboard_db_path.unlink(missing_ok=True)
    source = sqlite3.connect(config.db_path)
    target = sqlite3.connect(config.dashboard_db_path)
    for table in dashboard_tables:
        pd.read_sql(f"SELECT * FROM {table}", source).to_sql(table, target, index=False)
    # Sales charts only need these summaries
    sql_analysis.monthly_revenue().to_sql("monthly_revenue", target, index=False)
    sql_analysis.top_products().to_sql("top_products", target, index=False)
    sql_analysis.revenue_by_country().to_sql("revenue_by_country", target, index=False)
    source.close()
    target.close()
    print(f"Saved {config.dashboard_db_path}")


# Each step reads what the previous one saved to the database, so order matters
steps = [
    ("Load and clean data", load_data.main),
    ("SQL analysis", sql_analysis.main),
    ("Cluster products", cluster_products.main),
    ("Train models", train_models.main),
    ("Plan inventory", inventory_planning.main),
    ("Export dashboard data", export_dashboard_data),
]


def main():
    for name, step in steps:
        print(f"\n=== {name} ===")
        start = time.time()
        step()
        print(f"({time.time() - start:.0f}s)")


if __name__ == "__main__":
    main()
