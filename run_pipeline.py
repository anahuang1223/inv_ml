import time

from src import cluster_products, inventory_planning, load_data, sql_analysis, train_models

# Each step reads what the previous one saved to the database, so order matters
steps = [
    ("Load and clean data", load_data.main),
    ("SQL analysis", sql_analysis.main),
    ("Cluster products", cluster_products.main),
    ("Train models", train_models.main),
    ("Plan inventory", inventory_planning.main),
]


def main():
    for name, step in steps:
        print(f"\n=== {name} ===")
        start = time.time()
        step()
        print(f"({time.time() - start:.0f}s)")


if __name__ == "__main__":
    main()
