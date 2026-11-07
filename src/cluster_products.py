import sqlite3

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from src import build_features, config, sql_analysis


def product_profiles():
    # Only training weeks, so test demand can't influence the groups
    df = build_features.build_features()
    units = df[~df["is_test"]].groupby("stock_code")["units"]
    profiles = pd.DataFrame({
        "mean_units": units.mean(),
        # Relative volatility, so busy and quiet products can be compared
        "cv": units.std() / units.mean(),
    })
    prices = sql_analysis.run_query("SELECT stock_code, avg_price FROM products")
    return profiles.join(prices.set_index("stock_code")).reset_index()


def assign_clusters(profiles):
    # volume and price are heavily skewed; logs stop a few big products dominating
    x = pd.DataFrame({
        "log_units": np.log(profiles["mean_units"]),
        "cv": profiles["cv"],
        "log_price": np.log(profiles["avg_price"]),
    })
    # KMeans works on distances, so features need a shared scale
    x = StandardScaler().fit_transform(x)
    # group products into 3 clusters based on demand, volatility, and price
    kmeans = KMeans(n_clusters=config.n_clusters, n_init=10, random_state=42)
    # Return the profiles DataFrame with a new "cluster" column (0, 1, or 2) for each product
    return profiles.assign(cluster=kmeans.fit_predict(x))


def save_clusters(df):
    conn = sqlite3.connect(config.db_path)
    df.to_sql("product_clusters", conn, if_exists="replace", index=False)
    conn.close()


def main():
    df = assign_clusters(product_profiles())
    save_clusters(df)
    summary = df.groupby("cluster").agg(
        products=("stock_code", "count"),
        mean_units=("mean_units", "mean"),
        cv=("cv", "mean"),
        avg_price=("avg_price", "mean"),
    )
    print(summary.round(2))


if __name__ == "__main__":
    main()
