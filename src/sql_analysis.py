import sqlite3

import pandas as pd

from src import config


def run_query(sql, params=()):
    conn = sqlite3.connect(config.db_path)
    df = pd.read_sql(sql, conn, params=params)
    conn.close()
    return df


def monthly_revenue():
    return run_query("""
        SELECT strftime('%Y-%m', invoice_date) AS month,
               ROUND(SUM(revenue), 2)          AS revenue,
               COUNT(DISTINCT invoice_no)      AS orders,
               COUNT(DISTINCT customer_id)     AS customers
        FROM sales
        GROUP BY month
        ORDER BY month
    """)


def top_products(limit=10):
    return run_query("""
        SELECT stock_code,
               MAX(description)       AS description,
               SUM(quantity)          AS units_sold,
               ROUND(SUM(revenue), 2) AS revenue
        FROM sales
        GROUP BY stock_code
        ORDER BY revenue DESC
        LIMIT ?
    """, (limit,))


def revenue_by_country(limit=10):
    return run_query("""
        SELECT country,
               ROUND(SUM(revenue), 2)     AS revenue,
               COUNT(DISTINCT invoice_no) AS orders
        FROM sales
        GROUP BY country
        ORDER BY revenue DESC
        LIMIT ?
    """, (limit,))


def weekly_demand():
    # Units sold per product per week, for the selected products only.
    # Weeks run Monday to Sunday: jump to that week's Sunday, then back 6 days.
    df = run_query("""
        SELECT s.stock_code,
               date(s.invoice_date, 'weekday 0', '-6 days') AS week_start,
               SUM(s.quantity) AS units
        FROM sales s
        JOIN products p ON p.stock_code = s.stock_code
        GROUP BY s.stock_code, week_start
        ORDER BY s.stock_code, week_start
    """)
    df["week_start"] = pd.to_datetime(df["week_start"])
    return df


def main():
    print("Monthly revenue:\n", monthly_revenue(), "\n")
    print("Top products:\n", top_products(), "\n")
    print("Revenue by country:\n", revenue_by_country(), "\n")
    print("Weekly demand:\n", weekly_demand().head())


if __name__ == "__main__":
    main()
