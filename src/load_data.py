import sqlite3

import pandas as pd

from src import config


def load_raw():
    # Read codes as text so values like "85123A" and leading zeros stay intact
    return pd.read_excel(config.raw_xlsx, dtype={"InvoiceNo": str, "StockCode": str})


def clean(df):
    df = df.rename(columns={
        "InvoiceNo": "invoice_no",
        "StockCode": "stock_code",
        "Description": "description",
        "Quantity": "quantity",
        "InvoiceDate": "invoice_date",
        "UnitPrice": "unit_price",
        "CustomerID": "customer_id",
        "Country": "country",
    })

    # Cancelled orders have invoice numbers starting with "C"
    df = df[~df["invoice_no"].str.startswith("C")]

    # Quantity <= 0 means a return; price <= 0 means a free item or adjustment
    df = df[(df["quantity"] > 0) & (df["unit_price"] > 0)]

    # Real products have codes starting with 5 digits (e.g. 22423, 85123A);
    # codes like POST, DOT, M are postage, fees, or manual entries
    df = df[df["stock_code"].str.match(r"^\d{5}")].copy()

    # Some codes appear in both cases ("47566b" vs "47566B"); merge them
    df["stock_code"] = df["stock_code"].str.upper()
    # Remove stray spaces so the same product doesn't get two descriptions
    df["description"] = df["description"].str.strip()
    df["revenue"] = df["quantity"] * df["unit_price"]
    return df.reset_index(drop=True)


def test_start(dates):
    """Monday the test period begins: the last test_weeks full weeks."""
    # Data ends on a Friday, so the final week is partial and is left out.
    # Counting back from that week's Monday keeps the split on week boundaries.
    last_week_start = dates.max().to_period("W").start_time
    return last_week_start - pd.Timedelta(weeks=config.test_weeks)


def pick_top_products(df):
    # Ranking on the test weeks would leak future sales into the selection
    history = df[df["invoice_date"] < test_start(df["invoice_date"])]

    history = history.assign(week=history["invoice_date"].dt.to_period("W"))
    products = history.groupby("stock_code").agg(
        # A product can have slightly different descriptions; use the most common
        description=("description", lambda s: s.mode().iloc[0]),
        active_weeks=("week", "nunique"),
        total_quantity=("quantity", "sum"),
        total_revenue=("revenue", "sum"),
        avg_price=("unit_price", "mean"),
    )

    # Prefer products that sell every week (easier to forecast),
    # then the highest revenue among those (matter most to the business)
    products = products.sort_values(["active_weeks", "total_revenue"], ascending=False)
    return products.head(config.n_products).reset_index()


def save_to_db(sales, products):
    conn = sqlite3.connect(config.db_path)

    sales.to_sql("sales", conn, if_exists="replace", index=False)
    products.to_sql("products", conn, if_exists="replace", index=False)

    conn.execute("CREATE INDEX idx_sales_code ON sales(stock_code)")
    conn.execute("CREATE INDEX idx_sales_date ON sales(invoice_date)")
    conn.commit()
    conn.close()


def main():
    print("Loading Excel file...")
    raw = load_raw()
    sales = clean(raw)
    products = pick_top_products(sales)
    save_to_db(sales, products)
    print(f"Raw rows: {len(raw):,} -> clean rows: {len(sales):,}")
    print(f"Saved {len(products)} top products to {config.db_path}")


if __name__ == "__main__":
    main()
