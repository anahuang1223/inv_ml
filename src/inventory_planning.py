import sqlite3
from statistics import NormalDist

import numpy as np
import pandas as pd

from src import build_features, config, sql_analysis

# Weeks of demand each order has to cover
cover_weeks = config.lead_time_weeks + config.review_weeks


def order_up_to_levels():
    # Each policy sets a target inventory position per product and week
    levels = sql_analysis.run_query(
        'SELECT stock_code, week_start, q90, q95 FROM forecasts WHERE "window" = ?', (cover_weeks,)
    )
    levels["week_start"] = pd.to_datetime(levels["week_start"])

    # Textbook rule for comparison: assumes normally distributed weekly demand,
    # mean cover + z * std * sqrt(weeks) for 95% service. The spread uses 12
    # weeks of history, since 4 weeks badly understates it; the level uses the
    # last 4 weeks so it keeps up with changes in demand.
    weekly = build_features.build_features(1).sort_values(["stock_code", "week_start"])
    past = weekly.groupby("stock_code")["units"].shift(1)
    weekly["rolling_std_12"] = past.groupby(weekly["stock_code"]).transform(lambda s: s.rolling(12).std())
    # Simulate from the first forecast week: validation weeks first, then test
    weekly = weekly[weekly["week_start"] >= levels["week_start"].min()]
    weekly["period"] = np.where(weekly["is_test"], "test", "validation")
    # Calculate textbook target = expected 5-week demand + 95% safety stock.
    z = NormalDist().inv_cdf(0.95)
    weekly = weekly.assign(normal_95=weekly["rolling_mean_4"] * cover_weeks
                           + z * weekly["rolling_std_12"] * cover_weeks ** 0.5)

    levels = levels.merge(weekly[["stock_code", "week_start", "normal_95"]], on=["stock_code", "week_start"])
    levels = levels.rename(columns={"q90": "quantile_90", "q95": "quantile_95"})
    return levels, weekly[["stock_code", "week_start", "units", "period"]]


def simulate(demand, targets, start_factor=1):
    # Weekly review, order-up-to policy with backorders. Orders compare the
    # target with inventory position (on hand + on order - backorders), not
    # just on hand, otherwise each shortfall would be reordered every week
    # while earlier orders are still on the way.
    # Start at the first target, all of it on the shelf. Nothing ordered by the
    # policy arrives before lead_time_weeks, so those first weeks are warm-up.
    on_hand = np.ceil(targets.iloc[0] * start_factor)
    backorders = 0
    arrivals = {}
    rows = []
    for week, units in demand.items():
        on_hand += arrivals.pop(week, 0)

        # Order at the start of the week; it arrives lead_time_weeks later
        on_order = sum(arrivals.values())
        position = on_hand + on_order - backorders
        # Orders are whole units, rounded up so the target is still reached
        order = max(np.ceil(targets[week] - position), 0) if week in targets else 0
        # Schedule the order amt to arrive after the 4-week supplier lead time.
        if order:
            arrivals[week + pd.Timedelta(weeks=config.lead_time_weeks)] = order

        # Waiting backorders are filled first, then this week's demand
        owed = backorders + units
        shipped = min(on_hand, owed)
        on_hand -= shipped
        backorders = owed - shipped
        # Units of this week's demand sent straight from the shelf: shipped - (previous backorders)
        filled_now = max(shipped - (owed - units), 0)

        rows.append({"week_start": week, "demand": units, "order": order, "on_hand": on_hand,
                     "backorders": backorders, "filled_now": filled_now})
    sim = pd.DataFrame(rows)
    # Mark the first 4 weeks as warm-up while waiting for initial orders to arrive.
    sim["warmup"] = sim["week_start"] < demand.index[0] + pd.Timedelta(weeks=config.lead_time_weeks)
    return sim


def run_policies(start_factor=1):
    levels, weekly = order_up_to_levels()
    prices = sql_analysis.run_query("SELECT stock_code, avg_price FROM products")
    policies = ["quantile_90", "quantile_95", "normal_95"]

    results = []
    # code is the product ID, product contains all its weekly rows.
    for code, product in weekly.groupby("stock_code"):
        demand = product.set_index("week_start")["units"].sort_index()
        product_levels = levels[levels["stock_code"] == code].set_index("week_start").sort_index()
        # Simulate all three policies for each product and label the results.
        for policy in policies:
            sim = simulate(demand, product_levels[policy], start_factor)
            results.append(sim.assign(stock_code=code, policy=policy))
    # Combine all simulations and attach avg product prices and evaluation periods.
    sim = pd.concat(results, ignore_index=True).merge(prices, on="stock_code")
    sim = sim.merge(weekly[["stock_code", "week_start", "period"]], on=["stock_code", "week_start"])
    # Calculate weekly holding cost for inventory remaining on the shelf.
    sim["holding_cost"] = sim["on_hand"] * sim["avg_price"] * config.holding_rate / 52
    # Value of demand not shipped in the week it was ordered
    sim["short_value"] = (sim["demand"] - sim["filled_now"]) * sim["avg_price"]
    return sim


def summarize(sim, by):
    # Exclude the first 4 warm-up weeks from performance calculations.
    # Warm-up weeks run on the assumed starting stock rather than the policy
    sim = sim[~sim["warmup"]]
    summary = sim.groupby(by).agg(
        demand=("demand", "sum"),
        filled_now=("filled_now", "sum"),
        avg_on_hand=("on_hand", "mean"),
        avg_backorders=("backorders", "mean"),
        holding_cost=("holding_cost", "sum"),
        short_value=("short_value", "sum"),
        # Calculate the share of product-weeks ending with unfilled demand.
        stockout_weeks=("backorders", lambda b: (b > 0).mean()),
    )
    # Fill rate: share of demand shipped from the shelf in the week it was ordered
    summary["fill_rate"] = summary.pop("filled_now") / summary["demand"]
    short_value = summary.pop("short_value")
    # Calculate total inventory cost under different shortage penalties.
    for m in config.shortage_multiples:
        summary[f"total_cost_{m}x"] = summary["holding_cost"] + m * short_value
    return summary.reset_index()


def stock_balance_holds(sim):
    # Stock on hand minus backorders must equal starting stock plus arrivals
    # minus all demand so far, every week, including across validation -> test
    def holds(g):
        g = g.sort_values("week_start")
        # Recover starting inventory from the first week's ending balance.
        start = g["on_hand"].iloc[0] - g["backorders"].iloc[0] + g["demand"].iloc[0]
        arrived = g["order"].shift(config.lead_time_weeks, fill_value=0).cumsum()
        expected = start + arrived - g["demand"].cumsum()
        # Return True if all values are approximately equal, otherwise False if any differ beyond tolerance.
        return np.allclose(g["on_hand"] - g["backorders"], expected)

    return sim.groupby(["stock_code", "policy"]).apply(holds).all()


def save_results(sim, summary):
    conn = sqlite3.connect(config.db_path)
    sim.to_sql("inventory_sim", conn, if_exists="replace", index=False)
    summary.to_sql("inventory_summary", conn, if_exists="replace", index=False)
    conn.close()


def main():
    sim = run_policies()
    assert stock_balance_holds(sim), "stock balance check failed"
    clusters = sql_analysis.run_query("SELECT stock_code, cluster FROM product_clusters")
    # Add a column that contains each product's cluster num to compare inventory performance by segment.
    sim = sim.merge(clusters, on="stock_code")
    # Test weeks are the main result; validation weeks helped tune the models
    summary = pd.concat([
        # Summarize inventory performance across all products by period and policy.
        summarize(sim, ["period", "policy"]).assign(cluster="all"),
        # Summarize inventory performance by period, policy, and product cluster.
        summarize(sim, ["period", "policy", "cluster"]).astype({"cluster": str}),
    ], ignore_index=True)
    save_results(sim, summary)
    print(summary.round(3).to_string(index=False), "\n")
    # Test how different starting stock amounts affect inventory performance.
    # Results should barely move with the starting stock if the warm-up is long enough
    sensitivity = pd.concat([
        summarize(run_policies(f), ["period", "policy"]).assign(start_factor=f)
        for f in config.start_stock_factors
    ])
    sensitivity = sensitivity[sensitivity["period"] == "test"]
    print("Starting stock sensitivity (test weeks):")
    # pivot so that policies are rows, starting stock factors are columns,
    print(sensitivity.pivot(index="policy", columns="start_factor",
                            values=["fill_rate", "total_cost_1x"]).round(3).to_string())


if __name__ == "__main__":
    main()
