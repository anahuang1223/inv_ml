import sqlite3

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src import config

series_colors = ["#d9772b", "#8b5a3c", "#e3cfa6"]
ink = "#898781"

policy_names = {
    "quantile_95": "95th percentile forecast",
    "quantile_90": "90th percentile forecast",
    "normal_95": "Textbook formula (95%)",
}
policy_colors = dict(zip(policy_names.values(), series_colors))
cover_weeks = config.lead_time_weeks + config.review_weeks

st.set_page_config(page_title="Inventory Planning", layout="wide")


@st.cache_data
def query(sql, params=()):
    # Reads the slim dashboard copy of the database, so the app can be deployed
    conn = sqlite3.connect(config.dashboard_db_path)
    df = pd.read_sql(sql, conn, params=params)
    conn.close()
    if "week_start" in df:
        df["week_start"] = pd.to_datetime(df["week_start"])
    return df


def style(fig, height=380, hovermode="closest"):
    # Extra top margin keeps the title clear of the legend sitting above the plot
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=90, b=10),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, title=None),
                      hovermode=hovermode)
    fig.update_traces(selector=dict(type="scatter"), line_width=2)
    return fig


def show(container, fig):
    container.plotly_chart(fig, width="stretch", config={"displayModeBar": False})


def mark_test_start(fig, test_start):
    fig.add_vline(x=test_start, line_dash="dot", line_color=ink, line_width=1)
    fig.add_annotation(x=test_start, y=1, yref="paper", text="Test weeks start",
                       showarrow=False, xanchor="left", font_color=ink)


products = query("SELECT stock_code, description FROM products")
labels = dict(zip(products["stock_code"], products["stock_code"] + " – " + products["description"]))
summary = query("SELECT * FROM inventory_summary")
summary["policy"] = summary["policy"].map(policy_names)
test_start = query("SELECT MIN(week_start) AS week_start FROM forecasts WHERE period = 'test'")["week_start"][0]

# Cluster numbers are arbitrary and can change between runs, so names come from each
# group's profile: the most volatile group is "spiky", the busiest of the rest "high volume"
profiles = query("SELECT cluster, AVG(mean_units) AS units, AVG(cv) AS cv FROM product_clusters GROUP BY cluster")
profiles = profiles.set_index("cluster")
spiky = profiles["cv"].idxmax()
busy = profiles.drop(spiky)["units"].idxmax()
cluster_names = {c: f"Group {c}: lower volume" for c in profiles.index}
cluster_names[spiky] = f"Group {spiky}: spiky demand"
cluster_names[busy] = f"Group {busy}: high volume"

st.title("Demand Forecasting & Inventory Planning")
st.caption(f"Online Retail II (UK gift wholesaler, Dec 2009 – Dec 2011) · top {config.n_products} products · "
           f"{config.lead_time_weeks}-week supplier lead time, weekly reordering")

# Headline: the chosen policy on the held-out test weeks
test_all = summary[(summary["period"] == "test") & (summary["cluster"] == "all")].set_index("policy")
q95, textbook = test_all.loc[policy_names["quantile_95"]], test_all.loc[policy_names["normal_95"]]
cols = st.columns(4)
# The difference between two rates is in percentage points, not percent
cols[0].metric("Fill rate (test)", f"{q95['fill_rate']:.1%}",
               f"{(q95['fill_rate'] - textbook['fill_rate']) * 100:+.1f} pts vs textbook")
cols[1].metric("Product-weeks with backorders", f"{q95['stockout_weeks']:.1%}",
               help="Share of product-weeks that ended with customers still waiting for stock.")
cols[2].metric("Total cost, shortage at 1× price", f"£{q95['total_cost_1x']:,.0f}")
cols[3].metric("Cost vs textbook formula", f"{q95['total_cost_1x'] / textbook['total_cost_1x'] - 1:+.0%}",
               delta_color="off")
st.caption(
    "Simulated results under assumed costs, not measured savings. "
    f"Policy: order up to the 95th-percentile forecast of demand over the next {cover_weeks} weeks. "
    f"Stock is reviewed weekly and orders arrive after {config.lead_time_weeks} weeks; unmet demand is backordered. "
    f"Holding cost is {config.holding_rate:.0%} of stock value per year. Each unit not shipped on time costs "
    f"{', '.join(f'{m}×' for m in config.shortage_multiples)} its price (headline uses 1×)."
)

inventory_tab, forecast_tab, product_tab, sales_tab = st.tabs(
    ["Inventory policies", "Forecasts", "Product groups", "Sales"]
)

with inventory_tab:
    period = st.radio("Period", ["test", "validation"], horizontal=True,
                      format_func={"test": "Test (held out)", "validation": "Validation (used for tuning)"}.get)
    rows = summary[(summary["period"] == period) & (summary["cluster"] == "all")]

    left, right = st.columns(2)
    fig = px.bar(rows, x="fill_rate", y="policy", color="policy", orientation="h",
                 color_discrete_map=policy_colors, text_auto=".1%",
                 title="Fill rate: demand shipped from the shelf the same week")
    fig.update_layout(showlegend=False, xaxis_tickformat=".0%", xaxis_title=None, yaxis_title=None)
    show(left, style(fig, 300))

    costs = rows.melt(id_vars="policy", value_vars=[f"total_cost_{m}x" for m in config.shortage_multiples],
                      var_name="shortage", value_name="cost")
    costs["shortage"] = costs["shortage"].str.removeprefix("total_cost_").str.replace("x", "× price")
    # Costs differ by an order of magnitude across shortage levels, so each gets its own scale
    fig = px.bar(costs, x="policy", y="cost", color="policy", facet_col="shortage",
                 color_discrete_map=policy_colors, facet_col_spacing=0.1,
                 title="Total cost (holding + shortage) by shortage cost")
    fig.update_yaxes(matches=None, showticklabels=True, title=None, tickprefix="£")
    fig.update_xaxes(showticklabels=False, title=None)
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    # Legend below, so it doesn't collide with the shortage-level labels above each panel
    show(right, style(fig, 340).update_layout(legend=dict(yanchor="top", y=-0.05)))

    by_cluster = summary[summary["period"] == period].sort_values(["cluster", "policy"])
    by_cluster["cluster"] = by_cluster["cluster"].map(
        lambda c: "All products" if c == "all" else cluster_names[int(c)])
    cost_cols = ["holding_cost", *[f"total_cost_{m}x" for m in config.shortage_multiples]]
    # Whole pounds are enough here; "localized" adds thousands separators
    by_cluster[cost_cols] = by_cluster[cost_cols].round(0)
    pounds = {c: st.column_config.NumberColumn(
        "Holding cost (£)" if c == "holding_cost" else f"Total cost at {c[11:-1]}× (£)", format="localized")
        for c in cost_cols}
    st.dataframe(
        by_cluster[["cluster", "policy", "fill_rate", "stockout_weeks", "avg_on_hand", *cost_cols]],
        hide_index=True, width="stretch",
        column_config={"cluster": "Products", "policy": "Policy",
                       "fill_rate": st.column_config.NumberColumn("Fill rate", format="percent"),
                       "stockout_weeks": st.column_config.NumberColumn("Product-weeks with backorders",
                                                                      format="percent"),
                       "avg_on_hand": st.column_config.NumberColumn("Avg units on hand", format="%.0f"),
                       **pounds},
    )

    st.subheader("One product, week by week")
    left, right = st.columns(2)
    code = left.selectbox("Product", list(labels), format_func=labels.get, key="sim_product")
    policy = right.selectbox("Policy", list(policy_names), format_func=policy_names.get)
    sim = query("SELECT * FROM inventory_sim WHERE stock_code = ? AND policy = ? ORDER BY week_start",
                (code, policy))
    fig = go.Figure()
    fig.add_bar(x=sim["week_start"], y=sim["demand"], name="Demand", marker_color=ink, opacity=0.35)
    fig.add_scatter(x=sim["week_start"], y=sim["on_hand"], name="Stock on hand",
                    line_color=series_colors[2], mode="lines+markers")
    fig.add_scatter(x=sim["week_start"], y=sim["backorders"], name="Backorders",
                    line_color=series_colors[0], mode="lines+markers")
    # The ordering decision itself; each order arrives lead_time_weeks later
    orders = sim[sim["order"] > 0]
    fig.add_scatter(x=orders["week_start"], y=orders["order"], name="Order placed", mode="markers",
                    marker=dict(symbol="triangle-up", size=11, color=series_colors[1]))
    warmup_end = sim.loc[~sim["warmup"].astype(bool), "week_start"].min()
    fig.add_vrect(x0=sim["week_start"].min(), x1=warmup_end, fillcolor=ink, opacity=0.08, line_width=0)
    fig.add_annotation(x=sim["week_start"].min(), y=1, yref="paper", text="Warm-up (not scored)",
                       showarrow=False, xanchor="left", font_color=ink)
    mark_test_start(fig, test_start)
    fig.update_layout(yaxis_title="Units")
    show(st, style(fig, hovermode="x unified"))

with forecast_tab:
    left, right = st.columns(2)
    code = left.selectbox("Product", list(labels), format_func=labels.get, key="forecast_product")
    window = right.radio("Forecast", [1, cover_weeks], horizontal=True,
                         format_func=lambda w: "Next week" if w == 1 else f"Next {w} weeks (total)")
    fc = query('SELECT * FROM forecasts WHERE stock_code = ? AND "window" = ? ORDER BY week_start',
               (code, window))
    fig = go.Figure()
    fig.add_scatter(x=fc["week_start"], y=fc["units"], name="Actual", line_color=ink)
    fig.add_scatter(x=fc["week_start"], y=fc["linear_regression"], name="Forecast (linear regression)",
                    line_color=series_colors[2])
    # Only the multi-week total is used as the order-up-to target
    upper = "95th percentile forecast" + (" (order target)" if window == cover_weeks else "")
    fig.add_scatter(x=fc["week_start"], y=fc["q95"], name=upper,
                    line_color=series_colors[0], line_dash="dash")
    mark_test_start(fig, test_start)
    fig.update_layout(yaxis_title="Units", title="Actual demand vs forecasts")
    show(st, style(fig, hovermode="x unified"))

    left, right = st.columns(2)
    left.markdown("**Forecast accuracy, test weeks**")
    left.dataframe(query('SELECT model, mae, rmse, wape, bias FROM model_scores WHERE "window" = ?', (window,)),
                   hide_index=True, width="stretch",
                   column_config={c: st.column_config.NumberColumn(format="%.2f")
                                  for c in ["mae", "rmse", "wape", "bias"]})
    right.markdown("**Coverage of upper forecasts, test weeks** (target = quantile)")
    coverage = query('SELECT "group", quantile, coverage FROM quantile_scores WHERE "window" = ?', (window,))
    coverage["group"] = coverage["group"].map(
        lambda g: "All products" if g == "all" else cluster_names[int(g.split()[-1])])
    right.dataframe(coverage, hide_index=True, width="stretch",
                    column_config={"coverage": st.column_config.NumberColumn(format="percent")})
    st.caption("Settings were tuned on the 12 weeks before the test period. Model design choices were explored "
               "with test results visible, so test scores may be slightly optimistic.")

with product_tab:
    clusters = query("SELECT * FROM product_clusters").merge(products, on="stock_code")
    clusters["cluster"] = clusters["cluster"].map(cluster_names)
    fig = px.scatter(clusters, x="mean_units", y="cv", color="cluster", size="avg_price", log_x=True,
                     hover_name="description", color_discrete_sequence=series_colors,
                     hover_data={"cluster": False, "mean_units": ":.0f", "cv": ":.2f", "avg_price": ":.2f"},
                     category_orders={"cluster": sorted(clusters["cluster"].unique())},
                     labels={"mean_units": "Average weekly units (log scale)",
                             "cv": "Volatility (std / mean)", "avg_price": "Average price (£)"},
                     title="Products grouped by volume, volatility, and price (marker size = price)")
    fig.update_traces(marker=dict(opacity=0.8))
    show(st, style(fig, 450))
    st.dataframe(
        clusters.groupby("cluster").agg(products=("stock_code", "count"), mean_units=("mean_units", "mean"),
                                        cv=("cv", "mean"), avg_price=("avg_price", "mean")).round(2),
        width="stretch",
    )

with sales_tab:
    monthly = query("SELECT * FROM monthly_revenue")
    fig = px.line(monthly, x="month", y="revenue", markers=True, title="Monthly revenue, all products",
                  color_discrete_sequence=series_colors)
    fig.update_layout(yaxis_tickprefix="£", xaxis_title=None, yaxis_title=None)
    show(st, style(fig, hovermode="x unified"))

    left, right = st.columns(2)
    top = query("SELECT * FROM top_products")
    fig = px.bar(top, x="revenue", y="description", orientation="h", title="Top 10 products by revenue",
                 color_discrete_sequence=series_colors)
    fig.update_layout(yaxis=dict(autorange="reversed", title=None), xaxis_tickprefix="£", xaxis_title=None)
    show(left, style(fig, 420))

    countries = query("SELECT * FROM revenue_by_country")
    fig = px.bar(countries, x="revenue", y="country", orientation="h", title="Top 10 countries by revenue",
                 color_discrete_sequence=series_colors)
    fig.update_layout(yaxis=dict(autorange="reversed", title=None), xaxis_tickprefix="£", xaxis_title=None)
    show(right, style(fig, 420))
