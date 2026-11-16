import sqlite3

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_pinball_loss
from sklearn.model_selection import ParameterGrid

from src import build_features, config, sql_analysis

feature_cols = ["lag_1", "lag_2", "lag_3", "lag_4", "rolling_mean_4", "rolling_std_4", "lag_52"]

# Calendar features are left out: test weeks (Sep-Nov) have week numbers the
# training weeks never saw, and tree models did worse with them
candidates = {
    # Ridge is linear regression with a penalty that keeps coefficients small
    "linear_regression": (Ridge(), {"alpha": [0.01, 0.1, 1, 10]}),
    # Absolute-error loss stops a few huge bulk orders from pulling forecasts up
    "gradient_boosting": (
        GradientBoostingRegressor(loss="absolute_error", random_state=42),
        {"max_depth": [2, 3], "n_estimators": [100, 300], "learning_rate": [0.05, 0.1]},
    ),
}

# Upper demand forecasts, used to size safety stock for 90% and 95% service
quantiles = [0.9, 0.95]


def score(actual, pred):
    error = pred - actual
    return {
        "mae": np.abs(error).mean(),
        "rmse": np.sqrt((error ** 2).mean()),
        # Total error relative to total demand, comparable across products
        "wape": np.abs(error).sum() / actual.sum(),
        # Negative means under-forecasting, which leads to stockouts
        "bias": error.mean(),
    }


def scale(df):
    # Models learn demand relative to the recent average rather than raw units.
    # Trees can't predict above the range they trained on, but a ratio like
    # "1.3x recent demand" still applies when demand climbs before Christmas.
    # +1 avoids dividing by zero for weeks with no sales.
    return df["rolling_mean_4"] + 1


def fit(model, train):
    model = clone(model)
    # This converts raw sales into ratios relative to recent weekly demand.
    model.fit(train[feature_cols].div(scale(train), axis=0), train["units"] / scale(train))
    return model


def predict(model, df):
    # Multiply by scale to convert predicted ratios back into actual units.
    # clip(0) prevents negative demand forecasts.
    return (model.predict(df[feature_cols].div(scale(df), axis=0)) * scale(df)).clip(0)


def known_before(df, week, window):
    # A row's target covers `window` weeks from its start, so at the start of
    # `week` only rows ending before it have a known target
    return df[df["week_start"] < week - pd.Timedelta(weeks=window - 1)]


def forecast_weeks(model, df, weeks, window):
    # Retrain before each week on what is known by then, as it would run in practice
    preds = pd.Series(np.nan, index=df.index)
    for week in weeks:
        fitted = fit(model, known_before(df, week, window))
        in_week = df["week_start"] == week
        preds[in_week] = predict(fitted, df[in_week])
    return preds[df["week_start"].isin(weeks)], fitted


def tuning_data(df, window):
    # Last training weeks used for tuning; the test weeks are never used
    train = known_before(df, df.loc[df["is_test"], "week_start"].min(), window)
    weeks = sorted(train["week_start"].unique())[-config.validation_weeks:]
    return train, weeks


def tune(name, df, window):
    model, grid = candidates[name]
    train, weeks = tuning_data(df, window)
    # Actual demand for validation weeks
    actual = train.loc[train["week_start"].isin(weeks), "units"]

    results = []
    # ParameterGrid generates every possible hyperparameter combination.
    for params in ParameterGrid(grid):
        # forecast_weeks() retrains chronologically for each validation week.
        preds, _ = forecast_weeks(clone(model).set_params(**params), train, weeks, window)
        results.append({"window": window, "model": name, "params": str(params),
                        "mae": np.abs(preds - actual).mean(), "params_dict": params})
    best = min(results, key=lambda r: r["mae"])["params_dict"]
    # Return an unfitted model with winning settings, plus tuning details.
    return clone(model).set_params(**best), best, results


def run_window(window, clusters):
    # window=1 predicts one week's demand; window=5 predicts a 5-week total.
    df = build_features.build_features(window)
    # Forecast from the first tuning week, not just the test weeks, so the
    # inventory simulation can warm up before the test period. The tuning weeks
    # helped pick settings, so scores below use the test weeks only.
    # tuning_data() returns (train, weeks): [0] = training DataFrame, [1] = validation weeks.
    rows = df[df["week_start"] >= tuning_data(df, window)[1][0]]
    # Tune on validation weeks, then forecast validation and test weeks.
    weeks = sorted(rows["week_start"].unique())
    # Add window and period columns; labels True as "test", False as "validation".
    forecasts = rows[["stock_code", "week_start", "units"]].assign(
        window=window, period=np.where(rows["is_test"], "test", "validation")
    )
    # Simple baselines: any model has to beat these to be worth using.
    # Convert historical weekly sales into window-week totals.
    forecasts["naive"] = rows["lag_1"] * window
    forecasts["rolling_mean_4"] = rows["rolling_mean_4"] * window
    forecasts["rolling_median_4"] = rows["rolling_median_4"] * window

    cv_results = []
    best_params = {}
    for name in candidates:
        # tune() returns: configured model, winning settings, all CV results.
        model, best_params[name], results = tune(name, df, window)
        cv_results += results
        forecasts[name], fitted = forecast_weeks(model, df, weeks, window)
        # Keep the latest fit, trained on the most data
        joblib.dump(fitted, config.model_dir / f"{name}_w{window}.joblib")

    # Train separate upper-quantile models for q90 and q95.
    for q in quantiles:
        model = GradientBoostingRegressor(
            loss="quantile", alpha=q, random_state=42, **best_params["gradient_boosting"]
        )
        # Dynamic column names: q=0.9 -> "q90"; q=0.95 -> "q95".
        forecasts[f"q{round(q * 100)}"], _ = forecast_weeks(model, df, weeks, window)

    # Join each forecast row with its product's cluster using stock_code,
    # so quantile accuracy can be checked separately for each cluster.
    forecasts = forecasts.merge(clusters, on="stock_code")
    test = forecasts[forecasts["period"] == "test"]

    model_names = ["naive", "rolling_mean_4", "rolling_median_4", *candidates]
    # **score unpacks MAE, RMSE, WAPE, and bias into each result row(only the test weeks).
    scores = pd.DataFrame(
        [{"window": window, "model": name, **score(test["units"], test[name])}
         for name in model_names]
    ).sort_values("mae")

    # Evaluate quantiles across all products and separately by cluster.
    # groupby() yields (cluster_number, cluster_dataframe) pairs.
    groups = [("all", test)] + [
        (f"cluster {c}", g) for c, g in test.groupby("cluster")
    ]
    quantile_scores = pd.DataFrame([{
        "window": window,
        "quantile": q,
        "group": group,
        # Share of weeks where demand stayed under the forecast; ideally equals q
        # A well-calibrated q95 should have roughly 95% coverage.
        "coverage": (g["units"] <= g[f"q{round(q * 100)}"]).mean(),
        # Pinball loss penalizes underprediction more at high quantiles.
        # At q95, underprediction costs 19x as much per unit.
        "pinball_loss": mean_pinball_loss(g["units"], g[f"q{round(q * 100)}"], alpha=q),
    } for q in quantiles for group, g in groups])

    cv_scores = pd.DataFrame(cv_results).drop(columns="params_dict")
    return forecasts, scores, quantile_scores, cv_scores


def save_results(tables):
    conn = sqlite3.connect(config.db_path)
    for name, df in tables.items():
        df.to_sql(name, conn, if_exists="replace", index=False)
    conn.close()


def main():
    clusters = sql_analysis.run_query("SELECT stock_code, cluster FROM product_clusters")
    # Weekly forecasts for accuracy, and lead time + review week totals for reordering
    results = [run_window(w, clusters) for w in (1, config.lead_time_weeks + config.review_weeks)]
    forecasts, scores, quantile_scores, cv_scores = (pd.concat(t, ignore_index=True) for t in zip(*results))
    save_results({
        "forecasts": forecasts,
        "model_scores": scores,
        "quantile_scores": quantile_scores,
        "cv_scores": cv_scores,
    })
    print("Test scores:\n", scores.round(2).to_string(index=False), "\n")
    print("Quantile scores:\n", quantile_scores.round(3).to_string(index=False), "\n")
    best_cv = cv_scores.loc[cv_scores.groupby(["window", "model"])["mae"].idxmin()]
    print("Chosen settings:\n", best_cv.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
