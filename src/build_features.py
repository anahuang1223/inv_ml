import pandas as pd

from src import load_data, sql_analysis


def fill_missing_weeks(df):
    # A week with no sales has no row, but it is real demand of zero.
    # Without these rows, lag_1 could silently point two weeks back.
    weeks = pd.date_range(df["week_start"].min(), df["week_start"].max(), freq="7D")
    #cartesian product of stock and week
    full_index = pd.MultiIndex.from_product(
        [df["stock_code"].unique(), weeks], names=["stock_code", "week_start"]
    )
    df = df.set_index(["stock_code", "week_start"]).reindex(full_index, fill_value=0)
    return df.reset_index()


def add_features(df, window):
    df = df.sort_values(["stock_code", "week_start"])
    weekly = df.groupby("stock_code")["units"]
    # Shifting by one week means every feature was known when forecasting
    past = weekly.shift(1)

    df["lag_1"] = past
    df["lag_2"] = past.groupby(df["stock_code"]).shift(1)
    df["lag_3"] = past.groupby(df["stock_code"]).shift(2)
    df["lag_4"] = past.groupby(df["stock_code"]).shift(3)
    df["rolling_mean_4"] = past.groupby(df["stock_code"]).transform(lambda s: s.rolling(4).mean())
    df["rolling_std_4"] = past.groupby(df["stock_code"]).transform(lambda s: s.rolling(4).std())
    df["rolling_median_4"] = past.groupby(df["stock_code"]).transform(lambda s: s.rolling(4).median())

    # Target = total demand starting this week and spanning `window` weeks.
    # window=1: current week; window=4: current week + next 3 weeks.
    # Predict the full lead-time demand directly.
    # Summing weekly q95 forecasts does NOT generally give the q95 of the total.
    target = sum(weekly.shift(-k) for k in range(window))
    df["units"] = target
    # Same weeks last year, so the model can see seasonal peaks like Christmas
    df["lag_52"] = target.groupby(df["stock_code"]).shift(52)

    df["week_of_year"] = df["week_start"].dt.isocalendar().week.astype(int)
    df["month"] = df["week_start"].dt.month
    return df


# Get weekly demand, lag/rolling features, and train/test boolean labels
def build_features(window=1):
    df = fill_missing_weeks(sql_analysis.weekly_demand())
    split = load_data.test_start(df["week_start"])

    # The final week stops on a Friday, so its total would look like a demand drop
    df = df[df["week_start"] < df["week_start"].max()]
    # Remove rows with missing features/targets:
    # first 52 weeks (lag_52 unavailable) and last window-1 weeks
    # (incomplete forecast targets).
    df = add_features(df, window).dropna().reset_index(drop=True)
    df["is_test"] = df["week_start"] >= split
    return df


def main():
    df = build_features()
    print(df.head())
    print(f"Train rows: {(~df['is_test']).sum():,}, test rows: {df['is_test'].sum():,}")


if __name__ == "__main__":
    main()
