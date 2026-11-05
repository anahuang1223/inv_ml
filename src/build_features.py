import pandas as pd

from src import config, load_data, sql_analysis


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


def add_features(df):
    df = df.sort_values(["stock_code", "week_start"])
    # Shifting by the horizon means every feature was known when forecasting
    past = df.groupby("stock_code")["units"].shift(config.horizon_weeks)

    df["lag_1"] = past
    df["lag_2"] = past.groupby(df["stock_code"]).shift(1)
    df["lag_3"] = past.groupby(df["stock_code"]).shift(2)
    df["lag_4"] = past.groupby(df["stock_code"]).shift(3)
    df["rolling_mean_4"] = past.groupby(df["stock_code"]).transform(lambda s: s.rolling(4).mean())
    df["rolling_std_4"] = past.groupby(df["stock_code"]).transform(lambda s: s.rolling(4).std())

    df["week_of_year"] = df["week_start"].dt.isocalendar().week.astype(int)
    df["month"] = df["week_start"].dt.month
    return df


def build_features():
    df = fill_missing_weeks(sql_analysis.weekly_demand())
    split = load_data.test_start(df["week_start"])

    # The final week stops on a Friday, so its total would look like a demand drop
    df = df[df["week_start"] < df["week_start"].max()]

    df = add_features(df).dropna().reset_index(drop=True)
    df["is_test"] = df["week_start"] >= split
    return df


def main():
    df = build_features()
    print(df.head())
    print(f"Train rows: {(~df['is_test']).sum():,}, test rows: {df['is_test'].sum():,}")


if __name__ == "__main__":
    main()
