import marimo

__generated_with = "0.19.9"
app = marimo.App(width="full")


@app.cell
def _():
    import datetime as dt
    from typing import cast

    import altair as alt
    import marimo as mo
    import pandas as pd
    import polars as pl
    import polars.selectors as cs
    import statsmodels.tsa.stattools as tsa
    import statsmodels.api as sm

    return cast, cs, mo, pl


@app.cell
def _(cast, cs, pl):
    # Read uta_gas_usage.parquet using a Polars lazy frame
    lf = cast("pl.LazyFrame", pl.scan_parquet("/home/vandy/Work/DASC5309/scot-forge/data/silver/uta_gas_usage.parquet"))
    lf2 = lf.fill_null(0).with_columns((pl.col("Usage - 1") + pl.col("Usage - 2") + pl.col("Usage - 2_1")).alias("Total Usage"))
    pivot_data = lf2.select(pl.exclude(["Nom", "Delivery"])).unpivot(
        index="Date",
        on=cs.float(),
        variable_name="Usage Type",
        value_name="Usage",
    ).sort(by="Date")
    return (pivot_data,)


@app.cell
def _(mo, pivot_data):
    usage_types = pivot_data.select("Usage Type").unique().collect().to_series().to_list()
    multiselect_types = mo.ui.dropdown(options=usage_types, label="Select Usage Types", value="Total Usage")
    return (multiselect_types,)


@app.cell
def _(multiselect_types, pivot_data, pl):
    filtered_data = pivot_data.filter(pl.col("Usage Type") == multiselect_types.value).sort(by="Date").collect()
    return (filtered_data,)


@app.cell
def _(filtered_data, pl):
    train = filtered_data.filter(pl.col("Date").dt.year() < 2024)
    test = filtered_data.filter(~(pl.col("Date").dt.year() < 2024))
    return (train,)


@app.cell
def _(pl, train):
    average_usage = train.select(pl.col("Usage").mean()).item()
    return (average_usage,)


@app.cell
def _(average_usage, pl, train):
    monthly_idices = train.group_by([pl.col("Date").dt.month()]) \
        .agg(pl.col("Usage").mean()) \
        .with_columns(
            (pl.col("Usage")/average_usage)
                .alias("Monthly Index")
        )
    return


@app.cell
def _(average_usage, pl, train):
    dow_indices = train.group_by(pl.col("Date").dt.weekday())\
        .agg(pl.col("Usage").mean())\
        .with_columns(
            (pl.col("Usage") / average_usage)
                .alias("DOW Index")
        )
    return


@app.cell
def _(pl):
    naive_yesterday = pl.col("Usage").shift(1)
    naive_last_week = pl.col("Usage").shift(7)
    naive_last_year = pl.col("Usage").shift(364)
    def rolling_mean(period):
        return pl.mean("Usage").rolling(index_column="Date", period=period, closed="left")

    return naive_last_week, naive_last_year, naive_yesterday, rolling_mean


@app.cell
def _(filtered_data):
    filtered_data
    return


@app.cell
def _(filtered_data, pl):
    start = pl.date(day=1, month=1, year=2024)
    end = pl.date(day=1, month=1, year=2025)
    filtered_data.with_columns(pl.when((pl.col("Date") >= start) & (pl.col("Date") < end)).then(pl.lit(1)).otherwise(pl.lit(0)).alias("Test"))
    return


@app.cell
def _(
    filtered_data,
    naive_last_week,
    naive_last_year,
    naive_yesterday,
    pl,
    rolling_mean,
):
    models = {
        'naive_yesterday': naive_yesterday,
        'naive_last_week': naive_last_week,
        'naive_last_year': naive_last_year,
        'rolling_7': rolling_mean('7d'),
        'rolling_14': rolling_mean("14d"),
        'rolling_30': rolling_mean("1mo"),
        'rolling_90': rolling_mean("3mo"),
        'rolling_150': rolling_mean("5mo"),
        'rolling_365': rolling_mean("1y"),
        'rolling_730': rolling_mean("2y"),
    }
    def predict(start, end):
        predictions = filtered_data.with_columns(
            [pred_func.alias(model) for model, pred_func in models.items()]
        )

        return predictions

    predict(pl.date(day=1, month=1, year=2024), pl.date(day=1, month=1, year=2025))
    
    return


@app.cell
def _():
    # # =====================================================
    # # FULL YEAR FORECAST + ACTUAL COMPARISON
    # # USING total_usage = Usage-1 + Usage-2 + Usage2_1
    # # NaNs replaced with 0
    # # =====================================================

    # import pandas as pd
    # import numpy as np
    # import matplotlib.pyplot as plt
    # from sklearn.metrics import mean_absolute_error

    # # ------------------------------
    # # 1. LOAD & PREP
    # # ------------------------------
    # df = pd.read_csv("my_data_full.csv")
    # df['Date'] = pd.to_datetime(df['Date'])
    # df = df.sort_values('Date')
    # df.set_index('Date', inplace=True)
    # df = df.asfreq('D')

    # # Replace NaNs or nulls in usage columns with 0
    # for col in ['Usage - 1', 'Usage - 2', 'Usage - 2_1']:
    #     if col in df.columns:
    #         df[col] = df[col].fillna(0)

    # # Compute total_usage as sum of three columns
    # df['total_usage'] = df['Usage - 1'] + df['Usage - 2'] + df['Usage - 2_1']

    # # Add date features
    # df['dayofweek'] = df.index.dayofweek
    # df['month'] = df.index.month
    # df['year'] = df.index.year

    # # ------------------------------
    # # 2. SEASONALITY INDICES
    # # ------------------------------
    # overall_avg = df['total_usage'].mean()
    # monthly_index = df.groupby('month')['total_usage'].mean() / overall_avg
    # dow_index = df.groupby('dayofweek')['total_usage'].mean() / overall_avg

    # def seasonal_adjustment(date):
    #     return monthly_index[date.month] * dow_index[date.dayofweek]

    # # ------------------------------
    # # 3. BASE MODELS
    # # ------------------------------
    # def naive_yesterday(date):
    #     return df.loc[date - pd.Timedelta(days=1), 'total_usage']

    # def naive_last_week(date):
    #     return df.loc[date - pd.Timedelta(days=7), 'total_usage']

    # def naive_last_year(date):
    #     return df.loc[date - pd.Timedelta(days=364), 'total_usage']

    # def rolling_mean(date, window):
    #     return df.loc[:date - pd.Timedelta(days=1), 'total_usage'].tail(window).mean()

    # def predict_models(date):
    #     base = {
    #         'naive_yesterday': naive_yesterday(date),
    #         'naive_last_week': naive_last_week(date),
    #         'naive_last_year': naive_last_year(date),
    #         'rolling_7': rolling_mean(date, 7),
    #         'rolling_14': rolling_mean(date, 14),
    #         'rolling_30': rolling_mean(date, 30),
    #         'rolling_90': rolling_mean(date, 90),
    #         'rolling_150': rolling_mean(date, 150),
    #         'rolling_365': rolling_mean(date, 365),
    #         'rolling_730': rolling_mean(date, 730),
    #     }
    #     seasonal = {k + "_seasonal": v * seasonal_adjustment(date) for k, v in base.items()}
    #     return {**base, **seasonal}

    # # ------------------------------
    # # 4. BACKTEST TO GET WEIGHTS
    # # ------------------------------
    # start_test = "2024-01-01"
    # dates = df.loc[start_test:].index

    # model_names = list(predict_models(dates[0]).keys())
    # results = {m: [] for m in model_names}
    # actuals = []

    # for date in dates:
    #     if (date - pd.Timedelta(days=730)) not in df.index:
    #         continue
    #     preds = predict_models(date)
    #     actual = df.loc[date, 'total_usage']
    #     actuals.append(actual)
    #     for m in model_names:
    #         results[m].append(preds[m])

    # mae_scores = {m: mean_absolute_error(actuals, results[m]) for m in model_names}
    # mae_df = pd.DataFrame.from_dict(mae_scores, orient='index', columns=['MAE'])
    # weights = 1 / mae_df['MAE']
    # weights = weights / weights.sum()

    # def ensemble_predict(date):
    #     preds = predict_models(date)
    #     return sum(preds[m] * weights[m] for m in weights.index)

    # # ------------------------------
    # # 5. 1-YEAR FORECAST (365 DAYS)
    # # ------------------------------
    # forecast_start = pd.Timestamp("2024-01-01")
    # forecast_dates = pd.date_range(forecast_start, periods=365, freq='D')

    # predictions = []
    # actual_values = []

    # for date in forecast_dates:
    #     if date not in df.index:
    #         continue
    #     predictions.append(ensemble_predict(date))
    #     actual_values.append(df.loc[date, 'total_usage'])

    # # ------------------------------
    # # 6. CREATE COMPARISON TABLE & DAILY MAE
    # # ------------------------------
    # comparison_df = pd.DataFrame({
    #     "Date": forecast_dates[:len(predictions)],
    #     "Predicted": predictions,
    #     "Actual": actual_values
    # })

    # # Daily MAE
    # comparison_df["Daily_MAE"] = abs(comparison_df["Actual"] - comparison_df["Predicted"])

    # print("\n1-Year Forecast vs Actual with Daily MAE:")
    # print(comparison_df.head(15))  # print first 15 rows as sample

    # # ------------------------------
    # # 7. PLOT
    # # ------------------------------
    # plt.figure(figsize=(16,6))
    # plt.plot(comparison_df["Date"], comparison_df["Actual"], label='Actual')
    # plt.plot(comparison_df["Date"], comparison_df["Predicted"], label='Ensemble Forecast')
    # plt.title("1-Year Forecast vs Actual Usage (sum of usage-1 + usage-2 + usage2_1)")
    # plt.xlabel("Date")
    # plt.ylabel("Total Gas Usage")
    # plt.legend()
    # plt.grid(True)
    # plt.tight_layout()
    # plt.show()

    # # ------------------------------
    # # 8. OVERALL MAE
    # # ------------------------------
    # mae_year = mean_absolute_error(actual_values, predictions)
    # print("\n1-Year Overall MAE:", mae_year)

    # # ------------------------------
    # # 9. EXPORT TO CSV
    # # ------------------------------
    # # comparison_df.to_csv("1_year_forecast_vs_actual.csv", index=False)
    # # print("\nCSV saved as '1_year_forecast_vs_actual.csv'")

    return


@app.cell
def _(pl):
    dates = [
        "2020-01-01 13:45:48",
        "2020-01-01 16:42:13",
        "2020-01-01 16:45:09",
        "2020-01-02 18:12:48",
        "2020-01-03 19:45:32",
        "2020-01-08 23:16:43",
    ]
    df = pl.DataFrame({"dt": dates, "a": [3, 7, 5, 9, 2, 1]}).with_columns(
        pl.col("dt").str.strptime(pl.Datetime).set_sorted()
    )
    df.with_columns(
        sum_a=pl.sum("a").rolling(index_column="dt", period="2d"),
        min_a=pl.min("a").rolling(index_column="dt", period="2d"),
        max_a=pl.max("a").rolling(index_column="dt", period="2d"),
    )
    return


if __name__ == "__main__":
    app.run()
