import marimo

__generated_with = "0.19.9"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    import polars.selectors as cs
    import polars as pl
    import altair as alt
    import statsmodels.tsa.stattools as tsa
    import statsmodels.api as sm
    from datetime import date as dt
    import numpy as np

    return alt, cs, pl, tsa


@app.cell
def _(cs, pl):
    data_path = "data/silver/uta_gas_usage.parquet"
    gas_usage = pl.scan_parquet(data_path) \
        .fill_null(0) \
        .with_columns(
            (pl.col("Usage - 1") + pl.col("Usage - 2") + pl.col("Usage - 2_1")).alias("Total Usage")) \
        .select(
            pl.exclude(["Nom", "Delivery"])
        ).unpivot(
        index="Date",
        on=cs.float(),
        variable_name="Usage Type",
        value_name="Usage",
    )
    return (gas_usage,)


@app.cell(disabled=True)
def _(gas_usage):
    gas_usage.collect_schema()
    return


@app.cell
def _(gas_usage, mo):
    usage_types = gas_usage.select("Usage Type").unique().collect().to_series().to_list()
    dropdown = mo.ui.dropdown(usage_types, value="Total Usage")
    dropdown
    return (dropdown,)


@app.cell
def _(dropdown, gas_usage, pl):
    gas_usage_filtered = gas_usage.filter(pl.col("Usage Type") == dropdown.value).collect()
    return (gas_usage_filtered,)


@app.cell(disabled=True)
def _(gas_usage_filtered, pl):
    gas_usage_roll_up = gas_usage_filtered.group_by_dynamic(index_column="Date", every="1w", closed="right").agg(pl.col("Usage").mean())
    gas_usage_roll_up
    return (gas_usage_roll_up,)


@app.cell
def _(alt, gas_usage_roll_up):
    # replace _df with your data source
    _chart = (
        alt.Chart(gas_usage_roll_up)
        .mark_line()
        .encode(
            x=alt.X(field='Date', type='temporal', timeUnit='yearmonthdate'),
            y=alt.Y(field='Usage', type='quantitative'),
            tooltip=[
                alt.Tooltip(field='Date', timeUnit='yearmonthdate', title='Date'),
                alt.Tooltip(field='Usage', format=',.2f')
            ]
        )
        .properties(
            height=290,
            width='container',
            config={
                'axis': {
                    'grid': False
                }
            }
        ).interactive(bind_y=False)
    )
    _chart
    return


@app.cell
def _(alt, gas_usage_filtered):
    # replace _df with your data source
    _chart = (
        alt.Chart(gas_usage_filtered)
        .mark_line()
        .encode(
            x=alt.X(field='Date', type='temporal', timeUnit='yearmonthdate'),
            y=alt.Y(field='Usage', type='quantitative'),
            tooltip=[
                alt.Tooltip(field='Date', timeUnit='yearmonthdate', title='Date'),
                alt.Tooltip(field='Usage', format=',.2f')
            ]
        )
        .properties(
            height=290,
            width='container',
            config={
                'axis': {
                    'grid': False
                }
            }
        )
    )
    _chart
    return


@app.cell(hide_code=True)
def _(mo):
    danger = mo.icon(icon_name="jam:triangle-danger", color="#FF0000")
    mo.md(f"""
    # Check for stationarity

    - Constant Mean
    - Constant Variance
    - {danger}**<span style="color:#FF0000">Seasonality</span>**
    """)
    return


@app.cell
def _(gas_usage_filtered, pl, tsa):
    acf_tsa = tsa.acf(gas_usage_filtered.sort(by="Date")["Usage"].to_numpy(), nlags=3600, alpha=0.05)
    acf_df = pl.DataFrame({
        "lag": range(len(acf_tsa[0])),
        "correlation": acf_tsa[0],
        "lower_ci": acf_tsa[1][:, 0] - acf_tsa[0],
        "upper_ci": acf_tsa[1][:, 1] - acf_tsa[0],   
    })
    return (acf_df,)


@app.cell
def _(gas_usage_filtered, pl, tsa):
    pacf_tsa = tsa.pacf(gas_usage_filtered.sort(by="Date")["Usage"].to_numpy(), nlags=800, alpha=0.05)
    pacf_df = pl.DataFrame({
        "lag": range(len(pacf_tsa[0])),
        "correlation": pacf_tsa[0],
        "lower_ci": pacf_tsa[1][:, 0] - pacf_tsa[0],
        "upper_ci": pacf_tsa[1][:, 1] - pacf_tsa[0],   
    })
    return (pacf_df,)


@app.cell
def _(acf_df, mo, pacf_df):
    corr_dropdown = mo.ui.radio({"acf": acf_df, "pacf": pacf_df}, value="acf")
    corr_dropdown
    return (corr_dropdown,)


@app.cell
def _(alt, corr_dropdown):
    bars = (
        alt.Chart(corr_dropdown.value)
        .mark_bar()
        .encode(
            x=alt.X(field='lag', type='quantitative'),
            y=alt.Y(field='correlation', type='quantitative'),
            tooltip=[
                alt.Tooltip(field='lag', format=',.0f'),
                alt.Tooltip(field='correlation', format=',.2f')
            ]
        )
    )

    ci = (
        alt.Chart(corr_dropdown.value)
        .mark_area(color='steelblue', opacity=0.5)
        .encode(
            x=alt.X('lag:Q'),
            y=alt.Y('lower_ci:Q'),
            y2=alt.Y2('upper_ci:Q'),
            tooltip=[
                alt.Tooltip('lower_ci:Q', format=',.2f', title='lower_ci'),
                alt.Tooltip('upper_ci:Q', format=',.2f', title='upper_ci')
            ]
        )
    )

    _chart = alt.layer(bars, ci).properties(
        height=290,
        width='container',
    ).interactive(bind_y=False)
    _chart
    return


@app.cell
def _(acf_df, gas_usage_filtered, pl):
    lags_w_significant_positive_corr = acf_df.filter((pl.col("correlation") > pl.col("upper_ci").abs()) & (pl.col("lag") != 0)).select("lag").to_series().to_list()

    lags_w_significant_positive_corr_week = acf_df.filter((pl.col("correlation") > pl.col("upper_ci").abs()) & (pl.col("lag") != 0) & (pl.col("lag") % 7 == 0) ).select("lag").to_series().to_list()

    lag_exprs1 = [pl.col("Usage").shift(lag) for lag in lags_w_significant_positive_corr]
    lag_exprs2 = [pl.col("Usage").shift(lag) for lag in lags_w_significant_positive_corr_week]
    gas_usage_2024 = gas_usage_filtered.with_columns(
        pl.when(pl.col("Date").dt.year() == 2024).then(pl.mean_horizontal(lag_exprs1)).otherwise(pl.lit(0)).alias("avg_usage_plags"),
        pl.when(pl.col("Date").dt.year() == 2024).then(pl.mean_horizontal(lag_exprs2)).otherwise(pl.lit(0)).alias("avg_usage_plags_week"),
    )

    gas_usage_2024.filter(pl.col("Date").dt.year() == 2024)
    return (gas_usage_2024,)


@app.cell
def _(gas_usage_2024, pl):
    gas_usage_2024.filter(pl.col("Date").dt.year() == 2024).with_columns(
        (1 - (pl.col("avg_usage_plags")/pl.col("Usage"))).abs().alias("error_plags"),
        (1 - (pl.col("avg_usage_plags_week")/pl.col("Usage"))).abs().alias("error_plags_week"),
    )
    return


@app.cell
def _(gas_usage_2024, pl):
    predictions = gas_usage_2024.filter(pl.col("Date").dt.year() == 2024).select(["Date", "avg_usage_plags", "avg_usage_plags_week"]).unpivot(index="Date", variable_name="Type", value_name="Value")
    predictions
    return (predictions,)


@app.cell
def _(alt, predictions):
    # replace _df with your data source
    _chart = (
        alt.Chart(predictions)
        .mark_line()
        .encode(
            x=alt.X(field='Date', type='temporal', timeUnit='yearmonthdate'),
            y=alt.Y(field='Value', type='quantitative', aggregate='mean'),
            color=alt.Color(field='Type', type='nominal'),
            tooltip=[
                alt.Tooltip(field='Date', timeUnit='yearmonthdate', title='Date'),
                alt.Tooltip(field='Value', aggregate='mean'),
                alt.Tooltip(field='Type')
            ]
        )
        .properties(
            height=290,
            width='container',
        ).interactive()
    )
    _chart
    return


@app.cell
def _(gas_usage_2024, pl):
    # select only 2024 rows and the relevant columns
    g2024 = (
        gas_usage_2024
        .filter(pl.col("Date").dt.year() == 2024)
        .select(["Date", "Usage", "avg_usage_plags", "avg_usage_plags_week"])
        .drop_nulls()
    )


    # melt to long format for vectorized polars expressions
    melted = g2024.unpivot(
        index=["Date", "Usage"],
        variable_name="model",
        value_name="yhat",
    ).with_columns(
        (pl.col("yhat") - pl.col("Usage")).alias("err"),
        # denom for MAPE: None where actual == 0 to ignore those rows
        pl.when(pl.col("Usage") == 0).then(pl.lit(None)).otherwise(pl.col("Usage").abs()).alias("mape_denom"),
    ).with_columns(
        (2 * pl.col("err").abs() / (pl.col("Usage").abs() + pl.col("yhat").abs() + 1e-9)).alias("smape_elem"),
        (pl.col("err").abs() / pl.col("mape_denom")).alias("mape_elem")
    )

    errors_df = (
        melted
        .group_by("model")
        .agg([
            pl.col("err").abs().mean().alias("MAE"),
            pl.col("err").pow(2).mean().alias("MSE"),
            pl.col("err").pow(2).mean().sqrt().alias("RMSE"),
            (pl.col("mape_elem").mean() * 100).alias("MAPE_pct"),
            (pl.col("smape_elem").mean() * 100).alias("sMAPE_pct"),
            pl.col("err").mean().alias("Bias"),
            pl.len().alias("N"),
        ])
    )

    errors_df
    return


@app.cell
def _():
    # # =====================================================
    # # FULL MONTH (31 DAYS) FORECAST + ACTUAL COMPARISON
    # # USING total_usage = usage-1 + usage-2 + usage2_1
    # # NaNs replaced with 0 instead of interpolation
    # # =====================================================

    # import pandas as pd
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
    # # Corrected column names here
    # for col in ['Usage - 1', 'Usage - 2', 'Usage - 2_1']:
    #     if col in df.columns:
    #         df[col] = df[col].fillna(0)

    # # Compute total_usage as sum of three columns
    # # Corrected column names here
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
    # # 3. BASE MODELS (WITH MULTI-ROLLING)
    # # ------------------------------
    # def naive_yesterday(date):
    #     return df.loc[date - pd.Timedelta(days=1), 'total_usage']

    # def naive_last_week(date):
    #     return df.loc[date - pd.Timedelta(days=7), 'total_usage']

    # def naive_last_year(date):
    #     return df.loc[date - pd.Timedelta(days=365), 'total_usage']

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
    # start_test = "2024-09-01"
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
    # # 5. 31-DAY FORECAST
    # # ------------------------------
    # forecast_start = pd.Timestamp("2024-08-01")
    # forecast_dates = pd.date_range(forecast_start, periods=31, freq='D')

    # predictions = []
    # actual_values = []

    # for date in forecast_dates:
    #     if date not in df.index:
    #         continue
    #     predictions.append(ensemble_predict(date))
    #     actual_values.append(df.loc[date, 'total_usage'])

    # # ------------------------------
    # # 6. PRINT TABLE
    # # ------------------------------
    # comparison_df = pd.DataFrame({
    #     "Date": forecast_dates[:len(predictions)],
    #     "Predicted": predictions,
    #     "Actual": actual_values
    # })

    # print("\n31-Day Forecast vs Actual:")
    # print(comparison_df)

    # # ------------------------------
    # # 7. PLOT
    # # ------------------------------
    # plt.figure(figsize=(14,6))
    # plt.plot(comparison_df["Date"], comparison_df["Actual"], label='Actual')
    # plt.plot(comparison_df["Date"], comparison_df["Predicted"], label='Ensemble Forecast')

    # plt.title("31-Day Forecast vs Actual Usage (sum of usage-1 + usage-2 + usage2_1)")
    # plt.xlabel("Date")
    # plt.ylabel("Total Gas Usage")
    # plt.legend()
    # plt.grid(True)
    # plt.xticks(rotation=45)
    # plt.tight_layout()
    # plt.show()

    # # ------------------------------
    # # 8. ERROR METRIC
    # # ------------------------------
    # mae_31day = mean_absolute_error(actual_values, predictions)
    # print("\n31-Day MAE:", mae_31day)

    # comparison_df.to_csv("31_day_forecast_vs_actual.csv", index=False)
    return


if __name__ == "__main__":
    app.run()
