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
    import statsmodels.tsa.arima.model as arima 

    return alt, arima, cast, cs, mo, pd, pl


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
    )
    return (pivot_data,)


@app.cell
def _(mo, pivot_data):
    usage_types = pivot_data.select("Usage Type").unique().collect().to_series().to_list()
    multiselect_types = mo.ui.dropdown(options=usage_types, label="Select Usage Types", value="Total Usage")
    multiselect_types
    return (multiselect_types,)


@app.cell
def _(multiselect_types, pivot_data, pl):
    filtered_data = pivot_data.filter(pl.col("Usage Type") == multiselect_types.value).sort(by="Date").collect()
    return (filtered_data,)


@app.cell
def _(filtered_data, pl):
    train = filtered_data.filter(pl.col("Date").dt.year() < 2024)
    test = filtered_data.filter(~(pl.col("Date").dt.year() < 2024))
    return test, train


@app.cell
def _(arima, train):
    model_30 = arima.ARIMA(train["Usage"].to_numpy(), order=(35, 0, 0))
    model_60 = arima.ARIMA(train["Usage"].to_numpy(), order=(65, 0, 0))
    return


@app.cell
def _(arima, pl, test, train):
    history = train["Usage"].to_list()
    actuals = test["Usage"].to_list()
    test_dates = test["Date"].to_list()

    rolling_forecasts = []
    for actual in actuals:
        model = arima.ARIMA(history, order=(15, 0, 0))
        fitted = model.fit()
        forecast = float(fitted.forecast(steps=1)[0])
        rolling_forecasts.append(forecast)
        history.append(actual)

    rolling_results = pl.DataFrame(
        {
            "Date": test_dates,
            "Actual": actuals,
            "Forecast": rolling_forecasts,
        }
    ).with_columns((pl.col("Actual") - pl.col("Forecast")).alias("Residual"))

    mae = rolling_results.select(pl.col("Residual").abs().mean()).item()
    rmse = (
        rolling_results.select((pl.col("Residual") ** 2).mean()).item() ** 0.5
    )

    print(f"Rolling-origin MAE (AR(15)): {mae:.4f}")
    print(f"Rolling-origin RMSE (AR(15)): {rmse:.4f}")
    return (rolling_results,)


@app.cell
def _(alt, pd, rolling_results):
    # plot actual vs forecast using Altair
    results_df = rolling_results.to_pandas()
    results_df["Date"] = pd.to_datetime(results_df["Date"])
    chart = (alt.Chart(results_df)
        .mark_line()
        .encode(
            x="Date",
            y="Actual",
            color=alt.value("blue"),
            tooltip=["Date", "Actual"],
        )
        + alt.Chart(results_df)
        .mark_line()
        .encode(
            x="Date",
            y="Forecast",
            color=alt.value("red"),
            tooltip=["Date", "Forecast"],
        )
        .properties(title="Actual vs Forecasted Gas Usage"))
    chart = chart.interactive(bind_y=False)
    chart
    return


if __name__ == "__main__":
    app.run()
