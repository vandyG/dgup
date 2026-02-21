import marimo

__generated_with = "0.19.9"
app = marimo.App(width="full")


@app.cell
def _():
    import altair as alt
    import marimo as mo
    import polars as pl

    return alt, mo, pl


@app.cell
def _():
    data_path = "data/all_results.csv"
    return (data_path,)


@app.cell
def _(data_path, pl):
    df = pl.read_csv(data_path, try_parse_dates=True)
    df = df.rename({"Date": "date", "total_usage": "actual"})
    df = df.sort("date")

    model_cols = [
        "xgboost_pred",
        "ridge_pred",
        "en_pred",
        "prophet_pred",
        "avg_usage_plags",
        "avg_usage_plags_week",
        "ensemble_preds",
        "sarimax_preds"
    ]
    return df, model_cols


@app.cell
def _(mo, model_cols):
    model_picker = mo.ui.multiselect(
        options=model_cols,
        value=model_cols,
        label="Models",
    )
    return (model_picker,)


@app.cell
def _(df, model_cols, pl):
    metric_rows = []
    for col1 in model_cols:
        actual = df["actual"]
        pred = df[col1]
        error = pred - actual
        abs_error = error.abs()
        squared_error = error.pow(2)

        mae = abs_error.mean()
        mse = squared_error.mean()
        rmse = mse**0.5

        denom_smape = (actual.abs() + pred.abs()) / 2
        smape_pct = (abs_error / denom_smape).mean() * 100
        mape_pct = (abs_error / actual.abs()).mean() * 100
        bias = error.mean()

        metric_rows.append(
            {
                "model": col1,
                "mae": mae,
                "mse": mse,
                "rmse": rmse,
                "smape (%)": smape_pct,
                "mape (%)": mape_pct,
                "bias": bias,
            }
        )

    metrics_df = pl.DataFrame(metric_rows)
    return (metrics_df,)


@app.cell
def _(alt, df, model_picker, pl):
    selected_models = model_picker.value
    if not selected_models:
        selected_models = []

    series_frames = [
        df.select(
            [
                pl.col("date"),
                pl.lit("actual").alias("series"),
                pl.col("actual").alias("value"),
            ]
        )
    ]
    for col in selected_models:
        series_frames.append(
            df.select(
                [
                    pl.col("date"),
                    pl.lit(col).alias("series"),
                    pl.col(col).alias("value"),
                ]
            )
        )

    plot_df = pl.concat(series_frames, how="vertical")

    chart = (
        alt.Chart(plot_df.to_pandas())
        .mark_line()
        .encode(
            x=alt.X("date:T", title="Date"),
            y=alt.Y("value:Q", title="Usage"),
            color=alt.Color("series:N", title="Series"),
        )
        .properties(height=400)
        .interactive(bind_y=False)
    )
    return (chart,)


@app.cell
def _(chart, mo, model_picker):
    mo.vstack([model_picker, chart])
    return


@app.cell
def _(metrics_df):
    metrics_df
    return


if __name__ == "__main__":
    app.run()
