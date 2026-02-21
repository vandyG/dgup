import marimo

__generated_with = "0.19.9"
app = marimo.App(width="full")


@app.cell
def _():
    from pathlib import Path

    import marimo as mo
    import polars as pl
    import polars.selectors as cs
    import statsmodels.tsa.stattools as tsa

    return Path, cs, mo, pl, tsa


@app.cell
def _(pl):
    data_path = "data/silver/uta_gas_usage.parquet"
    lf = (
        pl.scan_parquet(data_path)
        .fill_null(0)
        .with_columns(
            (
                pl.col("Usage - 1")
                + pl.col("Usage - 2")
                + pl.col("Usage - 2_1")
            ).alias("Total Usage"),
        )
    )
    return data_path, lf


@app.cell
def _(cs, lf, pl):
    pivot_data = lf.select(pl.exclude(["Nom", "Delivery"])).unpivot(
        index="Date",
        on=cs.float(),
        variable_name="Usage Type",
        value_name="Usage",
    )
    return (pivot_data,)


@app.cell
def _(mo, pivot_data):
    usage_types = pivot_data.select("Usage Type").unique().collect().to_series().to_list()
    dropdown = mo.ui.dropdown(usage_types, value="Total Usage")
    return dropdown, usage_types


@app.cell
def _(dropdown, pivot_data, pl):
    gas_usage_filtered = (
        pivot_data.filter(pl.col("Usage Type") == dropdown.value).collect()
    )
    return (gas_usage_filtered,)


@app.cell
def _(gas_usage_filtered, pl, tsa):
    acf_tsa = tsa.acf(
        gas_usage_filtered.sort(by="Date")["Usage"].to_numpy(),
        nlags=3600,
        alpha=0.05,
    )
    acf_df = pl.DataFrame(
        {
            "lag": range(len(acf_tsa[0])),
            "correlation": acf_tsa[0],
            "lower_ci": acf_tsa[1][:, 0] - acf_tsa[0],
            "upper_ci": acf_tsa[1][:, 1] - acf_tsa[0],
        },
    )
    return (acf_df,)


@app.cell
def _(acf_df, pl):
    lags_w_significant_positive_corr = (
        acf_df.filter(
            (pl.col("correlation") > pl.col("upper_ci").abs())
            & (pl.col("lag") != 0),
        )
        .select("lag")
        .to_series()
        .to_list()
    )

    lags_w_significant_positive_corr_week = (
        acf_df.filter(
            (pl.col("correlation") > pl.col("upper_ci").abs())
            & (pl.col("lag") != 0)
            & (pl.col("lag") % 7 == 0),
        )
        .select("lag")
        .to_series()
        .to_list()
    )

    return lags_w_significant_positive_corr, lags_w_significant_positive_corr_week


@app.cell
def _(lags_w_significant_positive_corr, lags_w_significant_positive_corr_week, pl):
    model_lags = pl.DataFrame(
        {
            "model": ["all_lags"] * len(lags_w_significant_positive_corr)
            + ["weekly_lags"] * len(lags_w_significant_positive_corr_week),
            "lag": lags_w_significant_positive_corr
            + lags_w_significant_positive_corr_week,
        },
    )
    return (model_lags,)


@app.cell
def _(Path, mo):
    save_button = mo.ui.button(label="Save models")
    model_path = Path("data/silver/naive_lag_models.parquet")
    return model_path, save_button


@app.cell
def _(model_lags, model_path, mo, save_button):
    if save_button.value:
        model_lags.write_parquet(model_path)
        mo.callout(f"Saved model lags to {model_path}.", kind="success")
    else:
        mo.callout(
            "Click 'Save models' to write naive lag models to disk.",
            kind="info",
        )
    return


@app.cell
def _(model_lags, mo):
    mo.md(
        "# Naive Lag Models\n"
        "- **all_lags**: mean of all significant positive ACF lags.\n"
        "- **weekly_lags**: mean of significant positive lags that are multiples of 7.\n",
    )
    model_lags
    return


if __name__ == "__main__":
    app.run()
