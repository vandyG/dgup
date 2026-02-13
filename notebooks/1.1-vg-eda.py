import marimo

__generated_with = "0.19.9"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo

    return


@app.cell
def _():
    import polars as pl
    import altair as alt

    return (pl,)


@app.cell
def _(pl):
    data_path = "data/silver/uta_gas_usage.parquet"
    gas_usage = pl.scan_parquet(data_path)
    return (gas_usage,)


@app.cell
def _():
    import os
    os.getcwd()
    return


@app.cell
def _(gas_usage):
    gas_usage.collect_schema()
    return


@app.cell
def _(pl):
    def polars_acf(df, col_name, lags):
        expressions = [
            pl.corr(col_name, pl.col(col_name).shift(i)).alias(f"lag_{i}")
            for i in range(lags + 1)
        ]
        return df.select(expressions)


    return (polars_acf,)


@app.cell
def _(gas_usage, polars_acf):
    acf = polars_acf(gas_usage, "Usage - 1", 400).collect().unpivot(value_name="correlation", variable_name="lag")
    acf.head()
    return (acf,)


@app.cell
def _(acf):
    acf.plot.line(x="lag", y="correlation")
    return


if __name__ == "__main__":
    app.run()
