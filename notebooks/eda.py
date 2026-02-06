# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo>=0.19.0",
#     "pyzmq>=27.1.0",
# ]
# ///

import marimo

__generated_with = "0.19.7"
app = marimo.App(width="full")


@app.cell
def _():
    from typing import cast

    import altair as alt
    import marimo as mo
    import polars as pl
    import polars.selectors as cs
    import datetime as dt
    return alt, cast, cs, dt, mo, pl


@app.cell
def _(cast, pl):
    # Read uta_gas_usage.parquet using a Polars lazy frame
    lf= cast("pl.LazyFrame", pl.scan_parquet("/home/vandy/Work/DASC5309/scot-forge/data/silver/uta_gas_usage.parquet"))

    lf2 = lf.with_columns((pl.col("Usage - 1")+pl.col("Usage - 2")+pl.col("Usage - 2_1")).alias("Total Usage"))
    # lf2.collect_schema()
    return (lf2,)


@app.cell
def _(cs, lf2, pl):
    pivot_data = lf2 \
        .select(pl.exclude(["Nom", "Delivery"])) \
        .unpivot(index="Date", 
                 on= cs.float(), 
                 variable_name="Usage Type", 
                 value_name="Usage",
                )
    # pivot_data.collect_schema()
    return (pivot_data,)


@app.cell
def _(mo, pivot_data):
    usage_types = pivot_data.select("Usage Type").unique().collect().to_series().to_list()
    multiselect_types = mo.ui.dropdown(options=usage_types, label="Select Usage Types")
    return (multiselect_types,)


@app.cell
def _(mo):
    scale = ["Yearly", "Monthly", "Bi-weekly", "Weekly"]
    multiselect_scale = mo.ui.dropdown(options=scale, label="Select Scale", value="Yearly")
    return (multiselect_scale,)


@app.cell
def _(alt, dt, mo, multiselect_scale, multiselect_types, pivot_data, pl):
    def _():
        source = pivot_data.filter(pl.col("Usage Type").is_in([multiselect_types.value])).collect()
        rolling_avg = source.with_columns(rolling_avg = pl.col("Usage").rolling_mean(window_size=7))

        date_ranges = {
            "Yearly": (dt.date(2020,1,1), dt.date(2020,12,31)),
            "Monthly": (dt.date(2020,6,1), dt.date(2020,6,30)),
            "Weekly": (dt.date(2020,6,1), dt.date(2020,6,7)),
            "Bi-weekly": (dt.date(2020,6,1), dt.date(2020,6,14)),
        }
        # Define initial date range to select
        date_range = date_ranges.get(multiselect_scale.value, (dt.date(2020,1,1), dt.date(2020,12,31)))

        # Create interval selection with initial value
        brush = alt.selection_interval(
            encodings=['x'],
            value={'x': date_range}
        )

        # Create base chart for both panels
        base = alt.Chart(rolling_avg, width="container", height=200).mark_line(tooltip=True).encode(
            x = 'Date:T',
            y = 'Usage:Q'
        )

        # Upper panel shows detailed view filtered by the brush
        upper_base = base.encode(
            alt.X('Date:T').scale(domain=brush)
        )

        upper_avg = alt.Chart(rolling_avg, width="container", height=200).mark_line(tooltip=True, strokeDash=[4, 2], color='#d62728').encode(
            x = alt.X('Date:T').scale(domain=brush),
            y = 'rolling_avg:Q'
        )

        upper = alt.layer(upper_base, upper_avg)

        # Lower panel shows overview with the brush control
        lower = base.encode(
            alt.X('Date:T', axis=alt.Axis(format='%Y', tickCount='year'))
        ).properties(
            height=60
        ).add_params(brush)


        # Combine the two charts
        return upper & lower

    title = mo.md(r"""# Gas Usage Explorer""")
    desc = mo.md(r"""The lower panel provides an overview with a brush (selection) that controls the visible time window. The upper panel shows a detailed view of the selected time range. Use the dropdowns to choose which usage types and time scale to inspect. The y-axis is shared between panels to keep vertical scaling consistent when comparing ranges.""")

    mo.vstack([title, desc, mo.hstack([multiselect_types, multiselect_scale]), _()], gap="1")
    return


@app.cell
def _():
    # TODO: Add heatmaps.
    # TODO: Add lag visualizations.
    # TODO: Add box plots.
    return


@app.cell
def _(mo):
    mo.md(r"""
    # Usage 1
    - Usage peaks during winter season and dips during summer.
    - Dips on saturday.

    # Usage 2_1
    - Usage peaks during winter season and barely running during summer.
    - Overall decreasing trend.

    # Usage 2
    - Usage peaks during winter season and dips during summer.
    - Dips on saturday.
    - Anomaly towards late 2017 and early 2018 (January). Stagnation -> Sudden jump.

    Note: **More Noise**
    """)
    return


@app.cell
def _(alt, pivot_data):
    alt.Chart(data=pivot_data.collect()).mark_rect().encode(
        x="year(Date):Q",
        y="month(Date):O",
        color="sum(Usage):Q",
    )
    return


@app.cell
def _(lf2, pl):
    lf3 = lf2.with_columns((pl.col("Delivery").cum_sum() - pl.col("Total Usage").cum_sum()).shift(1).fill_null(0).alias("Total Gas Before"))
    lf3.collect().plot.line(x="Date", y="Total Gas Before")
    return


if __name__ == "__main__":
    app.run()
