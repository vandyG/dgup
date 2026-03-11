# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo>=0.19.10",
#     "pyzmq>=27.1.0",
# ]
# ///

import marimo

__generated_with = "0.20.4"
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
    import statsmodels.api as sm
    import statsmodels.tsa.stattools as tsa


    return alt, cast, cs, dt, mo, pd, pl, tsa


@app.cell
def _(cast, pl):
    # Read uta_gas_usage.parquet using a Polars lazy frame
    lf = cast("pl.LazyFrame", pl.scan_parquet("/home/vandy/Work/DASC5309/scot-forge/data/silver/uta_gas_usage.parquet"))

    lf2 = lf.fill_null(0).with_columns((pl.col("Usage - 1") + pl.col("Usage - 2") + pl.col("Usage - 2_1")).alias("Total Usage"))
    # lf2.collect_schema()
    return (lf2,)


@app.cell
def _(cs, lf2, pl):
    pivot_data = lf2.select(pl.exclude(["Nom", "Delivery"])).unpivot(
        index="Date",
        on=cs.float(),
        variable_name="Usage Type",
        value_name="Usage",
    )
    # pivot_data.collect_schema()
    return (pivot_data,)


@app.cell
def _(mo, pivot_data):
    usage_types = pivot_data.select("Usage Type").unique().collect().to_series().to_list()
    multiselect_types = mo.ui.dropdown(options=usage_types, label="Select Usage Types", value="Total Usage")
    return (multiselect_types,)


@app.cell
def _(mo):
    scale = ["Yearly", "Monthly", "Bi-weekly", "Weekly"]
    multiselect_scale = mo.ui.dropdown(options=scale, label="Select Scale")
    return (multiselect_scale,)


@app.cell
def _(alt, dt, multiselect_scale):
    date_ranges = {
        "Yearly": (dt.date(2020, 1, 1), dt.date(2020, 12, 31)),
        "Monthly": (dt.date(2020, 6, 1), dt.date(2020, 6, 30)),
        "Weekly": (dt.date(2020, 6, 1), dt.date(2020, 6, 7)),
        "Bi-weekly": (dt.date(2020, 6, 1), dt.date(2020, 6, 14)),
    }
    # Define initial date range to select
    date_range = date_ranges.get(multiselect_scale.value, (dt.date(2015, 8, 1), dt.date(2024, 11, 30)))

    # Create interval selection with initial value
    brush = alt.selection_interval(
        encodings=["x"],
        value={"x": date_range},
    )
    return (brush,)


@app.cell
def _(mo):
    number = mo.ui.number(start=7, stop=365, label="Rolling Average", value=365)
    return (number,)


@app.cell
def _(multiselect_types, number, pivot_data, pl):
    source_collected = pivot_data.collect()
    source = source_collected.filter(pl.col("Usage Type").is_in([multiselect_types.value]))
    rolling_avg = source.with_columns(rolling_avg=pl.col("Usage").rolling_mean(window_size=number.value))
    return rolling_avg, source


@app.cell
def _(
    alt,
    brush,
    mo,
    multiselect_scale,
    multiselect_types,
    number,
    rolling_avg,
    source,
):
    def _():
        # Create base chart for both panels
        base = (
            alt.Chart(source, width="container", height=200)
            .mark_line(tooltip=True)
            .encode(
                x="Date:T",
                y="Usage:Q",
            )
        )

        # Upper panel shows detailed view filtered by the brush
        upper_base = base.encode(
            alt.X("Date:T").scale(domain=brush),
        )

        upper_avg = (
            alt.Chart(rolling_avg, width="container", height=200)
            .mark_line(tooltip=True, strokeDash=[4, 2], color="#d62728")
            .encode(
                x=alt.X("Date:T").scale(domain=brush),
                y="rolling_avg:Q",
            )
        )

        upper = alt.layer(upper_base, upper_avg)

        # Lower panel shows overview with the brush control
        lower = (
            base.encode(
                alt.X("Date:T", axis=alt.Axis(format="%Y", tickCount="year")),
            )
            .properties(
                height=60,
            )
            .add_params(brush)
        )

        # Combine the two charts
        return upper & lower

    title = mo.md(r"""# Gas Usage Explorer""")
    desc = mo.md(
        r"""The lower panel provides an overview with a brush (selection) that controls the visible time window. The upper panel shows a detailed view of the selected time range. Use the dropdowns to choose which usage types and time scale to inspect. The y-axis is shared between panels to keep vertical scaling consistent when comparing ranges.""",
    )

    mo.vstack([title, desc, mo.hstack([multiselect_types, number, multiselect_scale]), _()], gap="1")
    return


@app.cell
def _(mo):
    usc_title = mo.md(r"""## Usage Stream Characteristics""")
    usc_desc = mo.md(r"""- **Usage-1** follows a standard seasonal cycle (Winter peaks/Summer dips). Notably, it shows a "Saturday Slump," suggesting a reduction in operational activity or a specific weekend shutdown process.
    - **Usage-2** represents the core consumption driver of the system and accounts for approximately 84% of total gas usage. This stream exhibits the highest degree of variability and stochastic noise, making it the dominant contributor to overall volatility. There is a stagnation period in late 2017 followed by sudden "jumps" in early 2018.
    - **Usage-2_1** is a minor consumption component that exhibits a persistent long-term decreasing trend. It remains largely inactive during summer months and shows a noticeable shift in activity levels after 2021, indicating a potential change in operational behavior or system configuration.""")

    stt_title = mo.md(r"""## Seasonality and Temporal Trends""")
    stt_desc = mo.md(r"""- **Annual Cyclicity:** Gas consumption is strongly driven by seasonal thermal demand. Peak usage occurs during winter months, with January averaging approximately 2,099 units and February approximately 2,034 units. In contrast, summer consumption drops significantly, with July through September forming a baseline range of roughly 1,452 to 1,622 units. This reflects an annual seasonal swing of approximately 600 units.

    - **Weekly Periodicity:** System-wide gas usage is not evenly distributed across the week. Multiple usage streams show a consistent reduction in consumption on Saturdays, confirming the presence of a weekly operational cycle. This pattern reinforces the importance of including day-of-week effects as a core modeling feature.
    """)

    mo.callout(mo.vstack([usc_title, usc_desc]), kind="info")
    return stt_desc, stt_title


@app.cell
def _(alt, multiselect_types, pivot_data, pl):
    def boxplot_chart():
        # 1. Prepare data
        df_box = (
            pivot_data.filter(pl.col("Usage Type") == multiselect_types.value)
            .with_columns(
                [
                    pl.col("Date").dt.strftime("%b").alias("Month"),
                ],
            )
            .collect()
        )

        # Define the explicit order for the X-axis
        month_order = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

        # 2. Build the Box Plot
        chart = (
            alt.Chart(df_box)
            .mark_boxplot(
                extent=1.5,  # Standard IQR whiskers
                outliers=True,
                size=40,
                color="#4c78a8",
            )
            .encode(
                x=alt.X("Month:N", title="Month", sort=month_order),  # Explicitly set the calendar order
                y=alt.Y("Usage:Q", title=f"Usage: {multiselect_types.value}"),
                tooltip=["Date:T", "Usage:Q"],
            )
            .properties(
                width="container",
                height=450,
                title=f"Monthly Box-plots: {multiselect_types.value}",
            )
            .interactive()
        )

        return chart

    # Display the result
    boxplot_chart()
    return


@app.cell
def _(alt, mo, pivot_data):
    hist = (
        alt.Chart(pivot_data.collect())
        .mark_bar(tooltip=True, opacity=0.6, binSpacing=0)
        .encode(
            alt.X("Usage:Q", axis=alt.Axis(labelAngle=45)).bin(maxbins=50),
            alt.Y("count()").stack(None),
            alt.Color("Usage Type:N"),
        )
    )

    hist_desc = mo.md(r"""
    The density chart reveals that **Usage-2** and **Usage_Total** share a similar broad distribution, with dominant peaks occurring between approximately 1,400 and 2,000 units.

    In contrast, **Usage-1** exhibits a much tighter distribution, clustering around roughly 200 units, which suggests more stable and narrowly bounded operational behavior. **Usage-2_1** shows the highest density concentrated near zero, indicating that this stream is frequently inactive or operating at a minimal baseline for extended periods.

    """)

    mo.hstack([hist, hist_desc], widths=[0.65, 0.35])
    return


@app.cell
def _(mo, stt_desc, stt_title):
    mo.callout(mo.vstack([stt_title, stt_desc]), kind="info")
    return


@app.cell
def _():
    # TODO: Add heatmaps.
    # TODO: Add lag visualizations.
    # TODO: Add box plots.
    return


@app.cell
def _(mo):
    options = ["Average Monthly Usage", "Total Monthly Usage"]
    radio = mo.ui.radio(options=options, value="Total Monthly Usage")
    return (radio,)


@app.cell
def _(alt, mo, multiselect_types, pivot_data, pl, radio):
    def heatmap_view(selected_type=multiselect_types.value, option="Total Monthly Usage"):
        # Process the data for the heatmap
        # We extract Year and Month parts from the Date

        agg_func = pl.Expr.sum if option == "Total Monthly Usage" else pl.Expr.mean
        df_hm = (
            pivot_data.filter(pl.col("Usage Type") == selected_type)
            .with_columns(
                [
                    pl.col("Date").dt.year().alias("Year"),
                    pl.col("Date").dt.month().alias("Month_Num"),
                    pl.col("Date").dt.strftime("%b").alias("Month"),
                ],
            )
            .group_by(["Year", "Month", "Month_Num"])
            .agg(agg_func(pl.col("Usage")).alias("Total_Usage"))
            .sort("Month_Num")
            .collect()
        )

        # Create the Heatmap Base
        base = (
            alt.Chart(df_hm)
            .mark_rect()
            .encode(
                x=alt.X("Year:O", title="Year", axis=alt.Axis(labelAngle=0)),
                y=alt.Y("Month:N", title="Month", sort=alt.SortField("Month_Num")),
                color=alt.Color(
                    "Total_Usage:Q",
                    scale=alt.Scale(scheme="yelloworangered"),
                    title="Usage Intensity",
                ),
                tooltip=["Year", "Month", "Total_Usage"],
            )
            .properties(
                width="container",
                height=350,
                title=f"{option} Intensity: {selected_type}",
            )
        )

        # Add Text Annotations (annot=True equivalent)
        text = base.mark_text(baseline="middle").encode(
            text=alt.Text("Total_Usage:Q", format=".0f"),
            # Dynamic text color: black on light cells, white on dark cells
            color=alt.condition(
                alt.datum.Total_Usage > df_hm["Total_Usage"].mean(),
                alt.value("white"),
                alt.value("black"),
            ),
        )

        return base + text

    mo.hstack([heatmap_view(option=radio.value), radio], widths=[1, 0.15])
    return


@app.cell
def _(alt, lf2, mo, multiselect_types, pd):
    col = multiselect_types.value

    # df = pd.read_parquet("/home/vandy/Work/DASC5309/scot-forge/data/silver/uta_gas_usage.parquet")
    # df["Total Usage"] = (
    #     df["Usage - 1"] +
    #     df["Usage - 2"] +
    #     df["Usage - 2_1"]
    # )
    # df.head()

    df = lf2.collect().to_pandas()

    # Create lag columns
    temp = pd.DataFrame(
        {
            col: df[col],
            "lag_1": df[col].shift(1),
            "lag_7": df[col].shift(7),
            "lag_30": df[col].shift(30),
        },
    )

    # Correlation matrix
    corr = temp[[col, "lag_1", "lag_7", "lag_30"]].corr()

    # Convert to long format
    corr_long = (
        corr.reset_index()
        .melt(id_vars="index", var_name="Variable", value_name="Correlation")
        .rename(columns={"index": "Lag"})
    )

    # Heatmap
    heatmap = (
        alt.Chart(corr_long)
        .mark_rect()
        .encode(
            x=alt.X("Variable:N", title=""),
            y=alt.Y("Lag:N", title=""),
            color=alt.Color(
                "Correlation:Q",
                scale=alt.Scale(scheme="blues"),
                title="Correlation",
            ),
            tooltip=[
                alt.Tooltip("Lag:N"),
                alt.Tooltip("Variable:N"),
                alt.Tooltip("Correlation:Q", format=".3f"),
            ],
        )
        .properties(
            title=f"Correlation Heatmap: {col} vs Lags",
            width=400,
            height=400,
        )
    )

    # Add text annotations (like annot=True)
    text = (
        alt.Chart(corr_long)
        .mark_text(color="black")
        .encode(
            x="Variable:N",
            y="Lag:N",
            text=alt.Text("Correlation:Q", format=".3f"),
            color=alt.condition(
                alt.datum.Correlation > corr_long["Correlation"].mean(),
                alt.value("white"),
                alt.value("black"),
            ),
        )
    )
    ch_title = mo.md(r"""
    # Lagged Correlations
    """)
    ch_desc = mo.md(r"""
    The heatmap shows the correlation between the selected usage type and its lagged values (1 day, 7 days, and 30 days). It displays the strength and direction of relationships between Gas Usage, accounting for time delays (lags) in their interaction.""")

    ch_obs = mo.md(r"""- **Usage-1** Usage-1 displays a stronger relationship with its 7-day lag (0.824) than with its 1-day lag (0.718), providing statistical evidence of a consistent weekly cycle. This pattern aligns with the observed “Saturday Slump,” where reduced weekend activity drives recurring weekly dips in consumption.

    - **Usage-2** The main consumption stream shows moderate dependence on both 1-day (0.610) and 7-day (0.636) lags but experiences a complete loss of predictive signal at the 30-day horizon (-0.042). This behavior indicates that Usage-2 is governed by immediate operational conditions rather than long-term temporal trends.

    - **Usage-2_1** This stream exhibits exceptionally strong correlation of 0.985 with its 1-day lag. Even at a 30-day horizon, Usage-2_1 retains a substantial correlation of 0.808, suggesting slow-moving behavior and limited sensitivity to short-term operational fluctuations.

    - **Total Usage** Usage_Total closely mirrors the dynamics of the Usage-2, with the 7-day lag (0.691) serving as the strongest predictive anchor. In contrast, the 30-day lag contributes virtually no explanatory power (0.015), reinforcing the conclusion that weekly patterns dominate system-wide gas usage behavior.
    """)

    mo.hstack(
        [
            mo.vstack([ch_title, ch_desc, ch_obs]),
            mo.vstack([multiselect_types.center(), mo.ui.altair_chart(heatmap + text).center()]),
        ],
    )
    return


@app.cell
def _(alt, lf2, mo, pl):
    # # Columns to include
    # columns_to_corr = ["Usage_Total","Usage - 1","Usage - 2","Usage - 2_1", "Nom", "Delivery"]

    # Compute correlation matrix (store in a new variable)
    # corr_matrix_alt = df[columns_to_corr].corr()
    corr_matrix_alt = lf2.select(pl.exclude("Date")).collect().to_pandas().corr()

    # Convert to long format
    corr_long_alt = (
        corr_matrix_alt.reset_index()
        .melt(id_vars="index", var_name="Variable", value_name="Correlation")
        .rename(columns={"index": "Row"})
    )

    # Altair heatmap chart (unique variable names)
    corr_heatmap_alt = (
        alt.Chart(corr_long_alt)
        .mark_rect()
        .encode(
            x=alt.X("Variable:N", title=""),
            y=alt.Y("Row:N", title=""),
            color=alt.Color("Correlation:Q", scale=alt.Scale(scheme="blues")),
            tooltip=[
                alt.Tooltip("Row:N", title="Row"),
                alt.Tooltip("Variable:N", title="Column"),
                alt.Tooltip("Correlation:Q", format=".2f"),
            ],
        )
        .properties(
            width=400,
            height=400,
            title="Correlation Matrix",
        )
    )

    # Add text annotations (unique variable names)
    corr_text_alt = (
        alt.Chart(corr_long_alt)
        .mark_text(color="black")
        .encode(
            x="Variable:N",
            y="Row:N",
            text=alt.Text("Correlation:Q", format=".2f"),
            color=alt.condition(
                alt.datum.Correlation > corr_long_alt["Correlation"].mean(),
                alt.value("white"),
                alt.value("black"),
            ),
        )
    )

    corr_chart = corr_heatmap_alt + corr_text_alt
    corr_title = mo.md(r"# Variable Correlations")
    corr_desc = mo.md(r"""**Usage_Total** demonstrates a dominant linear relationship with **Usage - 2** ($r = 0.98$) and a moderate association with **Usage - 1** ($r = 0.78$). In contrast, all other usage-related metrics show weak negative correlations with the delivery and supply data, highlighting a decoupling between operational consumption and supply measurements
    """)

    mo.hstack([mo.ui.altair_chart(corr_chart).center(), mo.vstack([corr_title, corr_desc])])
    return


@app.cell
def _(mo):
    mo.md(r"""
    # Final Thoughts
    - The observed instances of excess gas usage support our hypothesis that **temperature** plays a significant role in driving gas consumption patterns.
    - Integrate calendar-based features such as **month, week, weekday**, and **holidays** to capture seasonality and weekend effects.
    - Add a **long-term trend** and **rolling mean** to account for overarching consumption patterns.
    - Implement **lag-based features** (7-day and 30-day lags) to mitigate short-term volatility.
    - Consider using the **delivery-minus-usage gap** as a potential feature to model operational risk.
    """)
    return


@app.cell
def _(lf2, mo, pl):
    delivery_usage_diff = (pl.col("Delivery").cum_sum() - pl.col("Total Usage").cum_sum())
    lf3 = lf2.with_columns(
        delivery_usage_diff
        .shift(1)
        .fill_null(0)
        .alias("Total Gas Before"),
        delivery_usage_diff.fill_null(0).alias("Total Gas After"),
    )

    dug_title = mo.md(r"""# Delivery vs Usage Gap""")
    dug = lf3.collect().plot.line(x="Date", y="Total Gas Before")
    assump = mo.md(r"""
    - **Assumption:** The gas storage tank starts empty.
    - **Observation:** The "Total Gas Before" metric goes negative at times, which is physically impossible. This indicates that the assumption of starting with an empty tank is incorrect, and there must be some initial gas in storage""")

    mo.vstack([dug_title, mo.ui.altair_chart(dug), assump], gap="1")
    return (lf3,)


@app.cell
def _(lf3, pl):
    lf3.with_columns(
        pl.col("Total Gas Before") + 81569.123976,
        pl.col("Total Gas After") + 81569.123976,
    )
    return


@app.cell
def _(mo):
    danger = mo.icon(icon_name="jam:triangle-danger", color="#FF0000")
    mo.md(f"""
    # Check for stationarity

    - Constant Mean
    - Constant Variance
    - {danger}**<span style="color:#FF0000">Seasonality</span>**
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # ACF & PACF
    """)
    return


@app.cell
def _(mo):
    lag_slider = mo.ui.slider(7, 3500, value=400)
    return (lag_slider,)


@app.cell
def _(lag_slider, pl, source, tsa):
    acf_tsa = tsa.acf(source.sort(by="Date")["Usage"].to_numpy(), nlags=lag_slider.value, alpha=0.05)
    acf_df = pl.DataFrame({
        "lag": range(len(acf_tsa[0])),
        "correlation": acf_tsa[0],
        "lower_ci": acf_tsa[1][:, 0] - acf_tsa[0],
        "upper_ci": acf_tsa[1][:, 1] - acf_tsa[0],   
    })
    return (acf_df,)


@app.cell
def _(lag_slider, pl, source, tsa):
    pacf_tsa = tsa.pacf(source.sort(by="Date")["Usage"].to_numpy(), nlags=lag_slider.value, alpha=0.05)
    pacf_df = pl.DataFrame({
        "lag": range(len(pacf_tsa[0])),
        "correlation": pacf_tsa[0],
        "lower_ci": pacf_tsa[1][:, 0] - pacf_tsa[0],
        "upper_ci": pacf_tsa[1][:, 1] - pacf_tsa[0],   
    })
    return (pacf_df,)


@app.cell
def _(acf_df, mo, pacf_df):
    corr_dropdown = mo.ui.radio({"acf": acf_df, "pacf": pacf_df}, value="acf", inline=True)
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

    acf_chart = alt.layer(bars, ci).properties(
        height=290,
        width='container',
    ).interactive(bind_y=False)
    return (acf_chart,)


@app.cell
def _(acf_chart, corr_dropdown, lag_slider, mo, multiselect_types):
    mo.vstack([mo.hstack([corr_dropdown, lag_slider ,multiselect_types]), acf_chart])
    return


if __name__ == "__main__":
    app.run()
