import marimo

__generated_with = "0.19.9"
app = marimo.App(width="full")


@app.cell
def _():
    import datetime as dt
    from pathlib import Path

    import altair as alt
    import marimo as mo
    import polars as pl
    import polars.selectors as cs
    import statsmodels.tsa.stattools as tsa

    return Path, alt, cs, dt, mo, pl, tsa


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
    return (lf,)


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
    multiselect_types = mo.ui.dropdown(
        options=usage_types,
        label="Select Usage Type",
        value="Total Usage",
    )
    return (multiselect_types,)


@app.cell
def _(mo):
    scale = ["Yearly", "Monthly", "Bi-weekly", "Weekly"]
    multiselect_scale = mo.ui.dropdown(options=scale, label="Select Scale")
    return (multiselect_scale,)


@app.cell
def _(mo):
    number = mo.ui.number(start=7, stop=365, label="Rolling Average", value=365)
    return (number,)


@app.cell
def _(alt, dt, multiselect_scale):
    date_ranges = {
        "Yearly": (dt.date(2020, 1, 1), dt.date(2020, 12, 31)),
        "Monthly": (dt.date(2020, 6, 1), dt.date(2020, 6, 30)),
        "Weekly": (dt.date(2020, 6, 1), dt.date(2020, 6, 7)),
        "Bi-weekly": (dt.date(2020, 6, 1), dt.date(2020, 6, 14)),
    }
    date_range = date_ranges.get(
        multiselect_scale.value,
        (dt.date(2015, 8, 1), dt.date(2024, 11, 30)),
    )

    brush = alt.selection_interval(
        encodings=["x"],
        value={"x": date_range},
    )
    return (brush,)


@app.cell
def _(multiselect_types, number, pivot_data, pl):
    source_collected = pivot_data.collect()
    source = source_collected.filter(
        pl.col("Usage Type").is_in([multiselect_types.value]),
    )
    rolling_avg = source.with_columns(
        rolling_avg=pl.col("Usage").rolling_mean(window_size=number.value),
    )
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
        base = (
            alt.Chart(source, width="container", height=200)
            .mark_line(tooltip=True)
            .encode(
                x="Date:T",
                y="Usage:Q",
            )
        )

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

        lower = (
            base.encode(
                alt.X("Date:T", axis=alt.Axis(format="%Y", tickCount="year")),
            )
            .properties(height=60)
            .add_params(brush)
        )

        return upper & lower

    title = mo.md("# Gas Usage Explorer")
    desc = mo.md(
        "The lower panel provides an overview with a brush that controls the visible time window."
        " The upper panel shows a detailed view of the selected time range. Use the dropdowns to"
        " choose which usage type and time scale to inspect. The y-axis is shared between panels"
        " to keep vertical scaling consistent when comparing ranges.",
    )

    mo.vstack(
        [title, desc, mo.hstack([multiselect_types, number, multiselect_scale]), _()],
        gap="1",
    )
    return


@app.cell
def _(mo):
    usc_title = mo.md("## Usage Stream Characteristics")
    usc_desc = mo.md(
        "- **Usage-1** follows a standard seasonal cycle (Winter peaks/Summer dips)."
        " Notably, it shows a Saturday slump, suggesting a reduction in operational activity or"
        " a specific weekend shutdown process.\n"
        "- **Usage-2** represents the core consumption driver of the system and accounts for"
        " approximately 84% of total gas usage. This stream exhibits the highest degree of"
        " variability and stochastic noise, making it the dominant contributor to overall"
        " volatility. There is a stagnation period in late 2017 followed by sudden jumps in early"
        " 2018.\n"
        "- **Usage-2_1** is a minor consumption component that exhibits a persistent long-term"
        " decreasing trend. It remains largely inactive during summer months and shows a"
        " noticeable shift in activity levels after 2021, indicating a potential change in"
        " operational behavior or system configuration.",
    )

    stt_title = mo.md("## Seasonality and Temporal Trends")
    stt_desc = mo.md(
        "- **Annual Cyclicity:** Gas consumption is strongly driven by seasonal thermal demand."
        " Peak usage occurs during winter months, with January averaging approximately 2,099 units"
        " and February approximately 2,034 units. In contrast, summer consumption drops"
        " significantly, with July through September forming a baseline range of roughly 1,452 to"
        " 1,622 units. This reflects an annual seasonal swing of approximately 600 units.\n\n"
        "- **Weekly Periodicity:** System-wide gas usage is not evenly distributed across the week."
        " Multiple usage streams show a consistent reduction in consumption on Saturdays,"
        " confirming the presence of a weekly operational cycle. This pattern reinforces the"
        " importance of including day-of-week effects as a core modeling feature.",
    )

    mo.callout(mo.vstack([usc_title, usc_desc]), kind="info")
    return stt_desc, stt_title


@app.cell
def _(alt, multiselect_types, pivot_data, pl):
    def _():
        df_box = (
            pivot_data.filter(pl.col("Usage Type") == multiselect_types.value)
            .with_columns(
                pl.col("Date").dt.strftime("%b").alias("Month"),
            )
            .collect()
        )

        month_order = [
            "Jan",
            "Feb",
            "Mar",
            "Apr",
            "May",
            "Jun",
            "Jul",
            "Aug",
            "Sep",
            "Oct",
            "Nov",
            "Dec",
        ]

        chart = (
            alt.Chart(df_box)
            .mark_boxplot(
                extent=1.5,
                outliers=True,
                size=40,
                color="#4c78a8",
            )
            .encode(
                x=alt.X("Month:N", title="Month", sort=month_order),
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


    _()
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

    hist_desc = mo.md(
        "The density chart reveals that **Usage-2** and **Usage_Total** share a similar broad"
        " distribution, with dominant peaks occurring between approximately 1,400 and 2,000 units."
        "\n\n"
        "In contrast, **Usage-1** exhibits a much tighter distribution, clustering around roughly"
        " 200 units, which suggests more stable and narrowly bounded operational behavior."
        " **Usage-2_1** shows the highest density concentrated near zero, indicating that this"
        " stream is frequently inactive or operating at a minimal baseline for extended periods.",
    )

    mo.hstack([hist, hist_desc], widths=[0.65, 0.35])
    return


@app.cell
def _(mo, stt_desc, stt_title):
    mo.callout(mo.vstack([stt_title, stt_desc]), kind="info")
    return


@app.cell
def _(mo):
    options = ["Average Monthly Usage", "Total Monthly Usage"]
    radio = mo.ui.radio(options=options, value="Total Monthly Usage")
    return (radio,)


@app.cell
def _(alt, mo, multiselect_types, pivot_data, pl, radio):
    def heatmap_view(selected_type=multiselect_types.value, option="Total Monthly Usage"):
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

        text = base.mark_text(baseline="middle").encode(
            text=alt.Text("Total_Usage:Q", format=".0f"),
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
def _(alt, lf, mo, multiselect_types, pl):
    def _():
        col = multiselect_types.value
        df = lf.collect()

        temp = df.select(
            [
                pl.col(col).alias(col),
                pl.col(col).shift(1).alias("lag_1"),
                pl.col(col).shift(7).alias("lag_7"),
                pl.col(col).shift(30).alias("lag_30"),
            ],
        ).drop_nulls()

        cols = [col, "lag_1", "lag_7", "lag_30"]
        corr_rows = []
        for row_col in cols:
            row_vals = []
            for col_name in cols:
                row_vals.append(
                    temp.select(pl.corr(pl.col(row_col), pl.col(col_name))).item(),
                )
            corr_rows.append(row_vals)

        corr_df = pl.DataFrame(corr_rows, schema=cols).with_columns(pl.Series("Lag", cols))
        corr_long = corr_df.unpivot(
            index="Lag",
            variable_name="Variable",
            value_name="Correlation",
        )

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

        ch_title = mo.md("# Lagged Correlations")
        ch_desc = mo.md(
            "The heatmap shows the correlation between the selected usage type and its lagged values"
            " (1 day, 7 days, and 30 days). It displays the strength and direction of relationships"
            " between gas usage, accounting for time delays (lags) in their interaction.",
        )
        return mo.hstack(
            [
                mo.vstack([ch_title, ch_desc]),
                mo.vstack([multiselect_types.center(), mo.ui.altair_chart(heatmap + text).center()]),
            ],
        )


    _()
    return


@app.cell
def _(alt, lf, mo, pl):
    df_corr = lf.select(pl.exclude("Date")).collect()
    corr_cols = df_corr.columns

    corr_rows = []
    for row_col in corr_cols:
        row_vals = []
        for col_name in corr_cols:
            row_vals.append(
                df_corr.select(pl.corr(pl.col(row_col), pl.col(col_name))).item(),
            )
        corr_rows.append(row_vals)

    corr_df = pl.DataFrame(corr_rows, schema=corr_cols).with_columns(
        pl.Series("Row", corr_cols),
    )
    corr_long = corr_df.unpivot(
        index="Row",
        variable_name="Variable",
        value_name="Correlation",
    )

    corr_heatmap = (
        alt.Chart(corr_long)
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

    corr_text = (
        alt.Chart(corr_long)
        .mark_text(color="black")
        .encode(
            x="Variable:N",
            y="Row:N",
            text=alt.Text("Correlation:Q", format=".2f"),
            color=alt.condition(
                alt.datum.Correlation > corr_long["Correlation"].mean(),
                alt.value("white"),
                alt.value("black"),
            ),
        )
    )

    corr_chart = corr_heatmap + corr_text
    corr_title = mo.md("# Variable Correlations")
    corr_desc = mo.md(
        "**Total Usage** demonstrates a dominant linear relationship with **Usage - 2**"
        " ($r = 0.98$) and a moderate association with **Usage - 1** ($r = 0.78$). In contrast,"
        " all other usage-related metrics show weak negative correlations with delivery and"
        " supply data, highlighting a decoupling between operational consumption and supply"
        " measurements.",
    )

    mo.hstack([mo.ui.altair_chart(corr_chart).center(), mo.vstack([corr_title, corr_desc])])
    return


@app.cell
def _(mo):
    mo.md("""
    # Final Thoughts
    - The observed instances of excess gas usage support our hypothesis that temperature plays a significant role in driving gas consumption patterns.
    - Integrate calendar-based features such as month, week, weekday, and holidays to capture seasonality and weekend effects.
    - Add a long-term trend and rolling mean to account for overarching consumption patterns.
    - Implement lag-based features (7-day and 30-day lags) to mitigate short-term volatility.
    - Consider using the delivery-minus-usage gap as a potential feature to model operational risk.
    """)
    return


@app.cell
def _(alt, lf, mo, pl):
    def _():
        lf3 = lf.with_columns(
            (pl.col("Delivery").cum_sum() - pl.col("Total Usage").cum_sum())
            .shift(1)
            .fill_null(0)
            .alias("Total Gas Before"),
        )
        df_gap = lf3.select(["Date", "Total Gas Before"]).collect()

        chart = (
            alt.Chart(df_gap)
            .mark_line()
            .encode(
                x=alt.X("Date:T"),
                y=alt.Y("Total Gas Before:Q"),
                tooltip=[alt.Tooltip("Date:T"), alt.Tooltip("Total Gas Before:Q")],
            )
            .properties(width="container", height=300, title="Delivery vs Usage Gap")
        )

        assump = mo.md(
            "- **Assumption:** The gas storage tank starts empty.\n"
            "- **Observation:** The Total Gas Before metric goes negative at times, which is"
            " physically impossible. This indicates that the assumption of starting with an empty"
            " tank is incorrect, and there must be some initial gas in storage.",
        )
        return mo.vstack([mo.md("# Delivery vs Usage Gap"), chart, assump], gap="1")


    _()
    return


@app.cell(hide_code=True)
def _(mo):
    danger = mo.icon(icon_name="jam:triangle-danger", color="#FF0000")
    mo.md(
        "# Check for stationarity\n\n"
        "- Constant Mean\n"
        "- Constant Variance\n"
        f"- {danger}**<span style=\"color:#FF0000\">Seasonality</span>**",
    )
    return


@app.cell
def _(multiselect_types, pivot_data, pl):
    gas_usage_filtered = (
        pivot_data.filter(pl.col("Usage Type") == multiselect_types.value).collect()
    )
    return (gas_usage_filtered,)


@app.cell
def _(gas_usage_filtered, pl):
    gas_usage_roll_up = (
        gas_usage_filtered.group_by_dynamic(
            index_column="Date",
            every="1w",
            closed="right",
        )
        .agg(pl.col("Usage").mean())
    )
    return (gas_usage_roll_up,)


@app.cell
def _(alt, gas_usage_roll_up):
    def _():
        chart = (
            alt.Chart(gas_usage_roll_up)
            .mark_line()
            .encode(
                x=alt.X(field="Date", type="temporal", timeUnit="yearmonthdate"),
                y=alt.Y(field="Usage", type="quantitative"),
                tooltip=[
                    alt.Tooltip(field="Date", timeUnit="yearmonthdate", title="Date"),
                    alt.Tooltip(field="Usage", format=",.2f"),
                ],
            )
            .properties(
                height=290,
                width="container",
                config={"axis": {"grid": False}},
            )
            .interactive(bind_y=False)
        )
        return chart


    _()
    return


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
def _(gas_usage_filtered, pl, tsa):
    pacf_tsa = tsa.pacf(
        gas_usage_filtered.sort(by="Date")["Usage"].to_numpy(),
        nlags=650,
        alpha=0.05,
    )
    pacf_df = pl.DataFrame(
        {
            "lag": range(len(pacf_tsa[0])),
            "correlation": pacf_tsa[0],
            "lower_ci": pacf_tsa[1][:, 0] - pacf_tsa[0],
            "upper_ci": pacf_tsa[1][:, 1] - pacf_tsa[0],
        },
    )
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
            x=alt.X(field="lag", type="quantitative"),
            y=alt.Y(field="correlation", type="quantitative"),
            tooltip=[
                alt.Tooltip(field="lag", format=",.0f"),
                alt.Tooltip(field="correlation", format=",.2f"),
            ],
        )
    )

    ci = (
        alt.Chart(corr_dropdown.value)
        .mark_area(color="steelblue", opacity=0.5)
        .encode(
            x=alt.X("lag:Q"),
            y=alt.Y("lower_ci:Q"),
            y2=alt.Y2("upper_ci:Q"),
            tooltip=[
                alt.Tooltip("lower_ci:Q", format=",.2f", title="lower_ci"),
                alt.Tooltip("upper_ci:Q", format=",.2f", title="upper_ci"),
            ],
        )
    )

    chart = (
        alt.layer(bars, ci)
        .properties(height=290, width="container")
        .interactive(bind_y=False)
    )
    chart
    return


@app.cell
def _(Path, pl):
    model_path = Path("data/silver/naive_lag_models.parquet")
    if model_path.exists():
        model_lags = pl.read_parquet(model_path)
    else:
        model_lags = None
    return model_lags, model_path


@app.cell
def _(mo, model_lags, model_path):
    if model_lags is None:
        mo.callout(
            f"No saved naive lag models found at {model_path}. Run 1.2-vg-naive-lag-models.py"
            " to create the model artifacts.",
            kind="warn",
        )
    else:
        mo.callout(
            f"Loaded naive lag models from {model_path}.",
            kind="info",
        )
    return


app._unparsable_cell(
    r"""
    if model_lags is None:
        gas_usage_2024 = pl.DataFrame()
        return (gas_usage_2024,)

    lags_all = (
        model_lags.filter(pl.col("model") == "all_lags")
        .select("lag")
        .to_series()
        .to_list()
    )
    lags_week = (
        model_lags.filter(pl.col("model") == "weekly_lags")
        .select("lag")
        .to_series()
        .to_list()
    )

    def avg_lags_expr(lags):
        if not lags:
            return pl.lit(None)
        return pl.mean_horizontal([pl.col("Usage").shift(lag) for lag in lags])

    gas_usage_2024 = gas_usage_filtered.with_columns(
        pl.when(pl.col("Date").dt.year() == 2024)
        .then(avg_lags_expr(lags_all))
        .otherwise(pl.lit(None))
        .alias("avg_usage_plags"),
        pl.when(pl.col("Date").dt.year() == 2024)
        .then(avg_lags_expr(lags_week))
        .otherwise(pl.lit(None))
        .alias("avg_usage_plags_week"),
    )
    """,
    name="_"
)


@app.cell
def _(gas_usage_2024, pl):
    if gas_usage_2024.is_empty():
        predictions = pl.DataFrame()
    else:
        predictions = (
            gas_usage_2024.filter(pl.col("Date").dt.year() == 2024)
            .select(["Date", "Usage", "avg_usage_plags", "avg_usage_plags_week"])
            .unpivot(index="Date", variable_name="Type", value_name="Value")
        )
    return


app._unparsable_cell(
    r"""
    if predictions.is_empty():
        mo.callout("No predictions available. Generate models first.", kind="warn")
        return

    chart = (
        alt.Chart(predictions)
        .mark_line()
        .encode(
            x=alt.X(field="Date", type="temporal", timeUnit="yearmonthdate"),
            y=alt.Y(field="Value", type="quantitative", aggregate="mean"),
            color=alt.Color(field="Type", type="nominal"),
            tooltip=[
                alt.Tooltip(field="Date", timeUnit="yearmonthdate", title="Date"),
                alt.Tooltip(field="Value", aggregate="mean"),
                alt.Tooltip(field="Type"),
            ],
        )
        .properties(height=290, width="container")
        .interactive()
    )
    chart
    """,
    name="_"
)


app._unparsable_cell(
    r"""
    if gas_usage_2024.is_empty():
        errors_df = pl.DataFrame()
        return (errors_df,)

    g2024 = (
        gas_usage_2024.filter(pl.col("Date").dt.year() == 2024)
        .select(["Date", "Usage", "avg_usage_plags", "avg_usage_plags_week"])
        .drop_nulls()
    )

    melted = g2024.unpivot(
        index=["Date", "Usage"],
        variable_name="model",
        value_name="yhat",
    ).with_columns(
        (pl.col("yhat") - pl.col("Usage")).alias("err"),
        pl.when(pl.col("Usage") == 0)
        .then(pl.lit(None))
        .otherwise(pl.col("Usage").abs())
        .alias("mape_denom"),
    ).with_columns(
        (2 * pl.col("err").abs() / (pl.col("Usage").abs() + pl.col("yhat").abs() + 1e-9))
        .alias("smape_elem"),
        (pl.col("err").abs() / pl.col("mape_denom")).alias("mape_elem"),
    )

    errors_df = (
        melted.group_by("model")
        .agg(
            [
                pl.col("err").abs().mean().alias("MAE"),
                pl.col("err").pow(2).mean().alias("MSE"),
                pl.col("err").pow(2).mean().sqrt().alias("RMSE"),
                (pl.col("mape_elem").mean() * 100).alias("MAPE_pct"),
                (pl.col("smape_elem").mean() * 100).alias("sMAPE_pct"),
                pl.col("err").mean().alias("Bias"),
                pl.len().alias("N"),
            ],
        )
    )
    """,
    name="_"
)


@app.cell
def _(errors_df, mo):
    if errors_df.is_empty():
        mo.callout("No error metrics available. Generate models first.", kind="warn")
    else:
        errors_df
    return


if __name__ == "__main__":
    app.run()
