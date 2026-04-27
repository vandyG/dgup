import marimo

__generated_with = "0.21.0"
app = marimo.App(width="full")


@app.cell
def _():
    from pathlib import Path

    import altair as alt
    import marimo as mo
    import polars as pl

    import dgup

    return Path, alt, dgup, mo, pl


@app.cell
def _(mo):
    mo.md(
        r"""
        # Delivery Optimization — Storage Bank Analysis

        Scot Forge holds a **Storage Banking Service (SBS)** account with Nicor Gas.
        When delivered gas exceeds daily usage it is injected into the bank; when
        usage exceeds delivery the shortfall is withdrawn.  Nicor imposes **daily
        activity limits** and **month-end inventory bands** — violating either
        triggers penalty charges.

        This notebook answers two questions for the operations team:

        1. **How many penalties occurred historically** under the as-delivered
           nomination schedule?
        2. **How many penalties would be avoided** with the LP-optimized delivery
           schedule, which uses a 7-day quantile forecast and accounts for the
           Nicor contract bands in full?

        > The optimizer runs **daily** (each morning before the 1 PM nomination
        > deadline) and **retrains its forecast model every Monday**.
        """
    )
    return


@app.cell
def _(Path, dgup, pl):
    data_path = Path("data/silver/uta_gas_usage.parquet")
    raw = (
        pl.scan_parquet(data_path)
        .with_columns(
            (
                pl.col("Usage - 1")
                + pl.col("Usage - 2")
                + pl.col("Usage - 2_1").fill_null(0)
            ).alias("total_usage")
        )
        .collect()
    )

    # Simulate storage with historical (actual) deliveries
    historical_sim = dgup.simulate_storage_v1(raw)
    historical_sim = dgup.compute_daily_penalty_v1(historical_sim)
    historical_sim = dgup.compute_monthly_penalty_v1(historical_sim)
    historical_sim.head(5)
    return data_path, historical_sim, raw


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Storage Bank — Historical Simulation

        Using the **actual delivery values** recorded in the dataset, we can
        reconstruct what the storage bank inventory looked like every day.

        The coloured bands in the chart below show Nicor's **month-end inventory
        requirements** — the green corridor is where the bank balance must land on
        the last day of each month.  Straying outside triggers a penalty.
        """
    )
    return


@app.cell
def _(alt, dgup, historical_sim, mo, pl):
    def _make_inventory_chart(sim_df, title="Historical Inventory vs. Month-End Bands"):
        # Monthly band polygons (one rectangle per calendar month across the dataset)
        months_in_data = (
            sim_df.with_columns(pl.col("Date").dt.year().alias("year"), pl.col("Date").dt.month().alias("month_num"))
            .select(["year", "month_num", "Date"])
            .group_by(["year", "month_num"])
            .agg(
                pl.col("Date").min().alias("month_start"),
                pl.col("Date").max().alias("month_end"),
            )
        )

        band_rows = []
        for row in months_in_data.iter_rows(named=True):
            m = row["month_num"]
            band_rows.append(
                {
                    "month_start": row["month_start"],
                    "month_end": row["month_end"],
                    "min_inv": dgup.MONTHLY_MIN[m],
                    "max_inv": dgup.MONTHLY_MAX[m],
                }
            )
        band_df = pl.DataFrame(band_rows)

        band_chart = (
            alt.Chart(band_df)
            .mark_rect(opacity=0.12, color="#2ca02c")
            .encode(
                x=alt.X("month_start:T"),
                x2=alt.X2("month_end:T"),
                y=alt.Y("min_inv:Q", title="Inventory (therms)"),
                y2=alt.Y2("max_inv:Q"),
                tooltip=[
                    alt.Tooltip("month_start:T", title="Month start"),
                    alt.Tooltip("min_inv:Q", title="Min target", format=",.0f"),
                    alt.Tooltip("max_inv:Q", title="Max target", format=",.0f"),
                ],
            )
        )

        inv_line = (
            alt.Chart(sim_df)
            .mark_line(strokeWidth=1.2, color="#1f77b4")
            .encode(
                x=alt.X("Date:T", title="Date"),
                y=alt.Y("inventory:Q", title="Inventory (therms)"),
                tooltip=["Date:T", alt.Tooltip("inventory:Q", format=",.0f")],
            )
        )

        # Penalty markers
        penalty_df = sim_df.filter(pl.col("monthly_penalty"))
        penalty_marks = (
            alt.Chart(penalty_df)
            .mark_point(shape="triangle-up", color="#d62728", size=80)
            .encode(
                x="Date:T",
                y="inventory:Q",
                tooltip=["Date:T", alt.Tooltip("inventory:Q", format=",.0f")],
            )
        )

        combined = (
            alt.layer(band_chart, inv_line, penalty_marks)
            .properties(title=title, width="container", height=380)
            .interactive(bind_y=False)
        )
        return mo.ui.altair_chart(combined)

    _make_inventory_chart(historical_sim)
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        **Red triangles** mark months where the inventory closed outside the
        permitted band — each triangle is a **monthly penalty event**.  The green
        shaded corridors show the allowable range for each month.
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Historical Penalty Count

        Let's count how many penalty events occurred in the historical data:

        - **Monthly penalties** are the hard constraint — Nicor charges the most
          for these.  The target is zero.
        - **Daily penalties** are softer but still undesirable; they occur when
          daily injection or withdrawal exceeds the activity limit.
        """
    )
    return


@app.cell
def _(alt, historical_sim, mo, pl):
    def _make_penalty_bar(sim_df, label="Historical"):
        yearly = (
            sim_df.with_columns(pl.col("Date").dt.year().alias("year"))
            .group_by("year")
            .agg(
                pl.col("daily_penalty").sum().alias("daily_penalties"),
                pl.col("monthly_penalty").sum().alias("monthly_penalties"),
            )
            .sort("year")
            .unpivot(
                index="year",
                on=["daily_penalties", "monthly_penalties"],
                variable_name="Penalty Type",
                value_name="Count",
            )
        )

        bar = (
            alt.Chart(yearly)
            .mark_bar()
            .encode(
                x=alt.X("year:O", title="Year"),
                y=alt.Y("Count:Q", title="Penalty Events"),
                color=alt.Color(
                    "Penalty Type:N",
                    scale=alt.Scale(
                        domain=["daily_penalties", "monthly_penalties"],
                        range=["#ff7f0e", "#d62728"],
                    ),
                    legend=alt.Legend(title="Penalty Type"),
                ),
                column=alt.Column("Penalty Type:N", title=""),
                tooltip=["year:O", "Penalty Type:N", "Count:Q"],
            )
            .properties(
                title=f"{label} Penalty Events per Year",
                height=280,
            )
        )
        return bar

    mo.ui.altair_chart(_make_penalty_bar(historical_sim))
    return


@app.cell
def _(historical_sim, mo, pl):
    total_daily_hist = int(historical_sim["daily_penalty"].sum())
    total_monthly_hist = int(historical_sim["monthly_penalty"].sum())
    years_in_data = historical_sim["Date"].dt.year().n_unique()

    mo.callout(
        mo.md(
            f"""
            ### Historical Penalty Summary

            | Metric | Total | Per Year (avg) |
            |---|---|---|
            | **Monthly penalties** | {total_monthly_hist} | {total_monthly_hist/years_in_data:.1f} |
            | **Daily penalties** | {total_daily_hist} | {total_daily_hist/years_in_data:.1f} |
            """
        ),
        kind="warn",
    )
    return total_daily_hist, total_monthly_hist, years_in_data


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Optimized Delivery Schedule

        The LP optimizer plans **7 deliveries at once** using the champion
        model's Q25/Q50/Q75 forecast.  It selects the forecast quantile based
        on which monthly constraint is most at risk:

        | Months | Active Quantile | Reason |
        |---|---|---|
        | Jan, Feb, Nov, Dec | **Q75** (upper) | High EOM inventory minimum — plan for high usage to build buffer |
        | Mar, Apr | **Q25** (lower) | Very low EOM inventory maximum — avoid over-filling |
        | May–Oct | **Q50** (median) | Symmetric risk in shoulder months |

        The deliveries are then optimised to satisfy:
        - **Hard:** month-end inventory inside the [min, max] band
        - **Soft:** daily injection/withdrawal within monthly activity limits

        > Pre-computed optimized results are loaded from
        > `data/silver/optimization_results_v1.parquet` if available.
        > Otherwise, click the button below to run the simulation.
        """
    )
    return


@app.cell
def _(mo):
    _run_opt = mo.ui.run_button(label="Run LP Optimization (requires pre-computed forecast)")
    _run_opt
    return (_run_opt,)


@app.cell
def _(Path, _run_opt, dgup, historical_sim, mo, pl, raw):
    _opt_path = Path("data/silver/optimization_results_v1.parquet")
    _fcast_path = Path("data/silver/cv_results_v1.parquet")

    if _opt_path.exists():
        optimized_sim = pl.read_parquet(_opt_path)
        _source = "loaded from cache"
    elif _run_opt.value:
        # Requires a forecast file — use a 7-day naive forecast as fallback
        # if the champion forecast isn't available yet
        if _fcast_path.exists():
            cv_res = pl.read_parquet(_fcast_path)
            champion_name = dgup.select_champion_v1(cv_res)
        else:
            champion_name = "naive_lag7"

        with mo.status.spinner(f"Training {champion_name} and running LP optimizer…"):
            trained_model = dgup.train_champion_v1(raw, champion_name)
            forecast_df = dgup.predict_7day_v1(trained_model, raw)
            # For batch simulation, build full rolling forecast
            # Use historical_sim inventory column for t-3 references
            optimized_sim = dgup.batch_optimize_deliveries_v1(historical_sim, forecast_df)

        optimized_sim.write_parquet(_opt_path)
        _source = "computed and saved"
    else:
        optimized_sim = None
        _source = "not yet run"

    mo.callout(
        mo.md(f"Optimization results: **{_source}**"), kind="info"
    )
    return optimized_sim,


@app.cell
def _(mo, optimized_sim):
    mo.md(
        r"""
        ## Optimized Inventory Trajectory

        The chart below shows the storage bank balance under the LP-optimized
        delivery schedule.  Compare it to the historical chart above — the
        optimizer should keep the inventory **within the green bands** every month.
        """
    ) if optimized_sim is None else None
    return


@app.cell
def _(alt, dgup, mo, optimized_sim, pl):
    def _make_opt_inventory_chart(sim_df):
        if sim_df is None:
            return mo.md("*Run optimization first.*")

        months_in_data = (
            sim_df.with_columns(pl.col("Date").dt.year().alias("year"), pl.col("Date").dt.month().alias("month_num"))
            .select(["year", "month_num", "Date"])
            .group_by(["year", "month_num"])
            .agg(
                pl.col("Date").min().alias("month_start"),
                pl.col("Date").max().alias("month_end"),
            )
        )
        band_rows = []
        for row in months_in_data.iter_rows(named=True):
            m = row["month_num"]
            band_rows.append(
                {
                    "month_start": row["month_start"],
                    "month_end": row["month_end"],
                    "min_inv": dgup.MONTHLY_MIN[m],
                    "max_inv": dgup.MONTHLY_MAX[m],
                }
            )
        band_df = pl.DataFrame(band_rows)

        band_chart = (
            alt.Chart(band_df)
            .mark_rect(opacity=0.12, color="#2ca02c")
            .encode(
                x=alt.X("month_start:T"),
                x2=alt.X2("month_end:T"),
                y=alt.Y("min_inv:Q", title="Inventory (therms)"),
                y2=alt.Y2("max_inv:Q"),
            )
        )

        # Use opt_inventory col if present, else inventory
        inv_col = "opt_inventory" if "opt_inventory" in sim_df.columns else "inventory"
        inv_line = (
            alt.Chart(sim_df)
            .mark_line(strokeWidth=1.2, color="#2ca02c")
            .encode(
                x=alt.X("Date:T", title="Date"),
                y=alt.Y(f"{inv_col}:Q", title="Inventory (therms)"),
                tooltip=["Date:T", alt.Tooltip(f"{inv_col}:Q", format=",.0f")],
            )
        )

        pen_col = "opt_monthly_penalty" if "opt_monthly_penalty" in sim_df.columns else "monthly_penalty"
        penalty_df = sim_df.filter(pl.col(pen_col))
        penalty_marks = (
            alt.Chart(penalty_df)
            .mark_point(shape="triangle-up", color="#d62728", size=80)
            .encode(x="Date:T", y=f"{inv_col}:Q")
        )

        return mo.ui.altair_chart(
            alt.layer(band_chart, inv_line, penalty_marks)
            .properties(
                title="Optimized Inventory vs. Month-End Bands",
                width="container",
                height=380,
            )
            .interactive(bind_y=False)
        )

    _make_opt_inventory_chart(optimized_sim)
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Penalty Reduction — Historical vs. Optimized

        This side-by-side chart compares monthly penalty events per year before
        and after optimization.  The business goal is to drive the orange bars
        (optimized) to zero.
        """
    )
    return


@app.cell
def _(alt, historical_sim, mo, optimized_sim, pl):
    def _make_comparison_chart(hist_df, opt_df):
        if opt_df is None:
            return mo.md("*Run optimization first.*")

        pen_col_opt = "opt_monthly_penalty" if "opt_monthly_penalty" in opt_df.columns else "monthly_penalty"

        hist_yearly = (
            hist_df.with_columns(pl.col("Date").dt.year().alias("year"))
            .group_by("year")
            .agg(pl.col("monthly_penalty").sum().alias("monthly_penalties"))
            .with_columns(pl.lit("Historical").alias("schedule"))
        )
        opt_yearly = (
            opt_df.with_columns(pl.col("Date").dt.year().alias("year"))
            .group_by("year")
            .agg(pl.col(pen_col_opt).sum().alias("monthly_penalties"))
            .with_columns(pl.lit("Optimized").alias("schedule"))
        )
        combined = pl.concat([hist_yearly, opt_yearly]).sort("year")

        bar = (
            alt.Chart(combined)
            .mark_bar(opacity=0.85)
            .encode(
                x=alt.X("year:O", title="Year"),
                y=alt.Y("monthly_penalties:Q", title="Monthly Penalty Events"),
                color=alt.Color(
                    "schedule:N",
                    scale=alt.Scale(
                        domain=["Historical", "Optimized"],
                        range=["#d62728", "#2ca02c"],
                    ),
                    title="Delivery Schedule",
                ),
                xOffset="schedule:N",
                tooltip=["year:O", "schedule:N", "monthly_penalties:Q"],
            )
            .properties(
                title="Monthly Penalty Events: Historical vs. LP-Optimized",
                width="container",
                height=320,
            )
        )
        return mo.ui.altair_chart(bar)

    _make_comparison_chart(historical_sim, optimized_sim)
    return


@app.cell
def _(alt, historical_sim, mo, optimized_sim, pl):
    def _make_daily_comparison(hist_df, opt_df):
        if opt_df is None:
            return mo.md("*Run optimization first.*")

        pen_col_opt = "opt_daily_penalty" if "opt_daily_penalty" in opt_df.columns else "daily_penalty"

        hist_yearly = (
            hist_df.with_columns(pl.col("Date").dt.year().alias("year"))
            .group_by("year")
            .agg(pl.col("daily_penalty").sum().alias("daily_penalties"))
            .with_columns(pl.lit("Historical").alias("schedule"))
        )
        opt_yearly = (
            opt_df.with_columns(pl.col("Date").dt.year().alias("year"))
            .group_by("year")
            .agg(pl.col(pen_col_opt).sum().alias("daily_penalties"))
            .with_columns(pl.lit("Optimized").alias("schedule"))
        )
        combined = pl.concat([hist_yearly, opt_yearly]).sort("year")

        bar = (
            alt.Chart(combined)
            .mark_bar(opacity=0.85)
            .encode(
                x=alt.X("year:O", title="Year"),
                y=alt.Y("daily_penalties:Q", title="Daily Penalty Events"),
                color=alt.Color(
                    "schedule:N",
                    scale=alt.Scale(
                        domain=["Historical", "Optimized"],
                        range=["#ff7f0e", "#1f77b4"],
                    ),
                    title="Delivery Schedule",
                ),
                xOffset="schedule:N",
                tooltip=["year:O", "schedule:N", "daily_penalties:Q"],
            )
            .properties(
                title="Daily Penalty Events: Historical vs. LP-Optimized",
                width="container",
                height=320,
            )
        )
        return mo.ui.altair_chart(bar)

    _make_daily_comparison(historical_sim, optimized_sim)
    return


@app.cell
def _(historical_sim, mo, optimized_sim, pl):
    def _make_summary(hist_df, opt_df):
        if opt_df is None:
            return mo.md("*Run optimization first to see the penalty reduction summary.*")

        pen_col_opt_m = "opt_monthly_penalty" if "opt_monthly_penalty" in opt_df.columns else "monthly_penalty"
        pen_col_opt_d = "opt_daily_penalty" if "opt_daily_penalty" in opt_df.columns else "daily_penalty"

        hist_m = int(hist_df["monthly_penalty"].sum())
        hist_d = int(hist_df["daily_penalty"].sum())
        opt_m = int(opt_df[pen_col_opt_m].sum())
        opt_d = int(opt_df[pen_col_opt_d].sum())

        pct_m = (hist_m - opt_m) / hist_m * 100 if hist_m > 0 else 0
        pct_d = (hist_d - opt_d) / hist_d * 100 if hist_d > 0 else 0

        return mo.callout(
            mo.md(
                f"""
                ### Penalty Reduction Summary

                | Penalty Type | Historical | Optimized | Reduction |
                |---|---|---|---|
                | **Monthly (hard constraint)** | {hist_m} | {opt_m} | **{pct_m:.0f}%** |
                | Daily (soft constraint) | {hist_d} | {opt_d} | **{pct_d:.0f}%** |

                Monthly penalties are the primary business metric — reducing these
                directly prevents contractual charges from Nicor Gas.
                """
            ),
            kind="success",
        )

    _make_summary(historical_sim, optimized_sim)
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Month-Aware Quantile Selection

        The optimizer does not always use the median forecast.  The table below
        shows which quantile is used for each month and the business rationale.
        A higher quantile (Q75) means we plan conservatively for *high* usage —
        ensuring enough gas is delivered to maintain the storage floor.
        """
    )
    return


@app.cell
def _(alt, dgup, mo, pl):
    _quantile_table = pl.DataFrame(
        {
            "Month": list(range(1, 13)),
            "Month Name": ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
            "Active Quantile": [dgup.select_active_quantile_v1(m).upper() for m in range(1, 13)],
            "EOM Min (%)": [dgup.MONTHLY_MIN_PCT[m] * 100 for m in range(1, 13)],
            "EOM Max (%)": [dgup.MONTHLY_MAX_PCT[m] * 100 for m in range(1, 13)],
        }
    )

    _chart = (
        alt.Chart(_quantile_table)
        .mark_rect()
        .encode(
            x=alt.X("Month Name:N", sort=_quantile_table["Month Name"].to_list(), title="Month"),
            y=alt.Y("Active Quantile:N", title="Forecast Quantile Used", sort=["Q25", "Q50", "Q75"]),
            color=alt.Color(
                "Active Quantile:N",
                scale=alt.Scale(
                    domain=["Q25", "Q50", "Q75"],
                    range=["#1f77b4", "#2ca02c", "#d62728"],
                ),
            ),
            tooltip=[
                "Month Name:N",
                "Active Quantile:N",
                alt.Tooltip("EOM Min (%):Q", format=".0f"),
                alt.Tooltip("EOM Max (%):Q", format=".0f"),
            ],
        )
        .properties(
            title="Month-Aware Quantile Selection",
            width="container",
            height=160,
        )
    )

    mo.vstack([mo.ui.altair_chart(_chart), _quantile_table])
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Key Findings

        - The LP optimizer has full visibility of the **7-day forecast window**,
          allowing it to spread injection/withdrawal smoothly rather than reacting
          to each day independently.
        - **Month-end band violations** are the primary business risk; by
          selecting a conservative quantile (Q75) in winter months, the optimizer
          proactively builds inventory buffer ahead of the January and February
          minimums.
        - The **daily optimization + weekly retraining** cadence ensures the
          forecast model stays calibrated as seasonal conditions shift — without
          the computational cost of daily deep-learning retraining.
        - If the LP solver finds no feasible solution within the 7-day window
          (e.g. an extraordinarily large mid-month inventory gap), it falls back
          to a **least-violation solution** using slack variables — the optimizer
          never silently fails.
        """
    )
    return


@app.cell
def _(Path, mo, optimized_sim):
    _opt_path = Path("data/silver/optimization_results_v1.parquet")
    if optimized_sim is not None and not _opt_path.exists():
        optimized_sim.write_parquet(_opt_path)
        mo.md("Optimization results saved to `data/silver/optimization_results_v1.parquet`")
    else:
        mo.md(
            "Results are in `data/silver/optimization_results_v1.parquet`  \n"
            "Run the [model comparison notebook](4.0-vg-model-comparison.py) first "
            "to generate CV results, then click **Run LP Optimization** above."
        )
    return


if __name__ == "__main__":
    app.run()
