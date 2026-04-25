import marimo

__generated_with = "0.20.4"
app = marimo.App(
    width="full",
    app_title="Delivery Penalty Comparison Dashboard",
)


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import altair as alt

    return alt, mo, pd


@app.cell
def _(pd):
    CAPACITY = 144_841

    df = pd.read_csv(
        "/home/kumar/dgup/notebooks/gas_optimization_results_2022_2024_v4.csv",
        parse_dates=["Date"],
    )
    df = df.sort_values("Date").reset_index(drop=True)

    # Actual net flow per day  (+injection / -withdrawal)
    df["Actual_Flow"] = df["Actual_Delivery"] - df["Actual_Usage"]
    df["Opt_Flow"]    = df["Opt_Delivery"]    - df["Actual_Usage"]

    # Storage as % of capacity
    df["Actual_Storage_Pct"] = df["Total_Gas_After"]   / CAPACITY * 100
    df["Opt_Storage_Pct"]    = df["Opt_Storage_After"] / CAPACITY * 100

    # Violation type labels for colour-coding
    def daily_viol_type(row):
        if not row["Actual_Daily_Violation"] and not row["Opt_Daily_Violation"]:
            return "No Violation"
        if row["Actual_Daily_Violation"] and row["Opt_Daily_Violation"]:
            return "Both Violated"
        if row["Actual_Daily_Violation"]:
            return "Actual Only"
        return "Opt Only"

    df["Daily_Viol_Type"] = df.apply(daily_viol_type, axis=1)

    DATE_MIN = df["Date"].min().date()
    DATE_MAX = df["Date"].max().date()
    return CAPACITY, DATE_MAX, DATE_MIN, df


@app.cell
def _(DATE_MAX, DATE_MIN, mo):
    view_selector = mo.ui.radio(
        options=["Daily Penalties", "Monthly Penalties"],
        value="Daily Penalties",
        label="**Penalty View**",
        inline=True,
    )
    date_start = mo.ui.date(
        value=DATE_MIN, start=DATE_MIN, stop=DATE_MAX, label="Start Date"
    )
    date_end = mo.ui.date(
        value=DATE_MAX, start=DATE_MIN, stop=DATE_MAX, label="End Date"
    )
    return date_end, date_start, view_selector


@app.cell
def _(date_end, date_start, df, pd):
    import numpy as np

    _s = pd.Timestamp(date_start.value)
    _e = pd.Timestamp(date_end.value)
    if _s > _e:
        _s, _e = _e, _s

    fdf = df[(df["Date"] >= _s) & (df["Date"] <= _e)].copy()

    # EOM rows for monthly view
    eom_df = (
        fdf.assign(_YM=fdf["Date"].dt.to_period("M"))
        .sort_values("Date")
        .groupby("_YM")
        .last()
        .reset_index(drop=True)
    )
    return eom_df, fdf


@app.cell
def _(CAPACITY, alt, date_end, date_start, eom_df, fdf, mo, view_selector):

    # ── Altair theme ─────────────────────────────────────────────────────────
    _THEME = {
        "config": {
            "background": "transparent",
            "font": "Inter, system-ui, sans-serif",
            "axis": {
                "labelColor": "#9ca3af", "titleColor": "#d1d5db",
                "gridColor": "#ffffff", 
                "domainColor": "#4b5563",
                "tickColor": "#4b5563",
            },
            "legend": {
                "labelColor": "#d1d5db", "titleColor": "#9ca3af",
                "orient": "bottom", "columns": 4,
            },
            "title": {"color": "#f3f4f6", "fontSize": 13, "fontWeight": 600},
            "view": {"stroke": "transparent"},
        }
    }
    alt.themes.register("dark", lambda: _THEME)
    alt.themes.enable("dark")

    brush = alt.selection_interval(encodings=["x"])

    COLORS = {
        "actual":       "#60a5fa",   # blue  – actual delivery / storage
        "opt":          "#34d399",   # green – optimised delivery / storage
        "no_viol":      "#1f2937",   # dark grey – no violation
        "both":         "#ef4444",   # red   – both violated
        "actual_only":  "#f59e0b",   # amber – only actual violated
        "opt_only":     "#a78bfa",   # purple – only opt violated
        "eom_min":      "#34d399",   # green dashed – EOM min bound
        "eom_max":      "#f87171",   # red dashed   – EOM max bound
    }

    _window_str = f"{date_start.value} → {date_end.value}  |  **{len(fdf):,} days**"

    # ═══════════════════════════════════════════════════════════════════════════
    if view_selector.value == "Daily Penalties":

        viol_color_scale = alt.Scale(
            domain=["No Violation", "Both Violated", "Actual Only", "Opt Only"],
            range=[COLORS["no_viol"], COLORS["both"],
                   COLORS["actual_only"], COLORS["opt_only"]],
        )

        base = alt.Chart(fdf).encode(
            x=alt.X("Date:T", title=None, scale=alt.Scale(domain=brush))
        )

        # ── Panel 1: Actual vs Opt Delivery ──────────────────────────────────
        act_line = base.mark_line(
            color=COLORS["actual"], strokeWidth=1.5, opacity=0.85,
        ).encode(
            y=alt.Y("Actual_Delivery:Q", title="Delivery (units)"),
            tooltip=[
                alt.Tooltip("Date:T", format="%Y-%m-%d"),
                alt.Tooltip("Actual_Delivery:Q", title="Actual Delivery", format=".1f"),
            ],
        )

        opt_line = base.mark_line(
            color=COLORS["opt"], strokeWidth=1.5, opacity=0.85,
        ).encode(
            y=alt.Y("Opt_Delivery:Q", title="Delivery (units)"),
            tooltip=[
                alt.Tooltip("Date:T", format="%Y-%m-%d"),
                alt.Tooltip("Opt_Delivery:Q", title="Opt Delivery", format=".1f"),
            ],
        )

        # Violation dots on the delivery chart (colour = violation category)
        viol_pts = (
            alt.Chart(fdf[fdf["Daily_Viol_Type"] != "No Violation"])
            .mark_point(size=55, filled=True, opacity=0.9)
            .encode(
                x=alt.X("Date:T", scale=alt.Scale(domain=brush)),
                y=alt.Y("Actual_Delivery:Q"),
                color=alt.Color(
                    "Daily_Viol_Type:N",
                    scale=viol_color_scale,
                    title="Violation",
                ),
                tooltip=[
                    alt.Tooltip("Date:T",              format="%Y-%m-%d"),
                    alt.Tooltip("Daily_Viol_Type:N",   title="Violation Type"),
                    alt.Tooltip("Actual_Delivery:Q",   title="Actual Delivery",  format=".1f"),
                    alt.Tooltip("Opt_Delivery:Q",      title="Opt Delivery",     format=".1f"),
                    alt.Tooltip("Actual_Usage:Q",      title="Actual Usage",     format=".1f"),
                    alt.Tooltip("Max_Injection_Limit:Q",  title="Max Injection",  format=".1f"),
                    alt.Tooltip("Max_Withdrawal_Limit:Q", title="Max Withdrawal", format=".1f"),
                ],
            )
        )

        panel_delivery = (
            (act_line + opt_line + viol_pts)
            .properties(height=220,
                        title="Actual Delivery (blue) vs Optimised Delivery (green)"
                              "  ·  dots = penalty violation days")
        )

        # ── Panel 2: Net flow vs daily limits ────────────────────────────────
        # Positive = injection, Negative = withdrawal
        flow_actual = base.mark_line(
            color=COLORS["actual"], strokeWidth=1.2, opacity=0.7,
        ).encode(
            y=alt.Y("Actual_Flow:Q", title="Net Flow (units)"),
            tooltip=[
                alt.Tooltip("Date:T", format="%Y-%m-%d"),
                alt.Tooltip("Actual_Flow:Q", title="Actual Flow", format=".1f"),
            ],
        )

        flow_opt = base.mark_line(
            color=COLORS["opt"], strokeWidth=1.2, opacity=0.7,
        ).encode(
            y=alt.Y("Opt_Flow:Q"),
            tooltip=[
                alt.Tooltip("Date:T", format="%Y-%m-%d"),
                alt.Tooltip("Opt_Flow:Q", title="Opt Flow", format=".1f"),
            ],
        )

        # Max injection limit line (positive ceiling)
        inj_limit_line = base.mark_line(
            color="#a3e635", strokeWidth=1.5, strokeDash=[6, 3],
        ).encode(y=alt.Y("Max_Injection_Limit:Q"))

        # Max withdrawal limit line (negated — withdrawals are negative flow)
        wd_limit_line = (
            alt.Chart(fdf)
            .mark_line(color="#fb923c", strokeWidth=1.5, strokeDash=[6, 3])
            .encode(
                x=alt.X("Date:T", scale=alt.Scale(domain=brush)),
                y=alt.Y("_neg_wd:Q", title="Net Flow (units)"),
            )
            .transform_calculate(_neg_wd="-datum.Max_Withdrawal_Limit")
        )

        zero_rule = base.mark_rule(color="#4b5563", strokeWidth=1, opacity=0.5
                                   ).encode(y=alt.datum(0))

        panel_flow = (
            (zero_rule + flow_actual + flow_opt
             + inj_limit_line + wd_limit_line)
            .properties(
                height=200,
                title="Net Flow: Actual (blue) vs Opt (green)"
                      "  ·  - - green: max injection  |  - - orange: max withdrawal",
            )
        )

        # ── Panel 3: Overview brush ───────────────────────────────────────────
        overview = (
            alt.Chart(fdf)
            .mark_bar(size=2)
            .encode(
                x=alt.X("Date:T", title="Date  ·  drag to zoom"),
                y=alt.Y("Daily_Viol_Type:N", title=None),
                color=alt.Color("Daily_Viol_Type:N", scale=viol_color_scale,
                                title="Violation"),
            )
            .add_params(brush)
            .properties(height=70, title="Violation overview — drag to zoom")
        )

        chart = (panel_delivery & panel_flow & overview).configure_concat(spacing=8)

        # Stats
        _n_act  = int(fdf["Actual_Daily_Violation"].sum())
        _n_opt  = int(fdf["Opt_Daily_Violation"].sum())
        _n_both = int((fdf["Daily_Viol_Type"] == "Both Violated").sum())
        _n_saved = _n_act - _n_opt
        stat_row = mo.hstack([
            mo.stat(label="Actual Daily Violations",    value=str(_n_act),  bordered=True),
            mo.stat(label="Optimised Daily Violations", value=str(_n_opt),  bordered=True),
            mo.stat(label="Days Both Violated",         value=str(_n_both), bordered=True),
            mo.stat(label="Days Saved by Opt",
                    value=f"{_n_saved}" if _n_saved >= 0 else str(_n_saved),
                    bordered=True),
        ], gap=1)

    # ═══════════════════════════════════════════════════════════════════════════
    else:  # Monthly Penalties

        base_m = alt.Chart(eom_df).encode(
            x=alt.X("Date:T", title=None, scale=alt.Scale(domain=brush))
        )

        # Actual storage line
        act_stor = base_m.mark_line(
            color=COLORS["actual"], strokeWidth=2,
        ).encode(
            y=alt.Y("Total_Gas_After:Q",
                    title="End-of-Month Storage (units)",
                    scale=alt.Scale(domain=[0, CAPACITY])),
            tooltip=[
                alt.Tooltip("Date:T", format="%Y-%m"),
                alt.Tooltip("Total_Gas_After:Q", title="Actual Storage", format=".0f"),
            ],
        )

        # Optimised storage line
        opt_stor = base_m.mark_line(
            color=COLORS["opt"], strokeWidth=2,
        ).encode(
            y=alt.Y("Opt_Storage_After:Q",
                    scale=alt.Scale(domain=[0, CAPACITY])),
            tooltip=[
                alt.Tooltip("Date:T", format="%Y-%m"),
                alt.Tooltip("Opt_Storage_After:Q", title="Opt Storage", format=".0f"),
            ],
        )

        # EOM min/max bands
        eom_min_line = base_m.mark_line(
            color=COLORS["eom_min"], strokeWidth=1.5, strokeDash=[5, 3],
        ).encode(y=alt.Y("EOM_Min_Storage:Q"))

        eom_max_line = base_m.mark_line(
            color=COLORS["eom_max"], strokeWidth=1.5, strokeDash=[5, 3],
        ).encode(y=alt.Y("EOM_Max_Storage:Q"))

        # Violation markers — actual
        act_viol_pts = (
            alt.Chart(eom_df[eom_df["Actual_Monthly_Violation"]])
            .mark_point(size=100, shape="triangle-up", filled=True,
                        color=COLORS["actual_only"], opacity=0.9)
            .encode(
                x=alt.X("Date:T", scale=alt.Scale(domain=brush)),
                y=alt.Y("Total_Gas_After:Q"),
                tooltip=[
                    alt.Tooltip("Date:T",                format="%Y-%m"),
                    alt.Tooltip("Total_Gas_After:Q",     title="Actual Storage",  format=".0f"),
                    alt.Tooltip("EOM_Min_Storage:Q",     title="Min Required",    format=".0f"),
                    alt.Tooltip("EOM_Max_Storage:Q",     title="Max Allowed",     format=".0f"),
                ],
            )
        )

        # Violation markers — optimised
        opt_viol_pts = (
            alt.Chart(eom_df[eom_df["Opt_Monthly_Violation"]])
            .mark_point(size=100, shape="triangle-down", filled=True,
                        color=COLORS["opt_only"], opacity=0.9)
            .encode(
                x=alt.X("Date:T", scale=alt.Scale(domain=brush)),
                y=alt.Y("Opt_Storage_After:Q"),
                tooltip=[
                    alt.Tooltip("Date:T",                 format="%Y-%m"),
                    alt.Tooltip("Opt_Storage_After:Q",    title="Opt Storage",    format=".0f"),
                    alt.Tooltip("EOM_Min_Storage:Q",      title="Min Required",   format=".0f"),
                    alt.Tooltip("EOM_Max_Storage:Q",      title="Max Allowed",    format=".0f"),
                ],
            )
        )

        panel_eom = (
            (eom_min_line + eom_max_line + act_stor + opt_stor
             + act_viol_pts + opt_viol_pts)
            .add_params(brush)
            .properties(
                height=300,
                title="End-of-Month Storage: Actual (blue) vs Optimised (green)"
                      "  ·  - - green: EOM min  |  - - red: EOM max"
                      "  ·  ▲ actual violation  |  ▼ opt violation",
            ).interactive()
        )

        # Deviation bar chart
        eom_plot = eom_df.copy()
        eom_plot["Actual_Deviation"] = eom_plot.apply(
            lambda r: r["Total_Gas_After"] - r["EOM_Min_Storage"]
            if r["Actual_Monthly_Violation"] and r["Total_Gas_After"] < r["EOM_Min_Storage"]
            else (r["Total_Gas_After"] - r["EOM_Max_Storage"]
                  if r["Actual_Monthly_Violation"] and r["Total_Gas_After"] > r["EOM_Max_Storage"]
                  else 0),
            axis=1,
        )
        eom_plot["Opt_Deviation"] = eom_plot.apply(
            lambda r: r["Opt_Storage_After"] - r["EOM_Min_Storage"]
            if r["Opt_Monthly_Violation"] and r["Opt_Storage_After"] < r["EOM_Min_Storage"]
            else (r["Opt_Storage_After"] - r["EOM_Max_Storage"]
                  if r["Opt_Monthly_Violation"] and r["Opt_Storage_After"] > r["EOM_Max_Storage"]
                  else 0),
            axis=1,
        )

        act_dev = (
            alt.Chart(eom_plot)
            .mark_bar(color=COLORS["actual_only"], opacity=0.8, size=10)
            .encode(
                x=alt.X("Date:T", title="Month  ·  drag to zoom"),
                y=alt.Y("Actual_Deviation:Q", title="Deviation (units)"),
                tooltip=[
                    alt.Tooltip("Date:T",              format="%Y-%m"),
                    alt.Tooltip("Actual_Deviation:Q",  title="Actual Deviation", format=".0f"),
                ],
            )
        )
        opt_dev = (
            alt.Chart(eom_plot)
            .mark_bar(color=COLORS["opt_only"], opacity=0.8, size=5)
            .encode(
                x=alt.X("Date:T"),
                y=alt.Y("Opt_Deviation:Q"),
                tooltip=[
                    alt.Tooltip("Date:T",            format="%Y-%m"),
                    alt.Tooltip("Opt_Deviation:Q",   title="Opt Deviation",    format=".0f"),
                ],
            )
        )

        overview_m = (
            (act_dev + opt_dev)
            .add_params(brush)
            .properties(height=90, title="Monthly Violation Deviation — drag to zoom")
        )

        chart = (panel_eom & overview_m).configure_concat(spacing=8)

        _n_act_m  = int(eom_df["Actual_Monthly_Violation"].sum())
        _n_opt_m  = int(eom_df["Opt_Monthly_Violation"].sum())
        _n_saved_m = _n_act_m - _n_opt_m
        stat_row = mo.hstack([
            mo.stat(label="Actual Monthly Violations",    value=str(_n_act_m),  bordered=True),
            mo.stat(label="Optimised Monthly Violations", value=str(_n_opt_m),  bordered=True),
            mo.stat(label="Months Saved by Opt",
                    value=f"{_n_saved_m}" if _n_saved_m >= 0 else str(_n_saved_m),
                    bordered=True),
        ], gap=1)

    # ── Layout ───────────────────────────────────────────────────────────────
    mo.vstack([
        mo.md("# 📊 Delivery Penalty Comparison  —  Actual vs Optimised"),
        mo.hstack(
            [view_selector, mo.hstack([date_start, date_end], gap=2, align="end")],
            justify="start", gap=4, align="end",
        ),
        mo.md(f"_{_window_str}_"),
        stat_row,
        mo.ui.altair_chart(chart),
        mo.md(
            "**Legend:**"
            "🔴 Both violated  🟡 Actual only  🟣 Opt only"
        ),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Cashout Penalty Tier Comparison
    """)
    return


@app.cell
def _():
    import polars as pl
    # Long form (one row per tier)
    # Actual penalties per tier 
    all_penalties = pl.DataFrame({
        "Tier": ["1", "2", "3"],
        "Actual Total Penalties": [71, 47, 302],
        "Optimized Total Penalties": [22, 25, 109],
    })
    # optimized penalties per tier
    # tier 2 46.8 #tier  3-  63.90

    return all_penalties, pl


@app.cell
def _(all_penalties, pl):
    all_pen = all_penalties.with_columns(
        pl.format("{}%",(((pl.col("Actual Total Penalties")-pl.col("Optimized Total Penalties"))/(pl.col("Actual Total Penalties")))*100).round(2)).alias("% Savings")
    )

    all_pen
    return (all_pen,)


@app.cell
def _(all_pen, pl):
    all_pen_1 = all_pen.select(pl.exclude("% Savings")).unpivot(index="Tier")

    return (all_pen_1,)


@app.cell
def _(all_pen_1, alt):
    # replace _df with your data source
    _chart = (
        alt.Chart(all_pen_1)
        .mark_bar()
        .encode(
            x=alt.X(field='variable', type='nominal'),
            y=alt.Y(field='value', type='quantitative'),
            color=alt.Color(field='variable', type='nominal', scale={
                'scheme': 'bluegreen'
            }),
            column="Tier:N",
            tooltip=[
                alt.Tooltip(field='Tier'),
                alt.Tooltip(field='value', format=',.0f'),
                alt.Tooltip(field='variable')
            ]
        )
        .properties(
            title='Actual vs Optimized Penalties',
            width= 250,
            config={
                'axis': {
                    'grid': True
                }
            }
        )
    )
    _chart
    return


if __name__ == "__main__":
    app.run()
