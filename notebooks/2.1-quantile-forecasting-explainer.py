import marimo

__generated_with = "0.22.0"
app = marimo.App(width="full")


@app.cell
def _():
    import altair as alt
    import marimo as mo
    import numpy as np
    import pandas as pd

    return alt, mo, np, pd


@app.cell
def _(mo):
    mo.md(r"""
    # Quantile Forecasting Explainer

    This interactive app explains quantile forecasting in a business-friendly way.

    A point forecast gives one number. A quantile forecast gives a *risk-aware range*:

    - $p10$: low-demand scenario
    - $p50$: median scenario
    - $p95$: high-demand stress scenario

    In operations, choosing a planning quantile is a business risk decision.

    A quantile $\hat{y}_q$ is the value where approximately a fraction $q$ of outcomes are below it:

    $$
    P(Y \leq \hat{y}_q) \approx q
    $$
    """)
    return


@app.cell
def _(mo):
    demand_mean = mo.ui.slider(
        200,
        1800,
        value=800,
        step=25,
        label="Expected Daily Usage",
        show_value=True,
    )
    demand_sigma = mo.ui.slider(
        50,
        500,
        value=180,
        step=10,
        label="Demand Uncertainty (Std Dev)",
        show_value=True,
    )
    simulation_n = mo.ui.slider(
        300,
        5000,
        value=2000,
        step=100,
        label="Simulation Samples",
        show_value=True,
    )
    random_seed = mo.ui.slider(
        1,
        999,
        value=42,
        step=1,
        label="Random Seed",
        show_value=True,
    )

    planning_quantile = mo.ui.dropdown(
        options=["p10", "p50", "p95"],
        value="p95",
        label="Planning Quantile",
    )

    controls = mo.vstack(
        [
            mo.md("## 1) Configure Demand Scenario"),
            mo.hstack([demand_mean, demand_sigma]),
            mo.hstack([simulation_n, random_seed]),
            planning_quantile,
        ]
    )

    controls
    return (
        demand_mean,
        demand_sigma,
        planning_quantile,
        random_seed,
        simulation_n,
    )


@app.cell
def _(demand_mean, demand_sigma, np, pd, random_seed, simulation_n):
    rng1 = np.random.default_rng(random_seed.value)
    simulated_usage = np.maximum(
        rng1.normal(
            loc=demand_mean.value,
            scale=demand_sigma.value,
            size=simulation_n.value,
        ),
        0.0,
    )

    quantile_values = {
        "p10": float(np.quantile(simulated_usage, 0.10)),
        "p50": float(np.quantile(simulated_usage, 0.50)),
        "p95": float(np.quantile(simulated_usage, 0.95)),
    }

    distribution_df = pd.DataFrame({"usage": simulated_usage})
    quantile_df = pd.DataFrame(
        {
            "quantile": ["p10", "p50", "p95"],
            "usage": [
                quantile_values["p10"],
                quantile_values["p50"],
                quantile_values["p95"],
            ],
        }
    )
    return distribution_df, quantile_df, quantile_values


@app.cell
def _(alt, distribution_df, quantile_df):
    histogram = (
        alt.Chart(distribution_df)
        .mark_bar(opacity=0.45, binSpacing=0)
        .encode(
            x=alt.X("usage:Q", bin=alt.Bin(maxbins=45), title="Daily Usage"),
            y=alt.Y("count():Q", title="Simulated Days"),
        )
    )

    lines = (
        alt.Chart(quantile_df)
        .mark_rule(strokeWidth=3)
        .encode(
            x=alt.X("usage:Q"),
            color=alt.Color(
                "quantile:N",
                scale=alt.Scale(domain=["p10", "p50", "p95"], range=["#1f77b4", "#2ca02c", "#d62728"]),
                title="Forecast Quantile",
            ),
        )
    )

    labels_source = quantile_df.copy()
    labels_source["label"] = labels_source["quantile"] + ": " + labels_source["usage"].round(1).astype(str)

    labels = (
        alt.Chart(labels_source)
        .mark_text(align="left", dx=6, dy=-5, fontSize=12)
        .encode(
            x=alt.X("usage:Q"),
            y=alt.value(12),
            text=alt.Text("label:N"),
            color=alt.Color("quantile:N", legend=None),
        )
    )

    distribution_chart = (histogram + lines + labels).properties(
        title="Simulated Demand Distribution with Quantiles",
        height=360,
    )
    return (distribution_chart,)


@app.cell
def _(mo, planning_quantile, quantile_values):
    risk_text = {
        "p10": "Aggressive plan: assumes lower usage and higher risk of shortage if demand spikes.",
        "p50": "Balanced plan: median usage assumption with moderate risk posture.",
        "p95": "Conservative plan: protects against high-demand days and reduces violation risk.",
    }

    selected = planning_quantile.value

    summary = mo.vstack(
        [
            mo.md("## 2) Interpreting Quantiles"),
            mo.md(
                f"""
    Selected planning quantile: **{selected}**

    - p10 usage estimate: **{quantile_values['p10']:.1f}**
    - p50 usage estimate: **{quantile_values['p50']:.1f}**
    - p95 usage estimate: **{quantile_values['p95']:.1f}**

    Business interpretation:
    {risk_text[selected]}
    """
            ),
        ]
    )

    summary
    return


@app.cell
def _(distribution_chart, mo):
    mo.vstack([mo.md("## 3) Visual Distribution View"), distribution_chart])
    return


@app.cell
def _(mo):
    pinball_quantile = mo.ui.dropdown(
        options=["p10", "p50", "p95"],
        value="p95",
        label="Quantile in Pinball Loss",
    )
    true_usage = mo.ui.slider(
        0,
        2000,
        value=950,
        step=10,
        label="Observed Usage (Actual)",
        show_value=True,
    )
    predicted_usage = mo.ui.slider(
        0,
        2000,
        value=900,
        step=10,
        label="Predicted Quantile Value",
        show_value=True,
    )

    mo.vstack(
        [
            mo.md("## 4) Why Quantile Models Use Pinball Loss"),
            mo.hstack([pinball_quantile, true_usage, predicted_usage]),
            mo.md(
                r"""
    Pinball loss for quantile $q$:

    $$
    L_q(y, \hat{y}) =
    \begin{cases}
    q\,(y - \hat{y}), & y \ge \hat{y} \\
    (1-q)(\hat{y} - y), & y < \hat{y}
    \end{cases}
    $$

    This asymmetry is key: for high quantiles (like p95), under-predicting is penalized much more than over-predicting.
    """
            ),
        ]
    )
    return pinball_quantile, predicted_usage, true_usage


@app.cell
def _(np, pinball_quantile, predicted_usage, true_usage):
    quantile_map = {"p10": 0.10, "p50": 0.50, "p95": 0.95}
    q = quantile_map[pinball_quantile.value]

    y_actual = float(true_usage.value)
    y_hat = float(predicted_usage.value)

    error = y_actual - y_hat
    loss = q * max(error, 0.0) + (1.0 - q) * max(-error, 0.0)

    x = np.arange(0.0, 2001.0, 10.0)
    curve = q * np.maximum(y_actual - x, 0.0) + (1.0 - q) * np.maximum(x - y_actual, 0.0)
    return curve, loss, q, x, y_actual, y_hat


@app.cell
def _(alt, curve, np, pd, x, y_actual, y_hat):
    loss_curve_df = pd.DataFrame({"prediction": x, "loss": curve})
    marker_df = pd.DataFrame({"prediction": [y_hat], "loss": [float(np.interp(y_hat, x, curve))]})
    actual_df = pd.DataFrame({"actual": [y_actual]})

    base = (
        alt.Chart(loss_curve_df)
        .mark_line(strokeWidth=2)
        .encode(
            x=alt.X("prediction:Q", title="Predicted Quantile Value"),
            y=alt.Y("loss:Q", title="Pinball Loss"),
        )
    )

    actual_rule = alt.Chart(actual_df).mark_rule(strokeDash=[6, 4], color="#666").encode(x="actual:Q")
    marker = alt.Chart(marker_df).mark_point(size=140, color="#d62728").encode(x="prediction:Q", y="loss:Q")

    pinball_chart = (base + actual_rule + marker).properties(
        title="Pinball Loss Curve for the Selected Actual Usage",
        height=320,
    )
    return


@app.cell
def _(loss, mo, q, y_actual, y_hat):
    direction = "under-prediction" if y_hat < y_actual else "over-prediction"
    mo.md(
        rf"""
    Current selection:

    - Quantile $q$: **{q:.2f}**
    - Actual usage $y$: **{y_actual:.1f}**
    - Predicted value $\hat{{y}}$: **{y_hat:.1f}**
    - Error direction: **{direction}**
    - Pinball loss $L_q(y, \hat{{y}})$: **{loss:.2f}**
    """
    )
    return


@app.cell
def _(mo):
    max_injection = mo.ui.slider(
        200,
        3000,
        value=1300,
        step=25,
        label="Daily Injection Limit",
        show_value=True,
    )
    max_withdrawal = mo.ui.slider(
        200,
        3000,
        value=1600,
        step=25,
        label="Daily Withdrawal Limit",
        show_value=True,
    )

    mo.vstack(
        [
            mo.md("## 5) Planning Impact: From Quantile to Delivery Band"),
            mo.hstack([max_injection, max_withdrawal]),
        ]
    )
    return max_injection, max_withdrawal


@app.cell
def _(max_injection, max_withdrawal, mo, planning_quantile, quantile_values):
    demand_for_planning = quantile_values[planning_quantile.value]

    min_safe_delivery = max(0.0, demand_for_planning - max_withdrawal.value)
    max_safe_delivery = demand_for_planning + max_injection.value

    mo.md(
        rf"""
    Using **{planning_quantile.value} = {demand_for_planning:.1f}** as demand input:

    $$
    \text{{min safe delivery}} = \max(0, \hat{{y}}_q - \text{{wd limit}})
    $$

    $$
    \text{{max safe delivery}} = \hat{{y}}_q + \text{{inj limit}}
    $$

    Resulting delivery band for tomorrow:

    - **Min safe delivery:** {min_safe_delivery:.1f}
    - **Max safe delivery:** {max_safe_delivery:.1f}

    This is the same risk translation used in optimization: selecting a higher quantile shifts planning upward to reduce shortage/violation risk.
    """
    )
    return


@app.cell
def _(alt, np, pd, planning_quantile, random_seed):
    horizon = 30
    day = np.arange(1, horizon + 1)

    rng2 = np.random.default_rng(random_seed.value + 100)

    seasonal_mean = 760 + 130 * np.sin(2 * np.pi * day / 14)
    volatility = 110 + 30 * np.cos(2 * np.pi * day / 10)

    q10 = np.maximum(seasonal_mean - 1.2816 * volatility, 0.0)
    q50 = seasonal_mean
    q95 = np.maximum(seasonal_mean + 1.6449 * volatility, 0.0)
    actual = np.maximum(seasonal_mean + rng2.normal(0, volatility * 0.65), 0.0)

    selected_line = {"p10": q10, "p50": q50, "p95": q95}[planning_quantile.value]

    fan_df = pd.DataFrame(
        {
            "day": day,
            "q10": q10,
            "q50": q50,
            "q95": q95,
            "actual": actual,
            "selected": selected_line,
        }
    )

    band = (
        alt.Chart(fan_df)
        .mark_area(opacity=0.25, color="#1f77b4")
        .encode(x=alt.X("day:Q", title="Planning Day"), y=alt.Y("q10:Q", title="Usage"), y2="q95:Q")
    )

    p50_line = alt.Chart(fan_df).mark_line(color="#2ca02c", strokeWidth=2).encode(x="day:Q", y="q50:Q")
    actual_line = alt.Chart(fan_df).mark_line(color="#444", strokeWidth=2).encode(x="day:Q", y="actual:Q")
    selected_plan = alt.Chart(fan_df).mark_line(color="#d62728", strokeDash=[5, 3], strokeWidth=3).encode(x="day:Q", y="selected:Q")

    horizon_chart = (band + p50_line + actual_line + selected_plan).properties(
        title=f"30-Day Forecast Fan: Selected Planning Quantile = {planning_quantile.value}",
        height=340,
    )
    return (horizon_chart,)


@app.cell
def _(horizon_chart, mo):
    mo.vstack(
        [
            mo.md("## 6) 30-Day Forecast Fan (Concept View)"),
            mo.md(
                "Blue area is p10-p95 uncertainty, green is p50, gray is one realized path, and red dashed is the quantile chosen for planning."
            ),
            horizon_chart,
        ]
    )
    return


@app.cell
def _(mo, quantile_values):
    spread = quantile_values["p95"] - quantile_values["p10"]
    mo.md(
        f"""
    ## 7) Stakeholder Takeaways

    - Quantile forecasting turns uncertainty into explicit risk scenarios.
    - Planning at **p95** is conservative and reduces high-demand shortfall risk.
    - Planning at **p50** improves average efficiency but raises tail risk.
    - Current uncertainty width $(p95 - p10)$ is **{spread:.1f}**.

    A simple governance metric is to track this width over time and align quantile choice with business risk appetite.
    """
    )
    return


if __name__ == "__main__":
    app.run()
