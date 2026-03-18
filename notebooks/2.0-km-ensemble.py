import marimo

__generated_with = "0.20.4"
app = marimo.App(width="full")


@app.cell
def _():
    import altair as alt
    import marimo as mo

    from notebooks.forecast_benchmarks import run_all_backtests

    return alt, mo, run_all_backtests


@app.cell
def _(run_all_backtests):
    metrics_frame, predictions_frame, summary_frame, aggregate_metrics = run_all_backtests()
    return metrics_frame, predictions_frame, summary_frame


@app.cell
def _(mo, summary_frame):
    series_selector = mo.ui.dropdown(
        options=summary_frame["series"].drop_duplicates().tolist(),
        value="Total Usage",
        label="Series",
    )
    return (series_selector,)


@app.cell
def _(mo):
    intro = mo.md(
        """
        # Forecast Benchmarking

        This notebook benchmarks multiple model families on Scot Forge gas usage with rolling calendar-year backtests.
        The models are built from scratch for this project and cover linear autoregression, tree boosting,
        feed-forward neural nets, LSTM, and transformer sequence models.
        """,
    )
    context = mo.md(
        """
        ## Direct Total vs Summed Components

        The aggregate comparison below tests whether forecasting the component streams separately and summing them
        beats forecasting Total Usage directly.
        """,
    )
    return


@app.cell
def _(metrics_frame, series_selector, summary_frame):
    selected_summary = summary_frame.loc[summary_frame["series"] == series_selector.value].copy()
    selected_summary = selected_summary.sort_values(["rank", "mae", "smape"]).reset_index(drop=True)
    selected_metrics = metrics_frame.loc[metrics_frame["series"] == series_selector.value].copy()
    return selected_metrics, selected_summary


@app.cell
def _(alt, selected_metrics):
    metric_chart = (
        alt.Chart(selected_metrics)
        .mark_bar()
        .encode(
            x=alt.X("model:N", sort="-y", title="Model"),
            y=alt.Y("mae:Q", title="Fold MAE"),
            color=alt.Color("fold:N", title="Fold"),
            column=alt.Column("fold:N", title=None),
            tooltip=[
                alt.Tooltip("series:N", title="Series"),
                alt.Tooltip("model:N", title="Model"),
                alt.Tooltip("fold:N", title="Fold"),
                alt.Tooltip("mae:Q", format=",.2f"),
                alt.Tooltip("rmse:Q", format=",.2f"),
                alt.Tooltip("smape:Q", format=",.2f"),
            ],
        )
        .properties(width=130, height=260)
    )
    return (metric_chart,)


@app.cell
def _(metric_chart):
    metric_chart
    return


@app.cell
def _(predictions_frame, selected_summary, series_selector):
    best_model = selected_summary.iloc[0]["model"]
    best_predictions = predictions_frame.loc[
        (predictions_frame["series"] == series_selector.value)
        & (predictions_frame["model"] == best_model)
    ].copy()
    latest_fold = best_predictions["fold"].max()
    latest_predictions = best_predictions.loc[best_predictions["fold"] == latest_fold].copy()
    latest_predictions["absolute_error"] = (
        latest_predictions["actual"] - latest_predictions["predicted"]
    ).abs()
    return best_model, latest_predictions


@app.cell
def _(alt, latest_predictions):
    history_frame = latest_predictions.melt(
        id_vars=["Date", "fold", "absolute_error"],
        value_vars=["actual", "predicted"],
        var_name="line_name",
        value_name="usage",
    )
    line_chart = (
        alt.Chart(history_frame)
        .mark_line()
        .encode(
            x=alt.X("Date:T", title="Date"),
            y=alt.Y("usage:Q", title="Usage"),
            color=alt.Color("line_name:N", title="Series"),
            tooltip=[
                alt.Tooltip("Date:T"),
                alt.Tooltip("line_name:N", title="Line"),
                alt.Tooltip("usage:Q", format=",.2f"),
            ],
        )
        .properties(height=320, width="container")
    )
    error_chart = (
        alt.Chart(latest_predictions)
        .mark_bar(color="#cc5a37", opacity=0.45)
        .encode(
            x=alt.X("Date:T", title="Date"),
            y=alt.Y("absolute_error:Q", title="Absolute Error"),
            tooltip=[
                alt.Tooltip("Date:T"),
                alt.Tooltip("actual:Q", format=",.2f"),
                alt.Tooltip("predicted:Q", format=",.2f"),
                alt.Tooltip("absolute_error:Q", format=",.2f"),
            ],
        )
        .properties(height=140, width="container")
    )
    return


@app.cell
def _(best_model, mo, series_selector):
    heading = mo.md(f"## {series_selector.value}: best model is `{best_model}`")
    return


if __name__ == "__main__":
    app.run()
