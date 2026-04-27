from __future__ import annotations

from pathlib import Path

import altair as alt
import polars as pl
from statsmodels.tsa.stattools import acf, pacf


ROOT = Path(__file__).resolve().parents[2]
FIG_DIR = Path(__file__).resolve().parent / "figures"
CAPACITY = 144_841.0
INITIAL_INVENTORY = 81_569.123976

MONTH_ORDER = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

DAILY_LIMITS = {
    1: {"inj": 0.0030, "wd": 0.0100},
    2: {"inj": 0.0030, "wd": 0.0085},
    3: {"inj": 0.0030, "wd": 0.0060},
    4: {"inj": 0.0030, "wd": 0.0030},
    5: {"inj": 0.0045, "wd": 0.0030},
    6: {"inj": 0.0050, "wd": 0.0030},
    7: {"inj": 0.0045, "wd": 0.0030},
    8: {"inj": 0.0070, "wd": 0.0030},
    9: {"inj": 0.0070, "wd": 0.0030},
    10: {"inj": 0.0070, "wd": 0.0030},
    11: {"inj": 0.0030, "wd": 0.0040},
    12: {"inj": 0.0030, "wd": 0.0085},
}

MONTH_END_LIMITS = {
    1: {"min": 0.35, "max": 0.45},
    2: {"min": 0.10, "max": 0.25},
    3: {"min": 0.00, "max": 0.10},
    4: {"min": 0.00, "max": 0.10},
    5: {"min": 0.10, "max": 0.20},
    6: {"min": 0.20, "max": 0.30},
    7: {"min": 0.30, "max": 0.40},
    8: {"min": 0.50, "max": 0.60},
    9: {"min": 0.70, "max": 0.80},
    10: {"min": 0.85, "max": 1.00},
    11: {"min": 0.75, "max": 0.90},
    12: {"min": 0.55, "max": 0.70},
}

TIER_COUNTS = pl.DataFrame(
    {
        "Tier": ["Tier 1", "Tier 2", "Tier 3"],
        "Actual": [71, 47, 302],
        "Optimized": [22, 25, 109],
    }
)


@alt.theme.register("report", enable=True)
def report_theme() -> alt.theme.ThemeConfig:
    config = {
        "config": {
            "background": "white",
            "axis": {
                "labelFontSize": 10,
                "titleFontSize": 11,
                "gridColor": "#d9d9d9",
                "domainColor": "#666666",
                "tickColor": "#666666",
            },
            "legend": {"labelFontSize": 10, "titleFontSize": 10},
            "title": {"fontSize": 13, "anchor": "start", "fontWeight": "bold"},
            "view": {"stroke": None},
        }
    }
    return alt.theme.ThemeConfig(config)


def theme() -> None:
    alt.theme.enable("report")


def save(chart: alt.Chart, name: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    chart.save(FIG_DIR / name, scale_factor=2)


def load_usage() -> pl.DataFrame:
    return (
        pl.scan_parquet(ROOT / "data" / "silver" / "uta_gas_usage.parquet")
        .fill_null(0)
        .with_columns(
            [
                (pl.col("Usage - 1") + pl.col("Usage - 2") + pl.col("Usage - 2_1")).alias("Total Usage"),
                pl.col("Date").dt.strftime("%b").alias("Month"),
                pl.col("Date").dt.month().alias("Month_Num"),
                pl.col("Date").dt.year().alias("Year"),
            ]
        )
        .collect()
        .sort("Date")
    )


def load_results() -> pl.DataFrame:
    return (
        pl.read_csv(ROOT / "Report" / "gas_optimization_results_2022_2024_v4 1.csv")
        .with_columns(pl.col("Date").str.strptime(pl.Date, format="%m/%d/%Y", strict=False))
        .with_columns(
            [
                (pl.col("Actual_Delivery") - pl.col("Actual_Usage")).alias("Actual_Flow"),
                (pl.col("Opt_Delivery") - pl.col("Actual_Usage")).alias("Opt_Flow"),
                (pl.col("Total_Gas_After") / CAPACITY * 100).alias("Actual_Storage_Pct"),
                (pl.col("Opt_Storage_After") / CAPACITY * 100).alias("Opt_Storage_Pct"),
                pl.col("Date").dt.year().alias("Year"),
                pl.col("Date").dt.month().alias("Month_Num"),
            ]
        )
        .sort("Date")
    )


def usage_long(df: pl.DataFrame) -> pl.DataFrame:
    return (
        df.select(["Date", "Usage - 1", "Usage - 2", "Usage - 2_1", "Total Usage"])
        .unpivot(index="Date", variable_name="Usage Type", value_name="Usage")
        .with_columns(pl.col("Usage Type").replace({"Usage - 1": "Usage 1", "Usage - 2": "Usage 2", "Usage - 2_1": "Usage 2_1"}))
    )


def eda_usage_overview(df: pl.DataFrame) -> alt.VConcatChart:
    total = df.select(
        [
            "Date",
            "Total Usage",
            pl.col("Total Usage").rolling_mean(window_size=365, min_samples=30).alias("Rolling 365D"),
        ]
    )
    monthly = (
        df.group_by(["Year", "Month", "Month_Num"])
        .agg(pl.col("Total Usage").mean().alias("Average Usage"))
        .sort(["Year", "Month_Num"])
    )

    line = (
        alt.Chart(total)
        .transform_fold(["Total Usage", "Rolling 365D"], as_=["Series", "Usage"])
        .mark_line()
        .encode(
            x=alt.X("Date:T", title=None),
            y=alt.Y("Usage:Q", title="Daily gas usage"),
            color=alt.Color("Series:N", scale=alt.Scale(range=["#4c78a8", "#e45756"])),
        )
        .properties(width=720, height=220, title="Total usage with long-run rolling mean")
    )

    heat = (
        alt.Chart(monthly)
        .mark_rect()
        .encode(
            x=alt.X("Year:O", title="Year"),
            y=alt.Y("Month:N", title="Month", sort=MONTH_ORDER),
            color=alt.Color("Average Usage:Q", scale=alt.Scale(scheme="yelloworangered")),
            tooltip=["Year:O", alt.Tooltip("Month:N"), alt.Tooltip("Average Usage:Q", format=".1f")],
        )
        .properties(width=720, height=180, title="Average monthly usage")
    )

    return alt.vconcat(line, heat, spacing=14)


def eda_boxplot_total(df: pl.DataFrame) -> alt.Chart:
    return (
        alt.Chart(df)
        .mark_boxplot(extent=1.5, size=32, color="#4c78a8")
        .encode(
            x=alt.X("Month:N", sort=MONTH_ORDER, title="Month"),
            y=alt.Y("Total Usage:Q", title="Daily gas usage"),
        )
        .properties(width=720, height=360, title="Monthly boxplots for total usage")
    )


def eda_histogram(df_long: pl.DataFrame) -> alt.Chart:
    return (
        alt.Chart(df_long)
        .mark_bar(opacity=0.55, binSpacing=0)
        .encode(
            x=alt.X("Usage:Q", bin=alt.Bin(maxbins=45), title="Usage"),
            y=alt.Y("count():Q", title="Count", stack=None),
            color=alt.Color("Usage Type:N", scale=alt.Scale(range=["#4c78a8", "#f58518", "#54a24b", "#b279a2"])),
        )
        .properties(width=720, height=360, title="Usage distribution by stream")
    )


def eda_monthly_heatmap(df: pl.DataFrame) -> alt.Chart:
    monthly = (
        df.group_by(["Year", "Month", "Month_Num"])
        .agg(pl.col("Total Usage").sum().alias("Total Usage"))
        .sort(["Year", "Month_Num"])
    )
    return (
        alt.Chart(monthly)
        .mark_rect()
        .encode(
            x=alt.X("Year:O", title="Year"),
            y=alt.Y("Month:N", title="Month", sort=MONTH_ORDER),
            color=alt.Color("Total Usage:Q", scale=alt.Scale(scheme="yelloworangered"), title="Monthly total"),
        )
        .properties(width=720, height=360, title="Total monthly usage heatmap")
    )


def eda_lag_correlations(df: pl.DataFrame) -> alt.LayerChart:
    records: list[dict[str, float | str]] = []
    labels = {
        "Usage - 1": "Usage 1",
        "Usage - 2": "Usage 2",
        "Usage - 2_1": "Usage 2_1",
        "Total Usage": "Total",
    }
    for col, label in labels.items():
        temp = df.select(
            [
                pl.col(col).alias("current"),
                pl.col(col).shift(1).alias("Lag 1"),
                pl.col(col).shift(7).alias("Lag 7"),
                pl.col(col).shift(30).alias("Lag 30"),
            ]
        ).drop_nulls()
        for lag_name in ["Lag 1", "Lag 7", "Lag 30"]:
            records.append(
                {
                    "Series": label,
                    "Lag": lag_name,
                    "Correlation": float(temp.select(pl.corr("current", lag_name)).item()),
                }
            )

    corr_df = pl.DataFrame(records)
    base = (
        alt.Chart(corr_df)
        .mark_rect()
        .encode(
            x=alt.X("Lag:N", title=None),
            y=alt.Y("Series:N", title=None),
            color=alt.Color("Correlation:Q", scale=alt.Scale(scheme="blues")),
        )
        .properties(width=520, height=180, title="Lag correlations used for feature design")
    )
    text = base.mark_text().encode(
        text=alt.Text("Correlation:Q", format=".3f"),
        color=alt.condition(alt.datum.Correlation > 0.7, alt.value("white"), alt.value("black")),
    )
    return base + text


def eda_correlation_matrix(df: pl.DataFrame) -> alt.LayerChart:
    cols = ["Usage - 1", "Usage - 2", "Usage - 2_1", "Total Usage", "Nom", "Delivery"]
    records: list[dict[str, float | str]] = []
    for row_col in cols:
        for col_col in cols:
            value = float(df.select(pl.corr(row_col, col_col)).item())
            records.append({"Row": row_col, "Column": col_col, "Correlation": value})
    corr_df = pl.DataFrame(records)
    base = (
        alt.Chart(corr_df)
        .mark_rect()
        .encode(
            x=alt.X("Column:N", title=None),
            y=alt.Y("Row:N", title=None),
            color=alt.Color("Correlation:Q", scale=alt.Scale(scheme="blues")),
        )
        .properties(width=460, height=420, title="Full variable correlation matrix")
    )
    text = base.mark_text(fontSize=9).encode(
        text=alt.Text("Correlation:Q", format=".2f"),
        color=alt.condition(alt.datum.Correlation > 0.6, alt.value("white"), alt.value("black")),
    )
    return base + text


def eda_delivery_usage_gap(df: pl.DataFrame) -> alt.Chart:
    gap = df.select(
        [
            "Date",
            (pl.col("Delivery") - pl.col("Total Usage")).cum_sum().alias("Cumulative Gap"),
        ]
    )
    return (
        alt.Chart(gap)
        .mark_line(color="#4c78a8")
        .encode(
            x=alt.X("Date:T", title=None),
            y=alt.Y("Cumulative Gap:Q", title="Delivery - usage cumulative sum"),
        )
        .properties(width=720, height=320, title="Delivery versus usage gap under zero-inventory assumption")
    )


def policy_frame(df: pl.DataFrame) -> pl.DataFrame:
    policy = (
        df.select(["Date", "Total Usage", "Delivery"])
        .with_columns(
            [
                pl.col("Date").dt.month().alias("month"),
                (pl.col("Delivery") - pl.col("Total Usage")).alias("Net Flow"),
            ]
        )
        .with_columns(
            [
                (pl.col("month").replace_strict({month: limits["inj"] * CAPACITY for month, limits in DAILY_LIMITS.items()})).alias("Daily Upper"),
                (-pl.col("month").replace_strict({month: limits["wd"] * CAPACITY for month, limits in DAILY_LIMITS.items()})).alias("Daily Lower"),
                ((pl.col("Delivery") - pl.col("Total Usage")).cum_sum() + INITIAL_INVENTORY).alias("Storage"),
            ]
        )
        .with_columns(
            [
                (pl.col("Storage") / CAPACITY * 100).alias("Storage Pct"),
                (pl.col("month").replace_strict({month: lim["min"] * 100 for month, lim in MONTH_END_LIMITS.items()})).alias("Month End Lower"),
                (pl.col("month").replace_strict({month: lim["max"] * 100 for month, lim in MONTH_END_LIMITS.items()})).alias("Month End Upper"),
                pl.col("Date").dt.month_end().eq(pl.col("Date")).alias("Is Month End"),
            ]
        )
    )
    return policy


def eda_policy_constraints(df: pl.DataFrame) -> alt.VConcatChart:
    policy = policy_frame(df)
    month_end = policy.filter(pl.col("Is Month End"))

    flow = (
        alt.Chart(policy)
        .transform_fold(["Net Flow", "Daily Upper", "Daily Lower"], as_=["Series", "Value"])
        .mark_line()
        .encode(
            x=alt.X("Date:T", title=None),
            y=alt.Y("Value:Q", title="Net flow and daily limits"),
            color=alt.Color("Series:N", scale=alt.Scale(range=["#4c78a8", "#54a24b", "#e45756"])),
        )
        .properties(width=720, height=220, title="Daily flow relative to contract bounds")
    )
    storage = (
        alt.Chart(month_end)
        .transform_fold(["Storage Pct", "Month End Lower", "Month End Upper"], as_=["Series", "Value"])
        .mark_line(point=True)
        .encode(
            x=alt.X("Date:T", title="Date"),
            y=alt.Y("Value:Q", title="Storage percentage"),
            color=alt.Color("Series:N", scale=alt.Scale(range=["#4c78a8", "#54a24b", "#e45756"])),
        )
        .properties(width=720, height=220, title="Month-end storage position versus required bounds")
    )
    return alt.vconcat(flow, storage, spacing=14)


def corr_chart(values: pl.DataFrame, title: str) -> alt.LayerChart:
    bars = (
        alt.Chart(values)
        .mark_bar(color="#4c78a8")
        .encode(x=alt.X("lag:Q", title="Lag"), y=alt.Y("correlation:Q", title="Correlation"))
        .properties(width=720, height=300, title=title)
    )
    bounds = (
        alt.Chart(values)
        .transform_fold(["upper", "lower"], as_=["Band", "Value"])
        .mark_line(color="#e45756")
        .encode(x="lag:Q", y="Value:Q", detail="Band:N")
    )
    return bars + bounds


def eda_acf_pacf(df: pl.DataFrame) -> tuple[alt.LayerChart, alt.LayerChart]:
    series = df["Total Usage"].to_numpy()
    acf_vals = acf(series, nlags=60, alpha=0.05)
    pacf_vals = pacf(series, nlags=60, alpha=0.05)

    acf_df = pl.DataFrame(
        {
            "lag": list(range(len(acf_vals[0]))),
            "correlation": acf_vals[0].tolist(),
            "lower": (acf_vals[1][:, 0] - acf_vals[0]).tolist(),
            "upper": (acf_vals[1][:, 1] - acf_vals[0]).tolist(),
        }
    )
    pacf_df = pl.DataFrame(
        {
            "lag": list(range(len(pacf_vals[0]))),
            "correlation": pacf_vals[0].tolist(),
            "lower": (pacf_vals[1][:, 0] - pacf_vals[0]).tolist(),
            "upper": (pacf_vals[1][:, 1] - pacf_vals[0]).tolist(),
        }
    )
    return corr_chart(acf_df, "Autocorrelation function for total usage"), corr_chart(
        pacf_df, "Partial autocorrelation function for total usage"
    )


def opt_penalty_summary(df: pl.DataFrame) -> alt.HConcatChart:
    counts = pl.DataFrame(
        {
            "Metric": ["Daily", "Monthly"],
            "Actual": [int(df["Actual_Daily_Violation"].sum()), int(df["Actual_Monthly_Violation"].sum())],
            "Optimized": [int(df["Opt_Daily_Violation"].sum()), int(df["Opt_Monthly_Violation"].sum())],
        }
    ).unpivot(index="Metric", variable_name="Policy", value_name="Count")

    left = (
        alt.Chart(counts)
        .mark_bar()
        .encode(
            x=alt.X("Metric:N", title=None),
            y=alt.Y("Count:Q", title="Violation count"),
            color=alt.Color("Policy:N", scale=alt.Scale(range=["#4c78a8", "#54a24b"])),
            xOffset="Policy:N",
        )
        .properties(width=260, height=300, title="Backtest violation counts")
    )

    tiers = TIER_COUNTS.unpivot(index="Tier", variable_name="Policy", value_name="Count")
    right = (
        alt.Chart(tiers)
        .mark_bar()
        .encode(
            x=alt.X("Tier:N", title=None),
            y=alt.Y("Count:Q", title="Penalty events"),
            color=alt.Color("Policy:N", scale=alt.Scale(range=["#4c78a8", "#54a24b"])),
            xOffset="Policy:N",
        )
        .properties(width=320, height=300, title="Tier-wise cash-out penalties")
    )
    return alt.hconcat(left, right, spacing=20)


def opt_daily_penalties(df: pl.DataFrame) -> alt.VConcatChart:
    status = df.with_columns(
        pl.when(~pl.col("Actual_Daily_Violation") & ~pl.col("Opt_Daily_Violation"))
        .then(pl.lit("No violation"))
        .when(pl.col("Actual_Daily_Violation") & pl.col("Opt_Daily_Violation"))
        .then(pl.lit("Both violated"))
        .when(pl.col("Actual_Daily_Violation"))
        .then(pl.lit("Actual only"))
        .otherwise(pl.lit("Optimized only"))
        .alias("Violation Type")
    )

    delivery = (
        alt.Chart(status)
        .transform_fold(["Actual_Delivery", "Opt_Delivery"], as_=["Series", "Delivery"])
        .mark_line()
        .encode(
            x=alt.X("Date:T", title=None),
            y=alt.Y("Delivery:Q", title="Delivery"),
            color=alt.Color("Series:N", scale=alt.Scale(range=["#4c78a8", "#54a24b"])),
        )
        .properties(width=720, height=220, title="Actual versus optimized delivery")
    )
    flow = (
        alt.Chart(status)
        .transform_fold(["Actual_Flow", "Opt_Flow", "Max_Injection_Limit"], as_=["Series", "Flow"])
        .mark_line()
        .encode(
            x=alt.X("Date:T", title="Date"),
            y=alt.Y("Flow:Q", title="Net flow"),
            color=alt.Color("Series:N", scale=alt.Scale(range=["#4c78a8", "#54a24b", "#f58518"])),
        )
        .properties(width=720, height=220, title="Net flow relative to the injection limit")
    )
    return alt.vconcat(delivery, flow, spacing=14)


def opt_monthly_penalties(df: pl.DataFrame) -> alt.VConcatChart:
    eom = (
        df.with_columns(pl.col("Date").dt.month_end().eq(pl.col("Date")).alias("Is EOM"))
        .filter(pl.col("Is EOM"))
        .select(["Date", "Total_Gas_After", "Opt_Storage_After", "EOM_Min_Storage", "EOM_Max_Storage"])
    )

    lines = (
        alt.Chart(eom)
        .transform_fold(
            ["Total_Gas_After", "Opt_Storage_After", "EOM_Min_Storage", "EOM_Max_Storage"],
            as_=["Series", "Storage"],
        )
        .mark_line(point=True)
        .encode(
            x=alt.X("Date:T", title=None),
            y=alt.Y("Storage:Q", title="End-of-month storage"),
            color=alt.Color(
                "Series:N",
                scale=alt.Scale(range=["#4c78a8", "#54a24b", "#f58518", "#e45756"]),
            ),
        )
        .properties(width=720, height=240, title="Monthly storage compliance")
    )

    deviations = eom.with_columns(
        [
            pl.when(pl.col("Total_Gas_After") < pl.col("EOM_Min_Storage"))
            .then(pl.col("Total_Gas_After") - pl.col("EOM_Min_Storage"))
            .when(pl.col("Total_Gas_After") > pl.col("EOM_Max_Storage"))
            .then(pl.col("Total_Gas_After") - pl.col("EOM_Max_Storage"))
            .otherwise(0.0)
            .alias("Actual Deviation"),
            pl.when(pl.col("Opt_Storage_After") < pl.col("EOM_Min_Storage"))
            .then(pl.col("Opt_Storage_After") - pl.col("EOM_Min_Storage"))
            .when(pl.col("Opt_Storage_After") > pl.col("EOM_Max_Storage"))
            .then(pl.col("Opt_Storage_After") - pl.col("EOM_Max_Storage"))
            .otherwise(0.0)
            .alias("Optimized Deviation"),
        ]
    ).select(["Date", "Actual Deviation", "Optimized Deviation"]).unpivot(index="Date", variable_name="Policy", value_name="Deviation")

    bars = (
        alt.Chart(deviations)
        .mark_bar()
        .encode(
            x=alt.X("Date:T", title="Date"),
            y=alt.Y("Deviation:Q", title="Out-of-band deviation"),
            color=alt.Color("Policy:N", scale=alt.Scale(range=["#4c78a8", "#54a24b"])),
            xOffset="Policy:N",
        )
        .properties(width=720, height=200, title="Deviation from end-of-month bounds")
    )
    return alt.vconcat(lines, bars, spacing=14)


def opt_tier_comparison() -> alt.Chart:
    tier_long = TIER_COUNTS.unpivot(index="Tier", variable_name="Policy", value_name="Count")
    return (
        alt.Chart(tier_long)
        .mark_bar()
        .encode(
            x=alt.X("Tier:N", title=None),
            y=alt.Y("Count:Q", title="Penalty events"),
            color=alt.Color("Policy:N", scale=alt.Scale(range=["#4c78a8", "#54a24b"])),
            xOffset="Policy:N",
        )
        .properties(width=560, height=320, title="Tier-wise actual versus optimized penalties")
    )


def main() -> None:
    theme()
    usage = load_usage()
    usage_long_df = usage_long(usage)
    results = load_results()

    acf_chart, pacf_chart = eda_acf_pacf(usage)

    save(eda_usage_overview(usage), "eda_usage_overview.png")
    save(eda_boxplot_total(usage), "eda_boxplot_total_usage.png")
    save(eda_histogram(usage_long_df), "eda_histogram_streams.png")
    save(eda_monthly_heatmap(usage), "eda_monthly_heatmap_total.png")
    save(eda_lag_correlations(usage), "eda_lag_correlations.png")
    save(eda_correlation_matrix(usage), "eda_correlation_matrix.png")
    save(eda_delivery_usage_gap(usage), "eda_delivery_usage_gap.png")
    save(eda_policy_constraints(usage), "eda_policy_constraints.png")
    save(acf_chart, "eda_acf_total.png")
    save(pacf_chart, "eda_pacf_total.png")
    save(opt_penalty_summary(results), "opt_penalty_summary.png")
    save(opt_daily_penalties(results), "opt_daily_penalties.png")
    save(opt_monthly_penalties(results), "opt_monthly_penalties.png")
    save(opt_tier_comparison(), "opt_tier_comparison.png")


if __name__ == "__main__":
    main()