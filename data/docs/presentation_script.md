# Scot Forge Presentation Script

## Speaker 1: Introduction and Problem Setup

**Opening**
Good [morning/afternoon], everyone. We are presenting our work on forecasting natural-gas usage for Scot Forge and using that forecast to reason about storage constraints and potential tariff penalties.

**Team Introduction**
Our team members are Pratyaksh Mathur, Vandit Goel, and Kumar Mantha.

**Problem Overview**
Scot Forge operates across two plants, with more than 60 furnaces that may be active, but usage is only measured at the facility level through third-party meter readings. That means we do not have furnace-level detail, so we need to work with aggregated gas usage data.

The business problem is not just forecasting usage. Recent Illinois legislation also introduced tariff rules that can create penalties based on how much gas is injected into or withdrawn from the storage bank, as well as the inventory level at the end of each month. So our forecast has to be useful operationally, not just statistically accurate.

**Project Goal**
Our goal is to prepare and analyze the gas usage data, identify the strongest patterns in the time series, and develop a next-day usage model that can support decision-making under those storage constraints.

Alongside the analysis, we also prepared a comprehensive dashboard for business use so the key patterns, forecasts, and operational checks can be reviewed in a more interactive format.

**Transition**
With that context, we will start with the exploratory data analysis and the main usage patterns we found, and we will switch to the dashboard view as we go through those results.

---

## Speaker 2: Exploratory Data Analysis and Modeling

**EDA Overview**
We began by loading the gas usage data and reshaping it so we could compare the different usage streams consistently. The data shows three main usage components: Usage - 1, Usage - 2, and Usage - 2_1, along with a combined total usage series.

As we move through the EDA, we will use the dashboard to present these patterns in a business-friendly way, starting with the high-level usage explorer and then drilling into seasonality, lag structure, and correlations.

**Usage Patterns**
From the usage explorer, we found that Usage - 1 follows a fairly regular seasonal cycle, with winter peaks and summer dips. It also shows a clear Saturday slump, which suggests reduced operational activity on weekends.

Usage - 2 is the dominant contributor to total consumption. It is much more volatile than the other streams and explains most of the variability in total usage. We also saw a stagnation period in late 2017, followed by jumps in early 2018.

Usage - 2_1 is much smaller, but it still matters because it shows a long-term declining trend and periods of near inactivity, especially during summer months.

**Seasonality and Distribution**
The box plots and distribution charts reinforce those patterns. Monthly usage varies sharply by season, and the density plots show that Usage - 2 and total usage share a broad distribution centered around the high-consumption range, while Usage - 1 is much tighter and Usage - 2_1 is concentrated near zero.

We also looked at the average monthly usage intensity heatmap. That confirmed strong annual seasonality, with winter months much heavier than summer months. The weekday effect was also clear, especially the recurring Saturday reduction.

**Lag and Correlation Analysis**
Next, we explored lagged correlations. The lag heatmaps showed that Usage - 1 has a strong weekly relationship, with the 7-day lag stronger than the 1-day lag. Usage - 2 showed moderate short-term dependence but little long-term signal at 30 days. Usage - 2_1 had very strong persistence from day to day. Total usage largely followed the behavior of Usage - 2.

The full correlation matrix confirmed that Total Usage is strongly related to Usage - 2, with a much weaker relationship to the other variables and to the delivery and supply data.

**ACF and PACF**
We also examined ACF and PACF to understand how much historical information the series retains. The autocorrelation structure supported the idea that short-term and weekly history are useful inputs for forecasting.

**Transition**
Based on those patterns, we moved into modeling and evaluation, and we also defined the business metric so that model performance reflects the real cost of errors.

---

## Speaker 3: Forecasting, Storage Logic, and Conclusion

**Modeling Objective**
For the modeling work, we focused on predicting next-day usage. Because the business cares about the size of the forecasting error in actual gas units, we used RMSE as the main evaluation metric. RMSE penalizes large misses more heavily, which matches the operational risk of getting the forecast wrong by a large amount.

To make the forecast operationally realistic, we used a walk-forward retraining setup. Current-day usage is not fully available until midway through the next day, so each day we retrain the model on data through $t-2$, generate a two-day forecast, and then use the second forecasted day as the prediction we evaluate and act on. That gives us a fresh forecast every day as new actuals arrive, rather than relying on a stale model.

This approach is also flexible. If the business later wants a longer planning horizon, the same framework can be expanded from a 2-day forecast to a 7-day forecast without changing the overall retraining logic.

**Why RMSE Matters**
We also highlighted why absolute error matters more than percentage error in this context. A 200-unit miss is damaging whether the prediction was small or large. That is why our modeling choice emphasizes unit-level accuracy and why RMSE is a better fit than a percentage-based metric.

**Model Results**
Across the models we tested, performance varied by usage stream. For Usage - 1, the best results were in the high-30s to low-40s RMSE range. For Usage - 2, the errors were much larger because the series is more volatile, with RMSE values around the mid-200s. Usage - 2_1 was easier to forecast and produced the lowest RMSE values overall.

The main takeaway is that no single model dominated every stream. The ensemble and regime-aware approaches were competitive, but the dominant factor remained the intrinsic volatility of each usage stream.

**Penalty and Storage Analysis**
We then used the delivery-versus-usage gap to reason about storage behavior. The initial assumption that the tank starts empty did not hold, because the calculated inventory could go negative, which is physically impossible. That told us there must already be gas in storage at the start of the series.

After adjusting for an initial inventory, we modeled daily injection and withdrawal limits, plus month-end inventory bounds. This let us check when a day or month would trigger a penalty.

**Operational Policy Prototype**
To make the analysis more actionable, we built a simple policy prototype that recommends delivery based on forecast usage and the remaining days in the month. The policy weights month-end violations more heavily, because those penalties are more expensive and more operationally sensitive.

The policy charts show both the daily net flow and the month-end inventory trajectory, with penalty points marked when the recommended flow or inventory goes outside the allowed bounds.

**Final Takeaways**
Our main conclusions are:

- Temperature and seasonality are major drivers of gas usage.
- Calendar effects, especially month and weekday patterns, should be included as features.
- Rolling averages and lag features help capture persistence and weekly structure.
- Delivery-minus-usage inventory logic is important for penalty-aware planning.
- The most useful forecasting system is one that balances accuracy with the real cost of operational mistakes.

**Closing**
Thank you for listening. We are happy to answer any questions.
