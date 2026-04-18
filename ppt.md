SCOT FORGE PROJECT

Team Members:
Pratyaksh Mathur
Vandit Goel
Kumar Mantha

PROJEC T OVERVIEW

Prepare and analyse the provided natural-gas usage data and develop a time-series

•
model that predicts next-day usage.

• Across the two plants, more than 60 furnaces may be active, but furnace-level gas

usage is not currently measured. Instead, Scot Forge receives facility-level meter

•
readings from a third-party provider.

• Recent Illinois legislation changed how natural gas tariffs are calculated, adding
limits that can trigger penalties. These penalties depend on daily injections or

• withdrawals from the company’s natural-gas storage bank and the total amount
stored at the end of each month.

EXPLORATORY DATA ANALYSIS

GAS USAGE

MONTHLY BOX PLOT

USAGE DISTRIBUTION

AVG MONTHLY USAGE INTENSITY

CORRELATION HEATMAPS

MODEL TRAINING &
 EVALUATION

WALK FORWARD

1

2

3

4

Train the model
using all available
historical data
up-to-date (t-1)

Generate a 2-day
forecast everyday

Discard the
Forecast for Day t

keep the Forecast
for Day t+1

WALK FORWARD EXAMPLE

Data Till

11/30/2024

12/01/2024

12/02/2024

12/03/2024

12/04/2024

12/05/2024

12/01/2024

12/02/2024

12/03/2024

12/04/2024

12/05/2024

12/06/2024

12/07/2024

Current date (t) and (t+1)

D

K

D

K

D

K

D

K

D

K

D

K

ASSUMPTIONS

• We allow for more wiggle room regarding small, insignificant errors that don't

impact the bottom line.

• Large errors are not just "bigger”, they are disproportionately more costly and

damaging to our use case.

• Over-estimating and under-estimating carry the same penalty; we simply want

to be as close to the target as possible.

• We care about the raw size of the error (e.g., missing by 10 units) rather than

the error as a percentage of the total.

EX AMPLE

Assume the gas tank reserves are 2000 units and if we keep the units below 1900 and above 2100
we get penalise.

Case 1

• Predicted usage: 300 units and Actual usage: 100 units

• Absolute error = 200 units

• % Error = 200 / 300 = 66%

Case 2

• Predicted usage: 1500 units and Actual usage: 1300 units

• Absolute error = 200 units

• %Error = 200 / 1500 = 13%

In both case the absolute forecasting error is 200 units and will get penalise.

RMSE

• RMSE tells us how far off we are in actual gas units, while heavily penalizing

large forecasting mistakes that create operational and financial risk.

• RMSE fits our assumptions very well as we want to penalise few large errors

more than many small errors to avoid penalties

If errors are 5,10 and 100 RMSE will be ~58 and MAE will be ~38. So RMSE
penalise large error more which is what we want.

ALL MODELS

Model
Basic Catboost
Basic LGBM
Catboost + Regime
Ensemble
LGBM + Regime
LGBM + Temp
Timesnet
Basic Catboost
Basic LGBM
Catboost + Regime
Ensemble
LGBM + Regime
LGBM + Temp
Timesnet
Basic Catboost
Basic LGBM
Catboost + Regime
Ensemble
LGBM + Regime
LGBM + Temp
Timesnet

Usage Type
Usage - 1
Usage - 1
Usage - 1
Usage - 1
Usage - 1
Usage - 1
Usage - 1
Usage - 2
Usage - 2
Usage - 2
Usage - 2
Usage - 2
Usage - 2
Usage - 2
Usage - 2_1
Usage - 2_1
Usage - 2_1
Usage - 2_1
Usage - 2_1
Usage - 2_1
Usage - 2_1

RMSE
37.93
40.04
37.06
41.70
36.72
40.31
57.64
239.56
243.98
237.50
227.66
252.51
249.19
258.29
8.88
10.24
8.80
8.17
7.84
9.87
15.46

THANK YOU


