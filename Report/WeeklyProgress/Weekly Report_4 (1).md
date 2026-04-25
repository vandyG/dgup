Weekly Report 4

Project title: Scot Forge                    
Date: 03/02/2026

Team members:
| Name       | Student ID  |
| ------------------------ | ----------------------- |
| Vandit Goel        | 1002245699  |
| Pratyaksh Mathur   | 1002228261  |
| Kumar Mantha | 1002233682

Progress report:
As discussed in our last call, we trained multiple ML models this week using different variations and training techniques. Below is a summary of our key findings:

| Usage  Type | Model  | RMSE  |MAE  | MAPE  |Description  |
|---|---| ------------- | ---------- | ------------------ | --- |
|Total Usage| Xgboost  | 270.5  |203.77   | 17%  |Basic Xgboost|
|Total Usage| LGBM     | 262.14|  194.48  | 19%|  Basic LGBM|
|Total Usage| LGBM     | 271.12  |198.79  | 20%  |LGBM + Regime based training|
|Total Usage | Catboost  | 258.57|  191.82  | 20%|  Basic Catboost       | Kumar  |
|Total Usage | Ensemble  | 265.16 | 192.59  | 21% | Average ensemble of Xgboost lgbm and catboost|
|Total Usage| Catboost  | 258|  191.77  | 19%|  CatBoost Regime + Residual + Hyperparameter Tuning|
|Total Usage| LGBM  | 262.73|  198.07  | 19%|  LGBM +Temperature columns |
|Usage 1|TimesNet| 10.709| 8.9209| 3%| Base TimesNet (raw Forecast)|
|Usage 2|TimesNet |292.17| 289.17| 17%| Base TimesNet (raw Forecast)|
|Usage 2| TimesNet |234.42| 193.81| 11%| Caliberated TimeNet (TimesNet + Online Correction)|
|Usage 1| PatchTST |43.417| 32.1| 25%| Patch Time Series Transformer|
|Usage 2| PatchTST |255.14| 184.86| 20%| Patch Time Series Transformer|

We used a 1-day walk-forward approach where we retrained the models each day on all
historical data up to that day, predicted the next day only, and repeated this across the forecast period (2024) to simulate real production forecasting.
