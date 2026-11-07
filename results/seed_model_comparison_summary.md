# Model comparison summary by seed

This summary uses only the `pure` results. Each mean gives equal weight to all 57 stations within the same forecast frequency and seed. The standard deviations across stations are available in [seed_model_means.csv](forecast/seed_model_means.csv).

Differences are calculated as `main model − baseline`. Negative MAE and RMSE differences indicate lower error for the main model, and positive R² differences indicate higher R². Win, loss, and tie counts compare the main model with each baseline station by station.

## Daily forecasts

### Seed 42

| Model | Mean MAE | Mean RMSE | Mean R² |
|---|---:|---:|---:|
| SARIMA | 4421.65 | 5823.27 | -0.5583 |
| LSTM | 4975.23 | 6998.36 | -1.7713 |
| SARIMA-LSTM | 4409.26 | 5801.15 | -0.5358 |
| CEEMDAN-LSTM | 4800.76 | 6204.74 | -0.9195 |
| SARIMA-CEEMDAN-LSTM | 4592.25 | 5773.74 | -0.5025 |

| Baseline | ΔMAE | ΔRMSE | ΔR² | RMSE wins/losses/ties |
|---|---:|---:|---:|---:|
| SARIMA | 170.60 | -49.53 | 0.0558 | 26/31/0 |
| LSTM | -382.97 | -1224.62 | 1.2688 | 26/31/0 |
| SARIMA-LSTM | 182.99 | -27.41 | 0.0332 | 25/32/0 |
| CEEMDAN-LSTM | -208.51 | -431.00 | 0.4170 | 36/21/0 |

### Seed 43

| Model | Mean MAE | Mean RMSE | Mean R² |
|---|---:|---:|---:|
| SARIMA | 4421.65 | 5823.27 | -0.5583 |
| LSTM | 5004.07 | 7023.64 | -1.7109 |
| SARIMA-LSTM | 4404.32 | 5795.80 | -0.5340 |
| CEEMDAN-LSTM | 4844.56 | 6373.16 | -1.1066 |
| SARIMA-CEEMDAN-LSTM | 4504.77 | 5779.05 | -0.4772 |

| Baseline | ΔMAE | ΔRMSE | ΔR² | RMSE wins/losses/ties |
|---|---:|---:|---:|---:|
| SARIMA | 83.12 | -44.22 | 0.0812 | 28/29/0 |
| LSTM | -499.30 | -1244.59 | 1.2337 | 29/28/0 |
| SARIMA-LSTM | 100.45 | -16.75 | 0.0568 | 27/30/0 |
| CEEMDAN-LSTM | -339.79 | -594.11 | 0.6294 | 39/18/0 |

### Seed 44

| Model | Mean MAE | Mean RMSE | Mean R² |
|---|---:|---:|---:|
| SARIMA | 4421.65 | 5823.27 | -0.5583 |
| LSTM | 5179.39 | 7216.74 | -2.1201 |
| SARIMA-LSTM | 4407.48 | 5802.32 | -0.5400 |
| CEEMDAN-LSTM | 4889.47 | 6428.43 | -1.0625 |
| SARIMA-CEEMDAN-LSTM | 4537.64 | 5740.00 | -0.4674 |

| Baseline | ΔMAE | ΔRMSE | ΔR² | RMSE wins/losses/ties |
|---|---:|---:|---:|---:|
| SARIMA | 115.99 | -83.28 | 0.0910 | 31/26/0 |
| LSTM | -641.74 | -1476.75 | 1.6528 | 32/25/0 |
| SARIMA-LSTM | 130.17 | -62.32 | 0.0726 | 32/25/0 |
| CEEMDAN-LSTM | -351.83 | -688.43 | 0.5951 | 37/20/0 |

## Hourly forecasts

### Seed 42

| Model | Mean MAE | Mean RMSE | Mean R² |
|---|---:|---:|---:|
| SARIMA | 129.43 | 170.18 | 0.6246 |
| LSTM | 218.38 | 280.66 | 0.1631 |
| SARIMA-LSTM | 126.60 | 167.78 | 0.6369 |
| CEEMDAN-LSTM | 272.52 | 339.51 | 0.0996 |
| SARIMA-CEEMDAN-LSTM | 130.40 | 171.50 | 0.6435 |

| Baseline | ΔMAE | ΔRMSE | ΔR² | RMSE wins/losses/ties |
|---|---:|---:|---:|---:|
| SARIMA | 0.97 | 1.31 | 0.0188 | 28/29/0 |
| LSTM | -87.98 | -109.17 | 0.4804 | 53/4/0 |
| SARIMA-LSTM | 3.79 | 3.72 | 0.0066 | 24/33/0 |
| CEEMDAN-LSTM | -142.12 | -168.01 | 0.5439 | 55/2/0 |

### Seed 43

| Model | Mean MAE | Mean RMSE | Mean R² |
|---|---:|---:|---:|
| SARIMA | 129.43 | 170.18 | 0.6246 |
| LSTM | 225.10 | 290.78 | 0.1448 |
| SARIMA-LSTM | 127.65 | 169.03 | 0.6278 |
| CEEMDAN-LSTM | 274.39 | 347.21 | 0.0853 |
| SARIMA-CEEMDAN-LSTM | 130.39 | 171.91 | 0.6125 |

| Baseline | ΔMAE | ΔRMSE | ΔR² | RMSE wins/losses/ties |
|---|---:|---:|---:|---:|
| SARIMA | 0.96 | 1.73 | -0.0121 | 22/35/0 |
| LSTM | -94.72 | -118.86 | 0.4677 | 54/3/0 |
| SARIMA-LSTM | 2.73 | 2.89 | -0.0152 | 19/38/0 |
| CEEMDAN-LSTM | -144.00 | -175.30 | 0.5273 | 55/2/0 |

### Seed 44

| Model | Mean MAE | Mean RMSE | Mean R² |
|---|---:|---:|---:|
| SARIMA | 129.43 | 170.18 | 0.6246 |
| LSTM | 220.53 | 284.32 | 0.1744 |
| SARIMA-LSTM | 126.86 | 167.97 | 0.6327 |
| CEEMDAN-LSTM | 280.93 | 353.25 | 0.0521 |
| SARIMA-CEEMDAN-LSTM | 134.33 | 175.72 | 0.6046 |

| Baseline | ΔMAE | ΔRMSE | ΔR² | RMSE wins/losses/ties |
|---|---:|---:|---:|---:|
| SARIMA | 4.90 | 5.54 | -0.0200 | 21/36/0 |
| LSTM | -86.20 | -108.59 | 0.4302 | 52/5/0 |
| SARIMA-LSTM | 7.47 | 7.76 | -0.0281 | 19/38/0 |
| CEEMDAN-LSTM | -146.59 | -177.53 | 0.5525 | 51/6/0 |
