Hourly SARIMA-LSTM forecast figures.
57 individual station PNGs at400dpi and a vector PDF atlas are included. Only the best aggregate hourly model is plotted. Forecast lines are the arithmetic means of seeds42/43/44; reported metrics remain averages of per-run metrics. Blue: all720 observed hours. Red: fixed-origin forecasts for hours506-720. Gray dashed line: training/test boundary after hour505. No test observations are fed back and no model was retrained.
Run python code/forecast/plot_hourly_forecasts.py from the repository root to reproduce the figures.
