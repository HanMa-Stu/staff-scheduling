Daily LSTM forecast figures

Only LSTM is plotted, selected by the best mean MAE, RMSE and R2 across all 57 stations and seeds42/43/44.
Mean MAE 2020.511406; RMSE 3417.366291; R2 0.379370. These metrics average per-run metrics, not metrics recomputed on the mean curve.

Training: initial86 days; neural training targets days17-86, 70 samples in all four neural branches. SARIMA uses all86 observations for fitting.
Evaluation: observed-history one-step forecasts, days87-122. Observations become available only after their own forecasts. Frozen neural weights.

Style follows original Fig6: blue observed series, red forecast with point markers, boxed axes, inward ticks, white background, Time(day) and Cargo(pieces) labels. Thin gray dashed line marks train/test split. Forecast curve is the arithmetic mean of the three seeds; no best seed or best site is selected.

Files: stations/ contains57 individual PNGs at400dpi; Daily_LSTM_all_stations.pdf contains all stations in four-panel pages; Daily_LSTM_SC1_SC4 is the first four station illustration, selected by station number. CSV preserves each seed forecast and plotted mean.

Checks: all855 metric records were recomputed from30780 prediction rows; matching observations and wide-format predictions were verified. Source training slices were checked with instrumented stand-ins; no model was trained or estimated. Saved files do not independently prove which source revision was run.

Source prose still contains documentation inaccuracies: the endpoint slope is (last-third_last)/2, not the last-two-point slope; channel alignment precedes sorting. SARIMA is not trained from70 supervised pairs. These do not change the verified neural target alignment.
