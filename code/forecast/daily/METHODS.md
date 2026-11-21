# Daily implementation notes

## Data and protocol

All 57 stations have 122 observations. Train on indices [0,86), evaluate [86,122), with seeds 42,43,44. Predict y[t] before using y[t]. Later predictions may use previously observed test values. Neural weights and normalization parameters are fixed after training. SARIMA coefficients are fitted once; state.extend incorporates each observation only after its forecast.

## Model formulas

Let s[t] be the SARIMA prediction and e[t]=y[t]-s[t]. Training residuals use training fitted values; test residuals use sequential one-step predictions. Let D be the network-predicted change.

| Model | Input | Final forecast |
|---|---|---|
| SARIMA | Observed history | s[t] |
| LSTM | y[t-1] | y[t-1] + D_y[t] |
| SARIMA-LSTM | e[t-1] | s[t] + e[t-1] + D_e[t] |
| CEEMDAN-LSTM | Last sorted component and y[t-1] | y[t-1] + D_y[t] |
| SARIMA-CEEMDAN-LSTM | Last sorted residual component and e[t-1] | s[t] + e[t-1] + D_e[t] |

Both decomposition branches use two inputs, not every IMF. All neural branches use two LSTM layers, 64 hidden units, dropout 0.2, a scalar output, 400 epochs, Adam, MSE, batch size 32, learning rate 0.001, weight decay 0.0001, gradient clipping 5, and learning-rate multiplication by 0.1 at epochs 200 and 300. Input values share a training-only min/max scale; delta targets have a separate training-only scale. Adding the previous observation/residual in predict_delta reconstructs a level from a predicted change.

## Decomposition features

The observed feature history is series[max(0,s-32):s], with minimum length 16. Append one synthetic value using slope (window[-1]-window[-3])/2. Decompose with CEEMDAN (50 trials, epsilon 0.005, noise scale 1, serial execution, fixed seed). K is determined from the extended initial training sequence. Align to K first, then sort by descending zero-crossing rate. Select the penultimate time row and its last sorted channel, and concatenate the preceding observed level. Frequency sorting is a numerical ordering, not proof of invariant physical mode identities; the last sorted channel is not necessarily the original unsorted residual channel.

All four neural branches use the same70 supervised target dates: indices16:86 (days17-86). Their input at each target ends at the preceding observed day. The earlier15 targets are excluded from the standalone and residual-only branches to match the decomposition branches. SARIMA still fits all86 training observations; a likelihood-based series fit is not counted as70 supervised neural pairs. All three SARIMA-family models share this same86-observation fit. K uses the complete initial training segment, while individual feature windows end before their prediction targets; no held-out test targets enter training.

## SARIMA

Period 7; ordinary candidates (1,0,0), (0,1,1), (1,1,0), (1,1,1), crossed with no seasonal component, (1,0,0,7), (0,0,1,7), (1,0,1,7). Select the lowest training AIC+BIC among converged, finite candidates. All three SARIMA-family models share the same station fit and sequential prediction path.

## Documentation corrections and reproducibility

Some source comments describe a last-two-point slope and sorting before alignment. The executable source instead uses (last-third_last)/2 and aligns before sorting. Comments were corrected in English without changing computation. Two LSTM layers are used, and hybrid forecasts include the preceding residual. Reconstruction is checked before input scaling.

The delivery includes30780 daily prediction records (57 sites x3 seeds x5 models x36 days), plus a6156-row wide table. All855 station-seed-model metrics were recomputed from these predictions and matched the supplied metrics within3.64e-12. All actual values match the supplied station data. These checks do not independently establish the historical source/device used for training. No models were retrained during packaging. Native reruns write station_metrics.csv and model_summary.csv; add --save-pred for daily_predictions.csv and daily_predictions_wide.csv. Seed summaries and per-seed station files are packaging exports. Use --outdir to avoid overwriting saved results. Plots show only LSTM, selected by the best mean of all three metrics, with the mean prediction across all three seeds. The metrics remain averages of per-run metrics, not metrics of the plotted mean curve.

The70-target result tables were recovered from the original archived70target files. The runner restores the previously reviewed TR=MINW=16 slicing, and decomposition uses the preserved original70-target implementation without mirror padding. This source recovery is recorded in SOURCE_RECOVERY.json; it is not a new training run.
