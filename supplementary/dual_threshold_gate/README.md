# Supplementary dual-threshold gate (6/1)

**Additional experiment only. Excluded from the correction's five-model tables, figures, summaries and conclusions.**

The two saved branches predict next-day levels: LSTM uses the preceding observed value; CEEMDAN-LSTM uses the last sorted decomposition component and the preceding observed value. Both predict a change and add it to the preceding level. The common network and causal feature construction are documented in ../../code/forecast/daily/METHODS.md.

The lagged signal is z[t]=abs(y[t-1]-y[t-2])/std(diff(y[:86])). The scale uses training data; the numerator may use already observed test values. With z>=6, select LSTM. With z<1, select CEEMDAN-LSTM. Otherwise retain the previous state. This is output selection from already computed forecasts, not an implementation that avoids decomposition computation during the LSTM state.

The supplied 171-run mean is approximately MAE 2028.0474, RMSE 3414.7865, R2 0.3941. All supplied rule comparisons, threshold scans, station details, formal 6/1 metrics and daily branch predictions are retained with English names and column labels. Numerical values are unchanged. Source paired-test statistics remain supplementary descriptive records; no new significance claim is made here.

To recalculate gates from archived branch predictions without training:
```bash
python supplementary/dual_threshold_gate/code/gate_hyst_agg.py
```
This overwrites the supplementary rule tables only. To retrain the two branches (optional, not performed during packaging):
```bash
TORCH_DEVICE=cpu python supplementary/dual_threshold_gate/code/model_gated.py --workers 6
```
model_gated.py implements the older theta/cooldown comparison and writes branch forecasts. Run gate_hyst_agg.py afterward for the reported 6/1 hysteresis configuration. Shared data are read from the package's data/daily directory.
