# Decisions log

## 15 Oct 2026
- Dataset: NASA C-MAPSS from Kaggle mirror (includes the original paper). Citing Saxena et al. (2008) as the source.
- Starting with FD001 only because it has one operating condition, so I can build the pipeline first and extend later.
- Environment: Python 3.14, pandas 3.0.6, numpy 2.5.4, scikit-learn 1.9.1, xgboost 3.4.1, lightgbm 4.7.0, torch 2.14.1.
- Created a separate virtual environment because the first one belonged to my asteroid project.


## 14 Nov 2026
- Sensors s2 and s11 trend upward and s7 and s12 trend downward before failure. Early life is flat and noisy, so RUL will be capped at 125 and windows of recent cycles will be used.
- RUL capped at 125 cycles because early-life degradation is not visible.
- Dropped constant columns: [op3, s1, s5, s10, s16, s18, s19]. Kept [17] features.

--Split 80 train / 20 validation engines by engine id with seed 42; MinMax scaler fitted on training engines only to avoid leakage.


RMSE                        NASA                       
               mean   std    min    max    mean    std     min     max
model                                                                 
GRU           14.84  0.31  14.57  15.17  368.29  22.23  346.46  390.91
CNN1D         15.05  0.32  14.82  15.42  333.46  22.60  309.49  354.38
LightGBM      16.66  0.14  16.53  16.80  479.40   4.12  475.28  483.51
RandomForest  17.87  0.02  17.85  17.89  602.48   5.97  598.81  609.36
Ridge         17.90   NaN  17.90  17.90  542.38    NaN  542.38  542.38
model
CNN1D           3
GRU             3
LightGBM        3
RandomForest    3
Ridge           1
dtype: int64

Limits:
- Three seeds only. The standard deviations are rough estimates.
- One validation split and one test set of 100 engines. Seeds change training, not the split, so the spread understates the real uncertainty.
- No tuning for any model. Your protocol promised an equal tuning budget. In practice every model used default-style settings, which is equal but zero. Say that plainly.
- FD001 only, so no generalization claim yet.
- As far as I know, published FD001 results for tuned deep models are somewhat lower than 14.8, so call yours a solid untuned result, not a state-of-the-art one.