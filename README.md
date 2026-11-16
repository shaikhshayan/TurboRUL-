# TurboRUL: Turbofan Engine Remaining Useful Life Prediction

Final year project (FYP4202 Project II), FDSIT, INTI International University.
Author: Shayan Athar, BCSI. Supervisor: Dr Shadab Kalhoro.

## What this project does
Compares classical, gradient boosting and neural network models for predicting the Remaining Useful Life (RUL) of turbofan engines on the NASA C-MAPSS FD001 dataset, and provides a Streamlit dashboard that shows the predictions with a green, yellow or red maintenance alert.

## Setup (Mac or Linux)
    git clone https://github.com/shaikhshayan/TurboRUL-.git
    cd TurboRUL-
    python3 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt

## Data
The dataset is not included. Download NASA C-MAPSS (Saxena et al., 2008) and place train_FD001.txt, test_FD001.txt and RUL_FD001.txt in a folder named data/ at the project root.

## Run
Notebook: open notebooks/02_rul_pipeline.ipynb with the .venv kernel and run the cells in order.
Dashboard, from the project root:
    streamlit run app.py

## Method summary
Engine-level 80/20 train and validation split, RUL capped at 125, 17 informative features, MinMax scaling fitted on training engines only, 30-cycle windows. Models: Ridge, Random Forest, LightGBM, 1D-CNN, GRU. Metrics: RMSE and NASA score on the 100 test engines.

## Results (test set, mean over 3 seeds)
| Model | RMSE | NASA score |
|---|---|---|
| GRU (tuned) | 14.60 | 364.8 |
| GRU | 14.84 | 368.3 |
| 1D-CNN | 15.05 | 333.5 |
| LightGBM | 16.66 | 479.4 |
| Random Forest | 17.87 | 602.5 |
| Ridge | 17.90 | 542.4 |

Only the GRU was tuned. Differences under about 0.3 RMSE are within seed noise. The data is simulated and only subset FD001 was used. The dashboard uses the GRU trained with seed 0.

## Folders
notebooks/ (analysis), models/ (saved GRU, scaler, config), results/ (tables and figures), app.py (dashboard), logs/ (project logs).
