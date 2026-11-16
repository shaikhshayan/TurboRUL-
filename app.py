import os, json
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'                  # avoid the OpenMP clash seen earlier
import numpy as np, pandas as pd, joblib, torch, torch.nn as nn
import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(page_title="Turbofan RUL Dashboard", layout="wide")

class GRUNet(nn.Module):                                     # must match the model trained in the notebook
    def __init__(self, n_feat, hidden=64):
        super().__init__()
        self.gru = nn.GRU(n_feat, hidden, num_layers=2, batch_first=True, dropout=0.2)
        self.head = nn.Sequential(nn.Linear(hidden, 32), nn.ReLU(), nn.Linear(32, 1))
    def forward(self, x):
        out, _ = self.gru(x)
        return self.head(out[:, -1, :]).squeeze(1)

@st.cache_resource                                           # load the model once, not on every click
def load_model():
    with open('models/config_fd001.json') as f:
        cfg = json.load(f)                                   # features, window, cap, sizes
    scaler = joblib.load('models/scaler_fd001.joblib')       # the scaler fitted on training engines
    model = GRUNet(cfg['n_feat'], cfg['hidden'])
    model.load_state_dict(torch.load('models/gru_fd001.pt', map_location='cpu'))
    model.eval()
    return cfg, scaler, model

@st.cache_data
def load_data():
    cols = ['unit', 'cycle', 'op1', 'op2', 'op3'] + [f's{i}' for i in range(1, 22)]
    test = pd.read_csv('data/test_FD001.txt', sep=r'\s+', header=None, names=cols)
    rul_true = pd.read_csv('data/RUL_FD001.txt', header=None)[0].values   # true RUL at each engine's last cycle
    return test, rul_true

def predict_engine(model, scaler, cfg, eng):                 # predicted RUL at every cycle of one engine
    WIN, CAP = cfg['WIN'], cfg['CAP']
    scaled = scaler.transform(eng[cfg['features']]).astype(np.float32)
    padded = np.vstack([np.repeat(scaled[:1], WIN - 1, axis=0), scaled])   # pad the start so every cycle has a window
    wins = np.stack([padded[i:i + WIN] for i in range(len(scaled))])        # window ending at each cycle
    with torch.no_grad():
        p = model(torch.tensor(wins)).numpy() * CAP          # model outputs 0 to 1, scale back to cycles
    return np.clip(p, 0, CAP), scaled

cfg, scaler, model = load_model()
test, rul_true = load_data()
CAP, WIN = cfg['CAP'], cfg['WIN']

st.title("Turbofan Engine Health Monitor")
st.caption("Remaining Useful Life (RUL) predicted by a GRU model trained on NASA C-MAPSS FD001")

st.sidebar.header("Controls")
unit = st.sidebar.selectbox("Engine (test set)", sorted(test['unit'].unique()))
sensor_options = [f for f in cfg['features'] if f.startswith('s')]
sensors = st.sidebar.multiselect("Sensors to display", sensor_options,
                                 default=[s for s in ['s2', 's7', 's11', 's12'] if s in sensor_options])
threshold = st.sidebar.slider("Maintenance alert threshold (cycles)", 5, 100, 30)

tab1, tab2 = st.tabs(["Engine monitor", "Model comparison"])

with tab1:
    eng = test[test['unit'] == unit].reset_index(drop=True)
    pred, scaled = predict_engine(model, scaler, cfg, eng)
    cycles = eng['cycle'].values
    last_cycle = cycles.max()
    true_last = int(rul_true[unit - 1])                      # row i of the RUL file belongs to engine i+1
    true_curve = np.clip(true_last + (last_cycle - cycles), 0, CAP)   # reference curve, capped like training labels
    latest = float(pred[-1])

    c1, c2, c3 = st.columns(3)
    c1.metric("Latest cycle", int(last_cycle))
    c2.metric("Predicted RUL (cycles)", f"{latest:.0f}")
    c3.metric("Reference RUL from NASA file", true_last)

    if latest <= threshold:
        st.error(f"Maintenance recommended: predicted RUL ({latest:.0f}) is at or below the threshold ({threshold}). Take the engine out of service soon.")
    elif latest <= 2 * threshold:
        st.warning(f"Schedule maintenance: predicted RUL ({latest:.0f}) is within twice the threshold ({threshold}).")
    else:
        st.success(f"Normal operation: predicted RUL ({latest:.0f}) is well above the threshold ({threshold}).")

    left, right = st.columns(2)
    with left:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(cycles[WIN - 1:], pred[WIN - 1:], color='black', label='Predicted RUL')
        ax.plot(cycles, true_curve, color='gray', ls='--', label='Reference RUL (capped)')
        ax.axhline(threshold, color='red', ls=':', label='Alert threshold')
        ax.set_xlabel('Cycle'); ax.set_ylabel('RUL (cycles)'); ax.legend()
        ax.set_title(f'Engine {unit}: predicted RUL over time')
        st.pyplot(fig)
    with right:
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        for s in sensors:
            ax2.plot(cycles, scaled[:, cfg['features'].index(s)], label=s)
        ax2.set_xlabel('Cycle'); ax2.set_ylabel('Scaled value (0 to 1)'); ax2.legend()
        ax2.set_title(f'Engine {unit}: sensor trends')
        st.pyplot(fig2)
    st.caption("Predictions start after the first 30 cycles, once a full window is available. "
               "Test engines are cut off before failure, so the reference RUL is the value from the NASA file.")

with tab2:
    st.subheader("Test RMSE by model (FD001, lower is better)")
    runs = pd.read_csv('results/fd001_all_runs.csv')
    summ = runs.groupby('model')[['RMSE', 'NASA']].agg(['mean', 'std']).round(2)
    summ.columns = ['RMSE mean', 'RMSE std', 'NASA mean', 'NASA std']
    summ = summ.sort_values('RMSE mean')
    st.dataframe(summ)
    st.bar_chart(summ['RMSE mean'])
    st.caption("Mean and standard deviation over seeds. Ridge has a single deterministic run.")
    