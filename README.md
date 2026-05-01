# Pile Driving SRD Correction Tool 🌊

ML-based correction of offshore pile driving simulations for monopile installation.



---

## Overview

This tool corrects the systematic bias in pre-installation pile driving simulations
using a machine learning model trained on back-calculated data from 51 pile locations
across two North Sea wind farms:
- **Merkur** (German North Sea, MO locations) — primarily sand
- **Rentel** (Belgian North Sea, R locations) — primarily clay

**Key results (clay, leave-one-location-out evaluation):**
- 50% RMSE reduction (46.4 → 23.2 MN)
- 99% bias reduction (+30.9 → -0.2 MN)  
- 34/38 locations improved vs raw simulation
- Calibrated 90% conformal prediction intervals

---

## Repository Structure

```
pile_srd_app/
├── app.py                    ← Streamlit application (main entry point)
├── requirements.txt          ← Python dependencies
├── README.md
├── models/
│   ├── clay_model.pkl        ← Trained ANN for clay (copy from notebook)
│   ├── sand_model.pkl        ← Trained ANN for sand (copy from notebook)
│   └── training_stats.json   ← Conformal qhats, site mean K, metadata
├── data/
│   ├── sample_CPT_R3.csv     ← Sample CPT data for demo (copy from notebook)
│   └── sample_SIM_R3.csv     ← Sample simulation data for demo
└── utils/
    ├── __init__.py
    └── predictor.py          ← Core ML pipeline: features, prediction, intervals
```

---

## Setup

### 1. Train and export models (Jupyter notebook)

Run Cell 23 in the analysis notebook, then copy the exported files:

```bash
cp pile_driving_ml/models/clay_model.pkl      pile_srd_app/models/
cp pile_driving_ml/models/sand_model.pkl      pile_srd_app/models/
cp pile_driving_ml/models/training_stats.json pile_srd_app/models/
cp pile_driving_ml/data/sample_CPT_R3.csv     pile_srd_app/data/
cp pile_driving_ml/data/sample_SIM_R3.csv     pile_srd_app/data/
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the app locally

```bash
cd pile_srd_app
streamlit run app.py
```

The app will open at `http://localhost:8501`

---

## Deploy to Streamlit Cloud

1. Push this repository to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repo
4. Set the main file path to `app.py`
5. Deploy

**Note:** The `models/` directory with `.pkl` files must be committed to the repo
(or use Streamlit Secrets + cloud storage for larger models).

---

## How to Use the App

### Step 1 — Configure (sidebar)
- Select wind farm site (Rentel/Merkur)
- Select dominant soil type (clay/sand)
- Choose CPT file format
- Enter pile diameter
- Set prediction interval coverage (70–95%)

### Step 2 — Upload data
Upload your CPT file and simulation CSV, or use the sample data for a demo.

**CPT formats supported:**
- **R-format (.txt):** Tab/space delimited with header row (Depth, qc, fs, qt)
- **MO-format (.dat):** Space delimited, no header (col 0=depth, 2=qt, 4=fs)
- **Generic CSV:** Any CSV with columns named Depth, qt, fs (or similar)

**Simulation CSV:** Must have columns `Depth_actual` [m] and `SRD_sim` [kN].
Optional: `SoilGroup` (clay/sand) per row.

### Step 3 — Run prediction
Click "Run SRD Correction". The app will:
1. Merge CPT and simulation data by rounded depth key
2. Compute 11 engineered features
3. Compute K_std for screening
4. Apply ML or mean correction based on K_std ≥ 0.15
5. Compute conformal prediction intervals

### Step 4 — Download results
- Corrected SRD profile as CSV
- Publication-ready plot as PNG

---

## The Two-Stage Framework

```
New pile location
        │
        ▼
   Compute K_std
   (K variability from CPT + simulation)
        │
   K_std ≥ 0.15?
   ┌────┴────┐
  YES        NO
   │          │
   ▼          ▼
  ML       Mean K
correction  correction
(ANN predicts  (site mean K
 depth-varying K) applied uniformly)
   │          │
   └────┬─────┘
        ▼
  Apply correction
  Add conformal PI
        ▼
  Corrected SRD profile
  with uncertainty bounds
```

---

## Model Architecture

- **Type:** Artificial Neural Network (ANN / MLP)
- **Architecture:** (10, 10) — two hidden layers, 10 neurons each
- **Activation:** ReLU
- **Optimizer:** Adam (adaptive learning rate)
- **Target:** K = SRD_actual / SRD_simulated (bounded [0.2, 2.5])
- **Separate models:** trained for clay and sand independently
- **Uncertainty:** Split conformal prediction intervals from LOGO residuals

### Features (11 total)

| Feature | Description | Unit |
|---|---|---|
| Depth_actual | Penetration depth | m |
| SRD_sim | Pre-installation simulated SRD | kN |
| CPT_col3 (fs) | Sleeve friction | MPa |
| CPT_col5 (qt) | Corrected tip resistance | MPa |
| fs_cumulative | Cumulative sleeve friction integral | MPa·m |
| qc_cumulative | Cumulative tip resistance integral | MPa·m |
| friction_ratio | fs/qt ratio | — |
| ISBT | Soil Behaviour Type Index (Robertson 2010) | — |
| tip_area | Pile tip cross-sectional area | m² |
| max_diameter | Maximum pile outer diameter | m |
| pct_sand | Fraction of sandy layers above tip | 0–1 |

---

## Known Limitations

1. Clay model is more reliable than sand (LOGO R²: clay=0.24, sand=0.16)
2. Performance at sites outside Merkur/Rentel is uncertain
3. Shallow depth intervals (0–10 m) are under-calibrated (50% empirical coverage)
4. K_std screening accuracy is 63% — recommendations should be verified manually
5. Sand at Merkur (MO) and Rentel (R) have fundamentally different K distributions
   (mean K: 1.28 vs 0.76) — site flag matters for sand

---

## Citation

If you use this tool in research, please cite:

```
Philipp, J.M. (2026). Improving Pile Driving Predictions with Machine Learning.
Master's Thesis, Aalborg University, Denmark.
Collaboration: Florida Polytechnic University & COWI A/S.
```

---

## License

Research use only. Data provided by COWI A/S under NDA.
Contact Parisa Hajibabaee (Florida Polytechnic University) for access.
