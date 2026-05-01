"""
Core ML prediction pipeline for SRD correction.
Handles feature engineering, model loading, prediction, and conformal intervals.
"""
import numpy as np
import pandas as pd
import pickle
import json
import os

PA_KPA = 100.0
SANDY  = {"sand", "sand/silt"}
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

FEATURE_COLS = [
    "Depth_actual", "SRD_sim",
    "CPT_col3", "CPT_col5",
    "fs_cumulative", "qc_cumulative",
    "friction_ratio", "ISBT",
    "tip_area", "max_diameter", "pct_sand"
]

def compute_isbt(qt_mpa, fs_mpa):
    qt_norm = np.where(qt_mpa > 0, qt_mpa / PA_KPA, np.nan)
    Rf_pct  = np.where((qt_mpa > 0) & (fs_mpa > 0), (fs_mpa / qt_mpa) * 100, np.nan)
    isbt = np.sqrt((3.47 - np.log10(qt_norm))**2 + (np.log10(Rf_pct) + 1.22)**2)
    return np.nan_to_num(isbt, nan=2.0)

def engineer_features(df, tip_area=None, max_diameter=None):
    df = df.copy().sort_values("Depth_actual").reset_index(drop=True)
    df["friction_ratio"] = (df["CPT_col3"] / df["CPT_col5"].replace(0, np.nan)).clip(0, 0.1)
    df["ISBT"]           = compute_isbt(df["CPT_col5"].values, df["CPT_col3"].values)
    df["tip_area"]       = float(tip_area)     if tip_area     is not None else 28.27
    df["max_diameter"]   = float(max_diameter) if max_diameter is not None else 6.0
    depth   = df["Depth_actual"].to_numpy()
    dz      = np.diff(depth, prepend=depth[0])
    dz[0]   = 0.02
    fs_vals = df["CPT_col3"].fillna(0).to_numpy()
    qt_vals = df["CPT_col5"].fillna(0).to_numpy()
    sandy   = (df["SoilType"].str.lower().isin(SANDY).astype(float).to_numpy()
               if "SoilType" in df.columns else np.zeros(len(df)))
    df["fs_cumulative"] = np.cumsum(fs_vals * np.abs(dz))
    df["qc_cumulative"] = np.cumsum(qt_vals * np.abs(dz))
    cum_t = np.cumsum(np.abs(dz))
    df["pct_sand"] = np.where(cum_t > 0, np.cumsum(sandy * np.abs(dz)) / cum_t, 0.0)
    return df

def load_models():
    models, stats = {}, {}
    for soil in ["clay", "sand"]:
        path = os.path.join(MODEL_DIR, f"{soil}_model.pkl")
        if os.path.exists(path):
            with open(path, "rb") as f:
                models[soil] = pickle.load(f)
    stats_path = os.path.join(MODEL_DIR, "training_stats.json")
    if os.path.exists(stats_path):
        with open(stats_path) as f:
            stats = json.load(f)
    return models, stats

def predict_srd(depth, srd_sim, qt_mpa, fs_mpa,
                soil_type="clay", soil_labels=None,
                tip_area=None, max_diameter=None,
                coverage=0.90, site="R"):
    models, stats = load_models()
    depth   = np.array(depth,   dtype=float)
    srd_sim = np.array(srd_sim, dtype=float)
    qt_mpa  = np.array(qt_mpa,  dtype=float)
    fs_mpa  = np.array(fs_mpa,  dtype=float)
    if soil_labels is None:
        soil_labels = [soil_type] * len(depth)
    df = pd.DataFrame({
        "Depth_actual": depth, "SRD_sim": srd_sim,
        "CPT_col5": qt_mpa, "CPT_col3": fs_mpa, "SoilType": soil_labels
    })
    df = engineer_features(df, tip_area=tip_area, max_diameter=max_diameter)
    # K_std screening
    k_rough = srd_sim / np.clip(srd_sim.mean(), 1, None)
    k_std   = float(np.std(k_rough))
    recommendation = "ML" if k_std >= 0.15 else "MEAN"
    # Predict K
    if soil_type in models:
        X     = df[FEATURE_COLS].fillna(df[FEATURE_COLS].median()).values
        K_pred = models[soil_type].predict(X).clip(0.2, 2.5)
    else:
        K_pred = np.ones(len(depth))
    # Conformal interval
    qhats  = stats.get(soil_type, {}).get("qhats", {})
    qhat   = float(qhats.get(str(coverage), qhats.get("0.9", 0.30)))
    # Fallback mean K
    mean_k = float(stats.get("site_mean_k", {}).get(
        f"{soil_type}_{site}", stats.get("global_mean_k", 1.0)))
    K_applied     = K_pred if recommendation == "ML" else np.full_like(K_pred, mean_k)
    srd_corrected = K_applied * srd_sim
    srd_lower     = np.clip((K_applied - qhat) * srd_sim, 0, None)
    srd_upper     = (K_applied + qhat) * srd_sim
    return {
        "depth": depth, "srd_sim": srd_sim,
        "srd_corrected": srd_corrected,
        "srd_lower": srd_lower, "srd_upper": srd_upper,
        "K_pred": K_applied, "K_std": k_std, "qhat": qhat,
        "recommendation": recommendation,
        "mean_k_correction": mean_k,
        "soil_type": soil_type, "site": site, "coverage": coverage
    }

def read_cpt_mo(f):
    df = pd.read_csv(f, sep=r"\s+", header=None, engine="python")
    if df.shape[1] < 5:
        raise ValueError("MO CPT needs ≥5 columns (no header). Col 0=depth, 2=qt, 4=fs")
    return pd.DataFrame({
        "Depth":  pd.to_numeric(df.iloc[:,0], errors="coerce"),
        "qt_MPa": pd.to_numeric(df.iloc[:,2], errors="coerce"),
        "fs_MPa": pd.to_numeric(df.iloc[:,4], errors="coerce"),
    }).dropna()

def read_cpt_r(f):
    df = pd.read_csv(f, sep=r"\s+", engine="python")
    df.columns = df.columns.str.strip().str.lower()
    rename = {}
    for c in df.columns:
        if "depth" in c: rename[c] = "Depth"
        elif c == "qt":  rename[c] = "qt_MPa"
        elif c == "fs":  rename[c] = "fs_MPa"
    return df.rename(columns=rename)[["Depth","qt_MPa","fs_MPa"]].apply(
        pd.to_numeric, errors="coerce").dropna()
