"""
Direct Blowcount Prediction — prototype interface
Physics-guided ML for offshore monopile installation.

Companion tool to the manuscript. Loads the saved model bundle
(blowcount_model.joblib) and reproduces the paper's feature pipeline so
that predictions match the reported results exactly.

Run:  streamlit run blowcount_app.py
Requires in the same folder:  blowcount_model.joblib

The model bundle is expected to contain:
    model      : HistGradientBoostingRegressor (point prediction)
    q_lo, q_hi : quantile models (0.10, 0.90) for the prediction band
    features   : list of 11 feature names, in order
    median_impute : dict of median values for gap-filling
"""

import os
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import joblib

# -----------------------------------------------------------------------------
# CONSTANTS — must match the notebook pipeline exactly
# -----------------------------------------------------------------------------
PA_KPA = 100.0
GAMMA_W = 9.81
GAMMA_MIN, GAMMA_MAX = 14.0, 22.0
QT_CEILING = 110.0
WINDOW = 0.25            # 25 cm window above pile tip
IC_SAND_MAX = 2.60
K0_NC = 0.45
LAMBDA_10 = 0.045
MODEL_PATH = "blowcount_model.joblib"

# empirically measured on the blind holdout — the honest number
BAND_NOMINAL = 80
BAND_EMPIRICAL = 61

st.set_page_config(page_title="Blowcount Prediction", layout="wide")


# -----------------------------------------------------------------------------
# FEATURE PIPELINE  (mirrors the notebook — do not edit without re-verifying)
# -----------------------------------------------------------------------------
def isbt_correct(qt_mpa, fs_mpa):
    qt_kpa = np.asarray(qt_mpa, float) * 1000.0
    qtn = np.where(qt_kpa > 0, qt_kpa / PA_KPA, np.nan)
    rf = np.clip(np.where((qt_mpa > 0) & (fs_mpa > 0),
                          (fs_mpa / qt_mpa) * 100, np.nan), 0.01, 10.0)
    return np.sqrt((3.47 - np.log10(qtn)) ** 2 + (np.log10(rf) + 1.22) ** 2)


def _strict_increasing_mask(z):
    keep = np.zeros(len(z), bool)
    last = -np.inf
    for i, v in enumerate(z):
        if v > last:
            keep[i] = True
            last = v
    return keep


def clean_cpt(raw):
    """Restart removal (depth reversal), monotonic depth, fs floor, qt ceiling."""
    q = raw.reset_index(drop=True).copy()
    z = q["Depth"].values
    rev = np.zeros(len(q), bool)
    if len(q) > 1:
        rev[1:] = np.diff(z) <= 0
    bad = np.zeros(len(q), bool)
    for i in np.where(rev)[0]:
        bad |= (z >= z[i]) & (z <= z[i] + 0.10)
    k = q[~bad].copy()
    m = _strict_increasing_mask(k["Depth"].values)
    k = k[m].copy()
    k["fs_MPa"] = k["fs_raw"].fillna(0.0).clip(lower=0.001)
    k["qt_MPa"] = k["qt_raw"].clip(upper=QT_CEILING)
    return k.reset_index(drop=True)


def add_stress_and_norm(c, n_iter=12):
    d = c.sort_values("Depth").reset_index(drop=True).copy()
    z = d["Depth"].values
    qt = d["qt_MPa"].values * 1000.0
    fs = d["fs_MPa"].values * 1000.0
    Rf = np.clip(100.0 * fs / np.maximum(qt, 1e-6), 0.1, 10.0)
    gam = GAMMA_W * (0.27 * np.log10(Rf)
                     + 0.36 * np.log10(np.maximum(qt, 1e-6) / PA_KPA) + 1.236)
    gam = np.clip(gam, GAMMA_MIN, GAMMA_MAX)
    dz = np.diff(z, prepend=z[0])
    dz[0] = z[0]
    sv0 = np.cumsum(gam * dz)
    u0 = GAMMA_W * z
    sve = np.maximum(sv0 - u0, 1.0)
    qnet = np.maximum(qt - sv0, 1.0)
    Fr = np.clip(100.0 * fs / qnet, 0.1, 10.0)
    n = np.ones_like(z)
    for _ in range(n_iter):
        Qtn = np.maximum((qnet / PA_KPA) * (PA_KPA / sve) ** n, 1e-3)
        Ic = np.sqrt((3.47 - np.log10(Qtn)) ** 2 + (np.log10(Fr) + 1.22) ** 2)
        n_new = np.clip(0.381 * Ic + 0.05 * (sve / PA_KPA) - 0.15, 0.0, 1.0)
        if np.nanmax(np.abs(n_new - n)) < 1e-4:
            n = n_new
            break
        n = n_new
    Qtn = np.maximum((qnet / PA_KPA) * (PA_KPA / sve) ** n, 1e-3)
    Ic = np.sqrt((3.47 - np.log10(Qtn)) ** 2 + (np.log10(Fr) + 1.22) ** 2)
    d["sigma_v0_eff"] = sve
    d["Qtn"] = Qtn
    d["Fr_pct"] = Fr
    d["Ic"] = Ic
    return d


def window_mean(z_pile, z_cpt, v_cpt, w=WINDOW):
    out = np.full(len(z_pile), np.nan)
    for i, zp in enumerate(z_pile):
        m = (z_cpt <= zp) & (z_cpt >= zp - w)
        if m.any():
            out[i] = np.nanmean(v_cpt[m])
        else:
            j = np.argmin(np.abs(z_cpt - zp))
            out[i] = v_cpt[j]
    return out


def build_features(cpt_df, pile_depths, srd_sim, hammer_kj, feature_names):
    """cpt_df: Depth, qt_raw, fs_raw. Returns feature frame in model order."""
    clean = clean_cpt(cpt_df)
    norm = add_stress_and_norm(clean)
    zc = norm["Depth"].values
    zp = np.asarray(pile_depths, float)

    qt_w = window_mean(zp, zc, norm["qt_MPa"].values)
    fs_w = window_mean(zp, zc, norm["fs_MPa"].values)
    Qtn_w = window_mean(zp, zc, norm["Qtn"].values)
    Ic_w = window_mean(zp, zc, norm["Ic"].values)
    Fr_w = window_mean(zp, zc, norm["Fr_pct"].values)

    ic_row = np.interp(zp, zc, norm["Ic"].values)
    sand = (ic_row < IC_SAND_MAX).astype(float)
    dz = np.diff(zp, prepend=zp[0])
    dz[0] = WINDOW
    ct = np.cumsum(np.abs(dz))
    pct_sand = np.where(ct > 0, np.cumsum(sand * np.abs(dz)) / ct, 0.0)

    F = pd.DataFrame({
        "Depth_actual": zp,
        "SRD_sim": np.full(len(zp), srd_sim) if np.isscalar(srd_sim) else srd_sim,
        "nominal_energy_kJ": hammer_kj,
        "qt_w25": qt_w, "fs_w25": fs_w,
        "friction_ratio": np.clip(fs_w / np.maximum(qt_w, 1e-6), 0, 0.1),
        "ISBT": isbt_correct(qt_w, fs_w),
        "pct_sand_lab": pct_sand,
        "Qtn_w25": Qtn_w, "Ic_w25": Ic_w, "Fr_w25": Fr_w,
    })
    return F[feature_names], norm


# -----------------------------------------------------------------------------
# SYNTHETIC EXAMPLE  (sand with a clay interval, DanTysk-like)
# -----------------------------------------------------------------------------
def example_data():
    rng = np.random.default_rng(7)
    z = np.round(np.arange(0.02, 42.0, 0.02), 2)
    qt = np.zeros_like(z)
    for i, d in enumerate(z):
        if d < 6:
            base = 8 + 3.0 * d
        elif d < 18:                       # clay interval
            base = 1.3 + 0.05 * (d - 6)
        else:                              # dense sand
            base = 20 + 1.1 * (d - 18)
        qt[i] = max(0.2, base + rng.normal(0, base * 0.08))
    fs = np.clip(qt * np.where((z >= 6) & (z < 18), 0.028, 0.009), 0.001, None)
    cpt = pd.DataFrame({"Depth": z, "qt_raw": qt, "fs_raw": fs})
    pile = np.round(np.arange(3.0, 40.75, 0.25), 2)
    return cpt, pile


# -----------------------------------------------------------------------------
# LOAD MODEL
# -----------------------------------------------------------------------------
@st.cache_resource
def load_bundle():
    if not os.path.exists(MODEL_PATH):
        return None, f"Model file '{MODEL_PATH}' not found in this folder."
    b = joblib.load(MODEL_PATH)
    need = {"model", "features"}
    if not need.issubset(b.keys()):
        return None, f"Bundle missing keys; found {list(b.keys())}."
    return b, None


# =============================================================================
# UI
# =============================================================================
st.title("Direct Blowcount Prediction")
st.caption("Physics-guided machine learning for offshore monopile installation "
           "— research prototype accompanying the manuscript.")

bundle, err = load_bundle()
if err:
    st.error(err)
    st.info("Place blowcount_model.joblib (saved from the analysis notebook) "
            "in the same folder as this app.")
    st.stop()

FEATURES = bundle["features"]
model = bundle["model"]
qlo = bundle.get("q_lo")
qhi = bundle.get("q_hi")

with st.sidebar:
    st.header("Input")
    src = st.radio("Data source",
                   ["Load worked example", "Upload CPT file"])
    st.divider()
    st.subheader("Pre-installation inputs")
    hammer = st.selectbox("Hammer", ["IHC IQIP S-3000 (3000 kJ)",
                                     "IHC IQIP S-4000 (4000 kJ)"])
    hammer_kj = 3000.0 if "3000" in hammer else 4000.0
    srd_mode = st.radio("Simulated SRD (SRD_sim)",
                        ["Constant value", "From uploaded column"])
    srd_const = st.number_input("SRD_sim [kN]", value=120000.0, step=10000.0,
                                disabled=(srd_mode != "Constant value"))
    target_depth = st.number_input("Target embedment [m]", value=40.0, step=1.0)

    st.divider()
    st.caption(f"Prediction band is nominally {BAND_NOMINAL}% but achieved "
               f"**{BAND_EMPIRICAL}%** empirical coverage on the blind test "
               f"site. Treat the band as indicative, not calibrated, on "
               f"unseen geologies.")

# ---- assemble input ----------------------------------------------------------
cpt_df, pile_depths, srd_vec = None, None, None
if src == "Load worked example":
    cpt_df, pile_depths = example_data()
    srd_vec = srd_const
    st.success("Loaded a synthetic sand profile containing a clay interval "
               "(6–18 m), mimicking the blind test site.")
else:
    up = st.file_uploader("CPT file: whitespace/tab-delimited, columns "
                          "Depth, qt, fs (or Depth, qc, qc, fs, fs)",
                          type=["txt", "dat", "csv"])
    if up is not None:
        try:
            raw = pd.read_csv(up, sep=r"\s+", header=None, engine="python",
                              comment="#")
            if raw.shape[1] >= 5:
                cpt_df = pd.DataFrame({"Depth": raw.iloc[:, 0],
                                       "qt_raw": raw.iloc[:, 2],
                                       "fs_raw": raw.iloc[:, 4]})
            elif raw.shape[1] >= 3:
                cpt_df = pd.DataFrame({"Depth": raw.iloc[:, 0],
                                       "qt_raw": raw.iloc[:, 1],
                                       "fs_raw": raw.iloc[:, 2]})
            else:
                st.error("Need at least 3 columns (Depth, qt, fs).")
            if cpt_df is not None:
                cpt_df = cpt_df.apply(pd.to_numeric, errors="coerce") \
                               .dropna(subset=["Depth", "qt_raw"]) \
                               .query("Depth >= 0").reset_index(drop=True)
                pile_depths = np.round(
                    np.arange(3.0, min(target_depth, cpt_df.Depth.max()) + 0.25,
                              0.25), 2)
                srd_vec = srd_const
        except Exception as e:
            st.error(f"Could not parse file: {e}")

if cpt_df is None:
    st.info("Choose the worked example or upload a CPT file to begin.")
    st.stop()

# ---- predict -----------------------------------------------------------------
Feat, norm = build_features(cpt_df, pile_depths, srd_vec, hammer_kj, FEATURES)
med = bundle.get("median_impute", {})
Feat = Feat.fillna(pd.Series(med)) if med else Feat.fillna(Feat.median())

pred = model.predict(Feat.values)
pred = np.clip(pred, 0, None)
lo = qlo.predict(Feat.values) if qlo is not None else pred * 0.6
hi = qhi.predict(Feat.values) if qhi is not None else pred * 1.4
lo, hi = np.minimum(lo, hi), np.maximum(lo, hi)
lo = np.clip(lo, 0, None)

cum = np.cumsum(pred)
total_blows = cum[-1]

# ---- headline ----------------------------------------------------------------
c1, c2, c3 = st.columns(3)
c1.metric("Predicted total blows to target", f"{total_blows:,.0f}")
c2.metric("Target embedment", f"{pile_depths[-1]:.1f} m")
c3.metric("Prediction band (empirical)", f"{BAND_EMPIRICAL}%",
          help="Fraction of true values expected within the band on unseen sites.")

# ---- plots -------------------------------------------------------------------
plt.rcParams.update({"font.size": 10, "axes.linewidth": 1.2})
BLUE, BLACK, CLAY = "#2b6cb0", "#1a1a1a", "#e8dcc0"

colA, colB = st.columns(2)
with colA:
    fig, ax = plt.subplots(figsize=(4.6, 5.2))
    clay = norm[(norm.Ic > IC_SAND_MAX)]
    if len(clay):
        for _, seg in clay.groupby((clay.Depth.diff() > 0.5).cumsum()):
            ax.axhspan(seg.Depth.min(), seg.Depth.max(),
                       color=CLAY, alpha=0.6, zorder=0)
    ax.fill_betweenx(pile_depths, lo, hi, color=BLUE, alpha=0.15,
                     label=f"{BAND_NOMINAL}% band ({BAND_EMPIRICAL}% emp.)")
    ax.plot(pred, pile_depths, "-", color=BLUE, lw=2, label="Predicted")
    ax.invert_yaxis()
    ax.set_xlabel("Blowcount (blows / 25 cm)", fontweight="bold")
    ax.set_ylabel("Depth below seabed (m)", fontweight="bold")
    ax.set_xlim(left=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.set_title("Blowcount profile", fontweight="bold", loc="left")
    st.pyplot(fig)

with colB:
    fig2, ax2 = plt.subplots(figsize=(4.6, 5.2))
    ax2.plot(cum, pile_depths, "-", color=BLACK, lw=2.2)
    ax2.invert_yaxis()
    ax2.set_xlabel("Cumulative blows to depth", fontweight="bold")
    ax2.set_ylabel("Depth below seabed (m)", fontweight="bold")
    ax2.set_xlim(left=0)
    for s in ("top", "right"):
        ax2.spines[s].set_visible(False)
    ax2.annotate(f"{total_blows:,.0f} blows\nto target",
                 (cum[-1], pile_depths[-1]),
                 xytext=(cum[-1] * 0.45, pile_depths[-1] + 3),
                 fontsize=9, fontweight="bold", color=BLUE,
                 arrowprops=dict(arrowstyle="->", color=BLUE, lw=1.2))
    ax2.set_title("Cumulative blows", fontweight="bold", loc="left")
    st.pyplot(fig2)

# ---- table + download --------------------------------------------------------
out = pd.DataFrame({"Depth_m": pile_depths,
                    "Predicted_blowcount": np.round(pred, 1),
                    "Band_low": np.round(lo, 1),
                    "Band_high": np.round(hi, 1),
                    "Cumulative_blows": np.round(cum, 0)})
with st.expander("Prediction table"):
    st.dataframe(out, use_container_width=True, height=280)
st.download_button("Download predictions (CSV)",
                   out.to_csv(index=False).encode(),
                   "blowcount_predictions.csv", "text/csv")

st.caption("Prototype for research demonstration. Predictions on geologies "
           "unlike the training sites (Rentel, Merkur) may be unreliable; the "
           "prediction band is indicative only. Not validated for design use.")
