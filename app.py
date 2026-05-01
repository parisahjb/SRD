"""
Pile Driving SRD Correction Tool
"""
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pickle, json, os, io, sys
from pathlib import Path

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SRD Correction Tool",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Find models — check every possible location ───────────────────────────────
def find_file(filename):
    """Search for a file in all likely locations."""
    search_dirs = [
        Path(__file__).parent,                    # same folder as app.py
        Path(__file__).parent / "models",         # models/ subfolder
        Path(os.getcwd()),                        # working directory
        Path(os.getcwd()) / "models",             # working dir / models
        Path("/mount/src/srd"),                   # Streamlit Cloud root
        Path("/mount/src/srd") / "models",        # Streamlit Cloud models/
    ]
    for d in search_dirs:
        f = d / filename
        if f.exists() and f.stat().st_size > 200:
            return str(f)
    return None

@st.cache_resource
def load_all_models():
    models, stats = {}, {}
    for soil in ["clay", "sand"]:
        path = find_file(f"{soil}_model.pkl")
        if path:
            with open(path, "rb") as f:
                models[soil] = pickle.load(f)
    path = find_file("training_stats.json")
    if path:
        with open(path) as f:
            stats = json.load(f)
    return models, stats

models, stats = load_all_models()
model_loaded  = len(models) > 0

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600;700&display=swap');
html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
.title-block {
    background: linear-gradient(135deg, #0A1628 0%, #1565C0 100%);
    padding: 2rem 2.5rem; border-radius: 12px; margin-bottom: 1.5rem; color: white;
}
.title-block h1 {
    font-family: 'IBM Plex Mono', monospace; font-size: 1.8rem;
    font-weight: 600; margin: 0 0 0.3rem 0;
}
.title-block p { font-size: 0.95rem; opacity: 0.8; margin: 0; }
.metric-card {
    background: white; border: 1px solid #D0D7E8; border-radius: 10px;
    padding: 1.2rem 1.5rem; text-align: center;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
}
.metric-card .value {
    font-family: 'IBM Plex Mono', monospace; font-size: 1.5rem;
    font-weight: 600; color: #1565C0; line-height: 1;
}
.metric-card .label {
    font-size: 0.78rem; color: #666; margin-top: 0.3rem;
    text-transform: uppercase; letter-spacing: 0.5px;
}
.rec-ml {
    background: #E8F5E9; border: 2px solid #2E7D32; border-radius: 8px;
    padding: 1rem 1.5rem; color: #1B5E20; font-weight: 600;
}
.rec-mean {
    background: #FFF8E1; border: 2px solid #F9A825; border-radius: 8px;
    padding: 1rem 1.5rem; color: #795548; font-weight: 600;
}
.info-box {
    background: #E3F2FD; border-left: 4px solid #1565C0;
    padding: 0.8rem 1rem; border-radius: 0 8px 8px 0;
    font-size: 0.88rem; color: #1A237E; margin: 0.5rem 0;
}
.stButton > button {
    background: linear-gradient(135deg, #1565C0, #0D47A1); color: white;
    border: none; border-radius: 8px; padding: 0.6rem 2rem;
    font-weight: 600; font-size: 0.95rem; width: 100%;
}
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="title-block">
    <h1>🌊 Pile Driving SRD Correction Tool</h1>
    <p>ML-based correction of offshore pile driving simulations &nbsp;·&nbsp;
       Merkur (DE) + Rentel (BE) &nbsp;·&nbsp;
       Florida Polytechnic University × Aalborg University × COWI A/S</p>
</div>
""", unsafe_allow_html=True)

# ── Model check ───────────────────────────────────────────────────────────────
if not model_loaded:
    st.error("⚠️ Models not loaded yet.")
    with st.expander("🔍 Debug — click to see file search results"):
        for fname in ["clay_model.pkl", "sand_model.pkl", "training_stats.json"]:
            result = find_file(fname)
            if result:
                st.success(f"✅ {fname} found at: {result}")
            else:
                st.error(f"❌ {fname} NOT found")
        st.write("**app.py location:**", str(Path(__file__).resolve()))
        st.write("**cwd:**", os.getcwd())
        st.write("**cwd contents:**", sorted(os.listdir(os.getcwd())))
        sc_path = Path("/mount/src/srd")
        if sc_path.exists():
            st.write("**/mount/src/srd contents:**",
                     sorted(os.listdir(str(sc_path))))
    st.stop()

# ── Feature engineering ───────────────────────────────────────────────────────
PA_KPA = 100.0
SANDY  = {"sand", "sand/silt"}
FEATURE_COLS = [
    "Depth_actual", "SRD_sim", "CPT_col3", "CPT_col5",
    "fs_cumulative", "qc_cumulative", "friction_ratio",
    "ISBT", "tip_area", "max_diameter", "pct_sand"
]

def compute_isbt(qt, fs):
    qt_n  = np.where(qt > 0, qt / PA_KPA, np.nan)
    rf    = np.where((qt > 0) & (fs > 0), (fs / qt) * 100, np.nan)
    return np.nan_to_num(
        np.sqrt((3.47 - np.log10(qt_n))**2 + (np.log10(rf) + 1.22)**2), nan=2.0)

def make_features(depth, srd_sim, qt, fs, soil_labels,
                  tip_area=28.27, max_diameter=6.0):
    df = pd.DataFrame({
        "Depth_actual": depth, "SRD_sim": srd_sim,
        "CPT_col5": qt, "CPT_col3": fs, "SoilType": soil_labels
    }).sort_values("Depth_actual").reset_index(drop=True)
    df["friction_ratio"] = (df["CPT_col3"] / df["CPT_col5"].replace(0, np.nan)).clip(0, 0.1)
    df["ISBT"]           = compute_isbt(df["CPT_col5"].values, df["CPT_col3"].values)
    df["tip_area"]       = tip_area
    df["max_diameter"]   = max_diameter
    dz = np.diff(df["Depth_actual"].values, prepend=df["Depth_actual"].values[0])
    dz[0] = 0.02
    df["fs_cumulative"] = np.cumsum(df["CPT_col3"].fillna(0).values * np.abs(dz))
    df["qc_cumulative"] = np.cumsum(df["CPT_col5"].fillna(0).values * np.abs(dz))
    sandy = df["SoilType"].str.lower().isin(SANDY).astype(float).values
    cum_t = np.cumsum(np.abs(dz))
    df["pct_sand"] = np.where(cum_t > 0, np.cumsum(sandy * np.abs(dz)) / cum_t, 0.0)
    return df

def run_prediction(depth, srd_sim, qt, fs, soil_type,
                   soil_labels, tip_area, max_diameter, coverage, site):
    df     = make_features(depth, srd_sim, qt, fs, soil_labels, tip_area, max_diameter)
    k_std  = float(np.std(srd_sim / np.clip(srd_sim.mean(), 1, None)))
    rec    = "ML" if k_std >= 0.15 else "MEAN"
    if soil_type in models:
        X      = df[FEATURE_COLS].fillna(df[FEATURE_COLS].median()).values
        K_pred = models[soil_type].predict(X).clip(0.2, 2.5)
    else:
        K_pred = np.ones(len(depth))
    qhats  = stats.get(soil_type, {}).get("qhats", {})
    qhat   = float(qhats.get(str(coverage), qhats.get("0.9", 0.30)))
    mean_k = float(stats.get("site_mean_k", {}).get(
        f"{soil_type}_{site}", stats.get("global_mean_k", 1.0)))
    K_app  = K_pred if rec == "ML" else np.full_like(K_pred, mean_k)
    return {
        "depth": depth, "srd_sim": srd_sim,
        "srd_corrected": K_app * srd_sim,
        "srd_lower":     np.clip((K_app - qhat) * srd_sim, 0, None),
        "srd_upper":     (K_app + qhat) * srd_sim,
        "K_pred": K_app, "K_std": k_std, "qhat": qhat,
        "recommendation": rec, "mean_k": mean_k,
        "soil_type": soil_type, "site": site, "coverage": coverage
    }

# ── CPT readers ───────────────────────────────────────────────────────────────
def read_cpt_mo(f):
    df = pd.read_csv(f, sep=r"\s+", header=None, engine="python")
    if df.shape[1] < 5:
        raise ValueError("MO CPT needs ≥5 columns. Col 0=depth, 2=qt, 4=fs")
    return pd.DataFrame({
        "Depth":  pd.to_numeric(df.iloc[:,0], errors="coerce"),
        "qt_MPa": pd.to_numeric(df.iloc[:,2], errors="coerce"),
        "fs_MPa": pd.to_numeric(df.iloc[:,4], errors="coerce"),
    }).dropna()

def read_cpt_r(f):
    df = pd.read_csv(f, sep=r"\s+", engine="python")
    df.columns = df.columns.str.strip().str.lower()
    rename = {c: "Depth" for c in df.columns if "depth" in c}
    rename.update({c: "qt_MPa" for c in df.columns if c == "qt"})
    rename.update({c: "fs_MPa" for c in df.columns if c == "fs"})
    return df.rename(columns=rename)[["Depth","qt_MPa","fs_MPa"]].apply(
        pd.to_numeric, errors="coerce").dropna()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    site_label = st.selectbox("Wind farm site",
                              ["R — Rentel (Belgium)", "MO — Merkur (Germany)"])
    site_key   = "R" if site_label.startswith("R") else "MO"
    soil_type  = st.selectbox("Dominant soil type", ["clay", "sand"])
    cpt_format = st.radio("CPT file format", [
        "R-format (.txt with header)",
        "MO-format (.dat no header)",
        "Generic CSV (Depth, qt, fs)"
    ])
    diameter  = st.number_input("Max pile diameter [m]", 4.0, 12.0, 6.0, 0.1)
    tip_area  = np.pi / 4 * diameter**2
    st.caption(f"Tip area = {tip_area:.2f} m²")
    coverage  = st.select_slider(
        "Prediction interval coverage",
        options=[0.70, 0.80, 0.90, 0.95], value=0.90,
        format_func=lambda x: f"{int(x*100)}%")
    st.markdown("---")
    st.markdown("### 📊 Model Info")
    s = stats.get(soil_type, {})
    st.metric("Training locations", s.get("n_locations", "—"))
    st.metric("Training rows",      s.get("n_rows", "—"))
    st.metric("Mean K (training)",  f"{s.get('mean_K', 0):.3f}")
    qhats_disp = s.get("qhats", {})
    qhat_disp  = qhats_disp.get(str(coverage), qhats_disp.get("0.9", 0.3))
    st.metric(f"qhat ({int(coverage*100)}% PI)", f"±{float(qhat_disp):.3f} K")
    st.caption("K_std ≥ 0.15 → ML correction\nK_std < 0.15 → Mean correction")

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["📁 Upload & Predict", "📈 Results", "ℹ️ About"])

# ── TAB 1: UPLOAD ─────────────────────────────────────────────────────────────
with tab1:
    col_cpt, col_sim = st.columns(2)

    with col_cpt:
        st.markdown("#### 📂 CPT Data")
        st.markdown('<div class="info-box">Upload CPT file. '
                    'Columns: Depth [m], qt [MPa], fs [MPa]</div>',
                    unsafe_allow_html=True)
        cpt_file       = st.file_uploader("Upload CPT", type=["txt","dat","csv"])
        use_sample_cpt = st.checkbox("Use sample data (R3, Rentel clay)")
        cpt_df = None
        if use_sample_cpt:
            p = find_file("sample_CPT_R3.csv")
            if p:
                cpt_df = pd.read_csv(p)
                st.success(f"✓ Sample CPT: {len(cpt_df)} rows")
            else:
                st.warning("Sample CPT not found — upload a file instead.")
        elif cpt_file:
            try:
                if "R-format"  in cpt_format: cpt_df = read_cpt_r(cpt_file)
                elif "MO-format" in cpt_format: cpt_df = read_cpt_mo(cpt_file)
                else:
                    cpt_df = pd.read_csv(cpt_file)
                    cpt_df.columns = [c.strip() for c in cpt_df.columns]
                    rn = {}
                    for c in cpt_df.columns:
                        cl = c.lower()
                        if "depth" in cl: rn[c] = "Depth"
                        elif "qt"  in cl: rn[c] = "qt_MPa"
                        elif "fs"  in cl: rn[c] = "fs_MPa"
                    cpt_df = cpt_df.rename(columns=rn)
                st.success(f"✓ CPT: {len(cpt_df)} rows, "
                           f"depth {cpt_df['Depth'].min():.1f}–{cpt_df['Depth'].max():.1f} m")
            except Exception as e:
                st.error(f"CPT read error: {e}")
        if cpt_df is not None:
            with st.expander("Preview CPT"):
                st.dataframe(cpt_df.head(8), use_container_width=True)

    with col_sim:
        st.markdown("#### 📊 Simulation Data")
        st.markdown('<div class="info-box">Upload simulation CSV. '
                    'Columns: Depth_actual [m], SRD_sim [kN]</div>',
                    unsafe_allow_html=True)
        sim_file       = st.file_uploader("Upload Simulation CSV", type=["csv"])
        use_sample_sim = st.checkbox("Use sample simulation (R3)")
        sim_df = None
        if use_sample_sim:
            p = find_file("sample_SIM_R3.csv")
            if p:
                sim_df = pd.read_csv(p)
                st.success(f"✓ Sample sim: {len(sim_df)} rows")
            else:
                st.warning("Sample simulation not found — upload a file instead.")
        elif sim_file:
            try:
                sim_df = pd.read_csv(sim_file)
                sim_df.columns = [c.strip() for c in sim_df.columns]
                st.success(f"✓ Simulation: {len(sim_df)} rows")
            except Exception as e:
                st.error(f"Simulation read error: {e}")
        if sim_df is not None:
            with st.expander("Preview simulation"):
                st.dataframe(sim_df.head(8), use_container_width=True)

    st.markdown("---")
    ready   = cpt_df is not None and sim_df is not None
    run_btn = st.button("🚀 Run SRD Correction", disabled=not ready)
    if not ready:
        st.caption("Upload or select sample data for both CPT and simulation to run.")

    if run_btn and ready:
        with st.spinner("Computing features and running ML correction…"):
            try:
                sim_df = sim_df.sort_values("Depth_actual").reset_index(drop=True)
                cpt_df = cpt_df.sort_values("Depth").reset_index(drop=True)
                sim_df["_dk"] = sim_df["Depth_actual"].round(1)
                cpt_df["_dk"] = cpt_df["Depth"].round(1)
                cpt_agg = cpt_df.groupby("_dk", as_index=False).agg(
                    qt_MPa=("qt_MPa","mean"), fs_MPa=("fs_MPa","mean"))
                merged = sim_df.merge(cpt_agg, on="_dk", how="left")
                merged["qt_MPa"] = merged["qt_MPa"].interpolate().bfill().fillna(1.0)
                merged["fs_MPa"] = merged["fs_MPa"].interpolate().bfill().fillna(0.05)
                soil_labels = (merged["SoilGroup"].fillna(soil_type).tolist()
                               if "SoilGroup" in merged.columns
                               else [soil_type]*len(merged))
                result = run_prediction(
                    depth        = merged["Depth_actual"].values,
                    srd_sim      = merged["SRD_sim"].values,
                    qt           = merged["qt_MPa"].values,
                    fs           = merged["fs_MPa"].values,
                    soil_type    = soil_type,
                    soil_labels  = soil_labels,
                    tip_area     = tip_area,
                    max_diameter = diameter,
                    coverage     = coverage,
                    site         = site_key
                )
                st.session_state["result"] = result
                st.success("✓ Done — see the Results tab")
            except Exception as e:
                st.error(f"Prediction error: {e}")
                import traceback
                st.code(traceback.format_exc())

# ── TAB 2: RESULTS ────────────────────────────────────────────────────────────
with tab2:
    if "result" not in st.session_state:
        st.info("Run a prediction in the Upload tab first.")
    else:
        r        = st.session_state["result"]
        depth    = r["depth"];    srd_s = r["srd_sim"]
        srd_c    = r["srd_corrected"]
        srd_lo   = r["srd_lower"]; srd_hi = r["srd_upper"]
        K_pred   = r["K_pred"];   k_std  = r["K_std"]
        rec      = r["recommendation"]; qhat = r["qhat"]
        mean_k   = r["mean_k"]
        cov_pct  = int(r["coverage"] * 100)

        # Metrics row
        c1,c2,c3,c4,c5 = st.columns(5)
        for col, val, lbl in [
            (c1, str(len(depth)),          "Depth points"),
            (c2, f"{np.mean(K_pred):.3f}", "Mean K predicted"),
            (c3, f"±{qhat:.3f}",           f"qhat ({cov_pct}% PI)"),
            (c4, f"{k_std:.3f}",           "K_std (screening)"),
            (c5, rec,                      "Recommendation"),
        ]:
            clr = "#2E7D32" if (lbl=="Recommendation" and val=="ML") else \
                  "#C8860A" if (lbl=="Recommendation") else "#1565C0"
            col.markdown(f"""<div class="metric-card">
                <div class="value" style="color:{clr}">{val}</div>
                <div class="label">{lbl}</div></div>""",
                unsafe_allow_html=True)

        st.markdown("&nbsp;")
        if rec == "ML":
            st.markdown(f'<div class="rec-ml">✅ ML CORRECTION APPLIED &nbsp;|&nbsp; '
                        f'K_std={k_std:.3f} ≥ 0.15 → K varies with depth</div>',
                        unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="rec-mean">⚡ MEAN CORRECTION APPLIED &nbsp;|&nbsp; '
                        f'K_std={k_std:.3f} < 0.15 → site mean K={mean_k:.3f}</div>',
                        unsafe_allow_html=True)
        st.markdown("&nbsp;")

        # Plots
        fig = plt.figure(figsize=(16, 7))
        fig.patch.set_facecolor("#F8FAFF")
        gs  = gridspec.GridSpec(1, 3, figure=fig, wspace=0.35)
        CB, CG, CGR = "#1565C0", "#2E7D32", "#78909C"
        ck = CG if rec=="ML" else "#F9A825"
        corr_pct = (srd_c - srd_s) / srd_s * 100

        ax = fig.add_subplot(gs[0])
        ax.fill_betweenx(depth, srd_lo/1000, srd_hi/1000, alpha=0.2, color=CB)
        ax.plot(srd_c/1000, depth, "-",  color=CB,  lw=2.5, label="ML-corrected")
        ax.plot(srd_s/1000, depth, "--", color=CGR, lw=1.5, alpha=0.8, label="Simulated")
        ax.invert_yaxis(); ax.set_facecolor("white")
        ax.set_xlabel("SRD [MN]"); ax.set_ylabel("Depth [m]")
        ax.set_title("Corrected SRD Profile\n(shaded = prediction interval)",
                     fontweight="bold"); ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

        ax = fig.add_subplot(gs[1])
        ax.plot(K_pred, depth, "-", color=ck, lw=2.5)
        ax.fill_betweenx(depth, K_pred-qhat, K_pred+qhat, alpha=0.2, color=ck)
        ax.axvline(1.0,    color="red",     lw=1.5, ls="--", label="K=1 (perfect)")
        ax.axvline(mean_k, color="#C8860A", lw=1.5, ls=":",  label=f"Mean K={mean_k:.2f}")
        ax.invert_yaxis(); ax.set_facecolor("white")
        ax.set_xlabel("K [-]"); ax.set_ylabel("Depth [m]")
        ax.set_title(f"K Correction Factor\n({rec} applied)",
                     fontweight="bold", color=ck)
        ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

        ax = fig.add_subplot(gs[2])
        clrs = ["#2E7D32" if v < 0 else "#C62828" for v in corr_pct]
        ax.barh(depth, corr_pct, height=0.15, color=clrs, alpha=0.8)
        ax.axvline(0, color="black", lw=1.5)
        ax.axvline(float(np.mean(corr_pct)), color=CB, lw=1.5, ls="--",
                   label=f"Mean {np.mean(corr_pct):+.1f}%")
        ax.invert_yaxis(); ax.set_facecolor("white")
        ax.set_xlabel("SRD Change [%]"); ax.set_ylabel("Depth [m]")
        ax.set_title("Correction per Depth\n(green=reduction, red=increase)",
                     fontweight="bold")
        ax.legend(fontsize=9); ax.grid(True, alpha=0.3, axis="x")
        fig.suptitle(
            f"SRD Correction  |  Site: {r['site']}  |  Soil: {r['soil_type'].capitalize()}"
            f"  |  n={len(depth)}  |  {cov_pct}% PI",
            fontweight="bold", color="#0A1628", y=1.01)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        # Downloads
        st.markdown("---")
        st.markdown("#### 📥 Download")
        dl1, dl2 = st.columns(2)
        with dl1:
            res_df = pd.DataFrame({
                "Depth_m":             depth,
                "SRD_sim_kN":          srd_s,
                "SRD_corrected_kN":    srd_c,
                "SRD_lower_kN":        srd_lo,
                "SRD_upper_kN":        srd_hi,
                "K_factor":            K_pred,
                "SRD_change_pct":      corr_pct,
            })
            st.download_button("⬇️ Corrected SRD (CSV)",
                data=res_df.to_csv(index=False).encode(),
                file_name=f"srd_corrected_{r['site']}_{r['soil_type']}.csv",
                mime="text/csv")
        with dl2:
            buf = io.BytesIO()
            fig.savefig(buf, dpi=150, bbox_inches="tight")
            buf.seek(0)
            st.download_button("⬇️ Plot (PNG)",
                data=buf,
                file_name=f"srd_correction_{r['site']}_{r['soil_type']}.png",
                mime="image/png")

        # Summary
        st.markdown("---")
        st.markdown("#### 📋 Summary")
        c1, c2, c3 = st.columns(3)
        c1.markdown(f"**Simulation**\n- Mean: {np.mean(srd_s)/1000:.1f} MN\n"
                    f"- Min: {np.min(srd_s)/1000:.1f} MN\n"
                    f"- Max: {np.max(srd_s)/1000:.1f} MN")
        c2.markdown(f"**ML-Corrected**\n- Mean: {np.mean(srd_c)/1000:.1f} MN\n"
                    f"- Min: {np.min(srd_c)/1000:.1f} MN\n"
                    f"- Max: {np.max(srd_c)/1000:.1f} MN")
        c3.markdown(f"**{cov_pct}% Prediction Interval**\n"
                    f"- Lower mean: {np.mean(srd_lo)/1000:.1f} MN\n"
                    f"- Upper mean: {np.mean(srd_hi)/1000:.1f} MN\n"
                    f"- Mean width: {np.mean(srd_hi-srd_lo)/1000:.1f} MN")

# ── TAB 3: ABOUT ──────────────────────────────────────────────────────────────
with tab3:
    col_a, col_b = st.columns([3, 2])
    with col_a:
        st.markdown("""
        ### About This Tool
        ML-based correction of pre-installation pile driving simulations for offshore
        wind turbine monopiles. Developed as part of a research collaboration between
        **Florida Polytechnic University**, **Aalborg University**, and **COWI A/S**.

        #### Key results (clay, leave-one-location-out)
        - **50% RMSE reduction** (46.4 → 23.2 MN)
        - **99% bias reduction** (+30.9 → -0.2 MN)
        - **34/38 locations improved** vs raw simulation
        - **90% conformal intervals** with exact empirical coverage

        #### Two-stage decision rule
        - **K_std ≥ 0.15** → ML correction (depth-varying K)
        - **K_std < 0.15** → Site mean K correction (constant)

        #### Training data
        - 51 locations, 5,379 depth-matched rows
        - Rentel (Belgium, R): clay-dominated
        - Merkur (Germany, MO): sand-dominated
        """)
    with col_b:
        st.markdown("""
        ### File Format Guide

        **CPT — R format (.txt)**
        ```
        Depth  qc    fs     qt
        0.02   1.42  0.000  1.419
        ```

        **CPT — MO format (.dat)**
        ```
        0.000  0.164  0.164  0.000  0.000
        0.020  0.325  0.325  0.000  0.000
        ```
        Col 0=depth, 2=qt, 4=fs, no header.

        **Simulation CSV**
        ```
        Depth_actual,SRD_sim,SoilGroup
        10.5,68450,clay
        ```
        SRD in kN. SoilGroup optional.
        """)
    st.caption(
        "⚠️ For research and planning purposes only. "
        "All predictions should be reviewed by a qualified geotechnical engineer.")
