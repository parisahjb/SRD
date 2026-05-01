"""
Pile Driving SRD Correction Tool
ML-based correction of simulated Soil Resistance to Driving
Merkur + Rentel offshore wind farm dataset
"""
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
import io, json, os, sys

sys.path.insert(0, os.path.dirname(__file__))
from utils.predictor import predict_srd, read_cpt_mo, read_cpt_r, load_models

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SRD Correction Tool",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600;700&display=swap');

:root {
    --navy:   #0A1628;
    --blue:   #1565C0;
    --cyan:   #00ACC1;
    --sand:   #C8860A;
    --clay:   #1E6FA5;
    --green:  #2E7D32;
    --red:    #C62828;
    --light:  #F0F4FF;
    --border: #D0D7E8;
}

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
}

.main { background: #F8FAFF; }

.stApp header { background: transparent; }

.title-block {
    background: linear-gradient(135deg, #0A1628 0%, #1565C0 100%);
    padding: 2rem 2.5rem;
    border-radius: 12px;
    margin-bottom: 1.5rem;
    color: white;
}
.title-block h1 {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.8rem;
    font-weight: 600;
    margin: 0 0 0.3rem 0;
    letter-spacing: -0.5px;
}
.title-block p {
    font-size: 0.95rem;
    opacity: 0.8;
    margin: 0;
}

.metric-card {
    background: white;
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1.2rem 1.5rem;
    text-align: center;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
}
.metric-card .value {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 2rem;
    font-weight: 600;
    color: var(--blue);
    line-height: 1;
}
.metric-card .label {
    font-size: 0.78rem;
    color: #666;
    margin-top: 0.3rem;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

.rec-ml {
    background: #E8F5E9;
    border: 2px solid #2E7D32;
    border-radius: 8px;
    padding: 1rem 1.5rem;
    color: #1B5E20;
    font-weight: 600;
}
.rec-mean {
    background: #FFF8E1;
    border: 2px solid #F9A825;
    border-radius: 8px;
    padding: 1rem 1.5rem;
    color: #795548;
    font-weight: 600;
}

.info-box {
    background: #E3F2FD;
    border-left: 4px solid var(--blue);
    padding: 0.8rem 1rem;
    border-radius: 0 8px 8px 0;
    font-size: 0.88rem;
    color: #1A237E;
    margin: 0.5rem 0;
}
.warn-box {
    background: #FFF3E0;
    border-left: 4px solid #FF8F00;
    padding: 0.8rem 1rem;
    border-radius: 0 8px 8px 0;
    font-size: 0.88rem;
    color: #E65100;
    margin: 0.5rem 0;
}

.stButton > button {
    background: linear-gradient(135deg, #1565C0, #0D47A1);
    color: white;
    border: none;
    border-radius: 8px;
    padding: 0.6rem 2rem;
    font-family: 'IBM Plex Sans', sans-serif;
    font-weight: 600;
    font-size: 0.95rem;
    transition: all 0.2s;
    width: 100%;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #0D47A1, #0A2D6E);
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(21,101,192,0.35);
}

.step-header {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.75rem;
    color: var(--cyan);
    text-transform: uppercase;
    letter-spacing: 2px;
    margin-bottom: 0.3rem;
}
</style>
""", unsafe_allow_html=True)

# ── Load models once ──────────────────────────────────────────────────────────
@st.cache_resource
def get_models():
    return load_models()

models, stats = get_models()
model_loaded = len(models) > 0

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="title-block">
    <h1>🌊 Pile Driving SRD Correction Tool</h1>
    <p>ML-based correction of offshore pile driving simulations &nbsp;·&nbsp;
       Merkur (DE) + Rentel (BE) &nbsp;·&nbsp;
       Florida Polytechnic University × Aalborg University × COWI A/S</p>
</div>
""", unsafe_allow_html=True)

if not model_loaded:
    from utils.predictor import get_debug_info
    debug = get_debug_info()
    st.error("⚠️ No trained models found in: " + debug["model_dir"])
    with st.expander("🔍 Debug info (share this with developer)"):
        st.json(debug)
    st.markdown("""
    **To fix:** Make sure these files exist in your repo's `models/` folder:
    - `clay_model.pkl`
    - `sand_model.pkl`  
    - `training_stats.json`
    """)
    st.stop()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    st.markdown('<div class="step-header">Step 1 — Site & Soil</div>',
                unsafe_allow_html=True)
    site      = st.selectbox("Wind farm site", ["R (Rentel, Belgium)", "MO (Merkur, Germany)"])
    site_key  = "R" if site.startswith("R") else "MO"
    soil_type = st.selectbox("Dominant soil type", ["clay", "sand"])

    st.markdown("---")
    st.markdown('<div class="step-header">Step 2 — CPT Format</div>',
                unsafe_allow_html=True)
    cpt_format = st.radio("CPT file format",
                          ["R-format (.txt with header)",
                           "MO-format (.dat no header)",
                           "Generic CSV (Depth, qt, fs)"])

    st.markdown("---")
    st.markdown('<div class="step-header">Step 3 — Pile Geometry</div>',
                unsafe_allow_html=True)
    diameter    = st.number_input("Max pile diameter [m]", 4.0, 12.0, 6.0, 0.1)
    tip_area    = np.pi / 4 * diameter**2
    st.caption(f"Tip area = {tip_area:.2f} m²")

    st.markdown("---")
    st.markdown('<div class="step-header">Step 4 — Uncertainty</div>',
                unsafe_allow_html=True)
    coverage = st.select_slider(
        "Prediction interval coverage",
        options=[0.70, 0.80, 0.90, 0.95],
        value=0.90,
        format_func=lambda x: f"{int(x*100)}%"
    )

    st.markdown("---")
    st.markdown("### 📊 Model Info")
    soil_stats = stats.get(soil_type, {})
    st.metric("Training locations", soil_stats.get("n_locations", "—"))
    st.metric("Training rows",      soil_stats.get("n_rows", "—"))
    st.metric("Mean K (training)",  f"{soil_stats.get('mean_K', 0):.3f}")
    qhats = soil_stats.get("qhats", {})
    st.metric(f"qhat ({int(coverage*100)}% PI)",
              f"±{qhats.get(str(coverage), qhats.get('0.9', 0.3)):.3f} K")
    st.markdown(f"**K_std threshold:** 0.15")
    st.caption("K_std ≥ 0.15 → ML correction\nK_std < 0.15 → Mean correction")

# ── Main content ──────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    "📁 Upload & Predict",
    "📈 Results & Plots",
    "ℹ️ About & Methods"
])

# ──────────────────────────────────────────────────────────────────────────────
# TAB 1: UPLOAD
# ──────────────────────────────────────────────────────────────────────────────
with tab1:
    col_cpt, col_sim = st.columns(2)

    with col_cpt:
        st.markdown("#### 📂 CPT Data")
        st.markdown('<div class="info-box">Upload your CPT file. '
                    'Columns needed: Depth [m], qt [MPa], fs [MPa]</div>',
                    unsafe_allow_html=True)
        cpt_file = st.file_uploader("Upload CPT file",
                                    type=["txt","dat","csv"],
                                    key="cpt_upload")
        use_sample_cpt = st.checkbox("Use sample CPT (R3, Rentel clay)")

        cpt_df = None
        if use_sample_cpt:
            sample_path = os.path.join(os.path.dirname(__file__),
                                       "data", "sample_CPT_R3.csv")
            if os.path.exists(sample_path):
                cpt_df = pd.read_csv(sample_path)
                st.success(f"✓ Sample CPT loaded: {len(cpt_df)} rows")
        elif cpt_file is not None:
            try:
                if "R-format" in cpt_format:
                    cpt_df = read_cpt_r(cpt_file)
                elif "MO-format" in cpt_format:
                    cpt_df = read_cpt_mo(cpt_file)
                else:
                    cpt_df = pd.read_csv(cpt_file)
                    cpt_df.columns = [c.strip() for c in cpt_df.columns]
                    rename = {}
                    for c in cpt_df.columns:
                        cl = c.lower()
                        if "depth" in cl: rename[c] = "Depth"
                        elif "qt"  in cl: rename[c] = "qt_MPa"
                        elif "fs"  in cl: rename[c] = "fs_MPa"
                    cpt_df = cpt_df.rename(columns=rename)
                st.success(f"✓ CPT loaded: {len(cpt_df)} rows, "
                           f"depth {cpt_df['Depth'].min():.1f}–{cpt_df['Depth'].max():.1f} m")
            except Exception as e:
                st.error(f"Could not read CPT file: {e}")

        if cpt_df is not None:
            with st.expander("Preview CPT data"):
                st.dataframe(cpt_df.head(10), use_container_width=True)

    with col_sim:
        st.markdown("#### 📊 Simulation Data")
        st.markdown('<div class="info-box">Upload simulation output CSV. '
                    'Required columns: Depth_actual [m], SRD_sim [kN]</div>',
                    unsafe_allow_html=True)
        sim_file = st.file_uploader("Upload simulation CSV",
                                    type=["csv"],
                                    key="sim_upload")
        use_sample_sim = st.checkbox("Use sample simulation (R3, Rentel clay)")

        sim_df = None
        if use_sample_sim:
            sample_path = os.path.join(os.path.dirname(__file__),
                                       "data", "sample_SIM_R3.csv")
            if os.path.exists(sample_path):
                sim_df = pd.read_csv(sample_path)
                st.success(f"✓ Sample simulation loaded: {len(sim_df)} rows")
        elif sim_file is not None:
            try:
                sim_df = pd.read_csv(sim_file)
                sim_df.columns = [c.strip() for c in sim_df.columns]
                st.success(f"✓ Simulation loaded: {len(sim_df)} rows")
            except Exception as e:
                st.error(f"Could not read simulation file: {e}")

        if sim_df is not None:
            with st.expander("Preview simulation data"):
                st.dataframe(sim_df.head(10), use_container_width=True)

    st.markdown("---")

    # Run prediction
    ready = cpt_df is not None and sim_df is not None
    if not ready:
        st.markdown('<div class="warn-box">Upload or select sample data for both '
                    'CPT and simulation to enable prediction.</div>',
                    unsafe_allow_html=True)

    col_btn, col_note = st.columns([1, 3])
    with col_btn:
        run_btn = st.button("🚀 Run SRD Correction", disabled=not ready)
    with col_note:
        st.caption("This will: merge CPT→simulation by depth, compute features, "
                   "predict K correction factor, apply conformal intervals, "
                   "and display corrected SRD profiles.")

    if run_btn and ready:
        with st.spinner("Computing features and running ML correction…"):
            try:
                # Merge CPT to simulation depth
                sim_df = sim_df.sort_values("Depth_actual").reset_index(drop=True)
                cpt_df = cpt_df.sort_values("Depth").reset_index(drop=True)

                # Round-merge
                sim_df["Depth_key"] = sim_df["Depth_actual"].round(1)
                cpt_df["Depth_key"] = cpt_df["Depth"].round(1)
                cpt_agg = cpt_df.groupby("Depth_key", as_index=False).agg(
                    qt_MPa=("qt_MPa","mean"), fs_MPa=("fs_MPa","mean"))
                merged = sim_df.merge(cpt_agg, on="Depth_key", how="left")
                merged["qt_MPa"] = merged["qt_MPa"].interpolate().fillna(method="bfill").fillna(1.0)
                merged["fs_MPa"] = merged["fs_MPa"].interpolate().fillna(method="bfill").fillna(0.05)

                soil_labels = None
                if "SoilGroup" in merged.columns:
                    soil_labels = merged["SoilGroup"].fillna(soil_type).tolist()

                result = predict_srd(
                    depth       = merged["Depth_actual"].values,
                    srd_sim     = merged["SRD_sim"].values,
                    qt_mpa      = merged["qt_MPa"].values,
                    fs_mpa      = merged["fs_MPa"].values,
                    soil_type   = soil_type,
                    soil_labels = soil_labels,
                    tip_area    = tip_area,
                    max_diameter= diameter,
                    coverage    = coverage,
                    site        = site_key
                )
                st.session_state["result"] = result
                st.session_state["merged"] = merged
                st.success("✓ Prediction complete — see Results tab")
            except Exception as e:
                st.error(f"Prediction failed: {e}")
                import traceback
                st.code(traceback.format_exc())

# ──────────────────────────────────────────────────────────────────────────────
# TAB 2: RESULTS
# ──────────────────────────────────────────────────────────────────────────────
with tab2:
    if "result" not in st.session_state:
        st.info("Run a prediction in the Upload tab first.")
    else:
        result = st.session_state["result"]
        merged = st.session_state["merged"]

        depth    = result["depth"]
        srd_s    = result["srd_sim"]
        srd_c    = result["srd_corrected"]
        srd_lo   = result["srd_lower"]
        srd_hi   = result["srd_upper"]
        K_pred   = result["K_pred"]
        k_std    = result["K_std"]
        rec      = result["recommendation"]
        qhat     = result["qhat"]
        mean_k   = result["mean_k_correction"]
        n_rows   = len(depth)

        # ── Headline metrics ──────────────────────────────────────────────────
        rmse_sim = float(np.sqrt(np.mean((srd_s - srd_s)**2)))  # baseline = 0
        bias_sim = float(np.mean(srd_s - srd_s))
        bias_cor = float(np.mean(K_pred - 1.0) * np.mean(srd_s))
        mean_k_pred = float(np.mean(K_pred))

        c1, c2, c3, c4, c5 = st.columns(5)
        metrics = [
            (f"{n_rows}", "Data points"),
            (f"{mean_k_pred:.3f}", "Mean K predicted"),
            (f"±{qhat:.3f}", f"qhat ({int(coverage*100)}% PI)"),
            (f"{k_std:.3f}", "K_std (screening)"),
            (rec, "Recommendation"),
        ]
        for col, (val, lbl) in zip([c1,c2,c3,c4,c5], metrics):
            color = "#2E7D32" if (lbl=="Recommendation" and val=="ML") else \
                    "#C8860A" if (lbl=="Recommendation" and val=="MEAN") else "#1565C0"
            col.markdown(f"""
            <div class="metric-card">
                <div class="value" style="color:{color};font-size:1.5rem">{val}</div>
                <div class="label">{lbl}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("&nbsp;")

        # Recommendation box
        if rec == "ML":
            st.markdown(f"""
            <div class="rec-ml">
                ✅ ML CORRECTION APPLIED &nbsp;|&nbsp;
                K_std = {k_std:.3f} ≥ 0.15 &nbsp;·&nbsp;
                K varies with depth → ML can learn the correction pattern
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="rec-mean">
                ⚡ MEAN CORRECTION APPLIED &nbsp;|&nbsp;
                K_std = {k_std:.3f} < 0.15 &nbsp;·&nbsp;
                K is nearly constant → using site mean K = {mean_k:.3f}
            </div>""", unsafe_allow_html=True)

        st.markdown("&nbsp;")

        # ── Main plots ────────────────────────────────────────────────────────
        fig = plt.figure(figsize=(16, 8))
        fig.patch.set_facecolor("#F8FAFF")
        gs  = gridspec.GridSpec(1, 3, figure=fig, wspace=0.35)

        CBLUE = "#1565C0"
        CGRAY = "#78909C"
        CGRN  = "#2E7D32"

        # Plot 1: SRD profiles
        ax1 = fig.add_subplot(gs[0])
        ax1.fill_betweenx(depth, srd_lo/1000, srd_hi/1000,
                          alpha=0.2, color=CBLUE, label=f"{int(coverage*100)}% PI")
        ax1.plot(srd_c/1000, depth, "-", color=CBLUE, lw=2.5, label="ML-corrected")
        ax1.plot(srd_s/1000, depth, "--", color=CGRAY, lw=1.5, alpha=0.8, label="Simulated")
        ax1.invert_yaxis()
        ax1.set_xlabel("SRD [MN]", fontsize=11)
        ax1.set_ylabel("Penetration Depth [m]", fontsize=11)
        ax1.set_title("Corrected SRD Profile\n(with prediction interval)",
                      fontsize=11, fontweight="bold")
        ax1.legend(fontsize=9)
        ax1.grid(True, alpha=0.3)
        ax1.set_facecolor("white")

        # Plot 2: K factor profile
        ax2 = fig.add_subplot(gs[1])
        color_k = CGRN if rec == "ML" else "#F9A825"
        ax2.plot(K_pred, depth, "-", color=color_k, lw=2.5)
        ax2.fill_betweenx(depth, K_pred - qhat, K_pred + qhat,
                          alpha=0.2, color=color_k)
        ax2.axvline(1.0, color="red", lw=1.5, linestyle="--",
                    alpha=0.7, label="K=1 (perfect sim)")
        ax2.axvline(mean_k, color="#C8860A", lw=1.5, linestyle=":",
                    label=f"Site mean K={mean_k:.2f}")
        ax2.invert_yaxis()
        ax2.set_xlabel("K = SRD_corrected / SRD_sim [-]", fontsize=11)
        ax2.set_ylabel("Penetration Depth [m]", fontsize=11)
        ax2.set_title(f"K Correction Factor Profile\n({rec} correction applied)",
                      fontsize=11, fontweight="bold",
                      color=CGRN if rec == "ML" else "#C8860A")
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3)
        ax2.set_facecolor("white")

        # Plot 3: Correction % profile
        ax3 = fig.add_subplot(gs[2])
        correction_pct = (srd_c - srd_s) / srd_s * 100
        colors_bar = [CGRN if v < 0 else "#C62828" for v in correction_pct]
        ax3.barh(depth, correction_pct, height=0.2,
                 color=colors_bar, alpha=0.8)
        ax3.axvline(0, color="black", lw=1.5)
        bias_removal = float(np.mean(correction_pct))
        ax3.axvline(bias_removal, color=CBLUE, lw=1.5, linestyle="--",
                    label=f"Mean = {bias_removal:+.1f}%")
        ax3.invert_yaxis()
        ax3.set_xlabel("SRD Change from Simulation [%]", fontsize=11)
        ax3.set_ylabel("Penetration Depth [m]", fontsize=11)
        ax3.set_title("Correction Applied per Depth\n(negative = reduction in SRD)",
                      fontsize=11, fontweight="bold")
        ax3.legend(fontsize=9)
        ax3.grid(True, alpha=0.3, axis="x")
        ax3.set_facecolor("white")

        fig.suptitle(
            f"SRD Correction Results  |  Site: {site_key}  |  Soil: {soil_type.capitalize()}  |  "
            f"n={n_rows} depth points  |  {int(coverage*100)}% conformal PI",
            fontsize=11, y=1.02, fontweight="bold", color="#0A1628"
        )
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        # ── Download results ──────────────────────────────────────────────────
        st.markdown("---")
        st.markdown("#### 📥 Download Results")
        col_dl1, col_dl2 = st.columns(2)

        with col_dl1:
            results_df = pd.DataFrame({
                "Depth_m":              depth,
                "SRD_sim_kN":           srd_s,
                "SRD_corrected_kN":     srd_c,
                "SRD_lower_90pct_kN":   srd_lo,
                "SRD_upper_90pct_kN":   srd_hi,
                "K_correction_factor":  K_pred,
                "SRD_change_pct":       correction_pct,
            })
            csv_bytes = results_df.to_csv(index=False).encode()
            st.download_button(
                "⬇️ Download corrected SRD (CSV)",
                data=csv_bytes,
                file_name=f"srd_corrected_{site_key}_{soil_type}.csv",
                mime="text/csv"
            )

        with col_dl2:
            buf = io.BytesIO()
            fig2 = plt.figure(figsize=(16, 8))
            fig2.patch.set_facecolor("#F8FAFF")
            gs2  = gridspec.GridSpec(1, 3, figure=fig2, wspace=0.35)
            ax1b = fig2.add_subplot(gs2[0])
            ax1b.fill_betweenx(depth, srd_lo/1000, srd_hi/1000, alpha=0.2, color=CBLUE)
            ax1b.plot(srd_c/1000, depth, "-", color=CBLUE, lw=2.5, label="ML-corrected")
            ax1b.plot(srd_s/1000, depth, "--", color=CGRAY, lw=1.5, label="Simulated")
            ax1b.invert_yaxis(); ax1b.set_xlabel("SRD [MN]"); ax1b.set_ylabel("Depth [m]")
            ax1b.set_title("Corrected SRD Profile"); ax1b.legend(fontsize=8); ax1b.grid(True, alpha=0.3)
            ax2b = fig2.add_subplot(gs2[1])
            ax2b.plot(K_pred, depth, "-", color=color_k, lw=2.5)
            ax2b.fill_betweenx(depth, K_pred-qhat, K_pred+qhat, alpha=0.2, color=color_k)
            ax2b.axvline(1.0, color="red", lw=1.5, linestyle="--")
            ax2b.invert_yaxis(); ax2b.set_xlabel("K [-]"); ax2b.set_ylabel("Depth [m]")
            ax2b.set_title("K Correction Factor"); ax2b.grid(True, alpha=0.3)
            ax3b = fig2.add_subplot(gs2[2])
            ax3b.barh(depth, correction_pct, height=0.2, color=colors_bar, alpha=0.8)
            ax3b.axvline(0, color="black", lw=1.5)
            ax3b.invert_yaxis(); ax3b.set_xlabel("Change [%]"); ax3b.set_ylabel("Depth [m]")
            ax3b.set_title("Correction Applied"); ax3b.grid(True, alpha=0.3, axis="x")
            fig2.suptitle(f"SRD Correction | {site_key} | {soil_type} | {int(coverage*100)}% PI",
                          fontsize=10, fontweight="bold")
            fig2.savefig(buf, dpi=150, bbox_inches="tight")
            plt.close(fig2)
            buf.seek(0)
            st.download_button(
                "⬇️ Download plot (PNG)",
                data=buf,
                file_name=f"srd_correction_{site_key}_{soil_type}.png",
                mime="image/png"
            )

        # ── Summary stats ─────────────────────────────────────────────────────
        st.markdown("---")
        st.markdown("#### 📋 Correction Summary")
        c1, c2, c3 = st.columns(3)
        c1.markdown(f"""
        **Simulation (raw)**
        - Mean SRD: {np.mean(srd_s)/1000:.1f} MN
        - Min SRD: {np.min(srd_s)/1000:.1f} MN
        - Max SRD: {np.max(srd_s)/1000:.1f} MN
        """)
        c2.markdown(f"""
        **ML-Corrected**
        - Mean SRD: {np.mean(srd_c)/1000:.1f} MN
        - Min SRD: {np.min(srd_c)/1000:.1f} MN
        - Max SRD: {np.max(srd_c)/1000:.1f} MN
        """)
        c3.markdown(f"""
        **{int(coverage*100)}% Prediction Interval**
        - Mean lower: {np.mean(srd_lo)/1000:.1f} MN
        - Mean upper: {np.mean(srd_hi)/1000:.1f} MN
        - Mean width: {np.mean(srd_hi - srd_lo)/1000:.1f} MN
        """)

# ──────────────────────────────────────────────────────────────────────────────
# TAB 3: ABOUT
# ──────────────────────────────────────────────────────────────────────────────
with tab3:
    col_a, col_b = st.columns([3, 2])
    with col_a:
        st.markdown("""
        ### About This Tool

        This tool applies a machine learning correction to pre-installation pile driving
        simulations for offshore wind turbine monopiles. It was developed as part of a
        research collaboration between **Florida Polytechnic University**, **Aalborg University**,
        and **COWI A/S**.

        #### What the tool does
        1. **Reads** your CPT data and pre-installation simulation output
        2. **Engineers** 11 features: depth, simulated SRD, CPT tip resistance,
           sleeve friction, cumulative integrals, ISBT, geometry, and % sand
        3. **Screens** the location using K_std (within-location K variability):
           - K_std ≥ 0.15 → ML correction (ANN predicts depth-varying K)
           - K_std < 0.15 → Mean correction (site mean K applied uniformly)
        4. **Predicts** the K correction factor = SRD_actual / SRD_simulated
        5. **Applies** calibrated conformal prediction intervals

        #### Training data
        - 51 pile locations, 5,379 depth-matched rows
        - **Rentel** (Belgian North Sea): 42 R-locations, mostly clay
        - **Merkur** (German North Sea): 17 MO-locations, mostly sand
        - Models trained separately for sand and clay

        #### Key performance (clay, LOGO evaluation)
        - **50% RMSE reduction** (46.4 → 23.2 MN)
        - **99% bias reduction** (+30.9 → -0.2 MN)
        - **34/38 locations improved** vs raw simulation
        - **90% conformal intervals** calibrated to exact empirical coverage

        #### Limitations
        - Clay model is more reliable than sand
        - Performance at new wind farm sites (outside Merkur/Rentel) is uncertain
        - Shallow depth (0–10 m) prediction intervals are less reliable
        - K_std screening accuracy is 63% — verify recommendations manually
        """)

    with col_b:
        st.markdown("""
        ### File Format Guide

        **CPT — R format (.txt)**
        ```
        Depth  qc    fs     qt
        0.02   1.42  0.000  1.419
        0.04   1.08  0.004  1.081
        ...
        ```
        Tab or space delimited, with header row.

        **CPT — MO format (.dat)**
        ```
        0.000  0.164  0.164  0.000  0.000
        0.020  0.325  0.325  0.000  0.000
        ...
        ```
        Space delimited, no header. Use columns 0, 2, 4.

        **Simulation CSV**
        ```
        Depth_actual,SRD_sim,SoilGroup
        10.5,68450,clay
        10.75,72300,clay
        ...
        ```
        SRD in kN. SoilGroup is optional (clay/sand).

        ---
        ### References
        - Robertson (2010): Soil Behaviour Type Index ISBT
        - Angelopoulos et al. (2019): Conformal prediction
        - Philipp (2026): Improving Pile Driving Predictions with ML, Aalborg University

        ---
        ### Contact
        **Parisa Hajibabaee**
        Assistant Professor, Data Science & Business Analytics
        Florida Polytechnic University
        """)

    st.markdown("---")
    st.caption(
        "⚠️ This tool is intended for research and planning purposes. "
        "All predictions should be reviewed by a qualified geotechnical engineer "
        "before use in offshore pile installation decisions. "
        "Conformal intervals assume exchangeability and are calibrated on Rentel/Merkur data only."
    )
