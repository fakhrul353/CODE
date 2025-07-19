import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import streamlit as st

# ===============================
# STREAMLIT LSA APP
# ===============================
st.set_page_config(page_title="LSA Elevation Adjustment", layout="wide")
st.title("📏 Least Squares Adjustment (LSA) for Elevation")

st.markdown("""
🔗 **Versi Web Ini:** [Buka Aplikasi di Web](https://share.streamlit.io/) _(Sila muat naik ke Streamlit Cloud dahulu)_
""")

st.sidebar.header("Benchmark (Known Points)")
bm_count = st.sidebar.selectbox("Pilih bilangan benchmark:", [2, 3, 4, 5], index=0)
known_points = {}
for i in range(bm_count):
    label = st.sidebar.text_input(f"Nama BM {i+1}", f"BM{i+1}")
    value = st.sidebar.number_input(f"Nilai tinggi {label} (m)", key=f"bm_val_{i}")
    known_points[label] = value

unknown_points_input = st.text_input("Masukkan nama titik tak diketahui (pisahkan dengan koma)", "A,B,C")
unknown_points = [pt.strip() for pt in unknown_points_input.split(",") if pt.strip()]
point_index = {pt: i for i, pt in enumerate(unknown_points)}
u = len(unknown_points)

st.subheader("📊 Pemerhatian Tinggi")
n_obs = st.number_input("Bilangan pemerhatian", min_value=1, step=1)
observations = []
for i in range(int(n_obs)):
    cols = st.columns(3)
    frm = cols[0].text_input(f"From {i+1}", key=f"from_{i}")
    to = cols[1].text_input(f"To {i+1}", key=f"to_{i}")
    diff = cols[2].number_input(f"ΔH {i+1} (m)", format="%.4f", key=f"dh_{i}")
    if frm and to:
        observations.append((frm, to, diff))

if st.button("🧮 Jalankan LSA"):
    n = len(observations)
    r = n - u

    if r <= 0:
        st.error("❌ Tidak cukup pemerhatian. Redundancy ≤ 0")
    else:
        A = np.zeros((n, u))
        L = np.zeros((n, 1))

        for i, (frm, to, dh) in enumerate(observations):
            if frm in point_index:
                A[i, point_index[frm]] = -1
            elif frm in known_points:
                L[i] += known_points[frm]

            if to in point_index:
                A[i, point_index[to]] = 1
            elif to in known_points:
                L[i] -= known_points[to]

            L[i] += dh

        AT = A.T
        N = AT @ A
        U = AT @ L
        X = np.linalg.inv(N) @ U
        V = A @ X - L
        sigma0_squared = (V.T @ V)[0, 0] / r
        Cov = sigma0_squared * np.linalg.inv(N)
        std_dev = np.sqrt(np.diag(Cov))

        confidence_level = 0.95
        z_score = stats.norm.ppf(1 - (1 - confidence_level) / 2)
        ci_low = X.flatten() - z_score * std_dev
        ci_high = X.flatten() + z_score * std_dev

        # ===============================
        # HASIL DAN CSV
        # ===============================
        st.subheader("📈 Hasil Pelarasan")
        df_output = pd.DataFrame({
            'Point': unknown_points,
            'Adjusted Elevation (m)': X.flatten(),
            'Std Deviation (m)': std_dev,
            'CI Lower Bound': ci_low,
            'CI Upper Bound': ci_high
        })
        st.dataframe(df_output, use_container_width=True)
        st.download_button("📥 Muat Turun CSV", df_output.to_csv(index=False), file_name="adjusted_results.csv")

        # ===============================
        # GRAF
        # ===============================
        st.subheader("📊 Plot Pelarasan")
        elevation_points = unknown_points + list(known_points.keys())
        elevation_values = list(X.flatten()) + [known_points[k] for k in known_points]
        confidence_intervals = [z_score * e for e in std_dev] + [0 for _ in known_points]

        fig, ax = plt.subplots(figsize=(10, 5))
        x_pos = list(range(len(elevation_points)))
        ax.errorbar(x_pos, elevation_values, yerr=confidence_intervals, fmt='-o', color='blue', ecolor='gray', capsize=5)
        for i, pt in enumerate(elevation_points):
            ax.text(x_pos[i], elevation_values[i] + 0.05, pt, ha='center', fontsize=9)

        ax.set_title("Adjusted Elevations with 95% CI (Including Benchmarks)")
        ax.set_xlabel("Point Index")
        ax.set_ylabel("Elevation (m)")
        ax.set_xticks(x_pos)
        ax.set_xticklabels(elevation_points)
        ax.grid(True)
        st.pyplot(fig)

        # Residuals plot
        fig2, ax2 = plt.subplots(figsize=(8, 4))
        ax2.bar(range(len(V)), V.flatten(), color='skyblue')
        ax2.axhline(0, color='red', linestyle='--')
        ax2.set_title("Residuals of Elevation Differences")
        ax2.set_xlabel("Observation Index")
        ax2.set_ylabel("Residual (m)")
        ax2.grid(True)
        st.pyplot(fig2)
