# app.py
import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
from io import BytesIO
import os
import tempfile
from utils import (
    load_npz_image, compute_ndvi, ndvi_to_colormap_image,
    overlay_mask_on_rgb, read_sensor_csv, simple_fusion_alert
)

# ----------------- PAGE SETUP -----------------
st.set_page_config(layout="wide", page_title="Precision Agri Demo")

st.title("🌱 Precision Agriculture — Demo Prototype")
st.markdown(
    "Upload a **multispectral tile (`.npz`)** with `red` and `nir` bands "
    "and a **sensor CSV** to visualize NDVI, soil data, and stress alerts."
)

col1, col2 = st.columns([1, 1])

# ----------------- IMAGE UPLOAD + NDVI -----------------
with col1:
    uploaded_image = st.file_uploader("Upload tile (.npz)", type=["npz"])

    if uploaded_image is None:
        if st.button("Use sample tile"):
            # --- CORRECTED PATH LOGIC ---
            script_dir = os.path.dirname(__file__)
            sample_image_path = os.path.join(script_dir, "..", "sample_data", "sample_tile.npz")
            uploaded_image = open(sample_image_path, "rb")
            # --------------------------

    if uploaded_image is not None:
        # Handle uploaded file safely
        if hasattr(uploaded_image, "read"):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".npz") as tmp:
                tmp.write(uploaded_image.read())
                tmp_path = tmp.name
            imgdata = load_npz_image(tmp_path)
            os.unlink(tmp_path)
        else:
            imgdata = load_npz_image(uploaded_image)

        red = imgdata["red"]
        nir = imgdata["nir"]
        green = imgdata.get("green", red)

        # Compute NDVI
        ndvi = compute_ndvi(nir, red)
        ndvi_img = ndvi_to_colormap_image(ndvi)

        st.subheader("🌍 NDVI Map")
        st.image(ndvi_img, use_column_width=True)

        ndvi_mean = float(np.nanmean(ndvi))
        st.metric("Mean NDVI", f"{ndvi_mean:.3f}")

# ----------------- SENSOR UPLOAD -----------------
with col2:
    uploaded_csv = st.file_uploader(
        "Upload sensors CSV (timestamp,soil_moisture,air_temp,humidity,leaf_wetness)",
        type=["csv"],
    )

    if uploaded_csv is None:
        if st.button("Use sample sensors"):
            # --- CORRECTED PATH LOGIC ---
            script_dir = os.path.dirname(__file__)
            sample_csv_path = os.path.join(script_dir, "..", "sample_data", "sensors.csv")
            uploaded_csv = open(sample_csv_path, "rb")
            # --------------------------
            
    df = None
    if uploaded_csv is not None:
        try:
            df = pd.read_csv(uploaded_csv, parse_dates=["timestamp"])
            st.subheader("📊 Sensor Data Preview")
            st.dataframe(df.tail())

            st.subheader("📈 Sensor Trends")
            st.line_chart(df.set_index("timestamp")[["soil_moisture", "humidity", "air_temp"]])

            soil_mean = float(df["soil_moisture"].mean())
            st.metric("Mean Soil Moisture", f"{soil_mean:.3f}")
        except Exception as e:
            st.error(f"Could not read sensor file: {e}")

# ----------------- SETTINGS -----------------
st.sidebar.header("⚙️ Settings")
ndvi_thresh = st.sidebar.slider("NDVI threshold (stress below)", -0.5, 0.9, 0.30, 0.01)
soil_thresh = st.sidebar.slider("Soil moisture threshold", 0.0, 1.0, 0.20, 0.01)

# ----------------- ANALYSIS + FUSION ALERT -----------------
if "ndvi" in locals():
    mask = (ndvi < ndvi_thresh).astype(np.uint8)
    overlay = overlay_mask_on_rgb(red, green, nir, mask, alpha=0.4)

    st.subheader("🚨 Stressed Zones (Overlay)")
    st.image(overlay, use_column_width=True)
    
    # Defaults
    alert, score = False, 0.0
    ndvi_mean = float(np.nanmean(ndvi))

    if "df" in locals() and df is not None and not df.empty:
        # Full fusion: NDVI + Soil moisture
        soil_mean = float(df["soil_moisture"].mean())
        alert, score = simple_fusion_alert(ndvi_mean, soil_mean, ndvi_thresh, soil_thresh)

        if alert:
            st.error(f"⚠️ ALERT: Stress likely detected!\nRisk score: {score:.2f}")
        else:
            st.success(f"✅ No critical stress detected.\nRisk score: {score:.2f}")
    else:
        # Fallback: NDVI-only alert
        if ndvi_mean < ndvi_thresh:
            alert, score = True, (ndvi_thresh - ndvi_mean) / (ndvi_thresh + 1e-8)
            st.warning(f"⚠️ NDVI-only alert (no sensors): Vegetation stress detected!\nRisk score: {score:.2f}")
        else:
            st.info("ℹ️ No stress detected (NDVI-only, sensors missing).")

    # ----------------- SUMMARY REPORT -----------------
    summary = {
        "mean_ndvi": [ndvi_mean],
        "ndvi_thresh": [ndvi_thresh],
        "mean_soil_moisture": [float(df["soil_moisture"].mean())] if "df" in locals() and df is not None else [None],
        "soil_thresh": [soil_thresh],
        "alert": [bool(alert)],
        "risk_score": [float(score)],
    }
    summary_df = pd.DataFrame(summary)
    csv = summary_df.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Download Report CSV", csv, "report.csv", "text/csv")

    # Overlay download
    buf = BytesIO()
    overlay.save(buf, format="PNG")
    st.download_button("📥 Download Overlay PNG", buf.getvalue(), "overlay.png", "image/png")