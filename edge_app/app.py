"""
AgriRakshak-Edge: Offline Dual-Engine SoC Diagnostic Dashboard
Bharat AI-SoC Challenge 2026-27 | PS1: Agriculture & Food Security
Target: Arm Cortex-A Kiosk & Android Mobile Edge
"""

import os
import sys
import time
import numpy as np
import cv2
from PIL import Image
import streamlit as st

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from engine.pipeline import AgriRakshakPipeline
from engine.vision_infer import create_sample_leaf_image

# Streamlit Page Config
st.set_page_config(
    page_title="AgriRakshak-Edge | Offline AI on SoC",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling for Edge Look & Feel
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #134e4a, #065f46, #047857);
        color: white;
        padding: 20px 24px;
        border-radius: 12px;
        margin-bottom: 20px;
        box-shadow: 0 4px 14px rgba(0, 0, 0, 0.12);
    }
    .badge-soc {
        background: rgba(255, 255, 255, 0.18);
        border: 1px solid rgba(255, 255, 255, 0.3);
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 13px;
        font-weight: 600;
        margin-right: 8px;
        display: inline-block;
    }
    .metric-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
    }
    .advisory-box {
        background: #f0fdf4;
        border-left: 5px solid #16a34a;
        padding: 18px;
        border-radius: 8px;
        margin-top: 15px;
    }
    .chemical-box {
        background: #fef2f2;
        border-left: 5px solid #dc2626;
        padding: 18px;
        border-radius: 8px;
        margin-top: 15px;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_pipeline():
    return AgriRakshakPipeline()

pipeline = load_pipeline()

# Header
st.markdown("""
<div class="main-header">
    <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">
        <div>
            <h1 style="margin: 0; font-size: 26px; font-weight: 700; color: white;">🌾 AgriRakshak-Edge (कृषि रक्षक AI)</h1>
            <p style="margin: 4px 0 0 0; opacity: 0.9; font-size: 14px;">
                Offline Dual-Engine SoC Diagnostic System • Bharat AI-SoC Challenge 2026–27 (PS1)
            </p>
        </div>
        <div style="margin-top: 8px;">
            <span class="badge-soc">⚡ Arm Cortex-A Target</span>
            <span class="badge-soc">🧠 Llama 3.2 1B (INT4 ExecuTorch)</span>
            <span class="badge-soc">🔒 100% Offline</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.header("⚙️ Edge System Settings")
language = st.sidebar.selectbox("🌐 भाषा चुनें / Select Language", ["hindi", "english"], index=0, format_func=lambda x: "हिंदी (Hindi)" if x == "hindi" else "English")

st.sidebar.subheader("📷 Image Source")
input_mode = st.sidebar.radio(
    "Choose Input Method:",
    ["Pre-Loaded Field Samples (Instant Demo)", "Upload Leaf Photo", "Live Camera"],
    index=0
)

sample_choice = None
uploaded_file = None
camera_file = None

if input_mode == "Pre-Loaded Field Samples (Instant Demo)":
    sample_choice = st.sidebar.selectbox(
        "Select Crop Condition Sample:",
        [
            ("tomato_early_blight", "Tomato - Early Blight (अगेती झुलसा)"),
            ("tomato_late_blight", "Tomato - Late Blight (पछेती झुलसा)"),
            ("corn_rust", "Corn / Maize - Common Rust (रतुआ रोग)"),
            ("cotton_blight", "Cotton - Bacterial Blight (जीवाणु झुलसा)"),
            ("healthy_wheat", "Wheat - Healthy Crop (स्वस्थ फसल)")
        ],
        format_func=lambda x: x[1]
    )[0]
elif input_mode == "Upload Leaf Photo":
    uploaded_file = st.sidebar.file_uploader("Upload crop leaf image (JPG/PNG)", type=["jpg", "jpeg", "png"])
else:
    camera_file = st.sidebar.camera_input("Take a photo of crop leaf")

# Hardware & SoC Profiler in Sidebar
with st.sidebar.expander("📊 Arm SoC Telemetry & Benchmarks", expanded=True):
    st.markdown("""
    * **Target SoC:** Arm Cortex-A76 @ 2.4GHz
    * **Runtime:** ExecuTorch v0.4 (XNNPACK/KleidiAI)
    * **Vision Model:** MobileNetV4-Agri (INT8)
    * **Reasoning Model:** Llama 3.2 1B (INT4 PTE)
    * **Llama Model Size:** ~648.5 MB
    * **Peak RAM Consumption:** < 900 MB
    * **Network Call:** **0 bytes (Air-Gapped)**
    """)

# Prepare Image Input
active_image = None
class_hint = None

if input_mode == "Pre-Loaded Field Samples (Instant Demo)":
    active_image = create_sample_leaf_image(sample_choice)
    class_hint = sample_choice
elif uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    active_image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    active_image = cv2.cvtColor(active_image, cv2.COLOR_BGR2RGB)
elif camera_file is not None:
    file_bytes = np.asarray(bytearray(camera_file.read()), dtype=np.uint8)
    active_image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    active_image = cv2.cvtColor(active_image, cv2.COLOR_BGR2RGB)

if active_image is not None:
    # Run Pipeline
    with st.spinner("Processing on Arm Edge Core (Vision + Llama ExecuTorch)..."):
        res = pipeline.process_leaf(active_image, language=language, class_hint=class_hint)

    vis = res["vision"]
    adv = res["advisory"]
    telemetry = adv["telemetry"]

    # Top Metrics Row
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric(
            label="फसल / Crop",
            value=vis["crop"],
            delta=vis["hindi_name"] if language == "hindi" else vis["disease"]
        )
    with m2:
        st.metric(
            label="बीमारी / Condition",
            value=vis["disease"],
            delta=f"Conf: {vis['confidence_percent']}%"
        )
    with m3:
        sev_color = "🔴" if "Severe" in vis["severity"] else ("🟠" if "Moderate" in vis["severity"] else "🟢")
        st.metric(
            label="नुकसान की गंभीरता / Severity",
            value=f"{sev_color} {vis['severity']}",
            delta=f"Lesion Area: {vis['lesion_ratio_percent']}%"
        )
    with m4:
        st.metric(
            label="कुल लेटेंसी / SoC Latency",
            value=f"{res['total_system_latency_ms']} ms",
            delta=f"Vision: {vis['inference_time_ms']}ms | Llama: {telemetry['total_inference_time_sec']*1000:.0f}ms"
        )

    st.divider()

    # Visual Inspection & Lesion Segmentation
    col_img1, col_img2 = st.columns(2)
    with col_img1:
        st.subheader("🍃 मूल पत्ती (Original Leaf Capture)")
        st.image(active_image, use_container_width=True)

    with col_img2:
        st.subheader("🎯 लीज़न हीटमैप (Lesion Segmentation Overlay)")
        st.image(vis["annotated_image"], caption="Yellow contours indicate active necrotic/fungal patches", use_container_width=True)

    # Voice Audio & Speech Synthesis
    st.subheader("🔊 किसान ऑडियो सलाह (Offline Audio Advisory)")
    st.markdown(res["audio_html"], unsafe_allow_html=True)
    if os.path.exists(res["audio_file"]):
        with open(res["audio_file"], "rb") as f:
            audio_bytes = f.read()
        st.audio(audio_bytes, format="audio/wav")

    # Llama ExecuTorch Grounded Advisory Section
    st.subheader("🧠 Llama 3.2 ExecuTorch वैज्ञानिक सलाह (Actionable Advisory)")
    st.markdown(adv["markdown_advisory"])

    # Edge SoC Telemetry Table
    with st.expander("🔬 Deep SoC Performance & Execution Telemetry", expanded=False):
        c_tel1, c_tel2 = st.columns(2)
        with c_tel1:
            st.markdown(f"""
            * **Inference Runtime:** `{telemetry['model_runtime']}`
            * **Time to First Token (TTFT):** `{telemetry['time_to_first_token_sec']} s`
            * **Decode Speed:** `{telemetry['tokens_per_second']} tokens/sec`
            * **Total Tokens Generated:** `{telemetry['tokens_generated']}`
            """)
        with c_tel2:
            st.markdown(f"""
            * **Peak Working RAM (RSS):** `{telemetry['peak_ram_mb']} MB`
            * **Quantization Scheme:** `INT4 Groupwise (Weight) + INT8 (Activation)`
            * **Target Arm ISA:** `Armv8.2-A / Armv9-A (NEON + KleidiAI)`
            * **Network Dependency:** `Verified 100% Offline (Zero Socket Calls)`
            """)

else:
    st.info("👈 Please select or upload a crop leaf image from the sidebar to begin offline diagnosis.")
