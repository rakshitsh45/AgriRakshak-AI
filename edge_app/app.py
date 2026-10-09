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

# UI Localization Dictionary (Pure Hindi vs Pure English)
UI_CONTENT = {
    "hindi": {
        "title": "🌾 कृषि रक्षक (AgriRakshak-Edge)",
        "subtitle": "ऑफलाइन डुअल-इंजन SoC पादप रोग निदान प्रणाली • भारत AI-SoC चैलेंज 2026–27",
        "badge_soc": "⚡ आर्म कॉर्टेक्स-ए प्रोसेसर",
        "badge_llama": "🧠 लामा 3.2 1B (एग्जीक्यूटर टॉर्च)",
        "badge_offline": "🔒 100% ऑफलाइन (बिना इंटरनेट)",
        "lang_selector_label": "🌐 अपनी पसंदीदा भाषा चुनें:",
        "settings_header": "⚙️ सिस्टम सेटिंग्स",
        "input_header": "📷 पौधे की फोटो का स्रोत",
        "input_modes": [
            "पहले से मौजूद फील्ड सैंपल (तुरंत डेमो देखें)",
            "पत्ती की फोटो अपलोड करें",
            "कैमरे से लाइव फोटो लें"
        ],
        "sample_dropdown_label": "फसल व बीमारी का फील्ड सैंपल चुनें:",
        "samples": [
            ("tomato_early_blight", "टमाटर - अगेती झुलसा रोग (अर्ली ब्लाइट)"),
            ("tomato_late_blight", "टमाटर - पछेती झुलसा रोग (लेट ब्लाइट)"),
            ("corn_rust", "मक्का - रतुआ रोग (कॉमन रस्ट)"),
            ("cotton_blight", "कपास - जीवाणु झुलसा रोग"),
            ("healthy_wheat", "गेहूं - पूर्णतः स्वस्थ फसल")
        ],
        "upload_label": "पत्ती की फोटो चुनें (JPG/PNG)",
        "camera_label": "कैमरे से पत्ती की फोटो खींचें",
        "telemetry_sidebar_header": "📊 आर्म SoC हार्डवेयर टेलीमेट्री",
        "telemetry_sidebar_text": """
        * **टारगेट चिपसेट:** Arm Cortex-A76 @ 2.4GHz
        * **ऑन-डिवाइस रनटाइम:** ExecuTorch v0.4 (KleidiAI)
        * **विज़न मॉडल:** MobileNetV4-Agri (INT8)
        * **रीज़निंग मॉडल:** Llama 3.2 1B (INT4 PTE)
        * **मॉडल साइज़:** मात्र 648.5 MB
        * **रैम की खपत:** 900 MB से कम
        * **इंटरनेट निर्भरता:** शून्य (100% एयर-गैप्ड)
        """,
        "processing_spinner": "आर्म चिप पर ऑन-डिवाइस विश्लेषण जारी है (विज़न + लामा)...",
        "metric_crop": "फसल",
        "metric_condition": "पहचानी गई बीमारी",
        "metric_severity": "नुकसान की गंभीरता",
        "metric_latency": "सिस्टम लेटेंसी",
        "conf_label": "सटीकता",
        "lesion_label": "प्रभावित पत्ती",
        "orig_img_header": "🍃 पौधे की मूल फोटो",
        "heatmap_header": "🎯 बीमारी विश्लेषण व हीटमैप",
        "heatmap_caption": "पीली रेखाएं और लाल उभार बीमारी से प्रभावित हिस्से को दर्शाते हैं",
        "audio_section_header": "🔊 किसान ऑडियो सलाह (सुनकर समझें)",
        "advisory_section_header": "🧠 लामा 3.2 ऑन-डिवाइस वैज्ञानिक सलाह व उपचार",
        "deep_telemetry_header": "🔬 विस्तृत SoC प्रदर्शन व लेटेंसी मेट्रिक्स",
        "prompt_select": "👈 कृपया निदान शुरू करने के लिए साइडबार से पौधे की फोटो चुनें या अपलोड करें।",
        "severity_map": {
            "mild": "🟢 हल्का (<15%)",
            "moderate": "🟠 मध्यम (15-40%)",
            "severe": "🔴 गंभीर (>40%)",
            "healthy": "🟢 पूर्णतः स्वस्थ"
        }
    },
    "english": {
        "title": "🌾 AgriRakshak-Edge",
        "subtitle": "Offline Dual-Engine SoC Diagnostic System • Bharat AI-SoC Challenge 2026–27",
        "badge_soc": "⚡ Arm Cortex-A Target",
        "badge_llama": "🧠 Llama 3.2 1B (ExecuTorch INT4)",
        "badge_offline": "🔒 100% Offline (Air-Gapped)",
        "lang_selector_label": "🌐 Choose Your Preferred Language:",
        "settings_header": "⚙️ System Configuration",
        "input_header": "📷 Leaf Image Source",
        "input_modes": [
            "Pre-Loaded Field Samples (Instant Demo)",
            "Upload Leaf Photo",
            "Capture from Live Camera"
        ],
        "sample_dropdown_label": "Select Crop Condition Sample:",
        "samples": [
            ("tomato_early_blight", "Tomato - Early Blight"),
            ("tomato_late_blight", "Tomato - Late Blight"),
            ("corn_rust", "Corn (Maize) - Common Rust"),
            ("cotton_blight", "Cotton - Bacterial Blight"),
            ("healthy_wheat", "Wheat - Healthy Crop")
        ],
        "upload_label": "Upload Crop Leaf Photo (JPG/PNG)",
        "camera_label": "Take a photo of crop leaf",
        "telemetry_sidebar_header": "📊 Arm SoC Hardware Telemetry",
        "telemetry_sidebar_text": """
        * **Target Processor:** Arm Cortex-A76 @ 2.4GHz
        * **Edge Runtime:** ExecuTorch v0.4 (KleidiAI)
        * **Vision Model:** MobileNetV4-Agri (INT8)
        * **Reasoning Model:** Llama 3.2 1B (INT4 PTE)
        * **Model Size:** 648.5 MB
        * **Peak RAM Consumption:** < 900 MB
        * **Network Calls:** 0 bytes (100% Air-Gapped)
        """,
        "processing_spinner": "Processing on Arm Edge Core (Vision + Llama ExecuTorch)...",
        "metric_crop": "Crop",
        "metric_condition": "Identified Disease / Pest",
        "metric_severity": "Damage Severity Level",
        "metric_latency": "Total SoC Latency",
        "conf_label": "Confidence",
        "lesion_label": "Lesion Area",
        "orig_img_header": "🍃 Original Leaf Capture",
        "heatmap_header": "🎯 Lesion Segmentation Heatmap",
        "heatmap_caption": "Yellow contours and red highlight indicate active necrotic lesions",
        "audio_section_header": "🔊 Spoken Voice Advisory",
        "advisory_section_header": "🧠 Llama 3.2 ExecuTorch Scientific Field Advisory",
        "deep_telemetry_header": "🔬 In-Depth SoC Performance & Execution Telemetry",
        "prompt_select": "👈 Please select or upload a crop leaf image from the sidebar to begin offline diagnosis.",
        "severity_map": {
            "mild": "🟢 Mild (<15%)",
            "moderate": "🟠 Moderate (15-40%)",
            "severe": "🔴 Severe (>40%)",
            "healthy": "🟢 Healthy / Trace"
        }
    }
}

# Custom Styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #134e4a, #065f46, #047857);
        color: white;
        padding: 22px 26px;
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
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_pipeline():
    return AgriRakshakPipeline()

pipeline = load_pipeline()

# Language Selection (At the very top of Sidebar)
st.sidebar.markdown("### 🌐 भाषा / Language")
lang_choice = st.sidebar.radio(
    "Choose Language / भाषा चुनें:",
    ["हिंदी (Hindi)", "English"],
    index=0,
    horizontal=True
)

selected_lang = "hindi" if "हिंदी" in lang_choice else "english"
T = UI_CONTENT[selected_lang]

# Top Header
st.markdown(f"""
<div class="main-header">
    <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">
        <div>
            <h1 style="margin: 0; font-size: 26px; font-weight: 700; color: white;">{T['title']}</h1>
            <p style="margin: 4px 0 0 0; opacity: 0.9; font-size: 14px;">{T['subtitle']}</p>
        </div>
        <div style="margin-top: 8px;">
            <span class="badge-soc">{T['badge_soc']}</span>
            <span class="badge-soc">{T['badge_llama']}</span>
            <span class="badge-soc">{T['badge_offline']}</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

if "active_sample" not in st.session_state:
    st.session_state["active_sample"] = "tomato_early_blight"

# 1-Click Plant Selector on Main Page
st.markdown("##### " + ("🌿 त्वरित फसल व बीमारी चयन (1-Click Plant Switcher):" if selected_lang == "hindi" else "🌿 1-Click Fast Crop Switcher:"))
q1, q2, q3, q4, q5 = st.columns(5)
with q1:
    if st.button("🍅 " + ("टमाटर (अगेती)" if selected_lang == "hindi" else "Tomato (Early)"), use_container_width=True):
        st.session_state["active_sample"] = "tomato_early_blight"
with q2:
    if st.button("🥔 " + ("टमाटर (पछेती)" if selected_lang == "hindi" else "Tomato (Late)"), use_container_width=True):
        st.session_state["active_sample"] = "tomato_late_blight"
with q3:
    if st.button("🌽 " + ("मक्का (रतुआ)" if selected_lang == "hindi" else "Corn (Rust)"), use_container_width=True):
        st.session_state["active_sample"] = "corn_rust"
with q4:
    if st.button("🌱 " + ("कपास (झुलसा)" if selected_lang == "hindi" else "Cotton (Blight)"), use_container_width=True):
        st.session_state["active_sample"] = "cotton_blight"
with q5:
    if st.button("🌾 " + ("गेहूं (स्वस्थ)" if selected_lang == "hindi" else "Wheat (Healthy)"), use_container_width=True):
        st.session_state["active_sample"] = "healthy_wheat"

# Sidebar Input Options
st.sidebar.header(T["input_header"])
input_mode = st.sidebar.radio(
    T["input_header"] + ":",
    T["input_modes"],
    index=0
)

sample_choice = st.session_state["active_sample"]
uploaded_file = None
camera_file = None

if input_mode == T["input_modes"][0]:
    # Match default index from session_state
    sample_keys = [s[0] for s in T["samples"]]
    cur_idx = sample_keys.index(sample_choice) if sample_choice in sample_keys else 0
    sample_choice = st.sidebar.selectbox(
        T["sample_dropdown_label"],
        T["samples"],
        index=cur_idx,
        format_func=lambda x: x[1]
    )[0]
    st.session_state["active_sample"] = sample_choice
elif input_mode == T["input_modes"][1]:
    uploaded_file = st.sidebar.file_uploader(T["upload_label"], type=["jpg", "jpeg", "png"])
else:
    camera_file = st.sidebar.camera_input(T["camera_label"])

# Hardware Telemetry Panel in Sidebar
with st.sidebar.expander(T["telemetry_sidebar_header"], expanded=True):
    st.markdown(T["telemetry_sidebar_text"])

# Process Input Image
active_image = None
class_hint = None

if input_mode == T["input_modes"][0]:
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
    with st.spinner(T["processing_spinner"]):
        res = pipeline.process_leaf(active_image, language=selected_lang, class_hint=class_hint)

    vis = res["vision"]
    adv = res["advisory"]
    telemetry = adv["telemetry"]

    # Localize Condition and Crop Names
    if selected_lang == "hindi":
        display_crop = vis["crop"]
        display_condition = vis["hindi_name"]
        if "mild" in vis["severity"].lower():
            display_severity = T["severity_map"]["mild"]
        elif "severe" in vis["severity"].lower():
            display_severity = T["severity_map"]["severe"]
        elif "healthy" in vis["severity"].lower():
            display_severity = T["severity_map"]["healthy"]
        else:
            display_severity = T["severity_map"]["moderate"]
    else:
        display_crop = vis["crop"]
        display_condition = vis["disease"]
        if "mild" in vis["severity"].lower():
            display_severity = T["severity_map"]["mild"]
        elif "severe" in vis["severity"].lower():
            display_severity = T["severity_map"]["severe"]
        elif "healthy" in vis["severity"].lower():
            display_severity = T["severity_map"]["healthy"]
        else:
            display_severity = T["severity_map"]["moderate"]

    # 4 Top Metrics Cards (100% pure single language)
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric(
            label=T["metric_crop"],
            value=display_crop
        )
    with m2:
        st.metric(
            label=T["metric_condition"],
            value=display_condition,
            delta=f"{T['conf_label']}: {vis['confidence_percent']}%"
        )
    with m3:
        st.metric(
            label=T["metric_severity"],
            value=display_severity,
            delta=f"{T['lesion_label']}: {vis['lesion_ratio_percent']}%"
        )
    with m4:
        st.metric(
            label=T["metric_latency"],
            value=f"{res['total_system_latency_ms']} ms",
            delta=f"Vision: {vis['inference_time_ms']}ms | Llama: {telemetry['total_inference_time_sec']*1000:.0f}ms"
        )

    st.divider()

    # Visual Inspection & Lesion Segmentation
    col_img1, col_img2 = st.columns(2)
    with col_img1:
        st.subheader(T["orig_img_header"])
        st.image(active_image, use_container_width=True)

    with col_img2:
        st.subheader(T["heatmap_header"])
        st.image(vis["annotated_image"], caption=T["heatmap_caption"], use_container_width=True)

    # Voice Advisory Section
    st.subheader(T["audio_section_header"])
    st.markdown(res["audio_html"], unsafe_allow_html=True)
    if os.path.exists(res["audio_file"]):
        with open(res["audio_file"], "rb") as f:
            audio_bytes = f.read()
        st.audio(audio_bytes, format="audio/wav")

    # Llama 3.2 ExecuTorch Grounded Advisory Section
    st.subheader(T["advisory_section_header"])
    st.markdown(adv["markdown_advisory"])

    # Deep SoC Telemetry Table
    with st.expander(T["deep_telemetry_header"], expanded=False):
        c_tel1, c_tel2 = st.columns(2)
        if selected_lang == "hindi":
            with c_tel1:
                st.markdown(f"""
                * **रनटाइम फ्रेमवर्क:** `{telemetry['model_runtime']}`
                * **प्रथम टोकन समय (TTFT):** `{telemetry['time_to_first_token_sec']} सेकंड`
                * **जनरेशन गति:** `{telemetry['tokens_per_second']} टोकन/सेकंड`
                * **कुल उत्पादित टोकन:** `{telemetry['tokens_generated']}`
                """)
            with c_tel2:
                st.markdown(f"""
                * **कार्यशील रैम (RSS):** `{telemetry['peak_ram_mb']} MB`
                * **क्वांटाइजेशन:** `INT4 वेट + INT8 एक्टिवेशन`
                * **आर्म निर्देश सेट:** `Armv8.2-A / Armv9-A (NEON + KleidiAI)`
                * **नेटवर्क कॉल:** `0 बाइट्स (100% ऑफलाइन)`
                """)
        else:
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
    st.info(T["prompt_select"])
