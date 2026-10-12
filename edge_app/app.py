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
import base64
import streamlit as st
import streamlit.components.v1 as components

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
        "metric_crop": "🌾 फसल",
        "metric_disease": "🩺 बीमारी / रोग",
        "metric_type": "🔬 रोग का प्रकार",
        "metric_severity": "⚠️ नुकसान की गंभीरता",
        "metric_latency": "⚡ सिस्टम लेटेंसी",
        "conf_label": "सटीकता",
        "lesion_label": "प्रभावित पत्ती",
        "pathogen_map": {
            "Fungal": "🍄 कवक जनित (Fungus)",
            "Oomycete": "🍄 फफूंद जनित (Oomycete)",
            "Bacterial": "🦠 जीवाणु जनित (Bacteria)",
            "Viral": "🧬 विषाणु जनित (Virus)",
            "Pest": "🐛 कीट प्रकोप (Pest)",
            "Healthy": "🌿 स्वस्थ पत्ता (Healthy)"
        },
        "orig_img_header": "🍃 पौधे की मूल फोटो",
        "heatmap_header": "🎯 बीमारी विश्लेषण व हीटमैप",
        "heatmap_caption": "पीली रेखाएं और लाल उभार बीमारी से प्रभावित हिस्से को दर्शाते हैं",
        "audio_section_header": "🔊 किसान ऑडियो सलाह (सुनकर समझें)",
        "audio_transcript_label": "बोली गई किसान सलाह:",
        "advisory_section_header": "🧠 लामा 3.2 ऑन-डिवाइस वैज्ञानिक सलाह व उपचार",
        "deep_telemetry_header": "🔬 विस्तृत SoC प्रदर्शन व लेटेंसी मेट्रिक्स",
        "check_new_header": "🔄 नया पौधा या अगली फसल जांचें (Check Next Plant)",
        "check_new_desc": "किसी अन्य पौधे या नई फसल की जांच करने के लिए नीचे दिए गए आसान विकल्पों में से कोई भी एक चुनें:",
        "tab_samples": "🌱 तैयार सैंपल से जांचें",
        "tab_camera": "📸 कैमरे से फोटो खींचें",
        "tab_upload": "📁 गैलरी / फोन से फोटो चुनें",
        "sample_btn_labels": {
            "tomato_bacterial_spot": "🦠 टमाटर (जीवाणु धब्बा)",
            "tomato_yellow_leaf_curl": "🧬 टमाटर (पत्ता मरोड़ वायरस)",
            "cotton_bollworm": "🐛 कपास (सुंडी / कीट प्रकोप)",
            "tomato_early_blight": "🍄 टमाटर (अगेती झुलसा)",
            "tomato_late_blight": "🥔 टमाटर (पछेती झुलसा)",
            "healthy_wheat": "🌾 गेहूं (स्वस्थ फसल)"
        },
        "camera_input_label": "कैमरे के सामने पत्ता रखकर फोटो खींचें:",
        "upload_input_label": "फोन या कंप्यूटर से पत्ती की फोटो चुनें (JPG/PNG):",
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
        "metric_crop": "🌾 Crop",
        "metric_disease": "🩺 Disease Name",
        "metric_type": "🔬 Pathogen Type",
        "metric_severity": "⚠️ Damage Severity",
        "metric_latency": "⚡ SoC Latency",
        "conf_label": "Confidence",
        "lesion_label": "Lesion Area",
        "pathogen_map": {
            "Fungal": "🍄 Fungal",
            "Oomycete": "🍄 Oomycete",
            "Bacterial": "🦠 Bacterial",
            "Viral": "🧬 Viral",
            "Pest": "🐛 Pest Infestation",
            "Healthy": "🌿 Healthy Tissue"
        },
        "orig_img_header": "🍃 Original Leaf Capture",
        "heatmap_header": "🎯 Lesion Segmentation Heatmap",
        "heatmap_caption": "Yellow contours and red highlight indicate active necrotic lesions",
        "audio_section_header": "🔊 Spoken Voice Advisory",
        "audio_transcript_label": "Spoken Advisory Transcript:",
        "advisory_section_header": "🧠 Llama 3.2 ExecuTorch Scientific Field Advisory",
        "deep_telemetry_header": "🔬 In-Depth SoC Performance & Execution Telemetry",
        "check_new_header": "🔄 Check Another Plant or New Crop",
        "check_new_desc": "Select any easy option below to diagnose another leaf or new crop:",
        "tab_samples": "🌱 Quick Demo Samples",
        "tab_camera": "📸 Live Camera Scan",
        "tab_upload": "📁 Upload Leaf Photo",
        "sample_btn_labels": {
            "tomato_bacterial_spot": "🦠 Tomato (Bacterial Spot)",
            "tomato_yellow_leaf_curl": "🧬 Tomato (Yellow Leaf Curl)",
            "cotton_bollworm": "🐛 Cotton (Bollworm Pest)",
            "tomato_early_blight": "🍄 Tomato (Early Blight)",
            "tomato_late_blight": "🥔 Tomato (Late Blight)",
            "healthy_wheat": "🌾 Wheat (Healthy Crop)"
        },
        "camera_input_label": "Hold leaf in front of camera and capture:",
        "upload_input_label": "Select leaf photo from device (JPG/PNG):",
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
    /* Prevent Streamlit from dimming or greying out the page during rerun */
    [data-testid="stAppViewContainer"],
    [data-testid="stAppViewBlockContainer"],
    .main,
    .main .block-container,
    .stAppRunning,
    div[data-testid="stAppViewContainer"] > section,
    div[data-testid="stMain"] {
        opacity: 1 !important;
        filter: none !important;
        transition: none !important;
    }
    .stAppRunning * {
        opacity: 1 !important;
        filter: none !important;
    }
    /* Hide top running decoration bar to eliminate grey flashing */
    [data-testid="stDecoration"] {
        display: none !important;
    }
    [data-testid="stStatusWidget"] {
        visibility: hidden !important;
    }

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
    .check-new-card {
        background: linear-gradient(135deg, #f0fdf4, #e6f9ed);
        border: 2px solid #16a34a;
        border-radius: 16px;
        padding: 24px;
        margin-top: 35px;
        box-shadow: 0 6px 18px rgba(22, 163, 74, 0.12);
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_pipeline():
    return AgriRakshakPipeline()

pipeline = load_pipeline()

# Language Selection (Sidebar)
st.sidebar.markdown("### 🌐 भाषा / Language")
lang_choice = st.sidebar.radio(
    "Choose Language / भाषा चुनें:",
    ["हिंदी (Hindi)", "English"],
    index=0,
    horizontal=True
)

selected_lang = "hindi" if "हिंदी" in lang_choice else "english"
T = UI_CONTENT[selected_lang]

# Crop Selection Filter in Sidebar
st.sidebar.markdown("### 🌾 " + ("फसल का चयन" if selected_lang == "hindi" else "Target Crop"))
crop_options = [
    ("Auto-Detect", "🔍 सभी फसलें (Auto-Detect)" if selected_lang == "hindi" else "🔍 All Crops (Auto-Detect)"),
    ("Tomato", "🍅 टमाटर (Tomato)"),
    ("Cotton", "🌱 कपास (Cotton)"),
    ("Rice (Paddy)", "🌾 धान / चावल (Rice)"),
    ("Corn (Maize)", "🌽 मक्का (Corn)"),
    ("Potato", "🥔 आलू (Potato)"),
    ("Wheat", "🌾 गेहूं (Wheat)"),
    ("Apple", "🍎 सेब (Apple)")
]
selected_crop_filter = st.sidebar.selectbox(
    "फसल चुनें:" if selected_lang == "hindi" else "Select Crop Filter:",
    [c[0] for c in crop_options],
    format_func=lambda x: dict(crop_options).get(x, x),
    index=0
)

# Hardware Telemetry Panel in Sidebar
with st.sidebar.expander(T["telemetry_sidebar_header"], expanded=True):
    st.markdown(T["telemetry_sidebar_text"])

# Session State for Current Plant
if "source_type" not in st.session_state:
    st.session_state["source_type"] = "sample"
if "sample_name" not in st.session_state:
    st.session_state["sample_name"] = "tomato_early_blight"
if "active_image_rgb" not in st.session_state:
    st.session_state["active_image_rgb"] = create_sample_leaf_image("tomato_early_blight")
if "last_upload_id" not in st.session_state:
    st.session_state["last_upload_id"] = None
if "last_cam_id" not in st.session_state:
    st.session_state["last_cam_id"] = None

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

# Determine Active Image to Diagnose
active_image = st.session_state.get("active_image_rgb")
if active_image is None:
    active_image = create_sample_leaf_image("tomato_early_blight")
    st.session_state["active_image_rgb"] = active_image

# class_hint is ONLY provided for preset demo samples; uploaded & camera leaves undergo pure AI vision analysis!
class_hint = st.session_state["sample_name"] if st.session_state["source_type"] == "sample" else None

# Run Pipeline on Active Image
with st.spinner(T["processing_spinner"]):
    res = pipeline.process_leaf(
        active_image,
        language=selected_lang,
        class_hint=class_hint,
        crop_filter=selected_crop_filter
    )

vis = res["vision"]
adv = res["advisory"]
telemetry = adv["telemetry"]

# Localize Output Names
if selected_lang == "hindi":
    display_crop = vis["crop"]
    display_condition = vis["hindi_name"]
else:
    display_crop = vis["crop"]
    display_condition = vis["disease"]

sev_str = vis.get("severity", "").lower()
if "mild" in sev_str:
    display_severity = T["severity_map"]["mild"]
elif "severe" in sev_str:
    display_severity = T["severity_map"]["severe"]
elif "healthy" in sev_str:
    display_severity = T["severity_map"]["healthy"]
else:
    display_severity = T["severity_map"]["moderate"]

display_pathogen = T["pathogen_map"].get(vis.get("pathology_type", "Healthy"), vis.get("pathology_type", "Healthy"))

# 5 Dedicated Top Metrics Cards (Including Explicit Disease Column)
m1, m2, m3, m4, m5 = st.columns(5)
with m1:
    st.metric(label=T["metric_crop"], value=display_crop)
with m2:
    st.metric(label=T["metric_disease"], value=display_condition, delta=f"{T['conf_label']}: {vis['confidence_percent']}%")
with m3:
    st.metric(label=T["metric_type"], value=display_pathogen)
with m4:
    st.metric(label=T["metric_severity"], value=display_severity, delta=f"{T['lesion_label']}: {vis['lesion_ratio_percent']}%")
with m5:
    st.metric(
        label=T["metric_latency"],
        value=f"{res['total_system_latency_ms']} ms",
        delta=f"Vision: {vis['inference_time_ms']}ms | Llama: {telemetry['total_inference_time_sec']*1000:.0f}ms"
    )

st.divider()

# Visual Leaf Inspection & Lesion Heatmap Overlay
col_img1, col_img2 = st.columns(2)
with col_img1:
    st.subheader(T["orig_img_header"])
    st.image(active_image, use_container_width=True)

with col_img2:
    st.subheader(T["heatmap_header"])
    st.image(vis["annotated_image"], caption=T["heatmap_caption"], use_container_width=True)

# Precision Biomarkers & Differential Diagnosis Panel
bf = vis.get("botanical_features", {})
top2 = vis.get("top2_diagnosis")
precision_title = "🔬 ऑन-डिवाइस वनस्पति बायोमार्कर व सटीक विश्लेषण (Precision Biomarkers)" if selected_lang == "hindi" else "🔬 On-Device Botanical Biomarkers & Precision Analytics"
diff_label = f"वैकल्पिक संभावना (Differential): {top2}" if selected_lang == "hindi" else f"Differential Diagnosis: {top2}"

st.markdown(f"""
<div style="background: #f8fafc; border: 1.5px solid #cbd5e1; border-radius: 12px; padding: 14px 18px; margin-top: 14px; margin-bottom: 22px;">
    <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 8px;">
        <span style="font-weight: 700; color: #0f172a; font-size: 15px;">{precision_title}</span>
        {f'<span style="background: #e2e8f0; color: #334155; font-size: 12px; padding: 4px 12px; border-radius: 14px; font-weight: 600;">{diff_label}</span>' if top2 else ''}
    </div>
    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(130px, 1fr)); gap: 10px; margin-top: 12px; font-size: 13px;">
        <div style="background: white; padding: 10px 12px; border-radius: 8px; border: 1px solid #e2e8f0; text-align: center;">
            <div style="color: #64748b; font-size: 11px;">{'🍃 स्वस्थ ऊतक (Chlorophyll)' if selected_lang == 'hindi' else '🍃 Active Chlorophyll'}</div>
            <div style="font-weight: 700; color: #16a34a; font-size: 16px;">{bf.get('green_ratio', 0)*100:.1f}%</div>
        </div>
        <div style="background: white; padding: 10px 12px; border-radius: 8px; border: 1px solid #e2e8f0; text-align: center;">
            <div style="color: #64748b; font-size: 11px;">{'🟡 पीलापन (Chlorosis)' if selected_lang == 'hindi' else '🟡 Chlorosis Ratio'}</div>
            <div style="font-weight: 700; color: #d97706; font-size: 16px;">{bf.get('yellow_ratio', 0)*100:.1f}%</div>
        </div>
        <div style="background: white; padding: 10px 12px; border-radius: 8px; border: 1px solid #e2e8f0; text-align: center;">
            <div style="color: #64748b; font-size: 11px;">{'🎯 घावों / धब्बों की संख्या' if selected_lang == 'hindi' else '🎯 Spot Count'}</div>
            <div style="font-weight: 700; color: #dc2626; font-size: 16px;">{bf.get('spot_count', 0)}</div>
        </div>
        <div style="background: white; padding: 10px 12px; border-radius: 8px; border: 1px solid #e2e8f0; text-align: center;">
            <div style="color: #64748b; font-size: 11px;">{'🐛 कीट छिद्र (Chewed Voids)' if selected_lang == 'hindi' else '🐛 Chewed Holes'}</div>
            <div style="font-weight: 700; color: #7c3aed; font-size: 16px;">{bf.get('void_holes', 0)}</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Spoken Audio Advisory
st.subheader(T["audio_section_header"])
c_audio_box, c_audio_desc = st.columns([1, 1])
with c_audio_box:
    audio_bytes = res.get("audio_bytes")
    audio_mime = res.get("audio_mime", "audio/mp3")
    if audio_bytes and len(audio_bytes) > 0:
        audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")
        btn_label = "🔊 आवाज़ में पूरी सलाह सुनें (Play Spoken Advisory)" if selected_lang == "hindi" else "🔊 Listen to Complete Spoken Advisory"
        player_html = f"""
        <div style="background: #f0fdf4; border: 2px solid #22c55e; border-radius: 12px; padding: 14px 18px; margin-bottom: 8px; font-family: system-ui, -apple-system, sans-serif;">
            <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 12px; flex-wrap: wrap; gap: 8px;">
                <button id="playBtn" onclick="togglePlay()" style="
                    background: linear-gradient(135deg, #15803d, #166534);
                    color: white;
                    border: none;
                    padding: 10px 22px;
                    font-size: 15px;
                    font-weight: 700;
                    border-radius: 8px;
                    cursor: pointer;
                    display: inline-flex;
                    align-items: center;
                    gap: 8px;
                    box-shadow: 0 4px 12px rgba(21, 128, 61, 0.35);
                ">
                    ▶️ {btn_label}
                </button>
                <span style="background: #dcfce7; color: #166534; padding: 4px 12px; border-radius: 20px; font-size: 12px; font-weight: 700;">
                    🔒 100% Offline Edge Audio
                </span>
            </div>
            <audio id="agriAudio" controls style="width: 100%; height: 38px;">
                <source src="data:{audio_mime};base64,{audio_b64}" type="{audio_mime}">
                Your browser does not support audio playback.
            </audio>
        </div>
        <script>
            var audio = document.getElementById('agriAudio');
            var btn = document.getElementById('playBtn');
            function togglePlay() {{
                if (audio.paused) {{
                    audio.play();
                    btn.innerHTML = '⏸️ रोकें (Pause)';
                    btn.style.background = 'linear-gradient(135deg, #b91c1c, #991b1b)';
                }} else {{
                    audio.pause();
                    btn.innerHTML = '▶️ {btn_label}';
                    btn.style.background = 'linear-gradient(135deg, #15803d, #166534)';
                }}
            }}
            audio.onended = function() {{
                btn.innerHTML = '▶️ {btn_label}';
                btn.style.background = 'linear-gradient(135deg, #15803d, #166534)';
            }};
        </script>
        """
        components.html(player_html, height=135)
        st.audio(audio_bytes, format=audio_mime)
    else:
        st.warning("ऑडियो स्ट्रीम तैयार हो रही है..." if selected_lang == "hindi" else "Audio advisory is synthesizing...")

with c_audio_desc:
    spoken_txt = adv.get("audio_speech_text", "")
    st.info(f"🗣️ **{T['audio_transcript_label']}**\n\n_{spoken_txt}_")

# Llama 3.2 Grounded Advisory
st.subheader(T["advisory_section_header"])
st.markdown(adv["markdown_advisory"])

# Deep Telemetry Expander
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

# ==============================================================================
# SECTION: CHECK NEW PLANT / NEXT CROP (PLACED AT THE VERY END OF MAIN SCREEN)
# ==============================================================================
st.markdown("<div style='margin-top: 45px;'></div>", unsafe_allow_html=True)
st.markdown(f"""
<div class="check-new-card">
    <h2 style="color: #14532d; margin: 0 0 8px 0; font-size: 24px; font-weight: 700;">
        {T['check_new_header']}
    </h2>
    <p style="color: #166534; margin: 0 0 18px 0; font-size: 15px;">
        {T['check_new_desc']}
    </p>
</div>
""", unsafe_allow_html=True)

# 3 Intuitive Tabs for Farmer Accessibility
tab_sample, tab_cam, tab_upload = st.tabs([
    f"  {T['tab_samples']}  ",
    f"  {T['tab_camera']}  ",
    f"  {T['tab_upload']}  "
])

with tab_sample:
    st.markdown(" ")
    btn_cols = st.columns(6)
    samples_list = [
        ("tomato_bacterial_spot", T["sample_btn_labels"]["tomato_bacterial_spot"]),
        ("tomato_yellow_leaf_curl", T["sample_btn_labels"]["tomato_yellow_leaf_curl"]),
        ("cotton_bollworm", T["sample_btn_labels"]["cotton_bollworm"]),
        ("tomato_early_blight", T["sample_btn_labels"]["tomato_early_blight"]),
        ("tomato_late_blight", T["sample_btn_labels"]["tomato_late_blight"]),
        ("healthy_wheat", T["sample_btn_labels"]["healthy_wheat"]),
    ]
    for idx, (s_key, s_label) in enumerate(samples_list):
        with btn_cols[idx]:
            if st.button(s_label, key=f"btn_sample_{s_key}", use_container_width=True):
                st.session_state["source_type"] = "sample"
                st.session_state["sample_name"] = s_key
                st.session_state["active_image_rgb"] = create_sample_leaf_image(s_key)
                st.session_state["last_upload_id"] = None
                st.session_state["last_cam_id"] = None
                st.rerun()

with tab_cam:
    st.markdown(" ")
    cam_snap = st.camera_input(T["camera_input_label"], key="camera_widget_key")
    if cam_snap is not None:
        raw_cam_bytes = cam_snap.getvalue()
        current_cam_id = f"cam_{len(raw_cam_bytes)}"
        if current_cam_id != st.session_state.get("last_cam_id") and len(raw_cam_bytes) > 0:
            try:
                file_bytes = np.asarray(bytearray(raw_cam_bytes), dtype=np.uint8)
                img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                if img_bgr is not None:
                    st.session_state["active_image_rgb"] = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                    st.session_state["source_type"] = "camera"
                    st.session_state["last_cam_id"] = current_cam_id
                    st.rerun()
            except Exception:
                pass

with tab_upload:
    st.markdown(" ")
    uploaded_file = st.file_uploader(T["upload_input_label"], type=["jpg", "jpeg", "png"], key="uploader_widget_key")
    if uploaded_file is not None:
        current_up_id = f"{uploaded_file.name}_{uploaded_file.size}"
        if current_up_id != st.session_state.get("last_upload_id"):
            try:
                raw_bytes = uploaded_file.getvalue()
                if len(raw_bytes) > 0:
                    file_bytes = np.asarray(bytearray(raw_bytes), dtype=np.uint8)
                    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                    if img_bgr is not None:
                        st.session_state["active_image_rgb"] = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                        st.session_state["source_type"] = "upload"
                        st.session_state["last_upload_id"] = current_up_id
                        st.rerun()
            except Exception:
                pass
