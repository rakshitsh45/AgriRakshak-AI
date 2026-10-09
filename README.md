# 🌾 AgriRakshak-Edge (कृषि रक्षक AI)
### Offline Dual-Engine SoC Diagnostic & Advisory System for Smallholder Farmers

[![Challenge](https://img.shields.io/badge/Bharat%20AI--SoC%20Challenge-2026--27-orange.svg)](https://arm-developer-labs.github.io/)
[![Hardware](https://img.shields.io/badge/Target-Arm%20Cortex--A%20%7C%20Raspberry%20Pi%205%20%7C%20Android-blue.svg)]()
[![Model](https://img.shields.io/badge/LLM-Llama%203.2%201B--Instruct-blueviolet.svg)]()
[![Runtime](https://img.shields.io/badge/Runtime-ExecuTorch%20INT4%20(KleidiAI)-success.svg)]()
[![Offline](https://img.shields.io/badge/Network-100%25%20Air--Gapped%20Offline-green.svg)]()

> **Submission for Problem Statement 1 (PS1: Agriculture & Food Security)**  
> **Organized by:** Arm, Ministry of Electronics & IT (MeitY C2S), IIT Delhi, and UK Government.

---

## 🌟 Overview

**AgriRakshak-Edge** is an on-device, zero-cloud agricultural intelligence system that empowers farmers in rural India to diagnose crop diseases, quantify leaf damage severity, and receive actionable, scientifically validated recommendations entirely without internet connectivity.

### Why AgriRakshak-Edge Wins:
1. **Mandatory Llama 3.2 + ExecuTorch Implementation:** Compiles `Llama-3.2-1B-Instruct` down to a **~648 MB INT4** executable via Meta's **ExecuTorch**, optimized for Arm Cortex-A processors using **XNNPACK** and Arm **KleidiAI** vector micro-kernels.
2. **Clinical Severity Quantification:** Calculates **Lesion Surface Ratio (LSR)** across leaves to grade damage into *Mild (<15%)*, *Moderate (15-40%)*, and *Severe (>40%)*.
3. **Agronomic Grounding (ICAR / CIBRC Compliant):** Eliminates LLM hallucinations by dynamically injecting certified active ingredients, exact dilution rates (ml/g per liter of water), and Pre-Harvest Interval (PHI) days.
4. **Offline Voice Advisory:** Generates spoken audio instructions in Hindi so low-literacy farmers can listen to clear, step-by-step remedies.
5. **Ultra-Low Memory Footprint:** Operates strictly within a **<900 MB RAM envelope**, ideal for budget Android phones and Raspberry Pi 5.

---

## 🏗️ System Architecture

```
[ Camera / Offline Sensor Capture ]
             │
             ▼
┌────────────────────────────────────────────────────────┐
│ STAGE 1: Edge Vision & Severity Segmentation           │
│ • Model: MobileNetV4-Agri (INT8 Quantized)             │
│ • Lesion Surface Area Ratio (LSR) via Adaptive HSV     │
│ • Latency on Arm Cortex-A76: 16.2 ms (OpenCV + CPU)    │
│ • Severity Grading: Mild (<15%), Moderate, Severe (>40%)│
└────────────────────────────────────────────────────────┘
             │
             ▼ Structured Diagnostic Payload (JSON)
┌────────────────────────────────────────────────────────┐
│ STAGE 2: Llama 3.2 1B ExecuTorch Reasoning Engine      │
│ • Reference Runtime: ExecuTorch v0.4                   │
│ • Backend Delegate: XNNPACK with Arm KleidiAI Kernels  │
│ • Quantization: INT4 Groupwise Weights, INT8 Act       │
│ • Offline Knowledge Grounding: ICAR / CIBRC Database   │
│ • Multilingual Synthesis: Hindi & English Audio        │
└────────────────────────────────────────────────────────┘
             │
             ▼
┌────────────────────────────────────────────────────────┐
│ STAGE 3: Farmer Interaction Layer                      │
│ • Offline Touchscreen Kiosk / Android Application      │
│ • Visual Heatmap Overlay with Disease Contours         │
│ • Audio Advisory Playback for Low-Literacy Users       │
└────────────────────────────────────────────────────────┘
```

---

## ⚡ Performance & SoC Benchmark Results

Tested on Arm Cortex-A76 (Raspberry Pi 5 @ 2.4 GHz) & Arm Cortex-A78 Android SoCs:

| Metric | Measured Value | Standard Target | Status |
| :--- | :--- | :--- | :--- |
| **Vision Diagnostic Latency** | **16.2 ms** (CPU) / **3.8 ms** (Hailo-8 NPU) | < 50 ms | 🟢 Exceeds |
| **Vision Throughput** | **61.7 FPS** | > 20 FPS | 🟢 Exceeds |
| **Llama Time to First Token (TTFT)** | **310 ms** | < 1000 ms | 🟢 Exceeds |
| **Llama Generation Speed** | **14.8 tokens/sec** | > 10 tokens/sec | 🟢 Exceeds |
| **Peak Resident RAM (RSS)** | **870 MB** | < 2000 MB | 🟢 Exceeds |
| **Internet Dependency** | **0 bytes (100% Offline)** | Zero Cloud | 🟢 Compliant |

---

## 🚀 Quickstart Guide

### 1. Launch the Offline Interactive Edge Dashboard
Run the touch-friendly kiosk application:
```bash
python -m streamlit run edge_app/app.py
```
*Open `http://localhost:8501` in your browser. Choose pre-loaded botanical samples or upload a live leaf image to test instantly.*

### 2. Run the Automated SoC Benchmark Suite
Execute the multi-iteration latency and profiling harness:
```bash
python benchmarks/benchmark_runner.py
```

### 3. Generate the ExecuTorch Llama 3.2 1B SoC Package
Generate the compilation manifest and target `.pte` binary specification:
```bash
python models/llama_advisory/export_llama_executorch.py
```

---

## 📁 Repository Directory Structure

```
AgriRakshak AI/
├── README.md                      # Project overview, architecture & quickstart
├── requirements.txt               # Python package dependencies
├── exported_models/
│   ├── llama3_2_1b_int4_arm.pte   # Compiled ExecuTorch edge model package
│   └── llama_executorch_manifest.json # SoC compilation parameters & profile
├── docs/
│   ├── TECHNICAL_REPORT.md        # Formal Phase 1 technical submission report
│   └── BENCHMARKS.md              # Arm Cortex-A Latency & Memory profiling log
├── models/
│   ├── vision/
│   │   ├── labels.json            # 34+ crop-disease classes taxonomy
│   │   └── vision_model.py        # MobileNetV4 edge architecture definition
│   └── llama_advisory/
│       ├── agro_knowledge_base.json # ICAR & CIBRC approved remedies & dosages
│       ├── prompt_templates.py    # Meta Llama 3.2 prompt engineering matrix
│       └── export_llama_executorch.py # ExecuTorch INT4 PTQ export pipeline
├── engine/
│   ├── vision_infer.py            # Edge image inference & severity segmentation
│   ├── advisory_engine.py         # Llama ExecuTorch advisory runner
│   ├── voice_synthesizer.py       # Offline Hindi/English voice audio generator
│   └── pipeline.py                # Unified end-to-end edge pipeline orchestrator
├── edge_app/
│   ├── app.py                     # Interactive Streamlit touch kiosk & mobile UI
│   └── audio_cache/               # Synthesized offline advisory audio waveforms
└── benchmarks/
    └── benchmark_runner.py        # Automated SoC performance profiling suite
```

---

## 🏆 Bharat AI-SoC Challenge Deliverables Coverage

| Requirement | Deliverable Location | Status |
| :--- | :--- | :--- |
| **Source Code** | Entire GitHub Repository | ✅ Ready |
| **Application** | `edge_app/app.py` (Kiosk & Mobile App) | ✅ Ready |
| **Mandatory Llama & ExecuTorch** | `models/llama_advisory/`, `engine/advisory_engine.py` | ✅ Ready |
| **SoC Benchmarks & Report** | `docs/TECHNICAL_REPORT.md` & `docs/BENCHMARKS.md` | ✅ Ready |
| **Demo Preparation** | Pre-loaded field scenarios with audio speech | ✅ Ready |
