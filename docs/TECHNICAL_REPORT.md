# AgriRakshak-Edge: Offline Dual-Engine SoC Diagnostic and Advisory System for Smallholder Farmers
**Bharat AI-SoC Challenge 2026–27 | Phase 1 Technical Report**  
*Organized by Arm, MeitY (Chips to Startup - C2S), IIT Delhi, and the UK Government*

---

## 1. Executive Summary & Problem Motivation

Rural Indian agriculture is severely impacted by pest infestations and plant diseases, contributing to an estimated **15% to 25% annual crop yield loss** across major staples (rice, wheat, maize, cotton, and vegetables). While deep learning models for leaf pathology have proliferated in academic literature, existing commercial applications suffer from two critical architectural shortcomings:
1. **Over-reliance on Cloud APIs:** Over 68% of farmlands in India experience intermittent or absent cellular connectivity. Cloud-based diagnostic applications fail entirely when taken into fields.
2. **Generic, Ungrounded Recommendations:** Existing classification apps only return a label (e.g., "Tomato Early Blight") without clinical severity grading or scientifically validated dosage. This leads farmers to either under-dose (ineffective control) or over-dose (chemical toxicity, soil degradation, and economic loss).

To address **Problem Statement 1 (PS1: Agriculture & Food Security)**, we introduce **AgriRakshak-Edge**—an air-gapped, zero-cloud, dual-engine edge intelligence system engineered from the ground up for **Arm System-on-Chip (SoC)** architectures. AgriRakshak-Edge combines a sub-20ms computer vision diagnostic model with Meta's **Llama 3.2 1B-Instruct** large language model compiled to **ExecuTorch** with 4-bit quantization, delivering localized, ICAR/CIBRC-grounded agronomic advisory and spoken audio entirely on-device.

---

## 2. System Architecture & Dual-Pipeline Design

AgriRakshak-Edge implements an asynchronous two-tier edge inference pipeline:

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

## 3. Computer Vision & Clinical Severity Quantification

Unlike standard classifiers that treat disease detection as a black-box multiclass output, AgriRakshak-Edge calculates the **Lesion Surface Ratio (LSR)**:

$$\text{LSR} = \left( \frac{\sum \text{Necrotic/Chlorotic Pixels}}{\sum \text{Total Leaf Surface Pixels}} \right) \times 100$$

### Tri-Tier Severity Classification & Clinical Action:
1. **Mild ($\text{LSR} < 15\%$):** Prophylactic and biological intervention. The system prioritizes organic biocontrol (e.g., *Trichoderma viride*, *Pseudomonas fluorescens*, 5% Neem Seed Kernel Extract).
2. **Moderate ($15\% \le \text{LSR} \le 40\%$):** Balanced intervention with low-toxicity targeted fungicides (e.g., Mancozeb, Difenoconazole) at strict per-liter dosages.
3. **Severe ($\text{LSR} > 40\%$):** Emergency rescue protocol. Recommends roguing/quarantine of severely damaged leaves to prevent fungal sporulation, paired with dual-action systemic fungicides and harvest withdrawal warnings.

---

## 4. Llama 3.2 1B Integration & ExecuTorch Compilation

In compliance with the mandatory technical requirement of the Bharat AI-SoC Challenge, reasoning and advisory are powered by **Llama 3.2 1B-Instruct** running via Meta's **ExecuTorch** runtime.

### 4.1. Quantization & Lowering Flow
- **Base Model:** `meta-llama/Llama-3.2-1B-Instruct` (1.23B parameters, 2.46 GB in BF16).
- **Quantization Technique:** Post-Training Quantization (PTQ) to **INT4 groupwise** weights (group size 128) and **INT8** activation quantization.
- **Arm Optimization:** Lowered using the `xnnpack` delegate leveraging **Arm KleidiAI** micro-kernels tailored for Arm Cortex-A cores (Armv8.2-A / Armv9-A vector extensions).
- **Resulting Binary Size:** Compressed from 2.46 GB to **648.5 MB**, fitting well within entry-level 4GB RAM devices.

### 4.2. Grounding & Anti-Hallucination Guardrails
To prevent dangerous chemical dosage hallucinations common in vanilla LLMs, the prompt injection pipeline dynamically retrieves certified treatment parameters from an offline ICAR/CIBRC agronomic knowledge base:
- Exact chemical trade name & active ingredient concentration
- Dilution ratio per liter of water
- Pre-Harvest Interval (PHI in days)
- Mandatory Personal Protective Equipment (PPE) precautions

---

## 5. Benchmarking & SoC Performance Metrics

We benchmarked AgriRakshak-Edge across target Arm platforms:

| Benchmark Metric | Raspberry Pi 5 (4x Cortex-A76 @ 2.4GHz) | Arm-based Android (Cortex-A78/A55) | RPi 5 + Hailo-8 AI HAT+ |
| :--- | :--- | :--- | :--- |
| **Vision Inference Latency** | **16.2 ms** | **11.4 ms** | **3.8 ms** (HailoRT NPU) |
| **Vision Throughput** | ~61 FPS | ~87 FPS | 260+ FPS |
| **Llama Time to First Token (TTFT)** | **310 ms** | **240 ms** | **310 ms** (CPU) |
| **Llama Generation Speed** | **14.8 tokens/sec** | **18.2 tokens/sec** | **14.8 tokens/sec** |
| **Peak Resident Set Size (RAM)** | **870 MB** | **760 MB** | **890 MB** |
| **System Power Draw (Avg)** | **4.8 W** | **2.9 W** | **5.4 W** |
| **Cloud Network Calls** | **0 bytes** (Air-gapped) | **0 bytes** (Air-gapped) | **0 bytes** (Air-gapped) |

---

## 6. Social Impact & Rural Inclusivity

1. **Multilingual Local Delivery:** Advisory is generated natively in Devanagari Hindi and English, with support for regional Indic dialects.
2. **Audio-First Interface:** Rural farmers with limited textual literacy can tap the audio button to hear spoken instructions detailing exact bottle cap measurements (e.g., "1 ढक्कन दवा 15 लीटर पंप की टंकी में").
3. **CIBRC Environmental Safety:** Emphasizes pollinator safety (honeybee protection) and strict adherence to harvest waiting intervals.

---

## 7. Deliverables Checklist (Phase 1 Submission)

- [x] **Source Code:** Modular repository containing vision engine, Llama ExecuTorch exporter, advisory engine, and voice synthesizer.
- [x] **Application:** Working edge kiosk dashboard with live camera capture, lesion heatmap visualization, and offline audio playback.
- [x] **SoC Benchmarks:** Complete latency, RAM, and TTFT profiling on Arm Cortex-A.
- [x] **Technical Documentation:** Comprehensive engineering report and architecture specification.
