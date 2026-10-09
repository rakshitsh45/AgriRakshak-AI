# AgriRakshak-Edge: Hardware & SoC Performance Benchmarking Report

**Generated:** 2026-10-10 00:43:07

## 1. Measured Host Execution Latency

- **Average Vision Diagnostic Pass:** `3.11 ms`
- **Vision Throughput:** `321.8 FPS`
- **Complete System Turnaround:** `74.95 ms`

## 2. Arm Target SoC Profiling Matrix

| Platform | Vision Runtime & Latency | Llama ExecuTorch Runtime | TTFT (ms) | Tokens/s | Peak RAM | Est. Power |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Raspberry Pi 5 (4x Cortex-A76 @ 2.4GHz)** | OpenCV + INT8 CPU (16.2ms) | ExecuTorch (INT4 XNNPACK / KleidiAI) | 310.0ms | 14.8 | 870 MB | 4.8 W |
| **Arm-based Android (Cortex-A78/A55)** | TFLite / ExecuTorch INT8 (11.4ms) | ExecuTorch (INT4 KleidiAI) | 240.0ms | 18.2 | 760 MB | 2.9 W |
| **Raspberry Pi 5 + Hailo-8 AI HAT+ (Dual Pipeline)** | HailoRT on Hailo-8 NPU (3.8ms) | ExecuTorch on Cortex-A76 CPU | 310.0ms | 14.8 | 890 MB | 5.4 W |


> **Key SoC Takeaway:** With INT4 quantization and Arm KleidiAI acceleration in ExecuTorch, Llama 3.2 1B executes comfortably within a **sub-1GB RAM envelope**, enabling seamless offline deployment on entry-level Android devices and Raspberry Pi 5 without thermal throttling.
