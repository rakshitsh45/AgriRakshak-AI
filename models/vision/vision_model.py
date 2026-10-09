"""
AgriRakshak-Edge: MobileNetV4-Agri Lightweight Edge Classifier
Designed for ultra-low latency inference on Arm Cortex-A and edge NPUs.
"""

import json
import os
from typing import List, Dict, Any

class MobileNetV4AgriConfig:
    INPUT_SIZE = (224, 224)
    NUM_CLASSES = 34
    BACKBONE = "MobileNetV4-Conv-Small"
    QUANTIZATION = "INT8 Post-Training Quantization"
    TARGET_LATENCY_ARM_MS = 14.5

def get_class_labels() -> List[Dict[str, Any]]:
    """Loads class definitions from labels.json."""
    labels_file = os.path.join(os.path.dirname(__file__), "labels.json")
    if os.path.exists(labels_file):
        with open(labels_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data.get("classes", [])
    return []
