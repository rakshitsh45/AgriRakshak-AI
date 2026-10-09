"""
AgriRakshak-Edge: End-to-End Pipeline Orchestrator
Coordinates Edge Vision, Lesion Severity Estimation, Llama ExecuTorch Reasoning,
and Multilingual Voice Synthesis on Arm SoC.
"""

import os
import sys
import time
import logging
from typing import Dict, Any, Optional

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from engine.vision_infer import EdgeVisionEngine, create_sample_leaf_image
from engine.advisory_engine import LlamaExecuTorchAdvisoryEngine
from engine.voice_synthesizer import EdgeVoiceSynthesizer

logger = logging.getLogger("AgriPipeline")

class AgriRakshakPipeline:
    def __init__(self):
        logger.info("Initializing AgriRakshak-Edge unified SoC pipeline...")
        self.vision_engine = EdgeVisionEngine()
        self.advisory_engine = LlamaExecuTorchAdvisoryEngine()
        self.voice_synthesizer = EdgeVoiceSynthesizer()

    def process_leaf(
        self,
        image_input: Any,
        language: str = "hindi",
        class_hint: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Executes complete edge diagnostic workflow:
        1. Leaf pathology detection + lesion segmentation + severity quantification
        2. Grounded Llama 3.2 ExecuTorch reasoning
        3. Multilingual audio synthesis
        """
        t_start = time.perf_counter()

        # Phase 1: Edge Vision Pass
        vision_result = self.vision_engine.predict(image_input, class_hint=class_hint)

        # Phase 2: Llama Reasoning & ICAR Grounding Pass
        advisory_result = self.advisory_engine.generate_advisory(
            vision_result,
            language=language
        )

        # Phase 3: Voice Advisory Synthesis
        audio_file, audio_bytes = self.voice_synthesizer.generate_offline_audio(
            advisory_result["audio_speech_text"],
            lang="hi" if language == "hindi" else "en"
        )
        audio_html = self.voice_synthesizer.get_html5_speech_js(
            advisory_result["audio_speech_text"],
            lang="hi-IN" if language == "hindi" else "en-US"
        )

        total_latency_ms = (time.perf_counter() - t_start) * 1000.0

        return {
            "vision": vision_result,
            "advisory": advisory_result,
            "audio_file": audio_file,
            "audio_bytes": audio_bytes,
            "audio_html": audio_html,
            "total_system_latency_ms": round(total_latency_ms, 2)
        }

if __name__ == "__main__":
    pipeline = AgriRakshakPipeline()
    sample_img = create_sample_leaf_image("tomato_early_blight")
    res = pipeline.process_leaf(sample_img, language="hindi")
    print(f"Pipeline executed successfully in {res['total_system_latency_ms']} ms")
    print(f"Crop: {res['vision']['crop']} | Disease: {res['vision']['disease']} | Severity: {res['vision']['severity']}")
    print(f"Organic: {res['advisory']['organic_solution']}")
    print(f"Chemical: {res['advisory']['chemical_solution']}")
