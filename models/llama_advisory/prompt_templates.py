"""
AgriRakshak-Edge: Prompt Engineering Matrix for Llama 3.2 1B-Instruct
Engineered for edge deployment via ExecuTorch on Arm Cortex-A.
"""

from typing import Dict, Any

SYSTEM_PROMPT_HINDI = """आप 'कृषि रक्षक' (AgriRakshak AI) हैं - एक अत्यधिक सटीक और अनुभवी कृषि वैज्ञानिक एवं सलाहकार।
आपका कार्य किसान को उनके पौधे की बीमारी/कीट की पहचान, नुकसान की गंभीरता (Severity) और भारतीय कृषि अनुसंधान परिषद (ICAR) के मानकों के अनुसार सटीक, चरणबद्ध और सुरक्षित उपाय बताना है।
आपको जवाब सरल, स्पष्ट और व्यावहारिक हिंदी में देना है।

नियम:
1. हमेशा जैविक (ऑर्गेनिक) और रासायनिक (केमिकल) दोनों उपाय अलग-अलग बताएं।
2. प्रति लीटर पानी में सटीक दवा की मात्रा (Dosage) लिखें।
3. फसल की कटाई से पहले प्रतीक्षा अवधि (Pre-Harvest Interval) और सुरक्षा सावधानियां ज़रूर बताएं।
"""

SYSTEM_PROMPT_ENGLISH = """You are 'AgriRakshak AI' - an expert agronomic scientist and edge advisor for farmers.
Your task is to analyze crop disease/pest diagnostics, assess damage severity, and generate actionable, scientifically grounded recommendations aligned with ICAR/CIBRC standards.
Provide clear, structured, and farmer-friendly advice.

Rules:
1. Always offer distinct Organic/Biocontrol and Chemical Intervention strategies.
2. Provide exact dosage per liter of water.
3. State Pre-Harvest Interval (PHI) and critical application precautions.
"""

def build_llama_advisory_prompt(diagnostic_data: Dict[str, Any], language: str = "hindi") -> str:
    """
    Constructs the prompt for Llama 3.2 1B-Instruct including diagnosis, lesion area, and agronomic context.
    """
    crop = diagnostic_data.get("crop", "Unknown")
    disease = diagnostic_data.get("disease", "Unknown")
    severity = diagnostic_data.get("severity", "Moderate")
    lesion_ratio = diagnostic_data.get("lesion_ratio_percent", 20.0)
    confidence = diagnostic_data.get("confidence_percent", 90.0)
    context_remedy = diagnostic_data.get("kb_context", {})

    if language.lower() == "hindi":
        prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
{SYSTEM_PROMPT_HINDI}<|eot_id|>
<|start_header_id|>user<|end_header_id|>
मेरे खेत की फसल का डायग्नोसिस यह है:
- फसल (Crop): {crop}
- पहचानी गई बीमारी/कीट: {disease}
- पहचान सटीकता (Confidence): {confidence:.1f}%
- पत्ते पर नुकसान का स्तर (Severity): {severity} (पत्ते का {lesion_ratio:.1f}% भाग प्रभावित)
- ICAR संदर्भ रिकॉर्ड: {context_remedy.get('dosage_hindi', 'उपलब्ध नहीं')}

कृपया मुझे तुरंत 4 बिंदुओं में स्पष्ट सलाह दें:
1. समस्या का त्वरित विवरण (यह क्या है और क्यों फैलती है)
2. जैविक/देसी उपचार (Organic Solution)
3. रासायनिक उपचार एवं सटीक मात्रा (Chemical Remedy with exact dose per liter)
4. सिंचाई व सुरक्षा सलाह (Irrigation & Precaution)
<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
"""
    else:
        prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
{SYSTEM_PROMPT_ENGLISH}<|eot_id|>
<|start_header_id|>user<|end_header_id|>
Here is the edge diagnostic payload for the inspected crop:
- Target Crop: {crop}
- Identified Pathology/Pest: {disease}
- Classifier Confidence: {confidence:.1f}%
- Severity Level: {severity} ({lesion_ratio:.1f}% surface area damaged)
- Certified Agronomic Reference: {context_remedy.get('organic_treatment', 'Standard ICAR guidelines apply')}

Provide concise, structured guidance covering:
1. Pathology Overview & Vector/Condition triggers
2. Organic & Biocontrol Strategy (Immediate step)
3. Chemical Rescue Intervention (Active ingredient + Dosage per liter of water)
4. Irrigation adjustment, PHI days, and PPE Safety Protocol
<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
"""
    return prompt
