# app/integration/strands/agents/crop_health_agent.py
from typing import Dict, Any
import logging
from ..base_agent import BaseAgent
from ...tools.crop_health_tools import analyze_crop_image, get_disease_treatment_info, validate_crop_image

logger = logging.getLogger(__name__)


class CropHealthAgent(BaseAgent):
    """Specialized agent for crop disease diagnosis using AI vision"""
    
    def __init__(self):
        super().__init__(
            name="crop_health_specialist",
            description="Expert in crop disease diagnosis using AI vision analysis and providing practical treatment solutions",
            instruction="""You are an expert agricultural pathologist specializing in crop disease diagnosis for Indian farmers.

Your responsibilities:
1. Validate uploaded images are of crops/plants
2. Analyze crop images using AI vision
3. Identify diseases, pests, and nutrient deficiencies
4. Provide immediate treatment recommendations
5. Suggest locally available medicines and remedies
6. Offer preventive measures

Always respond in Hindi with practical, affordable solutions.
Prioritize treatments available in local agricultural stores.""",
            tools=[validate_crop_image, analyze_crop_image, get_disease_treatment_info]
        )
    
    async def process(self, message: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Process crop health queries"""
        try:
            logger.info(f"Crop health agent processing: {message[:100]}")
            
            # Check if image data is available
            image_data = context.get('image_data')
            location = context.get('user_location', 'India')
            
            tools_called = []
            diagnosis_data = {}
            
            if image_data:
                # Analyze image
                analysis_result = await self.invoke_tool(
                    'analyze_crop_image',
                    image_data=image_data,
                    location=location
                )
                diagnosis_data['analysis'] = analysis_result
                tools_called.append('analyze_crop_image')
                
                # Check if image validation failed
                if analysis_result.get('status') == 'invalid_image':
                    validation = analysis_result.get('validation', {})
                    return {
                        "status": "invalid_image",
                        "response": validation.get('hindi_message', 'यह फसल की तस्वीर नहीं है। कृपया फसल/पौधे की तस्वीर भेजें।'),
                        "tools_called": tools_called,
                        "data": diagnosis_data
                    }
                
                # If disease identified, get treatment info
                if analysis_result.get('status') == 'success':
                    analysis = analysis_result.get('analysis', {})
                    
                    # Handle both dict and string responses
                    if isinstance(analysis, dict):
                        diagnosis = analysis.get('diagnosis', {})
                        disease_name = diagnosis.get('primary_issue', '')
                        severity = diagnosis.get('severity_level', 'medium')
                    else:
                        # Fallback if analysis is a string
                        disease_name = ''
                        severity = 'medium'
                        logger.warning(f"Analysis returned as string: {str(analysis)[:100]}")
                    
                    if disease_name and disease_name != 'Analysis completed':
                        treatment_result = await self.invoke_tool(
                            'get_disease_treatment_info',
                            disease_name=disease_name,
                            severity=severity,
                            location=location,
                            farmer_budget='medium'
                        )
                        diagnosis_data['treatment'] = treatment_result
                        tools_called.append('get_disease_treatment_info')
            
            # Generate comprehensive response
            if diagnosis_data:
                response_prompt = f"""Based on this crop diagnosis, provide comprehensive advice in Hindi.

User Query: "{message}"
Location: {location}
Diagnosis Data: {diagnosis_data}

Provide:
1. Disease/pest identification
2. Severity assessment
3. Immediate actions to take
4. Treatment options (chemical and organic)
5. Where to buy medicines locally
6. Preventive measures

Keep response practical and actionable (max 300 words).
Use simple Hindi that farmers can understand."""
                
                response = await self.invoke_llm(response_prompt, temperature=0.7)
            else:
                response = """कृपया अपनी फसल की तस्वीर भेजें ताकि मैं बीमारी की पहचान कर सकूं।

तस्वीर लेते समय ध्यान दें:
- प्रभावित पत्ती/भाग को नजदीक से दिखाएं
- अच्छी रोशनी में फोटो लें
- साफ और स्पष्ट तस्वीर हो

मैं AI से विश्लेषण करके आपको तुरंत इलाज बताऊंगा।"""
            
            return {
                "status": "success",
                "response": response,
                "tools_called": tools_called,
                "data": diagnosis_data
            }
            
        except Exception as e:
            logger.error(f"Error in crop health agent: {e}", exc_info=True)
            return {
                "status": "error",
                "response": "मुझे खेद है, फसल विश्लेषण में समस्या हुई। कृपया दोबारा कोशिश करें।",
                "tools_called": []
            }


def create_crop_health_agent() -> CropHealthAgent:
    """Factory function to create crop health agent"""
    return CropHealthAgent()
