import base64
import io
from typing import Dict, Any, Optional
import logging
from PIL import Image
import os
import json

logger = logging.getLogger(__name__)

# Initialize Bedrock service
try:
    from app.google_adk_integration.aws_services import BedrockService
    bedrock_service = BedrockService()
    logger.info("✅ AWS Bedrock service initialized for crop health analysis")
except Exception as e:
    bedrock_service = None
    logger.error(f"❌ Failed to initialize Bedrock service: {e}")


class CropHealthService:
    """AI-powered service for crop disease diagnosis using AWS Bedrock Claude"""

    def __init__(self):
        self.bedrock = bedrock_service

    async def analyze_crop_image(
            self,
            image_data: str,
            location: Optional[str] = None,
            crop_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Analyze crop image using Claude Vision for disease/pest detection

        Args:
            image_data (str): Base64 encoded image data
            location (str, optional): Farmer's location for localized advice
            crop_type (str, optional): Type of crop for specific analysis

        Returns:
            Dict: Disease diagnosis and initial recommendations
        """
        try:
            if not self.bedrock:
                return {
                    "status": "error",
                    "message": "AI विश्लेषण सेवा उपलब्ध नहीं है। कृपया AWS Bedrock को कॉन्फ़िगर करें।"
                }

            # Decode and validate image
            try:
                if ',' in image_data:
                    image_data = image_data.split(',')[1]

                image_bytes = base64.b64decode(image_data)
                image = Image.open(io.BytesIO(image_bytes))

                # Validate image size
                if image.size[0] < 100 or image.size[1] < 100:
                    return {
                        "status": "error",
                        "message": "तस्वीर बहुत छोटी है। कृपया बेहतर quality की फोटो लें।"
                    }

            except Exception as e:
                logger.error(f"Image processing error: {e}")
                return {
                    "status": "error",
                    "message": "तस्वीर को process नहीं कर सकते। कृपया दूसरी फोटो try करें।"
                }

            # Create analysis prompt
            analysis_prompt = self._create_analysis_prompt(location, crop_type)

            # Call Claude via Bedrock
            try:
                result = await self.bedrock.analyze_crop_image_with_claude(
                    image_base64=image_data,
                    prompt=analysis_prompt
                )

                if result["status"] == "success":
                    analysis_text = result["text"]
                    analysis_result = self._parse_analysis_response(analysis_text)
                else:
                    return {
                        "status": "error",
                        "message": f"AI विश्लेषण में त्रुटि: {result.get('message')}"
                    }

            except Exception as e:
                logger.error(f"Bedrock API error: {e}")
                return {
                    "status": "error",
                    "message": f"AI विश्लेषण में त्रुटि: {str(e)}"
                }

            return {
                "status": "success",
                "analysis": analysis_result,
                "location": location,
                "crop_type": crop_type,
                "raw_ai_response": analysis_text,
                "timestamp": "analysis_completed"
            }

        except Exception as e:
            logger.error(f"Image analysis error: {e}")
            return {
                "status": "error",
                "message": f"तस्वीर का विश्लेषण नहीं हो सका: {str(e)}"
            }

    async def get_treatment_recommendations(
            self,
            disease_name: str,
            crop_type: Optional[str] = None,
            severity: str = "medium",
            location: Optional[str] = None,
            farmer_budget: Optional[str] = "medium"
    ) -> Dict[str, Any]:
        """Get AI-powered treatment recommendations using Claude"""
        try:
            if not self.bedrock:
                return {
                    "status": "error",
                    "message": "AI सेवा उपलब्ध नहीं है।"
                }

            treatment_prompt = self._create_treatment_prompt(
                disease_name, crop_type, severity, location, farmer_budget
            )

            try:
                result = await self.bedrock.generate_text(
                    prompt=treatment_prompt,
                    max_tokens=2048
                )

                if result["status"] == "success":
                    treatment_text = result["text"]
                    treatment_result = self._parse_treatment_response(treatment_text, disease_name)
                else:
                    return {
                        "status": "error",
                        "message": f"उपचार सुझाव प्राप्त करने में त्रुटि: {result.get('message')}"
                    }

            except Exception as e:
                logger.error(f"Bedrock API error: {e}")
                return {
                    "status": "error",
                    "message": f"उपचार सुझाव प्राप्त करने में त्रुटि: {str(e)}"
                }

            return {
                "status": "success",
                "treatment_plan": treatment_result,
                "disease_name": disease_name,
                "location": location,
                "severity": severity,
                "raw_ai_response": treatment_text
            }

        except Exception as e:
            logger.error(f"Treatment recommendation error: {e}")
            return {
                "status": "error",
                "message": f"उपचार जानकारी प्राप्त करने में त्रुटि: {str(e)}"
            }

    def _create_analysis_prompt(self, location: Optional[str], crop_type: Optional[str]) -> str:
        """Create comprehensive analysis prompt for Claude"""
        return f"""
        You are an expert agricultural pathologist specializing in Indian crop diseases. Analyze this crop image and provide a detailed, practical diagnosis.

        Context:
        - Location: {location or 'India (unspecified region)'}
        - Expected crop: {crop_type or 'Please identify from image'}
        - Farmer needs: Practical, affordable, locally available solutions in India

        Please analyze for:
        1. Disease identification (fungal, bacterial, viral, physiological)
        2. Pest infestation (insects, mites, nematodes)
        3. Nutrient deficiency symptoms
        4. Environmental stress indicators
        5. Overall plant health assessment

        Provide response in this JSON format:
        {{
            "diagnosis": {{
                "primary_issue": "Main disease/pest name in English",
                "hindi_name": "Disease/pest name in Hindi",
                "scientific_name": "Scientific name if applicable",
                "confidence_percentage": 85,
                "severity_level": "low/medium/high/critical",
                "crop_identified": "Identified crop type",
                "affected_plant_parts": ["leaves", "stem", "roots", "fruits"]
            }},
            "symptoms_observed": [
                "Specific visible symptoms you can see",
                "Color changes, spots, wilting, etc."
            ],
            "immediate_actions": [
                "तुरंत करने योग्य काम - in Hindi",
                "Immediate steps to prevent spread"
            ],
            "risk_assessment": {{
                "spread_risk": "low/medium/high",
                "potential_crop_loss": "5-10% / 20-30% / 50%+",
                "urgency_level": "immediate/urgent/moderate/low",
                "best_treatment_window": "next 24 hours / 3 days / 1 week"
            }},
            "next_steps": [
                "What farmer should do next",
                "When to seek expert help"
            ]
        }}

        Be very specific about visible symptoms. Focus on actionable advice that an Indian farmer can immediately implement.
        """

    def _create_treatment_prompt(
            self,
            disease_name: str,
            crop_type: Optional[str],
            severity: str,
            location: Optional[str],
            farmer_budget: Optional[str]
    ) -> str:
        """Create comprehensive treatment recommendation prompt"""
        return f"""
        You are an expert agricultural advisor specializing in practical, affordable crop disease management for Indian farmers.

        Provide comprehensive treatment recommendations for:
        - Disease/Pest: {disease_name}
        - Crop: {crop_type or 'general crop'}
        - Severity: {severity}
        - Location: {location or 'India'}
        - Farmer budget: {farmer_budget or 'medium'}

        Provide response in this JSON format:
        {{
            "disease_info": {{
                "name_hindi": "Disease name in Hindi",
                "name_english": "{disease_name}",
                "severity_impact": "What this severity means for the farmer",
                "expected_timeline": "How long treatment will take"
            }},
            "immediate_treatment": {{
                "chemical_options": [
                    {{
                        "product_name": "Specific fungicide/pesticide name available in India",
                        "active_ingredient": "Chemical name",
                        "dosage": "X ml/gram per liter water",
                        "application_method": "Spray/soil application",
                        "cost_estimate": "₹X-Y per acre",
                        "where_to_buy": "Agricultural stores, online platforms"
                    }}
                ],
                "organic_options": [
                    {{
                        "treatment_name": "Neem oil / Baking soda etc",
                        "preparation": "How to prepare at home",
                        "application": "How and when to apply",
                        "cost_estimate": "₹X per acre",
                        "effectiveness": "Expected success rate"
                    }}
                ]
            }},
            "treatment_schedule": {{
                "day_1_to_3": ["Specific actions for first 3 days"],
                "week_1": ["Actions for first week"],
                "week_2_to_4": ["Follow-up treatments"],
                "monitoring_signs": ["What to watch for improvement/worsening"]
            }},
            "follow_up_plan": {{
                "progress_check_timeline": "When to assess treatment success",
                "photo_follow_up": "When to take follow-up photos for monitoring",
                "expert_consultation": "When to call agricultural officer/KVK",
                "backup_plan": "Alternative treatment if first approach fails"
            }}
        }}

        Important guidelines:
        1. All product names should be real, commonly available in Indian agricultural markets
        2. Cost estimates should be realistic for Indian farmers
        3. Prioritize solutions based on {farmer_budget or 'medium'} budget constraints
        4. Include both chemical and organic solutions
        5. Focus on immediate actionability
        """

    def _parse_analysis_response(self, text: str) -> Dict[str, Any]:
        """Parse AI analysis response"""
        try:
            start_idx = text.find('{')
            end_idx = text.rfind('}') + 1
            if start_idx != -1 and end_idx > start_idx:
                json_str = text[start_idx:end_idx]
                return json.loads(json_str)
        except (json.JSONDecodeError, ValueError):
            pass

        return self._parse_text_analysis(text)

    def _parse_treatment_response(self, text: str, disease_name: str) -> Dict[str, Any]:
        """Parse AI treatment response"""
        try:
            start_idx = text.find('{')
            end_idx = text.rfind('}') + 1
            if start_idx != -1 and end_idx > start_idx:
                json_str = text[start_idx:end_idx]
                return json.loads(json_str)
        except (json.JSONDecodeError, ValueError):
            pass

        return self._parse_text_treatment(text, disease_name)

    def _parse_text_analysis(self, text: str) -> Dict[str, Any]:
        """Parse analysis when JSON extraction fails"""
        return {
            "diagnosis": {
                "primary_issue": "Analysis completed",
                "hindi_name": "फसल की जांच पूरी",
                "confidence_percentage": 75,
                "severity_level": "medium",
                "crop_identified": "Identified from image",
                "affected_plant_parts": ["leaves"]
            },
            "symptoms_observed": ["AI analysis has been completed"],
            "immediate_actions": ["Follow the detailed AI recommendations"],
            "risk_assessment": {
                "spread_risk": "medium",
                "potential_crop_loss": "10-20%",
                "urgency_level": "moderate",
                "best_treatment_window": "3-5 days"
            },
            "next_steps": ["Monitor crop daily", "Take follow-up photos in 3-5 days"],
            "full_ai_response": text
        }

    def _parse_text_treatment(self, text: str, disease_name: str) -> Dict[str, Any]:
        """Parse treatment text when JSON extraction fails"""
        return {
            "disease_info": {
                "name_hindi": f"फसल की समस्या - {disease_name}",
                "name_english": disease_name,
                "severity_impact": "AI has provided comprehensive treatment plan",
                "expected_timeline": "1-2 weeks for improvement"
            },
            "immediate_treatment": {
                "chemical_options": [
                    {
                        "product_name": "सामान्य कवकनाशी/कीटनाशक",
                        "active_ingredient": "As per product label",
                        "dosage": "Follow manufacturer instructions",
                        "application_method": "Foliar spray",
                        "cost_estimate": "₹300-500 per acre",
                        "where_to_buy": "Local agricultural stores"
                    }
                ],
                "organic_options": [
                    {
                        "treatment_name": "नीम तेल स्प्रे",
                        "preparation": "20ml neem oil per liter water",
                        "application": "Early morning or evening spray",
                        "cost_estimate": "₹100-200 per acre",
                        "effectiveness": "70-80% effective"
                    }
                ]
            },
            "follow_up_plan": {
                "progress_check_timeline": "Check improvement after 3-5 days",
                "photo_follow_up": "Take photos weekly",
                "expert_consultation": "Contact local agriculture officer if no improvement",
                "backup_plan": "Try alternative treatment if current approach fails"
            },
            "full_ai_response": text
        }
