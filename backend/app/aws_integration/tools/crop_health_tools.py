# app/aws_integration/tools/crop_health_tools.py
"""Crop health tools using AWS Bedrock for image analysis"""
from typing import Dict, Any, Optional
import logging
import base64
from ..bedrock.bedrock_client import bedrock_client

logger = logging.getLogger(__name__)


async def analyze_crop_image(
    image_data: str,
    location: Optional[str] = None,
    crop_type: Optional[str] = None
) -> Dict[str, Any]:
    """
    Analyze crop image using AWS Bedrock (Claude Vision)
    
    Args:
        image_data: Base64 encoded image data
        location: Farmer's location for localized advice
        crop_type: Type of crop for specific analysis
    
    Returns:
        Dict with disease diagnosis and recommendations
    """
    try:
        # Clean image data
        if ',' in image_data:
            image_data = image_data.split(',')[1]
        
        image_bytes = base64.b64decode(image_data)
        
        # Create analysis prompt
        analysis_prompt = f"""Analyze this crop image for disease, pest, or health issues.

Context:
- Location: {location or 'India'}
- Crop Type: {crop_type or 'Please identify'}

Provide detailed analysis in JSON format:
{{
    "diagnosis": {{
        "primary_issue": "Disease/pest name in English",
        "hindi_name": "Disease name in Hindi",
        "confidence_percentage": 85,
        "severity_level": "low/medium/high/critical",
        "crop_identified": "Crop type",
        "affected_plant_parts": ["leaves", "stem"]
    }},
    "symptoms_observed": ["Specific symptoms"],
    "immediate_actions": ["तुरंत करने योग्य काम in Hindi"],
    "risk_assessment": {{
        "spread_risk": "low/medium/high",
        "potential_crop_loss": "percentage",
        "urgency_level": "immediate/urgent/moderate"
    }},
    "next_steps": ["What farmer should do next"]
}}

Focus on practical, actionable advice for Indian farmers."""
        
        # Invoke Bedrock with image
        response = bedrock_client.invoke_model_with_image(
            prompt=analysis_prompt,
            image_data=image_bytes,
            image_format="jpeg"
        )
        
        if response['status'] == 'success':
            return {
                "status": "success",
                "analysis": response['content'],
                "location": location,
                "crop_type": crop_type
            }
        else:
            return {
                "status": "error",
                "message": "Image analysis failed"
            }
            
    except Exception as e:
        logger.error(f"Error analyzing crop image: {e}")
        return {
            "status": "error",
            "message": str(e)
        }


async def get_disease_treatment_info(
    disease_name: str,
    crop_type: Optional[str] = None,
    severity: str = "medium",
    location: Optional[str] = None,
    farmer_budget: str = "medium"
) -> Dict[str, Any]:
    """
    Get comprehensive treatment information for crop disease
    
    Args:
        disease_name: Name of the disease/pest
        crop_type: Type of crop affected
        severity: Severity level (low/medium/high/critical)
        location: Location for local context
        farmer_budget: Budget level (low/medium/high)
    
    Returns:
        Dict with comprehensive treatment plan
    """
    try:
        treatment_prompt = f"""Provide comprehensive treatment recommendations for this crop disease.

Disease: {disease_name}
Crop: {crop_type or 'general'}
Severity: {severity}
Location: {location or 'India'}
Farmer Budget: {farmer_budget}

Provide detailed treatment plan in JSON format:
{{
    "disease_info": {{
        "name_hindi": "Disease name in Hindi",
        "severity_impact": "What this means",
        "expected_timeline": "Treatment duration"
    }},
    "immediate_treatment": {{
        "chemical_options": [
            {{
                "product_name": "Available in India",
                "dosage": "X ml per liter",
                "cost_estimate": "₹X-Y per acre",
                "where_to_buy": "Local stores"
            }}
        ],
        "organic_options": [
            {{
                "treatment_name": "Neem oil etc",
                "preparation": "How to prepare",
                "cost_estimate": "₹X per acre"
            }}
        ],
        "home_remedies": ["Traditional remedies"]
    }},
    "treatment_schedule": {{
        "day_1_to_3": ["Actions"],
        "week_1": ["Actions"],
        "monitoring_signs": ["What to watch"]
    }},
    "prevention_strategy": {{
        "immediate_prevention": ["Stop spread"],
        "long_term_measures": ["Future prevention"]
    }}
}}

Focus on affordable, locally available solutions."""
        
        response = bedrock_client.invoke_model(
            prompt=treatment_prompt,
            temperature=0.7
        )
        
        if response['status'] == 'success':
            return {
                "status": "success",
                "treatment_plan": response['content'],
                "disease_name": disease_name
            }
        else:
            return {
                "status": "error",
                "message": "Treatment info retrieval failed"
            }
            
    except Exception as e:
        logger.error(f"Error getting treatment info: {e}")
        return {
            "status": "error",
            "message": str(e)
        }
