# app/integration/tools/government_schemes_tools.py
"""Government schemes tools using AWS Bedrock for AI-powered assistance"""
from typing import Dict, Any, Optional
import logging
from ..bedrock.bedrock_client import bedrock_client

logger = logging.getLogger(__name__)


async def search_government_schemes(
    query: str,
    state: Optional[str] = None,
    scheme_type: str = "all",
    farmer_category: Optional[str] = None
) -> Dict[str, Any]:
    """
    Search for relevant government schemes using AI
    
    Args:
        query: Search query
        state: State for state-specific schemes
        scheme_type: Type of scheme
        farmer_category: Farmer category
    
    Returns:
        Dict with relevant schemes
    """
    try:
        search_prompt = f"""Search for relevant Indian government agricultural schemes.

Query: {query}
State: {state or 'All India'}
Scheme Type: {scheme_type}
Farmer Category: {farmer_category or 'General'}

Provide comprehensive information about relevant schemes in Hindi including:
1. Scheme names (Central and State)
2. Main benefits and subsidy amounts
3. Basic eligibility criteria
4. Application process overview

Focus on currently active schemes (2024-2025)."""
        
        response = bedrock_client.invoke_model(
            prompt=search_prompt,
            temperature=0.7
        )
        
        if response['status'] == 'success':
            return {
                "status": "success",
                "query": query,
                "state": state,
                "ai_response": response['content']
            }
        else:
            return {
                "status": "error",
                "message": "Scheme search failed"
            }
            
    except Exception as e:
        logger.error(f"Error searching schemes: {e}")
        return {
            "status": "error",
            "message": str(e)
        }


async def get_scheme_details(
    scheme_name: str,
    state: Optional[str] = None,
    include_application_info: bool = True
) -> Dict[str, Any]:
    """
    Get detailed information about a specific scheme
    
    Args:
        scheme_name: Name of the scheme
        state: State for state-specific details
        include_application_info: Include application process
    
    Returns:
        Dict with detailed scheme information
    """
    try:
        details_prompt = f"""Provide detailed information about this government scheme.

Scheme: {scheme_name}
State: {state or 'All India'}
Include Application Info: {'Yes' if include_application_info else 'No'}

Provide comprehensive details in Hindi:
1. Full scheme name and purpose
2. Benefits and subsidy amounts
3. Eligibility criteria
4. Required documents
5. {'Application process (online/offline)' if include_application_info else ''}
6. Contact information and helpline
7. Important deadlines

Ensure information is accurate and current (2024-2025)."""
        
        response = bedrock_client.invoke_model(
            prompt=details_prompt,
            temperature=0.7
        )
        
        if response['status'] == 'success':
            return {
                "status": "success",
                "scheme_name": scheme_name,
                "ai_response": response['content']
            }
        else:
            return {
                "status": "error",
                "message": "Scheme details retrieval failed"
            }
            
    except Exception as e:
        logger.error(f"Error getting scheme details: {e}")
        return {
            "status": "error",
            "message": str(e)
        }


async def check_eligibility(
    scheme_name: str,
    farmer_profile: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Check farmer's eligibility for a scheme
    
    Args:
        scheme_name: Name of the scheme
        farmer_profile: Farmer's profile data
    
    Returns:
        Dict with eligibility assessment
    """
    try:
        eligibility_prompt = f"""Check eligibility for this government scheme.

Scheme: {scheme_name}
Farmer Profile:
- Land Size: {farmer_profile.get('land_size', 'Unknown')} acres
- Annual Income: ₹{farmer_profile.get('annual_income', 'Unknown')}
- Category: {farmer_profile.get('category', 'General')}
- Age: {farmer_profile.get('age', 'Unknown')} years
- State: {farmer_profile.get('state', 'Unknown')}
- Female Farmer: {'Yes' if farmer_profile.get('is_female') else 'No'}

Provide eligibility analysis in Hindi:
1. Eligibility Status: ✅ Eligible / ❌ Not Eligible / 🔄 Partially Eligible
2. Which criteria are met
3. Which criteria are not met
4. Suggestions to improve eligibility
5. Alternative schemes if not eligible
6. Next steps

Be specific and actionable."""
        
        response = bedrock_client.invoke_model(
            prompt=eligibility_prompt,
            temperature=0.7
        )
        
        if response['status'] == 'success':
            return {
                "status": "success",
                "scheme_name": scheme_name,
                "ai_response": response['content']
            }
        else:
            return {
                "status": "error",
                "message": "Eligibility check failed"
            }
            
    except Exception as e:
        logger.error(f"Error checking eligibility: {e}")
        return {
            "status": "error",
            "message": str(e)
        }


async def get_application_process(
    scheme_name: str,
    state: Optional[str] = None,
    application_type: str = "online"
) -> Dict[str, Any]:
    """
    Get step-by-step application process
    
    Args:
        scheme_name: Name of the scheme
        state: State for state-specific process
        application_type: online or offline
    
    Returns:
        Dict with application process
    """
    try:
        process_prompt = f"""Provide step-by-step application process for this scheme.

Scheme: {scheme_name}
State: {state or 'General Process'}
Application Type: {application_type}

Provide detailed process in Hindi:
1. Pre-application preparation (documents needed)
2. Step-by-step application process
3. Official website and direct links
4. Helpline numbers and office addresses
5. Timeline (how long for response)
6. How to check application status
7. Common mistakes to avoid
8. Tips for successful application

Make it practical and easy to follow."""
        
        response = bedrock_client.invoke_model(
            prompt=process_prompt,
            temperature=0.7
        )
        
        if response['status'] == 'success':
            return {
                "status": "success",
                "scheme_name": scheme_name,
                "application_type": application_type,
                "ai_response": response['content']
            }
        else:
            return {
                "status": "error",
                "message": "Application process retrieval failed"
            }
            
    except Exception as e:
        logger.error(f"Error getting application process: {e}")
        return {
            "status": "error",
            "message": str(e)
        }
