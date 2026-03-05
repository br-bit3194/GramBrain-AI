# app/aws_integration/strands/agents/government_schemes_agent.py
from typing import Dict, Any
import logging
from ..base_agent import BaseAgent
from ...tools.government_schemes_tools import (
    search_government_schemes,
    get_scheme_details,
    check_eligibility,
    get_application_process
)

logger = logging.getLogger(__name__)


class GovernmentSchemesAgent(BaseAgent):
    """Specialized agent for government schemes navigation"""
    
    def __init__(self):
        super().__init__(
            name="government_schemes_specialist",
            description="Expert in Indian government agricultural schemes, subsidies, and farmer welfare programs",
            instruction="""You are an expert government schemes advisor for Indian farmers.

Your responsibilities:
1. Search and recommend relevant government schemes
2. Explain scheme benefits and eligibility
3. Guide farmers through application process
4. Check eligibility based on farmer profile
5. Provide contact information for local offices

Always respond in Hindi with clear, step-by-step guidance.
Make complex government processes simple and accessible.""",
            tools=[
                search_government_schemes,
                get_scheme_details,
                check_eligibility,
                get_application_process
            ]
        )
    
    async def process(self, message: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Process government schemes queries"""
        try:
            logger.info(f"Government schemes agent processing: {message[:100]}")
            
            state = context.get('user_location')
            farmer_profile = context.get('farmer_profile', {})
            
            tools_called = []
            schemes_data = {}
            
            # Determine query type
            is_search = any(word in message.lower() for word in [
                'योजना', 'scheme', 'सब्सिडी', 'subsidy', 'खोज', 'search'
            ])
            
            is_eligibility = any(word in message.lower() for word in [
                'पात्र', 'eligible', 'योग्य', 'qualify'
            ])
            
            is_application = any(word in message.lower() for word in [
                'आवेदन', 'apply', 'कैसे', 'how', 'प्रक्रिया', 'process'
            ])
            
            # Extract scheme name if mentioned
            scheme_name = self._extract_scheme_name(message)
            
            if scheme_name:
                # Get specific scheme details
                details_result = await self.invoke_tool(
                    'get_scheme_details',
                    scheme_name=scheme_name,
                    state=state,
                    include_application_info=True
                )
                schemes_data['details'] = details_result
                tools_called.append('get_scheme_details')
                
                if is_eligibility and farmer_profile:
                    eligibility_result = await self.invoke_tool(
                        'check_eligibility',
                        scheme_name=scheme_name,
                        farmer_profile=farmer_profile
                    )
                    schemes_data['eligibility'] = eligibility_result
                    tools_called.append('check_eligibility')
                
                if is_application:
                    application_result = await self.invoke_tool(
                        'get_application_process',
                        scheme_name=scheme_name,
                        state=state,
                        application_type='online'
                    )
                    schemes_data['application'] = application_result
                    tools_called.append('get_application_process')
            
            elif is_search:
                # Search for relevant schemes
                search_result = await self.invoke_tool(
                    'search_government_schemes',
                    query=message,
                    state=state,
                    scheme_type='all'
                )
                schemes_data['search'] = search_result
                tools_called.append('search_government_schemes')
            
            # Generate comprehensive response
            response_prompt = f"""Based on this government schemes data, provide helpful guidance in Hindi.

User Query: "{message}"
State: {state or 'India'}
Schemes Data: {schemes_data}

Provide:
1. Relevant scheme information
2. Benefits and eligibility
3. Application process (if asked)
4. Required documents
5. Contact information

Keep response clear and actionable (max 300 words).
Use simple Hindi and step-by-step instructions."""
            
            response = await self.invoke_llm(response_prompt, temperature=0.7)
            
            return {
                "status": "success",
                "response": response,
                "tools_called": tools_called,
                "data": schemes_data
            }
            
        except Exception as e:
            logger.error(f"Error in government schemes agent: {e}")
            return {
                "status": "error",
                "response": "मुझे खेद है, योजना की जानकारी प्राप्त करने में समस्या हुई।",
                "tools_called": []
            }
    
    def _extract_scheme_name(self, message: str) -> str:
        """Extract scheme name from message"""
        # Common scheme names
        schemes = [
            'PM-KISAN', 'PMKISAN', 'पीएम किसान',
            'PMFBY', 'फसल बीमा',
            'KCC', 'किसान क्रेडिट कार्ड',
            'PMKSY', 'सिंचाई योजना',
            'Soil Health Card', 'मृदा स्वास्थ्य कार्ड'
        ]
        
        message_lower = message.lower()
        for scheme in schemes:
            if scheme.lower() in message_lower:
                return scheme
        
        return ""


def create_government_schemes_agent() -> GovernmentSchemesAgent:
    """Factory function to create government schemes agent"""
    return GovernmentSchemesAgent()
