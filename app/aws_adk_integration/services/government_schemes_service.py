from typing import Dict, Any, Optional
import logging
import os

logger = logging.getLogger(__name__)

# Initialize Bedrock service
try:
    from app.aws_adk_integration.aws_services import BedrockService
    bedrock_service = BedrockService()
    logger.info("✅ AWS Bedrock service initialized for Government Schemes")
except Exception as e:
    bedrock_service = None
    logger.error(f"❌ Failed to initialize Bedrock service: {e}")


class GovernmentSchemesService:
    """AI-powered service for government schemes using AWS Bedrock Claude"""

    def __init__(self):
        self.bedrock = bedrock_service

    async def search_schemes(
            self,
            query: str,
            state: Optional[str] = None,
            scheme_type: str = "all",
            farmer_category: Optional[str] = None
    ) -> Dict[str, Any]:
        """Search for relevant government schemes using Claude"""
        try:
            if not self.bedrock:
                return {
                    "status": "error",
                    "message": "AI सेवा उपलब्ध नहीं है।"
                }

            search_prompt = f"""
            आप एक सरकारी योजना विशेषज्ञ हैं। भारतीय किसान की निम्नलिखित आवश्यकता के लिए उपयुक्त सरकारी योजनाएं खोजें:

            किसान की आवश्यकता: {query}
            राज्य: {state or 'कोई विशिष्ट राज्य नहीं'}
            योजना प्रकार: {scheme_type}
            किसान श्रेणी: {farmer_category or 'सामान्य'}

            कृपया निम्नलिखित जानकारी प्रदान करें:

            1. **मुख्य योजनाएं** (केंद्र सरकार):
               - योजना का नाम
               - मुख्य लाभ
               - सब्सिडी राशि/प्रतिशत
               - बुनियादी पात्रता

            2. **राज्य सरकार की योजनाएं** (यदि state दी गई है):
               - राज्य-विशिष्ट योजनाएं
               - स्थानीय लाभ

            3. **व्यावहारिक सुझाव**:
               - कौन सी योजना सबसे बेहतर है
               - आवेदन की प्राथमिकता

            जानकारी सरल हिंदी में दें।
            """

            result = await self.bedrock.generate_text(prompt=search_prompt)

            if result["status"] == "success":
                return {
                    "status": "success",
                    "query": query,
                    "state": state,
                    "ai_response": result["text"]
                }
            else:
                return {
                    "status": "error",
                    "message": f"योजना खोजने में त्रुटि: {result.get('message')}"
                }

        except Exception as e:
            logger.error(f"Scheme search error: {e}")
            return {
                "status": "error",
                "message": f"योजना खोजने में त्रुटि: {str(e)}"
            }

    async def get_scheme_details(
            self,
            scheme_name: str,
            state: Optional[str] = None,
            include_application_info: bool = True
    ) -> Dict[str, Any]:
        """Get detailed information about a specific scheme"""
        try:
            if not self.bedrock:
                return {
                    "status": "error",
                    "message": "AI सेवा उपलब्ध नहीं है।"
                }

            details_prompt = f"""
            कृपया '{scheme_name}' योजना के बारे में विस्तृत जानकारी प्रदान करें:

            राज्य: {state or 'पूरे भारत के लिए'}
            आवेदन जानकारी शामिल करें: {'हां' if include_application_info else 'नहीं'}

            निम्नलिखित विवरण चाहिए:

            1. **योजना का पूरा नाम और उद्देश्य**
            2. **मुख्य लाभ और सुविधाएं**
            3. **पात्रता मानदंड**
            4. **आवश्यक दस्तावेज़**
            5. **योजना की अवधि और deadline**
            {'6. **आवेदन प्रक्रिया**' if include_application_info else ''}
            7. **संपर्क जानकारी**
            8. **महत्वपूर्ण बातें और सुझाव**

            सभी जानकारी सटीक, current और व्यावहारिक होनी चाहिए।
            """

            result = await self.bedrock.generate_text(prompt=details_prompt)

            if result["status"] == "success":
                return {
                    "status": "success",
                    "scheme_name": scheme_name,
                    "state": state,
                    "ai_response": result["text"]
                }
            else:
                return {
                    "status": "error",
                    "message": f"योजना विवरण प्राप्त करने में त्रुटि: {result.get('message')}"
                }

        except Exception as e:
            logger.error(f"Scheme details error: {e}")
            return {
                "status": "error",
                "message": f"योजना विवरण प्राप्त करने में त्रुटि: {str(e)}"
            }

    async def check_eligibility(
            self,
            scheme_name: str,
            farmer_profile: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Check farmer's eligibility for a scheme"""
        try:
            if not self.bedrock:
                return {
                    "status": "error",
                    "message": "AI सेवा उपलब्ध नहीं है।"
                }

            eligibility_prompt = f"""
            कृपया '{scheme_name}' योजना के लिए निम्नलिखित किसान की पात्रता की जांच करें:

            किसान का विवरण:
            - भूमि का आकार: {farmer_profile.get('land_size', 'अज्ञात')} एकड़
            - वार्षिक आय: ₹{farmer_profile.get('annual_income', 'अज्ञात')}
            - श्रेणी: {farmer_profile.get('category', 'सामान्य')}
            - आयु: {farmer_profile.get('age', 'अज्ञात')} वर्ष
            - राज्य: {farmer_profile.get('state', 'अज्ञात')}
            - महिला किसान: {'हां' if farmer_profile.get('is_female') else 'नहीं'}

            कृपया निम्नलिखित विश्लेषण करें:

            1. **पात्रता स्थिति**: ✅ पात्र / ❌ अपात्र / 🔄 आंशिक पात्र
            2. **विस्तृत विश्लेषण**
            3. **सुझाव**
            4. **अगले कदम**

            स्पष्ट और actionable सलाह दें।
            """

            result = await self.bedrock.generate_text(prompt=eligibility_prompt)

            if result["status"] == "success":
                return {
                    "status": "success",
                    "scheme_name": scheme_name,
                    "farmer_profile": farmer_profile,
                    "ai_response": result["text"]
                }
            else:
                return {
                    "status": "error",
                    "message": f"पात्रता जांच में त्रुटि: {result.get('message')}"
                }

        except Exception as e:
            logger.error(f"Eligibility check error: {e}")
            return {
                "status": "error",
                "message": f"पात्रता जांच में त्रुटि: {str(e)}"
            }

    async def get_application_process(
            self,
            scheme_name: str,
            state: Optional[str] = None,
            application_type: str = "online"
    ) -> Dict[str, Any]:
        """Get step-by-step application process"""
        try:
            if not self.bedrock:
                return {
                    "status": "error",
                    "message": "AI सेवा उपलब्ध नहीं है।"
                }

            process_prompt = f"""
            कृपया '{scheme_name}' योजना के लिए {application_type} आवेदन प्रक्रिया बताएं:

            राज्य: {state or 'सामान्य प्रक्रिया'}

            निम्नलिखित जानकारी चाहिए:

            1. **आवेदन से पहले की तैयारी**
            2. **चरणबद्ध आवेदन प्रक्रिया**
            3. **महत्वपूर्ण लिंक और संपर्क**
            4. **Application के बाद**
            5. **Common Mistakes और Tips**

            व्यावहारिक और step-by-step guidance दें।
            """

            result = await self.bedrock.generate_text(prompt=process_prompt)

            if result["status"] == "success":
                return {
                    "status": "success",
                    "scheme_name": scheme_name,
                    "application_type": application_type,
                    "ai_response": result["text"]
                }
            else:
                return {
                    "status": "error",
                    "message": f"आवेदन प्रक्रिया प्राप्त करने में त्रुटि: {result.get('message')}"
                }

        except Exception as e:
            logger.error(f"Application process error: {e}")
            return {
                "status": "error",
                "message": f"आवेदन प्रक्रिया प्राप्त करने में त्रुटि: {str(e)}"
            }

    def get_service_status(self) -> Dict[str, Any]:
        """Get service status"""
        return {
            "service_available": self.bedrock is not None,
            "ai_model": "Claude 3 Sonnet (AWS Bedrock)" if self.bedrock else None,
            "api_configured": os.getenv("AWS_REGION") is not None,
            "approach": "Pure AI-driven government schemes assistance via AWS Bedrock",
            "capabilities": [
                "Real-time scheme search using Claude",
                "Dynamic eligibility assessment",
                "Current application process guidance",
                "AI-powered office location assistance",
                "Latest updates from AI knowledge base"
            ] if self.bedrock else [],
            "data_source": "Claude AI knowledge base - no hardcoded data"
        }
