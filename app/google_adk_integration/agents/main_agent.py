# ============================================================================
# app/google_adk_integration/agents/main_agent.py - AWS Bedrock Claude Agent
# ============================================================================
import logging
from typing import Dict, Any, Optional
from app.google_adk_integration.aws_services import BedrockService

logger = logging.getLogger(__name__)


class MainFarmbotAgent:
    """Main FarmBot orchestrator using AWS Bedrock Claude"""

    def __init__(self):
        self.bedrock = BedrockService()
        self.name = "farmbot_main_orchestrator"
        self.model = "Claude 3 Sonnet (AWS Bedrock)"
        logger.info("✅ Main FarmBot agent initialized with AWS Bedrock")

    async def process_query(self, message: str, user_context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Process user query and route to appropriate specialist"""
        try:
            system_prompt = """
            You are FarmBot, the main agricultural intelligence system helping farmers across India.

            **Your specialized capabilities:**
            🌦️ **Weather**: Weather forecasts, climate advice, rainfall predictions
            📊 **Market**: Mandi prices, selling advice, price trends, market analysis
            🌱 **Crop Health**: Disease diagnosis, pest identification, treatment recommendations
            🏛️ **Government Schemes**: Subsidies, scheme eligibility, application process

            **Your approach:**
            • Greet farmers warmly in their context
            • Understand the farmer's specific need
            • Provide integrated advice when queries span multiple domains
            • Always consider the farmer's location, crops, and experience level
            • Respond in simple, practical language (primarily Hindi, but adapt to user's language)
            • Be encouraging and supportive

            **Query Classification:**
            - Weather queries: "आज बारिश होगी?", "इस सप्ताह का मौसम कैसा रहेगा?"
            - Market queries: "प्याज की कीमत क्या है?", "कब बेचना चाहिए?"
            - Crop health queries: "मेरी फसल में बीमारी है", "पत्तियों पर धब्बे हैं"
            - Government schemes: "सब्सिडी चाहिए", "योजना के बारे में बताएं"

            Provide practical, actionable advice that farmers can implement immediately.
            """

            result = await self.bedrock.generate_text(
                prompt=message,
                system_prompt=system_prompt,
                max_tokens=2048
            )

            if result["status"] == "success":
                return {
                    "status": "success",
                    "response": result["text"],
                    "agent_used": self.name,
                    "model": self.model
                }
            else:
                return {
                    "status": "error",
                    "message": result.get("message", "Error processing query")
                }

        except Exception as e:
            logger.error(f"❌ Error processing query: {e}")
            return {
                "status": "error",
                "message": f"माफ करें, एक त्रुटि हुई है: {str(e)}"
            }

    def get_service_status(self) -> Dict[str, Any]:
        """Get agent status"""
        return {
            "agent_name": self.name,
            "model": self.model,
            "status": "operational",
            "capabilities": [
                "Weather forecasting",
                "Market analysis",
                "Crop health diagnosis",
                "Government schemes assistance",
                "Multi-language support",
                "Image analysis"
            ]
        }


def create_main_farmbot_agent() -> MainFarmbotAgent:
    """Create and return the main FarmBot agent"""
    return MainFarmbotAgent()
