# app/google_adk_integration/farmbot_service.py - AWS Bedrock Integration
import os
import asyncio
from datetime import datetime
from typing import Dict, Any, Optional
import logging
import base64

from .agents.main_agent import create_main_farmbot_agent
from .services.elevenlabs_voice_service import ElevenLabsVoiceService
from .config.models import ChatResponse

# Configure logging
logging.basicConfig(
    level=getattr(logging, "INFO"),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


def get_logger(name: str) -> logging.Logger:
    """Get configured logger"""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


logger = get_logger(__name__)


class FarmBotService:
    """FarmBot service with AWS Bedrock and ElevenLabs voice integration"""

    def __init__(self):
        self.main_agent = None
        self.voice_service = ElevenLabsVoiceService()
        self.app_name = "farmbot_production"
        self.is_initialized = False

    async def initialize(self):
        """Initialize the FarmBot service with AWS Bedrock"""
        try:
            logger.info("Initializing FarmBot service with AWS Bedrock...")

            # Create main agent
            self.main_agent = create_main_farmbot_agent()
            logger.info(f"✅ Main agent '{self.main_agent.name}' created")

            # Test voice service
            voice_status = self.voice_service.get_service_status()
            if voice_status["api_configured"]:
                logger.info("✅ ElevenLabs voice service configured")
            else:
                logger.warning("⚠️ ElevenLabs API key not found - voice features will be limited")

            self.is_initialized = True
            logger.info("🌾 FarmBot service initialized successfully with AWS Bedrock!")

        except Exception as e:
            logger.error(f"❌ Failed to initialize FarmBot service: {e}")
            raise

    async def process_message_with_voice(
            self,
            message: str,
            session_id: str,
            user_context: Dict[str, Any] = None,
            message_type: str = "text",
            image_data: Optional[str] = None,
            include_voice: bool = True,
            voice_language: str = "hi"
    ) -> Dict[str, Any]:
        """Process message and generate both text and voice response"""
        if not self.is_initialized:
            raise RuntimeError("FarmBot service not initialized")

        try:
            logger.info(f"Processing {message_type} message with voice for session {session_id}")

            # Process message through main agent
            chat_response = await self.process_message(
                message=message,
                session_id=session_id,
                user_context=user_context,
                message_type=message_type,
                image_data=image_data
            )

            result = {
                "text_response": chat_response.response,
                "agent_used": chat_response.agent_used,
                "session_id": session_id,
                "timestamp": chat_response.timestamp,
                "response_type": "general"
            }

            # Generate voice if requested
            if include_voice and self.voice_service.get_service_status()["api_configured"]:
                logger.info("Generating voice response")

                voice_result = await self.voice_service.generate_voice_for_farming_response(
                    text=chat_response.response,
                    user_language=voice_language,
                    response_type="general"
                )

                if voice_result["status"] == "success":
                    result["voice_response"] = {
                        "audio_data": voice_result["audio_data"],
                        "audio_format": voice_result["audio_format"],
                        "voice_id": voice_result["voice_id"],
                        "size_bytes": voice_result["size_bytes"]
                    }
                    logger.info("✅ Voice generated successfully")
                else:
                    logger.warning(f"Voice generation failed: {voice_result.get('message')}")
                    result["voice_error"] = voice_result.get("message")

            return result

        except Exception as e:
            logger.error(f"❌ Error processing message with voice: {e}")
            return {
                "text_response": "मुझे खेद है, मैं अभी आपकी मदद करने में असमर्थ हूं। कृपया दोबारा कोशिश करें।",
                "session_id": session_id,
                "timestamp": datetime.now().isoformat(),
                "error": str(e)
            }

    async def process_message(
            self,
            message: str,
            session_id: str,
            user_context: Dict[str, Any] = None,
            message_type: str = "text",
            image_data: Optional[str] = None
    ) -> ChatResponse:
        """Process message through Bedrock agent"""
        if not self.is_initialized:
            raise RuntimeError("FarmBot service not initialized")

        try:
            logger.info(f"Processing {message_type} message for session {session_id}: {message[:100]}...")

            # Prepare enhanced message if image is provided
            if message_type == "image" and image_data:
                enhanced_message = self._create_enhanced_image_prompt(message, user_context)
                logger.info("🖼️ Image content prepared for crop health analysis")
            else:
                enhanced_message = message

            # Process through Bedrock agent
            result = await self.main_agent.process_query(enhanced_message, user_context)

            if result["status"] == "success":
                response = ChatResponse(
                    response=result["response"],
                    session_id=session_id,
                    agent_used=result.get("agent_used", "bedrock_claude"),
                    tools_called=[],
                    timestamp=datetime.now().isoformat()
                )
            else:
                response = ChatResponse(
                    response="मुझे खेद है, मैं अभी आपकी मदद करने में असमर्थ हूं। कृपया दोबारा कोशिश करें।",
                    session_id=session_id,
                    timestamp=datetime.now().isoformat()
                )

            logger.info(f"✅ Processed {message_type} message successfully")
            return response

        except Exception as e:
            logger.error(f"❌ Error processing {message_type} message: {e}")
            return ChatResponse(
                response="मुझे खेद है, मैं अभी आपकी मदद करने में असमर्थ हूं। कृपया दोबारा कोशिश करें।",
                session_id=session_id,
                timestamp=datetime.now().isoformat()
            )

    async def generate_voice_only(
            self,
            text: str,
            language: str = "hi",
            voice_type: str = "hindi_male"
    ) -> Dict[str, Any]:
        """Generate voice for any text"""
        try:
            return await self.voice_service.text_to_speech(
                text=text,
                voice_type=voice_type,
                language=language
            )
        except Exception as e:
            logger.error(f"Voice generation error: {e}")
            return {
                "status": "error",
                "message": str(e)
            }

    def _create_enhanced_image_prompt(self, message: str, user_context: Dict[str, Any] = None) -> str:
        """Create enhanced prompt for image analysis"""
        user_location = user_context.get('user_location') if user_context else None
        user_preferences = user_context.get('user_preferences', {}) if user_context else {}

        crop_preference = user_preferences.get('primary_crops', [])
        farming_scale = user_preferences.get('farming_scale', 'small')

        enhanced_prompt = f"""
        फसल की तस्वीर का विश्लेषण करने का अनुरोध:

        किसान का संदेश: {message}

        संदर्भ जानकारी:
        - स्थान: {user_location or 'भारत (स्थान अज्ञात)'}
        - मुख्य फसलें: {', '.join(crop_preference) if crop_preference else 'मिश्रित खेती'}
        - खेती का स्तर: {farming_scale}

        कृपया इस फसल की तस्वीर का विस्तृत विश्लेषण करें और:
        1. रोग/कीट की पहचान करें
        2. तत्काल करने योग्य उपाय बताएं  
        3. स्थानीय रूप से उपलब्ध उपचार सुझाएं
        4. लागत-प्रभावी समाधान दें
        """
        return enhanced_prompt

    def get_service_status(self) -> Dict[str, Any]:
        """Get comprehensive service status"""
        agent_status = self.main_agent.get_service_status() if self.main_agent else {}
        voice_status = self.voice_service.get_service_status()

        return {
            "initialized": self.is_initialized,
            "agent": agent_status,
            "voice_service": voice_status,
            "capabilities": {
                "text_processing": True,
                "image_analysis": True,
                "weather_integration": True,
                "market_analysis": True,
                "crop_health_diagnosis": True,
                "government_schemes": True,
                "multi_language_support": True,
                "voice_synthesis": voice_status["api_configured"],
                "high_quality_voice": voice_status["api_configured"]
            },
            "supported_image_formats": ["JPEG", "PNG", "WebP"],
            "max_image_size": "10MB",
            "supported_languages": ["Hindi", "English", "Marathi", "Gujarati"],
            "version": "3.0.0-aws-bedrock"
        }
