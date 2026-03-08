# app/integration/grambrain_service_aws.py
"""
GramBrain Service using AWS Stack:
- AWS Bedrock (Claude) for LLM
- DynamoDB for data storage
- Strands framework for multi-agent orchestration
"""
import logging
from typing import Dict, Any, Optional
from datetime import datetime

from .strands import (
    StrandsOrchestrator,
    create_weather_agent,
    create_market_agent,
    create_crop_health_agent,
    create_government_schemes_agent
)
from .database.dynamodb_client import dynamodb_client
from .config.models import ChatResponse

logger = logging.getLogger(__name__)


class GramBrainServiceAWS:
    """
    Enhanced GramBrain service using AWS stack
    - Bedrock for AI/LLM
    - DynamoDB for storage
    - Strands for multi-agent orchestration
    """
    
    def __init__(self):
        self.orchestrator = None
        self.is_initialized = False
        logger.info("GramBrain AWS Service initializing...")
    
    async def initialize(self):
        """Initialize the GramBrain service with AWS components"""
        try:
            logger.info("Initializing GramBrain with AWS Bedrock and Strands framework...")
            
            # Create specialized agents
            weather_agent = create_weather_agent()
            market_agent = create_market_agent()
            crop_health_agent = create_crop_health_agent()
            government_schemes_agent = create_government_schemes_agent()
            
            # Create orchestrator with all agents
            self.orchestrator = StrandsOrchestrator([
                weather_agent,
                market_agent,
                crop_health_agent,
                government_schemes_agent
            ])
            
            # Initialize DynamoDB tables
            dynamodb_client.create_tables()
            
            self.is_initialized = True
            logger.info("🌾 GramBrain AWS service initialized successfully!")
            logger.info("✅ Bedrock LLM ready")
            logger.info("✅ DynamoDB connected")
            logger.info("✅ Strands orchestrator ready with 4 agents")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize GramBrain AWS service: {e}")
            raise
    
    async def process_message(
        self,
        message: str,
        session_id: str,
        user_context: Dict[str, Any] = None,
        message_type: str = "text",
        image_data: Optional[str] = None
    ) -> ChatResponse:
        """
        Process a user message through the Strands orchestrator
        
        Args:
            message: User's input message
            session_id: Session identifier
            user_context: Additional user context
            message_type: Type of message (text, image)
            image_data: Base64 encoded image data for crop analysis
        
        Returns:
            ChatResponse with agent's reply
        """
        if not self.is_initialized:
            raise RuntimeError("GramBrain AWS service not initialized")
        
        try:
            logger.info(f"Processing {message_type} message for session {session_id}")
            
            # Ensure session exists
            session = await self._ensure_session_exists(session_id, user_context)
            
            # Prepare context for orchestrator
            session_context = session.get('state', {}) if session else {}
            session_context.update(user_context or {})
            
            # Add image data to context if present
            if image_data:
                session_context['image_data'] = image_data
            
            # Process through Strands orchestrator
            result = await self.orchestrator.process_query(
                message=message,
                session_context=session_context,
                message_type=message_type
            )
            
            # Update session
            if session:
                await self._update_session_interaction(session_id, result)
            
            # Create response
            response = ChatResponse(
                response=result.get('response', 'मुझे खेद है, मैं अभी आपकी मदद करने में असमर्थ हूं।'),
                session_id=session_id,
                agent_used=result.get('agent_used'),
                tools_called=result.get('tools_called', []),
                timestamp=datetime.now().isoformat()
            )
            
            logger.info(f"✅ Processed message. Agent: {result.get('agent_used')}")
            return response
            
        except Exception as e:
            logger.error(f"❌ Error processing message: {e}")
            return ChatResponse(
                response="मुझे खेद है, मैं अभी आपकी मदद करने में असमर्थ हूं। कृपया दोबारा कोशिश करें।",
                session_id=session_id,
                timestamp=datetime.now().isoformat()
            )
    
    async def _ensure_session_exists(
        self,
        session_id: str,
        user_context: Dict[str, Any] = None
    ) -> Optional[Dict[str, Any]]:
        """Ensure session exists in DynamoDB"""
        try:
            # Try to get existing session
            session = dynamodb_client.get_session(session_id)
            
            if not session:
                # Create new session
                initial_state = {
                    "initialized": True,
                    "user_context": user_context or {},
                    "timestamp": datetime.now().isoformat(),
                    "interaction_count": 0,
                    "capabilities": {
                        "weather_forecasting": True,
                        "market_analysis": True,
                        "crop_health_diagnosis": True,
                        "government_schemes": True,
                        "image_analysis": True
                    }
                }
                
                success = dynamodb_client.create_session(
                    session_id=session_id,
                    user_id="web_user",
                    initial_state=initial_state
                )
                
                if success:
                    logger.info(f"✅ Created new session: {session_id}")
                    return {"state": initial_state}
            
            return session
            
        except Exception as e:
            logger.error(f"❌ Session management error: {e}")
            return None
    
    async def _update_session_interaction(
        self,
        session_id: str,
        result: Dict[str, Any]
    ):
        """Update session with interaction data"""
        try:
            session = dynamodb_client.get_session(session_id)
            
            if session:
                state = session.get('state', {})
                state['interaction_count'] = state.get('interaction_count', 0) + 1
                state['last_agent_used'] = result.get('agent_used')
                state['last_interaction'] = datetime.now().isoformat()
                
                dynamodb_client.update_session_state(session_id, state)
                
        except Exception as e:
            logger.error(f"Error updating session: {e}")
    
    def get_service_status(self) -> Dict[str, Any]:
        """Get comprehensive service status"""
        return {
            "initialized": self.is_initialized,
            "orchestrator": "Strands Framework" if self.orchestrator else None,
            "database": "DynamoDB",
            "llm": "AWS Bedrock (Claude)",
            "agents": [
                "weather_specialist",
                "market_specialist",
                "crop_health_specialist",
                "government_schemes_specialist"
            ],
            "capabilities": {
                "text_processing": True,
                "image_analysis": True,
                "weather_integration": True,
                "market_analysis": True,
                "crop_health_diagnosis": True,
                "government_schemes": True,
                "multi_agent_orchestration": True
            },
            "supported_image_formats": ["JPEG", "PNG", "WebP"],
            "max_image_size": "5MB",
            "supported_languages": ["Hindi", "English"],
            "version": "3.0.0-aws-bedrock-strands"
        }
    
    async def get_session_analytics(self, session_id: str) -> Dict[str, Any]:
        """Get analytics for a specific session"""
        try:
            session = dynamodb_client.get_session(session_id)
            
            if not session:
                return {"error": "Session not found"}
            
            state = session.get('state', {})
            
            return {
                "session_info": {
                    "session_id": session_id,
                    "created": session.get('created_at'),
                    "interaction_count": state.get('interaction_count', 0),
                    "last_activity": state.get('last_interaction')
                },
                "user_context": state.get('user_context', {}),
                "last_agent_used": state.get('last_agent_used')
            }
            
        except Exception as e:
            logger.error(f"Error getting analytics: {e}")
            return {"error": str(e)}


# Global GramBrain AWS service instance
grambrain_service_aws = GramBrainServiceAWS()
