# app/main_aws.py - FastAPI application with AWS stack
"""
GramBrain - AWS Stack Version
- AWS Bedrock (Claude) for LLM
- DynamoDB for data storage
- Strands framework for multi-agent orchestration
"""
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from datetime import datetime
import json
import logging

from .aws_integration.grambrain_service_aws import grambrain_service_aws
from .aws_integration.config.aws_config import aws_config
from .websocket_conn import ConnectionManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="GramBrain - AWS Stack",
    description="AI-Powered Agricultural Assistant using AWS Bedrock and DynamoDB"
)

# Templates
templates = Jinja2Templates(directory="backend/app/templates")

# WebSocket connection manager
conn_manager = ConnectionManager()


@app.on_event("startup")
async def startup_event():
    """Initialize all services on startup"""
    try:
        logger.info("🚀 Starting GramBrain with AWS stack...")
        
        # Initialize GramBrain AWS service
        await grambrain_service_aws.initialize()
        logger.info("✅ GramBrain AWS service initialized")
        
        # Log configuration
        logger.info(f"AWS Region: {aws_config.aws_region}")
        logger.info(f"Bedrock Model: {aws_config.bedrock_model_id}")
        logger.info(f"DynamoDB Tables: {aws_config.dynamodb_sessions_table}")
        
    except Exception as e:
        logger.error(f"❌ Startup error: {e}")
        raise


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Home page"""
    try:
        context = {
            "request": request,
            "page_title": "GramBrain - AI Agricultural Assistant (AWS)",
            "market_preview": [],  # Can be populated from DynamoDB
            "enhanced_features_enabled": True,
            "websocket_enabled": True,
            "ai_crop_diagnosis": True,
            "version": "3.0 AWS Bedrock + Strands"
        }
        return templates.TemplateResponse("home_aws.html", context)
    
    except Exception as e:
        logger.error(f"Home page error: {e}")
        context = {
            "request": request,
            "page_title": "GramBrain",
            "market_preview": [],
            "version": "3.0"
        }
        return templates.TemplateResponse("home_aws.html", context)


@app.websocket("/ws/chat")
async def websocket_chat_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time chat"""
    session_id = f"session_{int(datetime.now().timestamp())}"
    
    try:
        await conn_manager.connect(websocket, session_id)
        logger.info(f"✅ WebSocket connected: {session_id}")
        
        while True:
            # Receive message from frontend
            data = await websocket.receive_text()
            message_data = json.loads(data)
            
            message_type = message_data.get("type", "text")
            content = message_data.get("content", "")
            user_location = message_data.get("user_location")
            user_preferences = message_data.get("user_preferences", {})
            additional_data = message_data.get("additional_data", {})
            
            # Update user session context
            if user_location:
                conn_manager.user_sessions[session_id]["user_location"] = user_location
            
            if user_preferences:
                conn_manager.user_sessions[session_id]["user_preferences"] = user_preferences
            
            conn_manager.user_sessions[session_id]["interaction_count"] += 1
            conn_manager.user_sessions[session_id]["last_activity"] = datetime.now().isoformat()
            
            # Process the query
            await handle_query(content, message_type, additional_data, session_id)
    
    except WebSocketDisconnect:
        logger.info(f"🔌 WebSocket disconnected: {session_id}")
        conn_manager.disconnect(session_id)
    
    except Exception as e:
        logger.error(f"❌ WebSocket error for {session_id}: {e}")
        conn_manager.disconnect(session_id)


async def handle_query(content: str, message_type: str, additional_data: dict, session_id: str):
    """Handle user query through AWS Bedrock and Strands orchestrator"""
    try:
        logger.info(f"Processing {message_type} query: {content[:100]}...")
        
        # Get user context from session
        user_context = conn_manager.user_sessions.get(session_id, {})
        
        # Prepare context for GramBrain
        enhanced_context = {
            "user_location": user_context.get("user_location"),
            "user_preferences": user_context.get("user_preferences", {}),
            "session_data": user_context
        }
        
        # Send thinking indicator
        await conn_manager.send_message({
            "type": "thinking",
            "content": "विश्लेषण हो रहा है...",
            "session_id": session_id
        }, session_id)
        
        # Process through GramBrain AWS service
        if message_type == "image":
            image_data = additional_data.get("image_data")
            if not image_data:
                await conn_manager.send_message({
                    "type": "error",
                    "content": "तस्वीर प्राप्त नहीं हुई। कृपया दोबारा कोशिश करें।",
                    "session_id": session_id
                }, session_id)
                return
            
            result = await grambrain_service_aws.process_message(
                message=content,
                session_id=session_id,
                user_context=enhanced_context,
                message_type="image",
                image_data=image_data
            )
        else:
            result = await grambrain_service_aws.process_message(
                message=content,
                session_id=session_id,
                user_context=enhanced_context,
                message_type="text"
            )
        
        # Send response
        await conn_manager.send_message({
            "type": "response",
            "content": result.response,
            "agent_used": result.agent_used,
            "tools_called": result.tools_called,
            "session_id": session_id,
            "timestamp": result.timestamp
        }, session_id)
        
        logger.info(f"✅ Successfully processed query. Agent: {result.agent_used}")
    
    except Exception as e:
        logger.error(f"❌ Query processing error: {e}")
        await conn_manager.send_message({
            "type": "error",
            "content": f"माफ करें, एक त्रुटि हुई है। कृपया दोबारा कोशिश करें।",
            "session_id": session_id
        }, session_id)


@app.get("/api/service-status")
async def get_service_status():
    """Get status of all AWS services"""
    try:
        grambrain_status = grambrain_service_aws.get_service_status()
        
        return {
            "status": "operational",
            "stack": "AWS",
            "services": {
                "grambrain_ai": grambrain_status,
                "bedrock": "running",
                "dynamodb": "connected",
                "websocket": "running"
            },
            "features": grambrain_status.get("capabilities", {}),
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }


@app.get("/api/session/{session_id}/analytics")
async def get_session_analytics(session_id: str):
    """Get analytics for a specific session"""
    try:
        analytics = await grambrain_service_aws.get_session_analytics(session_id)
        return analytics
    
    except Exception as e:
        return {
            "error": f"Analytics not available: {str(e)}",
            "session_id": session_id
        }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "GramBrain AWS",
        "timestamp": datetime.now().isoformat()
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main_aws:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        ws_ping_interval=20,
        ws_ping_timeout=20
    )
