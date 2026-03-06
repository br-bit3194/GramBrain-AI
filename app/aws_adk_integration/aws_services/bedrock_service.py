import boto3
import json
import logging
from typing import Dict, Any, Optional
import os

logger = logging.getLogger(__name__)

class BedrockService:
    """AWS Bedrock service for Claude AI models"""
    
    def __init__(self):
        self.region = os.getenv("AWS_REGION", "us-east-1")
        # Using Mistral 7B which doesn't require Anthropic use case form
        self.model_id = "mistral.mistral-7b-instruct-v0:2"
        
        try:
            self.client = boto3.client('bedrock-runtime', region_name=self.region)
            logger.info(f"✅ AWS Bedrock service initialized in region: {self.region}")
        except Exception as e:
            logger.error(f"❌ Failed to initialize Bedrock: {e}")
            raise
    
    async def invoke_claude(
            self,
            messages: list,
            system_prompt: Optional[str] = None,
            max_tokens: int = 2048,
            temperature: float = 0.7
    ) -> Dict[str, Any]:
        """Invoke Mistral model via Bedrock"""
        try:
            # Format messages for Mistral
            formatted_messages = ""
            for msg in messages:
                if isinstance(msg, dict):
                    role = msg.get("role", "user")
                    content = msg.get("content", "")
                    if isinstance(content, list):
                        # Handle content arrays
                        for item in content:
                            if isinstance(item, dict) and item.get("type") == "text":
                                formatted_messages += f"[{role.upper()}] {item.get('text', '')}\n"
                    else:
                        formatted_messages += f"[{role.upper()}] {content}\n"
            
            if system_prompt:
                formatted_messages = f"[SYSTEM] {system_prompt}\n" + formatted_messages
            
            formatted_messages += "[ASSISTANT]"
            
            body = {
                "prompt": formatted_messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": 0.9
            }
            
            response = self.client.invoke_model(
                modelId=self.model_id,
                body=json.dumps(body)
            )
            
            result = json.loads(response['body'].read())
            
            if result.get('outputs'):
                text_content = result['outputs'][0].get('text', '')
                logger.info("✅ Mistral invocation successful")
                return {
                    "status": "success",
                    "text": text_content,
                    "usage": result.get('usage', {})
                }
            else:
                logger.error("No content in Mistral response")
                return {
                    "status": "error",
                    "message": "No content in response"
                }
        
        except Exception as e:
            logger.error(f"❌ Error invoking Mistral: {e}")
            return {
                "status": "error",
                "message": str(e)
            }
    
    async def analyze_crop_image_with_claude(
            self,
            image_base64: str,
            prompt: str,
            image_type: str = "image/jpeg"
    ) -> Dict[str, Any]:
        """Analyze crop image using Mistral (text-based with image description)"""
        try:
            # Mistral doesn't support direct image input, so we use text-based analysis
            enhanced_prompt = f"""
            You are analyzing a crop image. The user has provided the following description and analysis request:
            
            {prompt}
            
            Please provide a detailed analysis based on the description provided.
            """
            
            messages = [
                {
                    "role": "user",
                    "content": enhanced_prompt
                }
            ]
            
            return await self.invoke_claude(messages, max_tokens=2048)
        
        except Exception as e:
            logger.error(f"❌ Error analyzing image with Mistral: {e}")
            return {
                "status": "error",
                "message": str(e)
            }
    
    async def generate_text(
            self,
            prompt: str,
            system_prompt: Optional[str] = None,
            max_tokens: int = 2048
    ) -> Dict[str, Any]:
        """Generate text using Claude"""
        try:
            messages = [
                {
                    "role": "user",
                    "content": prompt
                }
            ]
            
            return await self.invoke_claude(
                messages,
                system_prompt=system_prompt,
                max_tokens=max_tokens
            )
        
        except Exception as e:
            logger.error(f"❌ Error generating text: {e}")
            return {
                "status": "error",
                "message": str(e)
            }
    
    def get_service_status(self) -> Dict[str, Any]:
        """Get Bedrock service status"""
        return {
            "service_available": True,
            "model": "Mistral 7B Instruct",
            "model_id": self.model_id,
            "region": self.region,
            "capabilities": [
                "Text generation",
                "Crop health analysis",
                "Multi-turn conversations",
                "JSON output"
            ]
        }
