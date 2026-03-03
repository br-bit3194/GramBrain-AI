# app/aws_integration/bedrock/bedrock_client.py
import boto3
import json
import logging
from typing import Dict, Any, List, Optional
from ..config.aws_config import get_aws_credentials, get_bedrock_config

logger = logging.getLogger(__name__)


class BedrockClient:
    """AWS Bedrock client for LLM interactions"""
    
    def __init__(self):
        credentials = get_aws_credentials()
        self.bedrock_runtime = boto3.client(
            service_name='bedrock-runtime',
            **credentials
        )
        self.config = get_bedrock_config()
        logger.info(f"Bedrock client initialized with model: {self.config['model_id']}")
    
    def invoke_model(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stop_sequences: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Invoke Bedrock model (Claude)"""
        try:
            # Prepare messages
            messages = [
                {
                    "role": "user",
                    "content": prompt
                }
            ]
            
            # Prepare request body for Claude
            body = {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": max_tokens or self.config['max_tokens'],
                "temperature": temperature or self.config['temperature'],
                "messages": messages
            }
            
            if system_prompt:
                body["system"] = system_prompt
            
            if stop_sequences:
                body["stop_sequences"] = stop_sequences
            
            # Invoke model
            response = self.bedrock_runtime.invoke_model(
                modelId=self.config['model_id'],
                body=json.dumps(body)
            )
            
            # Parse response
            response_body = json.loads(response['body'].read())
            
            return {
                "status": "success",
                "content": response_body['content'][0]['text'],
                "stop_reason": response_body.get('stop_reason'),
                "usage": response_body.get('usage', {})
            }
            
        except Exception as e:
            logger.error(f"Error invoking Bedrock model: {e}")
            return {
                "status": "error",
                "message": str(e)
            }
    
    def invoke_model_with_image(
        self,
        prompt: str,
        image_data: bytes,
        image_format: str = "jpeg",
        system_prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        """Invoke Bedrock model with image (Claude Vision)"""
        try:
            import base64
            
            # Encode image to base64
            image_base64 = base64.b64encode(image_data).decode('utf-8')
            
            # Prepare messages with image
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": f"image/{image_format}",
                                "data": image_base64
                            }
                        },
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ]
                }
            ]
            
            # Prepare request body
            body = {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": self.config['max_tokens'],
                "temperature": self.config['temperature'],
                "messages": messages
            }
            
            if system_prompt:
                body["system"] = system_prompt
            
            # Invoke model
            response = self.bedrock_runtime.invoke_model(
                modelId=self.config['model_id'],
                body=json.dumps(body)
            )
            
            # Parse response
            response_body = json.loads(response['body'].read())
            
            return {
                "status": "success",
                "content": response_body['content'][0]['text'],
                "stop_reason": response_body.get('stop_reason'),
                "usage": response_body.get('usage', {})
            }
            
        except Exception as e:
            logger.error(f"Error invoking Bedrock model with image: {e}")
            return {
                "status": "error",
                "message": str(e)
            }
    
    def stream_model(
        self,
        prompt: str,
        system_prompt: Optional[str] = None
    ):
        """Stream response from Bedrock model"""
        try:
            messages = [{"role": "user", "content": prompt}]
            
            body = {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": self.config['max_tokens'],
                "temperature": self.config['temperature'],
                "messages": messages
            }
            
            if system_prompt:
                body["system"] = system_prompt
            
            response = self.bedrock_runtime.invoke_model_with_response_stream(
                modelId=self.config['model_id'],
                body=json.dumps(body)
            )
            
            # Stream events
            for event in response['body']:
                chunk = json.loads(event['chunk']['bytes'])
                
                if chunk['type'] == 'content_block_delta':
                    if 'delta' in chunk and 'text' in chunk['delta']:
                        yield chunk['delta']['text']
                elif chunk['type'] == 'message_stop':
                    break
                    
        except Exception as e:
            logger.error(f"Error streaming from Bedrock: {e}")
            yield f"Error: {str(e)}"


# Global Bedrock client instance
bedrock_client = BedrockClient()
