# app/integration/strands/base_agent.py
from typing import Dict, Any, List, Optional, Callable
from abc import ABC, abstractmethod
import logging
from ..bedrock.bedrock_client import bedrock_client

logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    """Base class for all Strands agents"""
    
    def __init__(
        self,
        name: str,
        description: str,
        instruction: str,
        tools: Optional[List[Callable]] = None
    ):
        self.name = name
        self.description = description
        self.instruction = instruction
        self.tools = tools or []
        self.tool_map = {tool.__name__: tool for tool in self.tools}
        
        logger.info(f"Initialized agent: {name}")
    
    @abstractmethod
    async def process(self, message: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Process a message and return response"""
        pass
    
    async def invoke_llm(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7
    ) -> str:
        """Invoke Bedrock LLM"""
        try:
            system = system_prompt or self.instruction
            
            response = bedrock_client.invoke_model(
                prompt=prompt,
                system_prompt=system,
                temperature=temperature
            )
            
            if response['status'] == 'success':
                return response['content']
            else:
                logger.error(f"LLM invocation failed: {response.get('message')}")
                return "मुझे खेद है, मैं अभी आपकी मदद करने में असमर्थ हूं।"
                
        except Exception as e:
            logger.error(f"Error invoking LLM: {e}")
            return "मुझे खेद है, एक त्रुटि हुई है।"
    
    async def invoke_tool(self, tool_name: str, **kwargs) -> Any:
        """Invoke a tool by name"""
        try:
            if tool_name not in self.tool_map:
                logger.error(f"Tool not found: {tool_name}")
                return {"status": "error", "message": f"Tool {tool_name} not found"}
            
            tool = self.tool_map[tool_name]
            result = await tool(**kwargs)
            
            logger.info(f"Tool {tool_name} executed successfully")
            return result
            
        except Exception as e:
            logger.error(f"Error invoking tool {tool_name}: {e}")
            return {"status": "error", "message": str(e)}
    
    def get_tool_descriptions(self) -> str:
        """Get formatted tool descriptions for LLM"""
        if not self.tools:
            return "No tools available."
        
        descriptions = []
        for tool in self.tools:
            tool_name = tool.__name__
            tool_doc = tool.__doc__ or "No description available"
            descriptions.append(f"- {tool_name}: {tool_doc.strip()}")
        
        return "\n".join(descriptions)
