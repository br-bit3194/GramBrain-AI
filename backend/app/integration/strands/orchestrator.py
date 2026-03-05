# app/integration/strands/orchestrator.py
from typing import Dict, Any, List, Optional
import logging
import json
import re
from .base_agent import BaseAgent
from ..bedrock.bedrock_client import bedrock_client

logger = logging.getLogger(__name__)


class StrandsOrchestrator:
    """
    Multi-agent orchestrator using Strands framework pattern
    Routes queries to appropriate specialized agents
    """
    
    def __init__(self, agents: List[BaseAgent]):
        self.agents = {agent.name: agent for agent in agents}
        self.agent_list = agents
        
        logger.info(f"Orchestrator initialized with {len(agents)} agents")
    
    async def process_query(
        self,
        message: str,
        session_context: Dict[str, Any],
        message_type: str = "text"
    ) -> Dict[str, Any]:
        """
        Process user query by routing to appropriate agent(s)
        
        Args:
            message: User's message
            session_context: Session state and context
            message_type: Type of message (text, image, etc.)
        
        Returns:
            Dict with response and metadata
        """
        try:
            logger.info(f"Processing {message_type} query: {message[:100]}...")
            
            # Step 1: Determine which agent(s) to use
            routing_decision = await self._route_query(message, message_type, session_context)
            
            # Step 2: Execute agent(s)
            if routing_decision['type'] == 'single':
                result = await self._execute_single_agent(
                    routing_decision['agent'],
                    message,
                    session_context
                )
            elif routing_decision['type'] == 'multi':
                result = await self._execute_multi_agent(
                    routing_decision['agents'],
                    message,
                    session_context
                )
            else:
                result = await self._handle_general_query(message, session_context)
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return {
                "status": "error",
                "response": "मुझे खेद है, मैं अभी आपकी मदद करने में असमर्थ हूं। कृपया दोबारा कोशिश करें।",
                "error": str(e)
            }
    
    async def _route_query(
        self,
        message: str,
        message_type: str,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Determine which agent(s) should handle the query"""
        try:
            # Image queries always go to crop health agent
            if message_type == "image":
                return {
                    "type": "single",
                    "agent": "crop_health_specialist"
                }
            
            # Build routing prompt
            agent_descriptions = "\n".join([
                f"- {agent.name}: {agent.description}"
                for agent in self.agent_list
            ])
            
            routing_prompt = f"""Analyze this farmer's query and determine which specialist agent(s) should handle it.

Available Agents:
{agent_descriptions}

Farmer's Query: "{message}"

Context:
- Location: {context.get('user_location', 'Unknown')}
- Previous queries: {context.get('interaction_count', 0)}

Respond with JSON only:
{{
    "type": "single" | "multi" | "general",
    "agent": "agent_name" (if single),
    "agents": ["agent1", "agent2"] (if multi),
    "reasoning": "brief explanation"
}}

Examples:
- "आज बारिश होगी?" → {{"type": "single", "agent": "weather_specialist"}}
- "प्याज की कीमत क्या है?" → {{"type": "single", "agent": "market_specialist"}}
- "मेरी फसल में बीमारी है" → {{"type": "single", "agent": "crop_health_specialist"}}
- "बारिश के बाद कब बेचूं?" → {{"type": "multi", "agents": ["weather_specialist", "market_specialist"]}}
- "नमस्ते" → {{"type": "general"}}
"""
            
            response = bedrock_client.invoke_model(
                prompt=routing_prompt,
                temperature=0.3
            )
            
            if response['status'] == 'success':
                # Extract JSON from response
                content = response['content']
                json_match = re.search(r'\{.*\}', content, re.DOTALL)
                
                if json_match:
                    routing_decision = json.loads(json_match.group())
                    logger.info(f"Routing decision: {routing_decision}")
                    return routing_decision
            
            # Fallback to general
            return {"type": "general"}
            
        except Exception as e:
            logger.error(f"Error in routing: {e}")
            return {"type": "general"}
    
    async def _execute_single_agent(
        self,
        agent_name: str,
        message: str,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute a single agent"""
        try:
            if agent_name not in self.agents:
                logger.warning(f"Agent not found: {agent_name}")
                return await self._handle_general_query(message, context)
            
            agent = self.agents[agent_name]
            result = await agent.process(message, context)
            
            return {
                "status": "success",
                "response": result.get('response', ''),
                "agent_used": agent_name,
                "tools_called": result.get('tools_called', []),
                "data": result.get('data', {})
            }
            
        except Exception as e:
            logger.error(f"Error executing agent {agent_name}: {e}")
            return {
                "status": "error",
                "response": "मुझे खेद है, एक त्रुटि हुई है।",
                "error": str(e)
            }
    
    async def _execute_multi_agent(
        self,
        agent_names: List[str],
        message: str,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute multiple agents and synthesize response"""
        try:
            results = []
            tools_called = []
            
            # Execute each agent
            for agent_name in agent_names:
                if agent_name in self.agents:
                    agent = self.agents[agent_name]
                    result = await agent.process(message, context)
                    results.append({
                        "agent": agent_name,
                        "response": result.get('response', ''),
                        "data": result.get('data', {})
                    })
                    tools_called.extend(result.get('tools_called', []))
            
            # Synthesize responses
            synthesis_prompt = f"""You are GramBrain, an agricultural assistant. Multiple specialists have provided information for this farmer's query.

Farmer's Query: "{message}"

Specialist Responses:
{json.dumps(results, indent=2, ensure_ascii=False)}

Synthesize these responses into a single, coherent answer in Hindi that:
1. Integrates all relevant information
2. Provides actionable advice
3. Is easy for farmers to understand
4. Addresses the complete query

Provide only the synthesized response, no meta-commentary."""
            
            synthesis = bedrock_client.invoke_model(
                prompt=synthesis_prompt,
                temperature=0.7
            )
            
            return {
                "status": "success",
                "response": synthesis.get('content', ''),
                "agent_used": "multi_agent",
                "agents_consulted": agent_names,
                "tools_called": list(set(tools_called)),
                "data": {"individual_results": results}
            }
            
        except Exception as e:
            logger.error(f"Error in multi-agent execution: {e}")
            return {
                "status": "error",
                "response": "मुझे खेद है, एक त्रुटि हुई है।",
                "error": str(e)
            }
    
    async def _handle_general_query(
        self,
        message: str,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle general queries that don't need specialist agents"""
        try:
            general_prompt = f"""You are GramBrain, a friendly agricultural assistant for Indian farmers.

User Context:
- Location: {context.get('user_location', 'India')}
- Interaction count: {context.get('interaction_count', 0)}

User Message: "{message}"

Respond warmly and helpfully in Hindi. If this is a greeting, welcome them and explain your capabilities.
If it's a general question, provide a helpful response and guide them on how to use your services.

Your capabilities:
🌱 Crop health diagnosis (send crop photos)
🌤️ Weather forecasts and farming advice
📊 Market prices and selling advice
🏛️ Government schemes and subsidies

Keep response concise and friendly."""
            
            response = bedrock_client.invoke_model(
                prompt=general_prompt,
                temperature=0.8
            )
            
            return {
                "status": "success",
                "response": response.get('content', ''),
                "agent_used": "general",
                "tools_called": []
            }
            
        except Exception as e:
            logger.error(f"Error handling general query: {e}")
            return {
                "status": "error",
                "response": "नमस्ते! मैं आपका कृषि सहायक हूं। कृपया अपना सवाल पूछें।"
            }
    
    def add_agent(self, agent: BaseAgent):
        """Add a new agent to the orchestrator"""
        self.agents[agent.name] = agent
        self.agent_list.append(agent)
        logger.info(f"Added agent: {agent.name}")
    
    def get_agent(self, name: str) -> Optional[BaseAgent]:
        """Get agent by name"""
        return self.agents.get(name)
