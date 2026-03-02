"""Orchestrator Agent - Master coordinator for all specialized agents."""

import asyncio
from typing import Dict, List, Optional
from datetime import datetime
from .agent_base import Agent, AgentOutput, Query, UserContext
from .agent_registry import get_registry


class OrchestratorAgent(Agent):
    """Master agent that coordinates all specialized agents."""
    
    def __init__(self):
        """Initialize orchestrator."""
        super().__init__("orchestrator")
        self.registry = get_registry()
        self.agent_timeout = 10  # seconds
    
    async def analyze(self, query: Query, context: UserContext) -> AgentOutput:
        """
        Process query by coordinating specialized agents.
        
        Args:
            query: User query
            context: User context
            
        Returns:
            Synthesized recommendation from all agents
        """
        # Determine which agents to invoke based on query intent
        relevant_agents = self._determine_relevant_agents(query)
        
        # Dispatch queries to agents in parallel
        agent_outputs = await self._dispatch_to_agents(
            relevant_agents, query, context
        )
        
        # Synthesize outputs using LLM
        synthesis = await self._synthesize_with_llm(agent_outputs, query, context)
        
        # Create final output
        return self._create_output(
            query=query,
            analysis={
                "agent_outputs": [o.to_dict() for o in agent_outputs],
                "synthesis_process": synthesis.get("process", []),
            },
            recommendation=synthesis.get("recommendation", ""),
            confidence=synthesis.get("confidence", 0.0),
            data_sources=self._collect_data_sources(agent_outputs),
            rag_context=self._collect_rag_context(agent_outputs),
            reasoning_chain=synthesis.get("reasoning_chain", []),
        )
    
    def _determine_relevant_agents(self, query: Query) -> List[str]:
        """
        Determine which agents should analyze this query.
        
        Args:
            query: User query
            
        Returns:
            List of agent names to invoke
        """
        intent_to_agents = {
            "irrigation": ["weather_agent", "soil_agent", "irrigation_agent"],
            "pest_detection": ["pest_agent", "weather_agent"],
            "yield_forecast": ["yield_agent", "crop_advisory_agent", "weather_agent"],
            "market_advice": ["market_agent", "yield_agent"],
            "soil_health": ["soil_agent", "sustainability_agent"],
            "crop_planning": ["crop_advisory_agent", "soil_agent", "weather_agent"],
            "general": ["weather_agent", "soil_agent", "crop_advisory_agent"],
        }
        
        # Default to general agents if intent not recognized
        agents = intent_to_agents.get(query.intent, intent_to_agents["general"])
        
        # Filter to only registered agents
        available = set(self.registry.list_agents())
        return [a for a in agents if a in available]
    
    async def _dispatch_to_agents(
        self,
        agent_names: List[str],
        query: Query,
        context: UserContext,
    ) -> List[AgentOutput]:
        """
        Dispatch query to multiple agents in parallel.
        
        Args:
            agent_names: List of agent names to invoke
            query: User query
            context: User context
            
        Returns:
            List of agent outputs
        """
        tasks = []
        
        for agent_name in agent_names:
            agent = self.registry.get_agent(agent_name)
            if not agent:
                # Try to instantiate if not already active
                try:
                    agent = self.registry.instantiate_agent(agent_name)
                except ValueError:
                    continue
            
            # Create task with timeout
            task = asyncio.wait_for(
                agent.analyze(query, context),
                timeout=self.agent_timeout
            )
            tasks.append(task)
        
        # Execute all tasks in parallel
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out exceptions and return valid outputs
        outputs = []
        for result in results:
            if isinstance(result, AgentOutput):
                outputs.append(result)
            elif isinstance(result, Exception):
                # Log error but continue
                pass
        
        return outputs
    
    async def _synthesize_with_llm(
        self,
        agent_outputs: List[AgentOutput],
        query: Query,
        context: UserContext,
    ) -> Dict:
        """
        Use LLM to synthesize agent outputs into coherent recommendation.
        
        Args:
            agent_outputs: Outputs from all agents
            query: Original query
            context: User context
            
        Returns:
            Synthesized recommendation with reasoning
        """
        if not agent_outputs:
            return {
                "recommendation": "Unable to generate recommendation due to agent failures",
                "confidence": 0.0,
                "reasoning_chain": ["All agents failed to produce output"],
                "process": [],
            }
        
        # Build prompt for LLM synthesis
        prompt = self._build_synthesis_prompt(agent_outputs, query, context)
        
        try:
            response = await self.call_llm(prompt, temperature=0.5, max_tokens=1500)
            synthesis = self._parse_synthesis_response(response)
        except Exception as e:
            # Fallback to simple aggregation if LLM fails
            synthesis = self._fallback_synthesis(agent_outputs)
        
        return synthesis
    
    def _build_synthesis_prompt(
        self,
        agent_outputs: List[AgentOutput],
        query: Query,
        context: UserContext,
    ) -> str:
        """Build prompt for LLM synthesis."""
        agent_summaries = "\n".join([
            f"- {o.agent_name}: {o.recommendation} (confidence: {o.confidence:.2f})"
            for o in agent_outputs
        ])
        
        prompt = f"""You are an agricultural advisor synthesizing recommendations from multiple AI agents.

User Query: {query.text}
Intent: {query.intent}

Agent Outputs:
{agent_summaries}

User Context:
- Farm Location: {context.farm_location}
- Farm Size: {context.farm_size_hectares} hectares
- Crop: {context.crop_type}
- Growth Stage: {context.growth_stage}
- Soil Type: {context.soil_type}

Task: Synthesize a single, actionable recommendation that:
1. Resolves any conflicts between agents
2. Prioritizes farmer safety and crop health
3. Provides clear reasoning
4. Includes confidence level (0-100)

Format your response as:
RECOMMENDATION: [clear action]
REASONING: [step-by-step explanation]
CONFIDENCE: [0-100]
TRADE_OFFS: [any considerations]"""
        
        return prompt
    
    def _parse_synthesis_response(self, response: str) -> Dict:
        """Parse LLM synthesis response."""
        lines = response.split("\n")
        result = {
            "recommendation": "",
            "confidence": 0.0,
            "reasoning_chain": [],
            "process": [],
        }
        
        current_section = None
        for line in lines:
            line = line.strip()
            if line.startswith("RECOMMENDATION:"):
                result["recommendation"] = line.replace("RECOMMENDATION:", "").strip()
            elif line.startswith("REASONING:"):
                current_section = "reasoning"
            elif line.startswith("CONFIDENCE:"):
                try:
                    conf_str = line.replace("CONFIDENCE:", "").strip()
                    result["confidence"] = float(conf_str) / 100.0
                except ValueError:
                    result["confidence"] = 0.5
            elif line.startswith("TRADE_OFFS:"):
                current_section = "trade_offs"
            elif current_section == "reasoning" and line:
                result["reasoning_chain"].append(line)
        
        return result
    
    def _fallback_synthesis(self, agent_outputs: List[AgentOutput]) -> Dict:
        """Fallback synthesis when LLM fails."""
        # Simple aggregation: take highest confidence recommendation
        best_output = max(agent_outputs, key=lambda o: o.confidence)
        
        return {
            "recommendation": best_output.recommendation,
            "confidence": best_output.confidence,
            "reasoning_chain": [
                f"Synthesized from {len(agent_outputs)} agents",
                f"Primary recommendation from {best_output.agent_name}",
            ],
            "process": ["Fallback aggregation due to LLM unavailability"],
        }
    
    def _collect_data_sources(self, agent_outputs: List[AgentOutput]) -> List[str]:
        """Collect all data sources from agent outputs."""
        sources = set()
        for output in agent_outputs:
            sources.update(output.data_sources)
        return list(sources)
    
    def _collect_rag_context(self, agent_outputs: List[AgentOutput]) -> List[str]:
        """Collect all RAG context from agent outputs."""
        context = []
        for output in agent_outputs:
            context.extend(output.rag_context)
        return context
