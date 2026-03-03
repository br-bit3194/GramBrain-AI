"""Orchestrator Agent - Master coordinator for all specialized agents."""

import asyncio
import json
import time
from typing import Dict, List, Optional, Any
from datetime import datetime
from .agent_base import Agent, AgentOutput, Query, UserContext
from .agent_registry import get_registry


class AgentExecutionMetrics:
    """Metrics for agent execution tracking."""
    
    def __init__(self):
        """Initialize metrics storage."""
        self.executions: Dict[str, List[Dict[str, Any]]] = {}
        self.total_executions = 0
        self.total_failures = 0
        self.total_timeouts = 0
    
    def record_execution(
        self,
        agent_name: str,
        duration_ms: float,
        success: bool,
        timeout: bool = False,
        error: Optional[str] = None
    ) -> None:
        """Record an agent execution."""
        if agent_name not in self.executions:
            self.executions[agent_name] = []
        
        self.executions[agent_name].append({
            "timestamp": datetime.now().isoformat(),
            "duration_ms": duration_ms,
            "success": success,
            "timeout": timeout,
            "error": error,
        })
        
        self.total_executions += 1
        if not success:
            self.total_failures += 1
        if timeout:
            self.total_timeouts += 1
    
    def get_agent_stats(self, agent_name: str) -> Dict[str, Any]:
        """Get statistics for a specific agent."""
        if agent_name not in self.executions:
            return {
                "total_executions": 0,
                "success_rate": 0.0,
                "avg_duration_ms": 0.0,
                "timeout_rate": 0.0,
            }
        
        executions = self.executions[agent_name]
        total = len(executions)
        successes = sum(1 for e in executions if e["success"])
        timeouts = sum(1 for e in executions if e["timeout"])
        durations = [e["duration_ms"] for e in executions]
        
        return {
            "total_executions": total,
            "success_rate": successes / total if total > 0 else 0.0,
            "avg_duration_ms": sum(durations) / len(durations) if durations else 0.0,
            "timeout_rate": timeouts / total if total > 0 else 0.0,
        }
    
    def get_overall_stats(self) -> Dict[str, Any]:
        """Get overall statistics."""
        return {
            "total_executions": self.total_executions,
            "total_failures": self.total_failures,
            "total_timeouts": self.total_timeouts,
            "failure_rate": self.total_failures / self.total_executions if self.total_executions > 0 else 0.0,
            "timeout_rate": self.total_timeouts / self.total_executions if self.total_executions > 0 else 0.0,
        }


class OrchestratorAgent(Agent):
    """Master agent that coordinates all specialized agents."""
    
    def __init__(self):
        """Initialize orchestrator."""
        super().__init__("orchestrator")
        self.registry = get_registry()
        self.agent_timeout = 10  # seconds
        self.metrics = AgentExecutionMetrics()
        self._initialized = False
        self._active_agents: Dict[str, Agent] = {}
    
    async def initialize(self) -> None:
        """Initialize orchestrator and prepare agents."""
        if self._initialized:
            return
        
        # Pre-instantiate commonly used agents for faster response
        common_agents = ["weather_agent", "soil_agent", "crop_advisory_agent"]
        for agent_name in common_agents:
            try:
                if agent_name in self.registry.list_agents():
                    agent = self.registry.instantiate_agent(agent_name)
                    # Inject dependencies if available
                    if self.llm_client:
                        agent.set_llm_client(self.llm_client)
                    if self.rag_client:
                        agent.set_rag_client(self.rag_client)
                    if self.data_client:
                        agent.set_data_client(self.data_client)
                    self._active_agents[agent_name] = agent
            except Exception as e:
                # Log but don't fail initialization
                pass
        
        self._initialized = True
    
    async def shutdown(self) -> None:
        """Shutdown orchestrator and cleanup agents."""
        self._active_agents.clear()
        self.registry.shutdown_all()
        self._initialized = False
    
    def get_metrics(self) -> Dict:
        """Get execution metrics."""
        return {
            "overall": self.metrics.get_overall_stats(),
            "by_agent": {
                agent_name: self.metrics.get_agent_stats(agent_name)
                for agent_name in self.registry.list_agents()
            }
        }
    
    def _validate_agent_output(self, output: Any, agent_name: str) -> bool:
        """
        Validate agent output for proper serialization.
        
        Args:
            output: Agent output to validate
            agent_name: Name of the agent that produced the output
            
        Returns:
            True if valid, False otherwise
        """
        if not isinstance(output, AgentOutput):
            return False
        
        try:
            # Test serialization
            serialized = output.to_dict()
            
            # Validate required fields
            required_fields = [
                "agent_name", "query_id", "timestamp", "analysis",
                "recommendation", "confidence"
            ]
            for field in required_fields:
                if field not in serialized:
                    return False
            
            # Validate types
            if not isinstance(serialized["analysis"], dict):
                return False
            if not isinstance(serialized["recommendation"], str):
                return False
            if not isinstance(serialized["confidence"], (int, float)):
                return False
            if not (0 <= serialized["confidence"] <= 1):
                return False
            
            # Test JSON serialization
            json.dumps(serialized)
            
            return True
        except Exception:
            return False
    
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
        Dispatch query to multiple agents in parallel with error handling.
        
        Args:
            agent_names: List of agent names to invoke
            query: User query
            context: User context
            
        Returns:
            List of agent outputs (only successful ones)
        """
        tasks = []
        task_agent_map = {}
        
        for agent_name in agent_names:
            # Get or instantiate agent
            agent = self._active_agents.get(agent_name)
            if not agent:
                agent = self.registry.get_agent(agent_name)
                if not agent:
                    # Try to instantiate if not already active
                    try:
                        agent = self.registry.instantiate_agent(agent_name)
                        # Inject dependencies
                        if self.llm_client:
                            agent.set_llm_client(self.llm_client)
                        if self.rag_client:
                            agent.set_rag_client(self.rag_client)
                        if self.data_client:
                            agent.set_data_client(self.data_client)
                        self._active_agents[agent_name] = agent
                    except ValueError:
                        # Agent class not registered
                        self.metrics.record_execution(
                            agent_name, 0, False, False, "Agent not registered"
                        )
                        continue
            
            # Create task with timeout and error handling
            task = self._execute_agent_with_validation(agent, query, context)
            tasks.append(task)
            task_agent_map[id(task)] = agent_name
        
        # Execute all tasks in parallel
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter valid outputs
        outputs = []
        for result in results:
            if isinstance(result, AgentOutput):
                outputs.append(result)
            # Exceptions already logged in _execute_agent_with_validation
        
        return outputs
    
    async def _execute_agent_with_validation(
        self,
        agent: Agent,
        query: Query,
        context: UserContext
    ) -> AgentOutput:
        """
        Execute agent with timeout, error handling, validation, and metrics tracking.
        
        Args:
            agent: Agent to execute
            query: User query
            context: User context
            
        Returns:
            Agent output (only if valid)
            
        Raises:
            Exception: If agent execution fails or output is invalid
        """
        agent_name = agent.agent_name
        start_time = time.time()
        
        try:
            # Execute with timeout
            output = await asyncio.wait_for(
                agent.analyze(query, context),
                timeout=self.agent_timeout
            )
            
            # Validate output
            if not self._validate_agent_output(output, agent_name):
                # Record validation failure
                duration_ms = (time.time() - start_time) * 1000
                self.metrics.record_execution(
                    agent_name, duration_ms, False, False, "Invalid output format"
                )
                raise ValueError(f"Invalid output from {agent_name}")
            
            # Record success
            duration_ms = (time.time() - start_time) * 1000
            self.metrics.record_execution(agent_name, duration_ms, True, False)
            
            return output
            
        except asyncio.TimeoutError:
            # Record timeout
            duration_ms = (time.time() - start_time) * 1000
            self.metrics.record_execution(
                agent_name, duration_ms, False, True, "Timeout"
            )
            raise
            
        except Exception as e:
            # Record failure (if not already recorded)
            duration_ms = (time.time() - start_time) * 1000
            if not isinstance(e, ValueError) or "Invalid output" not in str(e):
                error_msg = f"{type(e).__name__}: {str(e)}"
                self.metrics.record_execution(
                    agent_name, duration_ms, False, False, error_msg
                )
            raise
    
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
