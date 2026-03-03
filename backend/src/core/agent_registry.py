"""Agent registry for managing agent lifecycle and discovery."""

from typing import Dict, List, Optional, Type
from .agent_base import Agent


class AgentRegistry:
    """Registry for managing all agents in the system."""
    
    def __init__(self):
        """Initialize empty registry."""
        self._agents: Dict[str, Agent] = {}
        self._agent_classes: Dict[str, Type[Agent]] = {}
    
    def register_agent_class(self, agent_name: str, agent_class: Type[Agent]) -> None:
        """
        Register an agent class.
        
        Args:
            agent_name: Unique agent identifier
            agent_class: Agent class (must inherit from Agent)
        """
        if not issubclass(agent_class, Agent):
            raise TypeError(f"{agent_class} must inherit from Agent")
        
        self._agent_classes[agent_name] = agent_class
    
    def instantiate_agent(self, agent_name: str) -> Agent:
        """
        Instantiate an agent from registered class.
        
        Args:
            agent_name: Agent identifier
            
        Returns:
            Agent instance
            
        Raises:
            ValueError: If agent class not registered
        """
        if agent_name not in self._agent_classes:
            raise ValueError(f"Agent '{agent_name}' not registered")
        
        agent_class = self._agent_classes[agent_name]
        # Agents initialize themselves with their own name
        agent = agent_class()
        self._agents[agent_name] = agent
        return agent
    
    def get_agent(self, agent_name: str) -> Optional[Agent]:
        """Get instantiated agent by name."""
        return self._agents.get(agent_name)
    
    def list_agents(self) -> List[str]:
        """List all registered agent names."""
        return list(self._agent_classes.keys())
    
    def list_active_agents(self) -> List[str]:
        """List all instantiated agents."""
        return list(self._agents.keys())
    
    def shutdown_agent(self, agent_name: str) -> None:
        """Shutdown an agent instance."""
        if agent_name in self._agents:
            del self._agents[agent_name]
    
    def shutdown_all(self) -> None:
        """Shutdown all agent instances."""
        self._agents.clear()


# Global registry instance
_global_registry = AgentRegistry()


def get_registry() -> AgentRegistry:
    """Get global agent registry."""
    return _global_registry
