"""Village Intelligence Agent - Provides collective insights."""

from typing import Dict, Any, List
from ..core.agent_base import Agent, Query, UserContext


class VillageAgent(Agent):
    """Aggregates village-level data and provides collective insights."""
    
    def __init__(self):
        """Initialize village agent."""
        super().__init__("village_agent")
    
    async def analyze(self, query: Query, context: UserContext) -> Any:
        """Analyze village-level data and provide collective insights."""
        try:
            # Retrieve village management knowledge
            rag_context = await self.retrieve_rag_context(
                "village-level agricultural planning and resource management",
                top_k=3
            )
            
            # Aggregate village data
            analysis = await self._aggregate_village_data(context)
            
            # Generate recommendation
            recommendation = await self._generate_recommendation(
                analysis, context, rag_context
            )
            
            return self._create_output(
                query=query,
                analysis=analysis,
                recommendation=recommendation,
                confidence=analysis.get("confidence", 0.75),
                data_sources=["Village Farmer Data", "Resource Inventory", "Risk Assessment"],
                rag_context=rag_context,
                reasoning_chain=analysis.get("reasoning", []),
            )
        except Exception as e:
            return self._create_output(
                query=query,
                analysis={"error": str(e)},
                recommendation="Unable to analyze village data",
                confidence=0.0,
            )
    
    async def _aggregate_village_data(self, context: UserContext) -> Dict[str, Any]:
        """Aggregate village-level data."""
        reasoning = [
            "Aggregating village-level agricultural data",
        ]
        
        # Mock village statistics
        total_farmers = 250
        total_area_hectares = 500
        average_farm_size = total_area_hectares / total_farmers
        
        reasoning.append(f"Total farmers: {total_farmers}")
        reasoning.append(f"Total cultivated area: {total_area_hectares} hectares")
        reasoning.append(f"Average farm size: {average_farm_size:.1f} hectares")
        
        # Crop distribution
        crop_distribution = {
            "wheat": 0.35,
            "rice": 0.25,
            "cotton": 0.20,
            "pulses": 0.20,
        }
        
        reasoning.append("Crop distribution:")
        for crop, percentage in crop_distribution.items():
            area = total_area_hectares * percentage
            reasoning.append(f"  - {crop}: {percentage:.0%} ({area:.0f} hectares)")
        
        # Collective risks
        risks = {
            "water_scarcity": "medium",
            "pest_outbreak": "low",
            "market_volatility": "high",
            "soil_degradation": "medium",
        }
        
        reasoning.append("Collective risks:")
        for risk, level in risks.items():
            reasoning.append(f"  - {risk}: {level}")
        
        # Resource optimization opportunities
        opportunities = [
            "Implement collective water harvesting",
            "Coordinate crop rotation across village",
            "Bulk purchase of inputs at reduced cost",
            "Collective marketing for better prices",
        ]
        
        reasoning.extend(opportunities)
        
        return {
            "total_farmers": total_farmers,
            "total_area_hectares": total_area_hectares,
            "average_farm_size_hectares": average_farm_size,
            "crop_distribution": crop_distribution,
            "collective_risks": risks,
            "optimization_opportunities": opportunities,
            "confidence": 0.75,
            "reasoning": reasoning,
        }
    
    async def _generate_recommendation(
        self,
        analysis: Dict[str, Any],
        context: UserContext,
        rag_context: List[str],
    ) -> str:
        """Generate village-level recommendation."""
        if not self.llm_client:
            opportunities = analysis.get("optimization_opportunities", [])
            if opportunities:
                return "Village opportunities: " + "; ".join(opportunities[:2])
            else:
                return "Continue monitoring village-level metrics."
        
        opportunities_text = "\n".join(analysis.get("optimization_opportunities", []))
        risks_text = "\n".join([
            f"- {k}: {v}" for k, v in analysis.get("collective_risks", {}).items()
        ])
        
        prompt = f"""Provide village-level agricultural recommendations for collective benefit.

Village Statistics:
- Total Farmers: {analysis.get('total_farmers')}
- Total Area: {analysis.get('total_area_hectares')} hectares
- Average Farm Size: {analysis.get('average_farm_size_hectares'):.1f} hectares

Collective Risks:
{risks_text}

Optimization Opportunities:
{opportunities_text}

Knowledge Context:
{rag_context[0] if rag_context else 'No specific knowledge available'}

Provide 3-4 specific village-level recommendations including:
1. Collective resource management
2. Coordinated crop planning
3. Risk mitigation strategies
4. Market coordination initiatives"""
        
        try:
            return await self.call_llm(prompt, temperature=0.5, max_tokens=400)
        except Exception:
            opportunities = analysis.get("optimization_opportunities", [])
            if opportunities:
                return "Village opportunities: " + "; ".join(opportunities[:2])
            else:
                return "Continue monitoring village-level metrics."
