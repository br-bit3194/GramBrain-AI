"""Sustainability Agent - Monitors environmental impact."""

from typing import Dict, Any, List
from ..core.agent_base import Agent, Query, UserContext


class SustainabilityAgent(Agent):
    """Monitors environmental impact and promotes sustainable practices."""
    
    def __init__(self):
        """Initialize sustainability agent."""
        super().__init__("sustainability_agent")
    
    async def analyze(self, query: Query, context: UserContext) -> Any:
        """Analyze sustainability metrics."""
        try:
            # Retrieve sustainability knowledge
            rag_context = await self.retrieve_rag_context(
                f"sustainable farming practices for {context.crop_type}",
                top_k=3
            )
            
            # Calculate sustainability metrics
            analysis = await self._calculate_sustainability_metrics(context)
            
            # Generate recommendation
            recommendation = await self._generate_recommendation(
                analysis, context, rag_context
            )
            
            return self._create_output(
                query=query,
                analysis=analysis,
                recommendation=recommendation,
                confidence=analysis.get("confidence", 0.75),
                data_sources=["Soil Health Data", "Water Usage Records", "Input Records"],
                rag_context=rag_context,
                reasoning_chain=analysis.get("reasoning", []),
            )
        except Exception as e:
            return self._create_output(
                query=query,
                analysis={"error": str(e)},
                recommendation="Unable to calculate sustainability metrics",
                confidence=0.0,
            )
    
    async def _calculate_sustainability_metrics(self, context: UserContext) -> Dict[str, Any]:
        """Calculate sustainability metrics."""
        reasoning = [
            f"Calculating sustainability metrics for {context.crop_type}",
        ]
        
        # Soil health score (0-100)
        soil_health_score = 65  # Mock value
        reasoning.append(f"Soil health score: {soil_health_score}/100")
        
        # Water efficiency score (0-100)
        water_efficiency_score = 72  # Mock value
        reasoning.append(f"Water efficiency score: {water_efficiency_score}/100")
        
        # Carbon footprint (kg CO2/hectare)
        carbon_footprint = 850  # Mock value
        reasoning.append(f"Carbon footprint: {carbon_footprint} kg CO2/hectare")
        
        # Overall sustainability index (0-100)
        sustainability_index = (soil_health_score + water_efficiency_score) / 2
        reasoning.append(f"Overall sustainability index: {sustainability_index:.0f}/100")
        
        # Recommendations for improvement
        improvements = []
        if soil_health_score < 70:
            improvements.append("Increase organic matter through composting")
        if water_efficiency_score < 75:
            improvements.append("Consider drip irrigation for water savings")
        if carbon_footprint > 800:
            improvements.append("Reduce chemical inputs and adopt conservation agriculture")
        
        reasoning.extend(improvements)
        
        return {
            "crop_type": context.crop_type,
            "soil_health_score": soil_health_score,
            "water_efficiency_score": water_efficiency_score,
            "carbon_footprint_kg_co2_per_ha": carbon_footprint,
            "sustainability_index": sustainability_index,
            "improvement_recommendations": improvements,
            "confidence": 0.75,
            "reasoning": reasoning,
        }
    
    async def _generate_recommendation(
        self,
        analysis: Dict[str, Any],
        context: UserContext,
        rag_context: List[str],
    ) -> str:
        """Generate sustainability recommendation."""
        if not self.llm_client:
            improvements = analysis.get("improvement_recommendations", [])
            if improvements:
                return "Recommendations: " + "; ".join(improvements[:2])
            else:
                return "Current practices are sustainable. Continue monitoring."
        
        improvements_text = "\n".join(analysis.get("improvement_recommendations", []))
        
        prompt = f"""Provide sustainability recommendations based on environmental metrics.

Crop: {context.crop_type}
Soil Health Score: {analysis.get('soil_health_score')}/100
Water Efficiency Score: {analysis.get('water_efficiency_score')}/100
Carbon Footprint: {analysis.get('carbon_footprint_kg_co2_per_ha')} kg CO2/hectare
Sustainability Index: {analysis.get('sustainability_index'):.0f}/100

Improvement Areas:
{improvements_text if improvements_text else 'No critical areas identified'}

Knowledge Context:
{rag_context[0] if rag_context else 'No specific knowledge available'}

Provide 3-4 specific, actionable sustainability recommendations including:
1. Soil conservation practices
2. Water management improvements
3. Carbon reduction strategies
4. Organic farming transitions"""
        
        try:
            return await self.call_llm(prompt, temperature=0.5, max_tokens=400)
        except Exception:
            improvements = analysis.get("improvement_recommendations", [])
            if improvements:
                return "Recommendations: " + "; ".join(improvements[:2])
            else:
                return "Current practices are sustainable. Continue monitoring."
