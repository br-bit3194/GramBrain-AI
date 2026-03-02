"""Soil Intelligence Agent - Analyzes soil health and provides recommendations."""

from typing import Dict, Any, List
from ..core.agent_base import Agent, AgentOutput, Query, UserContext


class SoilAgent(Agent):
    """Analyzes soil health and provides soil management recommendations."""
    
    def __init__(self):
        """Initialize soil agent."""
        super().__init__("soil_agent")
    
    async def analyze(self, query: Query, context: UserContext) -> AgentOutput:
        """
        Analyze soil health and generate recommendations.
        
        Args:
            query: User query
            context: User context with farm data
            
        Returns:
            Soil analysis and recommendations
        """
        try:
            # Fetch soil health data
            soil_data = await self._fetch_soil_data(context)
            
            # Retrieve RAG context about soil management
            rag_context = await self.retrieve_rag_context(
                f"soil amendments for {context.soil_type} and {context.crop_type}",
                top_k=3
            )
            
            # Analyze soil health
            analysis = await self._analyze_soil_health(soil_data, context)
            
            # Generate recommendation using LLM
            recommendation = await self._generate_recommendation(
                analysis, soil_data, context, rag_context
            )
            
            return self._create_output(
                query=query,
                analysis=analysis,
                recommendation=recommendation,
                confidence=analysis.get("confidence", 0.75),
                data_sources=["Soil Health Card", "Farm Records"],
                rag_context=rag_context,
                reasoning_chain=analysis.get("reasoning", []),
            )
        
        except Exception as e:
            return self._create_output(
                query=query,
                analysis={"error": str(e)},
                recommendation="Unable to analyze soil health due to data unavailability",
                confidence=0.0,
                data_sources=[],
                rag_context=[],
            )
    
    async def _fetch_soil_data(self, context: UserContext) -> Dict[str, Any]:
        """
        Fetch soil health data for farm.
        
        Args:
            context: User context with farm ID
            
        Returns:
            Soil data dictionary
        """
        # TODO: Integrate with Soil Health Card database
        # For now, return mock data
        return {
            "farm_id": context.farm_id,
            "soil_type": context.soil_type,
            "nitrogen_kg_per_ha": 180,
            "phosphorus_kg_per_ha": 25,
            "potassium_kg_per_ha": 280,
            "ph_level": 6.8,
            "organic_carbon_percent": 0.45,
            "electrical_conductivity": 0.35,
            "micronutrients": {
                "zinc": 0.8,
                "iron": 4.2,
                "copper": 0.6,
                "manganese": 3.5,
                "boron": 0.5,
            },
            "test_date": "2024-01-15",
        }
    
    async def _analyze_soil_health(
        self,
        soil_data: Dict[str, Any],
        context: UserContext,
    ) -> Dict[str, Any]:
        """
        Analyze soil health and identify deficiencies.
        
        Args:
            soil_data: Soil test data
            context: User context
            
        Returns:
            Analysis results
        """
        # Optimal ranges for different crops
        optimal_ranges = {
            "wheat": {"N": (200, 280), "P": (20, 40), "K": (200, 400)},
            "rice": {"N": (240, 320), "P": (25, 45), "K": (250, 450)},
            "cotton": {"N": (160, 240), "P": (15, 30), "K": (150, 300)},
            "pulses": {"N": (120, 200), "P": (15, 30), "K": (150, 300)},
        }
        
        crop = context.crop_type or "wheat"
        ranges = optimal_ranges.get(crop, optimal_ranges["wheat"])
        
        # Calculate health score (0-100)
        deficiencies = []
        reasoning = [f"Analyzed soil for {crop} cultivation"]
        
        # Check nitrogen
        n_level = soil_data.get("nitrogen_kg_per_ha", 0)
        n_min, n_max = ranges["N"]
        if n_level < n_min:
            deficiencies.append(f"Nitrogen deficient: {n_level}kg/ha (optimal: {n_min}-{n_max})")
        elif n_level > n_max:
            deficiencies.append(f"Nitrogen excess: {n_level}kg/ha (optimal: {n_min}-{n_max})")
        
        # Check phosphorus
        p_level = soil_data.get("phosphorus_kg_per_ha", 0)
        p_min, p_max = ranges["P"]
        if p_level < p_min:
            deficiencies.append(f"Phosphorus deficient: {p_level}kg/ha (optimal: {p_min}-{p_max})")
        
        # Check potassium
        k_level = soil_data.get("potassium_kg_per_ha", 0)
        k_min, k_max = ranges["K"]
        if k_level < k_min:
            deficiencies.append(f"Potassium deficient: {k_level}kg/ha (optimal: {k_min}-{k_max})")
        
        # Check pH
        ph = soil_data.get("ph_level", 7.0)
        if ph < 6.0 or ph > 8.0:
            deficiencies.append(f"pH level suboptimal: {ph} (optimal: 6.0-8.0)")
        
        # Check organic carbon
        oc = soil_data.get("organic_carbon_percent", 0)
        if oc < 0.5:
            deficiencies.append(f"Low organic carbon: {oc}% (optimal: >0.5%)")
        
        reasoning.extend(deficiencies)
        
        # Calculate health score
        health_score = 100
        health_score -= len(deficiencies) * 15
        health_score = max(0, min(100, health_score))
        
        return {
            "health_score": health_score,
            "deficiencies": deficiencies,
            "npk_levels": {
                "nitrogen": n_level,
                "phosphorus": p_level,
                "potassium": k_level,
            },
            "ph_level": ph,
            "organic_carbon": oc,
            "confidence": 0.85,
            "reasoning": reasoning,
        }
    
    async def _generate_recommendation(
        self,
        analysis: Dict[str, Any],
        soil_data: Dict[str, Any],
        context: UserContext,
        rag_context: List[str],
    ) -> str:
        """
        Generate soil management recommendation using LLM.
        
        Args:
            analysis: Soil analysis
            soil_data: Soil data
            context: User context
            rag_context: Retrieved knowledge
            
        Returns:
            Recommendation text
        """
        if not self.llm_client:
            # Fallback recommendation
            deficiencies = analysis.get("deficiencies", [])
            if deficiencies:
                return f"Address soil deficiencies: {', '.join(deficiencies[:2])}"
            else:
                return "Soil health is good. Maintain current practices."
        
        deficiencies_text = "\n".join(analysis.get("deficiencies", ["None detected"]))
        
        prompt = f"""Based on soil analysis, provide soil management recommendations.

Soil Analysis:
- Health Score: {analysis.get('health_score')}/100
- Deficiencies: {deficiencies_text}
- pH Level: {analysis.get('ph_level')}
- Organic Carbon: {analysis.get('organic_carbon')}%
- Crop: {context.crop_type}

Knowledge Context:
{rag_context[0] if rag_context else 'No specific knowledge available'}

Provide 2-3 specific, actionable soil management recommendations."""
        
        try:
            recommendation = await self.call_llm(prompt, temperature=0.5, max_tokens=300)
            return recommendation.strip()
        except Exception:
            # Fallback
            deficiencies = analysis.get("deficiencies", [])
            if deficiencies:
                return f"Address soil deficiencies: {', '.join(deficiencies[:2])}"
            else:
                return "Soil health is good. Maintain current practices."
