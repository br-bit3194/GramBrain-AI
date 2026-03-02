"""Yield Prediction Agent - Forecasts crop yields."""

from typing import Dict, Any, List
from ..core.agent_base import Agent, Query, UserContext


class YieldAgent(Agent):
    """Forecasts crop yields for harvest planning."""
    
    def __init__(self):
        """Initialize yield agent."""
        super().__init__("yield_agent")
    
    async def analyze(self, query: Query, context: UserContext) -> Any:
        """Predict crop yield."""
        try:
            # Retrieve yield prediction knowledge
            rag_context = await self.retrieve_rag_context(
                f"yield prediction for {context.crop_type}",
                top_k=3
            )
            
            # Predict yield
            analysis = await self._predict_yield(context)
            
            # Generate recommendation
            recommendation = await self._generate_recommendation(
                analysis, context, rag_context
            )
            
            return self._create_output(
                query=query,
                analysis=analysis,
                recommendation=recommendation,
                confidence=analysis.get("confidence", 0.75),
                data_sources=["Crop Data", "Historical Yields", "Soil Health"],
                rag_context=rag_context,
                reasoning_chain=analysis.get("reasoning", []),
            )
        except Exception as e:
            return self._create_output(
                query=query,
                analysis={"error": str(e)},
                recommendation="Unable to predict yield",
                confidence=0.0,
            )
    
    async def _predict_yield(self, context: UserContext) -> Dict[str, Any]:
        """Predict crop yield based on conditions."""
        # Average yields by crop (kg/hectare)
        average_yields = {
            "wheat": 4500,
            "rice": 5000,
            "cotton": 1800,
            "pulses": 1500,
        }
        
        crop = context.crop_type or "wheat"
        base_yield = average_yields.get(crop, 4000)
        
        reasoning = [
            f"Predicting yield for {crop}",
            f"Base yield: {base_yield} kg/hectare",
        ]
        
        # Adjust yield based on growth stage
        stage_multipliers = {
            "germination": 0.3,
            "vegetative": 0.5,
            "tillering": 0.6,
            "stem_elongation": 0.7,
            "flowering": 0.85,
            "grain_filling": 0.95,
            "maturity": 1.0,
        }
        
        stage = context.growth_stage or "vegetative"
        multiplier = stage_multipliers.get(stage, 0.5)
        predicted_yield = base_yield * multiplier
        
        reasoning.append(f"Growth stage: {stage} (confidence: {multiplier:.0%})")
        
        # Calculate confidence based on stage
        confidence = 0.5 + (multiplier * 0.5)  # 0.5-1.0 confidence
        
        # Estimate range
        lower_bound = predicted_yield * 0.85
        upper_bound = predicted_yield * 1.15
        
        reasoning.append(f"Predicted yield: {predicted_yield:.0f} kg/hectare")
        reasoning.append(f"Range: {lower_bound:.0f} - {upper_bound:.0f} kg/hectare")
        
        return {
            "crop_type": crop,
            "predicted_yield_kg_per_ha": predicted_yield,
            "lower_bound_kg_per_ha": lower_bound,
            "upper_bound_kg_per_ha": upper_bound,
            "confidence": confidence,
            "reasoning": reasoning,
        }
    
    async def _generate_recommendation(
        self,
        analysis: Dict[str, Any],
        context: UserContext,
        rag_context: List[str],
    ) -> str:
        """Generate yield forecast recommendation."""
        if not self.llm_client:
            yield_pred = analysis.get("predicted_yield_kg_per_ha", 0)
            lower = analysis.get("lower_bound_kg_per_ha", 0)
            upper = analysis.get("upper_bound_kg_per_ha", 0)
            return f"Expected yield: {yield_pred:.0f} kg/hectare (range: {lower:.0f}-{upper:.0f})"
        
        prompt = f"""Provide yield forecast and harvest planning recommendations.

Crop: {context.crop_type}
Growth Stage: {context.growth_stage}
Farm Size: {context.farm_size_hectares} hectares

Yield Forecast:
- Predicted: {analysis.get('predicted_yield_kg_per_ha'):.0f} kg/hectare
- Range: {analysis.get('lower_bound_kg_per_ha'):.0f} - {analysis.get('upper_bound_kg_per_ha'):.0f}
- Confidence: {analysis.get('confidence'):.0%}

Knowledge Context:
{rag_context[0] if rag_context else 'No specific knowledge available'}

Provide harvest planning recommendations including:
1. Expected harvest date
2. Harvest logistics
3. Post-harvest handling"""
        
        try:
            return await self.call_llm(prompt, temperature=0.5, max_tokens=350)
        except Exception:
            yield_pred = analysis.get("predicted_yield_kg_per_ha", 0)
            lower = analysis.get("lower_bound_kg_per_ha", 0)
            upper = analysis.get("upper_bound_kg_per_ha", 0)
            return f"Expected yield: {yield_pred:.0f} kg/hectare (range: {lower:.0f}-{upper:.0f})"
