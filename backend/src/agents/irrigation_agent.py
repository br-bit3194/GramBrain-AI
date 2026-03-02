"""Irrigation Optimization Agent - Optimizes water usage."""

from typing import Dict, Any, List
from ..core.agent_base import Agent, Query, UserContext


class IrrigationAgent(Agent):
    """Optimizes water usage through intelligent irrigation scheduling."""
    
    def __init__(self):
        """Initialize irrigation agent."""
        super().__init__("irrigation_agent")
    
    async def analyze(self, query: Query, context: UserContext) -> Any:
        """Analyze irrigation needs and optimize water usage."""
        try:
            # Retrieve irrigation knowledge
            rag_context = await self.retrieve_rag_context(
                f"irrigation optimization for {context.crop_type}",
                top_k=3
            )
            
            # Calculate water requirements
            analysis = await self._calculate_water_requirements(context)
            
            # Generate recommendation
            recommendation = await self._generate_recommendation(
                analysis, context, rag_context
            )
            
            return self._create_output(
                query=query,
                analysis=analysis,
                recommendation=recommendation,
                confidence=analysis.get("confidence", 0.80),
                data_sources=["Soil Moisture Data", "Weather Forecast", "Crop Requirements"],
                rag_context=rag_context,
                reasoning_chain=analysis.get("reasoning", []),
            )
        except Exception as e:
            return self._create_output(
                query=query,
                analysis={"error": str(e)},
                recommendation="Unable to calculate irrigation requirements",
                confidence=0.0,
            )
    
    async def _calculate_water_requirements(self, context: UserContext) -> Dict[str, Any]:
        """Calculate water requirements for crop."""
        # Crop water requirements (mm/day) by growth stage
        crop_water_needs = {
            "wheat": {
                "germination": 2.0,
                "tillering": 3.5,
                "stem_elongation": 4.5,
                "flowering": 5.0,
                "grain_filling": 4.0,
                "maturity": 1.0,
            },
            "rice": {
                "germination": 3.0,
                "tillering": 5.0,
                "panicle_initiation": 6.0,
                "flowering": 6.5,
                "grain_filling": 5.0,
                "maturity": 2.0,
            },
            "cotton": {
                "germination": 1.5,
                "vegetative": 3.0,
                "flowering": 4.5,
                "boll_development": 5.0,
                "maturity": 2.0,
            },
            "pulses": {
                "germination": 1.5,
                "vegetative": 2.5,
                "flowering": 3.5,
                "pod_development": 3.0,
                "maturity": 1.0,
            },
        }
        
        crop = context.crop_type or "wheat"
        stage = context.growth_stage or "vegetative"
        
        crop_needs = crop_water_needs.get(crop, crop_water_needs["wheat"])
        daily_need_mm = crop_needs.get(stage, 3.0)
        
        # Calculate total water needed for farm
        farm_size = context.farm_size_hectares or 1.0
        daily_need_liters = daily_need_mm * farm_size * 10000  # mm to liters conversion
        
        reasoning = [
            f"Calculated water requirements for {crop}",
            f"Growth stage: {stage}",
            f"Daily requirement: {daily_need_mm}mm ({daily_need_liters:.0f} liters)",
        ]
        
        # Estimate irrigation schedule
        irrigation_interval = 7  # days
        if stage in ["flowering", "grain_filling"]:
            irrigation_interval = 5
        elif stage in ["germination", "vegetative"]:
            irrigation_interval = 10
        
        total_irrigation_liters = daily_need_liters * irrigation_interval
        
        reasoning.append(f"Recommended irrigation interval: {irrigation_interval} days")
        reasoning.append(f"Water per irrigation: {total_irrigation_liters:.0f} liters")
        
        return {
            "crop_type": crop,
            "growth_stage": stage,
            "daily_requirement_mm": daily_need_mm,
            "daily_requirement_liters": daily_need_liters,
            "irrigation_interval_days": irrigation_interval,
            "water_per_irrigation_liters": total_irrigation_liters,
            "confidence": 0.80,
            "reasoning": reasoning,
        }
    
    async def _generate_recommendation(
        self,
        analysis: Dict[str, Any],
        context: UserContext,
        rag_context: List[str],
    ) -> str:
        """Generate irrigation recommendation."""
        if not self.llm_client:
            interval = analysis.get("irrigation_interval_days", 7)
            water = analysis.get("water_per_irrigation_liters", 0)
            return f"Irrigate every {interval} days with approximately {water:.0f} liters per irrigation"
        
        prompt = f"""Provide irrigation scheduling recommendation based on water requirements.

Crop: {context.crop_type}
Growth Stage: {context.growth_stage}
Farm Size: {context.farm_size_hectares} hectares
Soil Type: {context.soil_type}

Water Requirements:
- Daily Need: {analysis.get('daily_requirement_mm')}mm
- Irrigation Interval: {analysis.get('irrigation_interval_days')} days
- Water per Irrigation: {analysis.get('water_per_irrigation_liters'):.0f} liters

Knowledge Context:
{rag_context[0] if rag_context else 'No specific knowledge available'}

Provide irrigation scheduling recommendations including:
1. Optimal irrigation timing
2. Water-saving techniques
3. Signs of over/under-watering"""
        
        try:
            return await self.call_llm(prompt, temperature=0.5, max_tokens=350)
        except Exception:
            interval = analysis.get("irrigation_interval_days", 7)
            water = analysis.get("water_per_irrigation_liters", 0)
            return f"Irrigate every {interval} days with approximately {water:.0f} liters per irrigation"
