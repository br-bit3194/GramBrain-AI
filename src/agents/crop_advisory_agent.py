"""Crop Advisory Agent - Provides crop-specific guidance."""

from typing import Dict, Any, List
from ..core.agent_base import Agent, Query, UserContext


class CropAdvisoryAgent(Agent):
    """Provides crop-specific guidance on planting, care, and harvest."""
    
    def __init__(self):
        """Initialize crop advisory agent."""
        super().__init__("crop_advisory_agent")
    
    async def analyze(self, query: Query, context: UserContext) -> Any:
        """Analyze crop conditions and generate recommendations."""
        try:
            # Retrieve crop-specific knowledge
            rag_context = await self.retrieve_rag_context(
                f"crop management for {context.crop_type} at {context.growth_stage} stage",
                top_k=3
            )
            
            # Analyze crop conditions
            analysis = await self._analyze_crop_conditions(context)
            
            # Generate recommendation
            recommendation = await self._generate_recommendation(
                analysis, context, rag_context
            )
            
            return self._create_output(
                query=query,
                analysis=analysis,
                recommendation=recommendation,
                confidence=analysis.get("confidence", 0.75),
                data_sources=["Crop Phenology Database", "Regional Crop Data"],
                rag_context=rag_context,
                reasoning_chain=analysis.get("reasoning", []),
            )
        except Exception as e:
            return self._create_output(
                query=query,
                analysis={"error": str(e)},
                recommendation="Unable to generate crop advisory",
                confidence=0.0,
            )
    
    async def _analyze_crop_conditions(self, context: UserContext) -> Dict[str, Any]:
        """Analyze current crop conditions."""
        crop_stages = {
            "wheat": ["germination", "tillering", "stem_elongation", "flowering", "grain_filling", "maturity"],
            "rice": ["germination", "tillering", "panicle_initiation", "flowering", "grain_filling", "maturity"],
            "cotton": ["germination", "vegetative", "flowering", "boll_development", "maturity"],
            "pulses": ["germination", "vegetative", "flowering", "pod_development", "maturity"],
        }
        
        crop = context.crop_type or "wheat"
        stages = crop_stages.get(crop, crop_stages["wheat"])
        current_stage_idx = stages.index(context.growth_stage) if context.growth_stage in stages else 0
        
        reasoning = [
            f"Analyzing {crop} at {context.growth_stage} stage",
            f"Progress: {current_stage_idx + 1}/{len(stages)} stages",
        ]
        
        # Recommend next actions based on stage
        stage_actions = {
            "germination": "Ensure proper soil moisture and temperature for seed germination",
            "tillering": "Monitor for nutrient deficiencies and pest activity",
            "stem_elongation": "Apply growth-stage appropriate fertilizers",
            "flowering": "Ensure adequate water and monitor for flower abortion",
            "grain_filling": "Maintain soil moisture and monitor for diseases",
            "maturity": "Prepare for harvest",
        }
        
        next_action = stage_actions.get(context.growth_stage, "Monitor crop health")
        reasoning.append(f"Recommended action: {next_action}")
        
        return {
            "crop_type": crop,
            "current_stage": context.growth_stage,
            "stage_progress": f"{current_stage_idx + 1}/{len(stages)}",
            "next_action": next_action,
            "confidence": 0.85,
            "reasoning": reasoning,
        }
    
    async def _generate_recommendation(
        self,
        analysis: Dict[str, Any],
        context: UserContext,
        rag_context: List[str],
    ) -> str:
        """Generate crop advisory recommendation."""
        if not self.llm_client:
            return analysis.get("next_action", "Monitor crop health")
        
        prompt = f"""Provide crop management advice based on current conditions.

Crop: {context.crop_type}
Growth Stage: {context.growth_stage}
Farm Size: {context.farm_size_hectares} hectares
Soil Type: {context.soil_type}

Current Analysis:
- Stage Progress: {analysis.get('stage_progress')}
- Recommended Action: {analysis.get('next_action')}

Knowledge Context:
{rag_context[0] if rag_context else 'No specific knowledge available'}

Provide 2-3 specific, actionable crop management recommendations."""
        
        try:
            return await self.call_llm(prompt, temperature=0.5, max_tokens=300)
        except Exception:
            return analysis.get("next_action", "Monitor crop health")
