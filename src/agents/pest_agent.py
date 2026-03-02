"""Pest and Disease Agent - Detects and predicts pest/disease outbreaks."""

from typing import Dict, Any, List
from ..core.agent_base import Agent, Query, UserContext


class PestAgent(Agent):
    """Detects and predicts pest and disease outbreaks."""
    
    def __init__(self):
        """Initialize pest agent."""
        super().__init__("pest_agent")
    
    async def analyze(self, query: Query, context: UserContext) -> Any:
        """Analyze pest and disease risks."""
        try:
            # Retrieve pest management knowledge
            rag_context = await self.retrieve_rag_context(
                f"pest and disease management for {context.crop_type}",
                top_k=3
            )
            
            # Analyze pest risk
            analysis = await self._analyze_pest_risk(context)
            
            # Generate recommendation
            recommendation = await self._generate_recommendation(
                analysis, context, rag_context
            )
            
            return self._create_output(
                query=query,
                analysis=analysis,
                recommendation=recommendation,
                confidence=analysis.get("confidence", 0.70),
                data_sources=["Weather Data", "Regional Pest Reports"],
                rag_context=rag_context,
                reasoning_chain=analysis.get("reasoning", []),
            )
        except Exception as e:
            return self._create_output(
                query=query,
                analysis={"error": str(e)},
                recommendation="Unable to analyze pest risk",
                confidence=0.0,
            )
    
    async def _analyze_pest_risk(self, context: UserContext) -> Dict[str, Any]:
        """Analyze pest and disease risk."""
        # Pest risk factors by crop
        pest_risks = {
            "wheat": {
                "high_risk": ["armyworm", "aphids", "rust"],
                "medium_risk": ["thrips", "mites"],
            },
            "rice": {
                "high_risk": ["stem_borer", "leaf_folder", "blast"],
                "medium_risk": ["brown_spot", "sheath_blight"],
            },
            "cotton": {
                "high_risk": ["bollworm", "whitefly", "leaf_curl"],
                "medium_risk": ["spider_mites", "jassids"],
            },
            "pulses": {
                "high_risk": ["pod_borer", "leaf_miner"],
                "medium_risk": ["aphids", "thrips"],
            },
        }
        
        crop = context.crop_type or "wheat"
        risks = pest_risks.get(crop, pest_risks["wheat"])
        
        reasoning = [
            f"Analyzing pest risk for {crop}",
            f"Growth stage: {context.growth_stage}",
        ]
        
        # Determine risk level based on conditions
        risk_level = "low"
        if context.growth_stage in ["flowering", "grain_filling"]:
            risk_level = "high"
        elif context.growth_stage in ["vegetative", "stem_elongation"]:
            risk_level = "medium"
        
        reasoning.append(f"Risk level: {risk_level}")
        
        high_risk_pests = risks.get("high_risk", [])
        medium_risk_pests = risks.get("medium_risk", [])
        
        return {
            "crop_type": crop,
            "risk_level": risk_level,
            "high_risk_pests": high_risk_pests,
            "medium_risk_pests": medium_risk_pests,
            "confidence": 0.75,
            "reasoning": reasoning,
        }
    
    async def _generate_recommendation(
        self,
        analysis: Dict[str, Any],
        context: UserContext,
        rag_context: List[str],
    ) -> str:
        """Generate pest management recommendation."""
        if not self.llm_client:
            risk_level = analysis.get("risk_level", "low")
            if risk_level == "high":
                return "High pest risk detected. Implement preventive measures and monitor closely."
            elif risk_level == "medium":
                return "Moderate pest risk. Monitor crop regularly and prepare treatment options."
            else:
                return "Low pest risk. Continue regular monitoring."
        
        high_risk = ", ".join(analysis.get("high_risk_pests", []))
        medium_risk = ", ".join(analysis.get("medium_risk_pests", []))
        
        prompt = f"""Provide pest management recommendations based on risk analysis.

Crop: {context.crop_type}
Growth Stage: {context.growth_stage}
Risk Level: {analysis.get('risk_level')}
High Risk Pests: {high_risk}
Medium Risk Pests: {medium_risk}

Knowledge Context:
{rag_context[0] if rag_context else 'No specific knowledge available'}

Provide specific pest management recommendations including:
1. Preventive measures
2. Monitoring strategies
3. Treatment options if needed"""
        
        try:
            return await self.call_llm(prompt, temperature=0.5, max_tokens=400)
        except Exception:
            risk_level = analysis.get("risk_level", "low")
            if risk_level == "high":
                return "High pest risk detected. Implement preventive measures and monitor closely."
            else:
                return "Monitor crop regularly for pest activity."
