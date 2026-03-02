"""Marketplace Agent - Manages product listings and Pure Product Scores."""

from typing import Dict, Any, List
from ..core.agent_base import Agent, Query, UserContext


class MarketplaceAgent(Agent):
    """Manages product listings and calculates Pure Product Scores."""
    
    def __init__(self):
        """Initialize marketplace agent."""
        super().__init__("marketplace_agent")
    
    async def analyze(self, query: Query, context: UserContext) -> Any:
        """Calculate Pure Product Score for products."""
        try:
            # Retrieve marketplace knowledge
            rag_context = await self.retrieve_rag_context(
                "organic certification and product authenticity",
                top_k=3
            )
            
            # Calculate Pure Product Score
            analysis = await self._calculate_pure_product_score(context)
            
            # Generate recommendation
            recommendation = await self._generate_recommendation(
                analysis, context, rag_context
            )
            
            return self._create_output(
                query=query,
                analysis=analysis,
                recommendation=recommendation,
                confidence=analysis.get("confidence", 0.80),
                data_sources=["Farm Data", "Soil Health", "Sustainability Metrics"],
                rag_context=rag_context,
                reasoning_chain=analysis.get("reasoning", []),
            )
        except Exception as e:
            return self._create_output(
                query=query,
                analysis={"error": str(e)},
                recommendation="Unable to calculate Pure Product Score",
                confidence=0.0,
            )
    
    async def _calculate_pure_product_score(self, context: UserContext) -> Dict[str, Any]:
        """Calculate Pure Product Score (0-100)."""
        reasoning = [
            f"Calculating Pure Product Score for {context.crop_type}",
        ]
        
        # Traceability score (0-30)
        traceability_score = 25  # Mock value
        reasoning.append(f"Traceability score: {traceability_score}/30")
        
        # Sustainability score (0-40)
        sustainability_score = 32  # Mock value
        reasoning.append(f"Sustainability score: {sustainability_score}/40")
        
        # Quality score (0-30)
        quality_score = 24  # Mock value
        reasoning.append(f"Quality score: {quality_score}/30")
        
        # Overall Pure Product Score
        overall_score = traceability_score + sustainability_score + quality_score
        
        # Determine category
        if overall_score >= 85:
            category = "Pure"
        elif overall_score >= 70:
            category = "Organic"
        elif overall_score >= 50:
            category = "Sustainable"
        else:
            category = "Conventional"
        
        reasoning.append(f"Overall Pure Product Score: {overall_score}/100 ({category})")
        
        # Score breakdown
        breakdown = {
            "traceability": {
                "score": traceability_score,
                "max": 30,
                "factors": ["Farm location verified", "Harvest date recorded", "Farmer profile complete"],
            },
            "sustainability": {
                "score": sustainability_score,
                "max": 40,
                "factors": ["Soil health maintained", "Water efficient", "Low chemical inputs"],
            },
            "quality": {
                "score": quality_score,
                "max": 30,
                "factors": ["No visible defects", "Proper storage", "Timely harvest"],
            },
        }
        
        return {
            "crop_type": context.crop_type,
            "overall_score": overall_score,
            "category": category,
            "traceability_score": traceability_score,
            "sustainability_score": sustainability_score,
            "quality_score": quality_score,
            "score_breakdown": breakdown,
            "confidence": 0.80,
            "reasoning": reasoning,
        }
    
    async def _generate_recommendation(
        self,
        analysis: Dict[str, Any],
        context: UserContext,
        rag_context: List[str],
    ) -> str:
        """Generate marketplace recommendation."""
        if not self.llm_client:
            score = analysis.get("overall_score", 0)
            category = analysis.get("category", "Conventional")
            return f"Pure Product Score: {score}/100 ({category}). Product is suitable for marketplace listing."
        
        breakdown = analysis.get("score_breakdown", {})
        breakdown_text = "\n".join([
            f"- {k.title()}: {v['score']}/{v['max']}"
            for k, v in breakdown.items()
        ])
        
        prompt = f"""Provide marketplace listing recommendations based on Pure Product Score.

Crop: {context.crop_type}
Pure Product Score: {analysis.get('overall_score')}/100
Category: {analysis.get('category')}

Score Breakdown:
{breakdown_text}

Knowledge Context:
{rag_context[0] if rag_context else 'No specific knowledge available'}

Provide recommendations including:
1. Marketplace listing strategy
2. Target consumer segments
3. Pricing recommendations
4. Areas for score improvement"""
        
        try:
            return await self.call_llm(prompt, temperature=0.5, max_tokens=350)
        except Exception:
            score = analysis.get("overall_score", 0)
            category = analysis.get("category", "Conventional")
            return f"Pure Product Score: {score}/100 ({category}). Product is suitable for marketplace listing."
