"""Farmer Interaction Agent - Handles multilingual voice and text interactions."""

from typing import Dict, Any, List
from ..core.agent_base import Agent, Query, UserContext


class FarmerInteractionAgent(Agent):
    """Handles multilingual voice and text interactions with farmers."""
    
    def __init__(self):
        """Initialize farmer interaction agent."""
        super().__init__("farmer_interaction_agent")
    
    async def analyze(self, query: Query, context: UserContext) -> Any:
        """Process farmer interaction and adapt communication."""
        try:
            # Retrieve communication knowledge
            rag_context = await self.retrieve_rag_context(
                "farmer communication and agricultural terminology",
                top_k=2
            )
            
            # Process interaction
            analysis = await self._process_interaction(query, context)
            
            # Generate recommendation
            recommendation = await self._generate_recommendation(
                analysis, query, context, rag_context
            )
            
            return self._create_output(
                query=query,
                analysis=analysis,
                recommendation=recommendation,
                confidence=analysis.get("confidence", 0.85),
                data_sources=["Language Processing", "Farmer Profiles"],
                rag_context=rag_context,
                reasoning_chain=analysis.get("reasoning", []),
            )
        except Exception as e:
            return self._create_output(
                query=query,
                analysis={"error": str(e)},
                recommendation="Unable to process interaction",
                confidence=0.0,
            )
    
    async def _process_interaction(self, query: Query, context: UserContext) -> Dict[str, Any]:
        """Process farmer interaction."""
        reasoning = [
            f"Processing farmer interaction in {context.language_preference}",
        ]
        
        # Supported languages
        supported_languages = {
            "hi": "Hindi",
            "mr": "Marathi",
            "pa": "Punjabi",
            "ta": "Tamil",
            "te": "Telugu",
            "kn": "Kannada",
            "ml": "Malayalam",
            "gu": "Gujarati",
            "bn": "Bengali",
            "en": "English",
        }
        
        language = context.language_preference or "en"
        language_name = supported_languages.get(language, "English")
        
        reasoning.append(f"Language: {language_name}")
        
        # Detect if voice input
        is_voice = query.text.startswith("VOICE:")
        if is_voice:
            reasoning.append("Input type: Voice")
            query_text = query.text.replace("VOICE:", "").strip()
        else:
            reasoning.append("Input type: Text")
            query_text = query.text
        
        # Simplify technical terms for low-literacy users
        simplified_query = self._simplify_technical_terms(query_text)
        reasoning.append(f"Simplified query: {simplified_query}")
        
        return {
            "language": language,
            "language_name": language_name,
            "is_voice_input": is_voice,
            "original_query": query_text,
            "simplified_query": simplified_query,
            "confidence": 0.85,
            "reasoning": reasoning,
        }
    
    def _simplify_technical_terms(self, text: str) -> str:
        """Simplify technical agricultural terms."""
        simplifications = {
            "irrigation": "pani dena",
            "fertilizer": "khad",
            "pesticide": "keet nashi",
            "soil health": "mitti ki sehat",
            "yield": "fasal",
            "crop": "fasal",
            "disease": "bimari",
            "pest": "keet",
        }
        
        simplified = text.lower()
        for technical, simple in simplifications.items():
            simplified = simplified.replace(technical, simple)
        
        return simplified
    
    async def _generate_recommendation(
        self,
        analysis: Dict[str, Any],
        query: Query,
        context: UserContext,
        rag_context: List[str],
    ) -> str:
        """Generate farmer-friendly recommendation."""
        if not self.llm_client:
            language = analysis.get("language_name", "English")
            return f"Recommendation will be provided in {language} with simple language and visual aids."
        
        prompt = f"""Provide a farmer-friendly response adapted for low-literacy users.

Original Query: {analysis.get('original_query')}
Simplified Query: {analysis.get('simplified_query')}
Language: {analysis.get('language_name')}
Farmer Role: {context.role}

Guidelines:
1. Use simple, everyday language
2. Avoid technical jargon
3. Use local examples and references
4. Provide step-by-step instructions
5. Include visual descriptions

Knowledge Context:
{rag_context[0] if rag_context else 'No specific knowledge available'}

Provide a clear, actionable response suitable for a farmer with basic literacy."""
        
        try:
            return await self.call_llm(prompt, temperature=0.6, max_tokens=300)
        except Exception:
            language = analysis.get("language_name", "English")
            return f"Recommendation will be provided in {language} with simple language and visual aids."
