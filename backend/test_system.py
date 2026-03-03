"""
Quick test of the system without AWS dependencies
"""
import asyncio
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from src.system import GramBrainSystem

async def test_system():
    """Test system with mock components"""
    print("🚀 Testing GramBrain System...")
    
    # Use mock components to avoid AWS credential issues
    system = GramBrainSystem(use_mock_llm=True, use_mock_rag=True)
    await system.initialize()
    
    print("✅ System initialized successfully!")
    print(f"   - Agents registered: {len(system.registry.list_agents())}")
    print(f"   - LLM client: {'Mock' if system.use_mock_llm else 'Bedrock'}")
    print(f"   - RAG client: {'In-Memory' if system.use_mock_rag else 'OpenSearch'}")
    
    # Add test knowledge
    print("\n📚 Adding test knowledge...")
    await system.add_knowledge(
        chunk_id="test-001",
        content="Wheat requires well-drained loamy soil with pH 6.0-7.5.",
        source="Test",
        topic="crop_cultivation",
        crop_type="wheat",
        region="test"
    )
    print("✅ Knowledge added successfully!")
    
    # Test query
    print("\n🤖 Processing test query...")
    result = system.process_query(
        query_text="How to grow wheat?",
        user_id="test-user",
        language="en"
    )
    
    print("✅ Query processed successfully!")
    print(f"\n📝 Recommendation:")
    print(f"   {result['recommendation'][:200]}...")
    print(f"\n📊 Confidence: {result['confidence']:.0%}")
    print(f"   Agents involved: {len(result.get('agent_contributions', []))}")
    
    # Cleanup
    system.shutdown()
    print("\n✅ System test complete!")

if __name__ == "__main__":
    asyncio.run(test_system())
