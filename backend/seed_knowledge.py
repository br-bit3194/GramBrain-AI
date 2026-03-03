"""
Seed knowledge base with agricultural data
"""
import asyncio
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from src.system import GramBrainSystem

# Sample agricultural knowledge
KNOWLEDGE_DATA = [
    {
        "chunk_id": "wheat-001",
        "content": "Wheat requires well-drained loamy soil with pH 6.0-7.5. Best sowing time is October-November in North India. Requires 4-5 irrigations during growing season.",
        "source": "ICAR Guidelines",
        "topic": "crop_cultivation",
        "crop_type": "wheat",
        "region": "north_india"
    },
    {
        "chunk_id": "rice-001",
        "content": "Rice cultivation requires flooded conditions. Transplanting should be done 25-30 days after sowing. Maintain 5-10cm water level during vegetative stage.",
        "source": "ICAR Guidelines",
        "topic": "crop_cultivation",
        "crop_type": "rice",
        "region": "all_india"
    },
    {
        "chunk_id": "tomato-001",
        "content": "Tomatoes need well-drained soil rich in organic matter. Spacing: 60cm x 45cm. Apply NPK fertilizer at 120:80:80 kg/ha. Protect from fruit borer and leaf curl virus.",
        "source": "Horticulture Department",
        "topic": "vegetable_cultivation",
        "crop_type": "tomato",
        "region": "all_india"
    },
    {
        "chunk_id": "pest-001",
        "content": "Aphids can be controlled using neem oil spray (5ml/liter water). Apply early morning or evening. Repeat every 7-10 days. Avoid chemical pesticides during flowering.",
        "source": "IPM Guidelines",
        "topic": "pest_management",
        "crop_type": "all",
        "region": "all_india"
    },
    {
        "chunk_id": "irrigation-001",
        "content": "Drip irrigation saves 40-60% water compared to flood irrigation. Install drippers at 30cm spacing. Irrigate daily for 1-2 hours. Monitor soil moisture regularly.",
        "source": "Water Management Institute",
        "topic": "irrigation",
        "crop_type": "all",
        "region": "all_india"
    },
    {
        "chunk_id": "soil-001",
        "content": "Soil testing should be done every 2-3 years. NPK levels, pH, and organic carbon are key parameters. Add compost or FYM at 10-15 tons/ha to improve soil health.",
        "source": "Soil Health Card Scheme",
        "topic": "soil_management",
        "crop_type": "all",
        "region": "all_india"
    },
    {
        "chunk_id": "cotton-001",
        "content": "Cotton requires 180-200 frost-free days. Sow in April-May. Spacing: 90cm x 60cm. Watch for pink bollworm and whitefly. Use pheromone traps for monitoring.",
        "source": "Cotton Corporation",
        "topic": "crop_cultivation",
        "crop_type": "cotton",
        "region": "central_india"
    },
    {
        "chunk_id": "sugarcane-001",
        "content": "Sugarcane needs 1500-2500mm rainfall or equivalent irrigation. Plant spacing: 90cm rows. Apply nitrogen in 3 splits. Harvest at 10-12 months for maximum sugar content.",
        "source": "Sugar Institute",
        "topic": "crop_cultivation",
        "crop_type": "sugarcane",
        "region": "all_india"
    },
    {
        "chunk_id": "organic-001",
        "content": "Organic farming uses no synthetic chemicals. Use vermicompost, green manure, and bio-fertilizers. Crop rotation and mixed cropping improve soil health naturally.",
        "source": "Organic Farming Association",
        "topic": "organic_farming",
        "crop_type": "all",
        "region": "all_india"
    },
    {
        "chunk_id": "weather-001",
        "content": "Monitor weather forecasts daily. Avoid irrigation before expected rain. Harvest before heavy rain to prevent crop damage. Use weather-based agro-advisories.",
        "source": "IMD Agro-Met",
        "topic": "weather_management",
        "crop_type": "all",
        "region": "all_india"
    },
]

async def seed_knowledge():
    """Seed knowledge base with agricultural data"""
    print("Initializing GramBrain system...")
    system = GramBrainSystem(use_mock_llm=False, use_mock_rag=False)
    await system.initialize()
    
    print(f"\nAdding {len(KNOWLEDGE_DATA)} knowledge chunks to OpenSearch...")
    
    added = 0
    for item in KNOWLEDGE_DATA:
        try:
            await system.add_knowledge(
                chunk_id=item["chunk_id"],
                content=item["content"],
                source=item["source"],
                topic=item["topic"],
                crop_type=item.get("crop_type"),
                region=item.get("region"),
            )
            added += 1
            print(f"✓ Added: {item['chunk_id']}")
        except Exception as e:
            print(f"✗ Failed to add {item['chunk_id']}: {e}")
    
    print(f"\n✅ Successfully added {added}/{len(KNOWLEDGE_DATA)} knowledge chunks")
    
    # Test search
    print("\n🔍 Testing search...")
    results = await system.rag_client.search("wheat cultivation", top_k=3)
    print(f"Found {len(results)} results for 'wheat cultivation'")
    for i, result in enumerate(results, 1):
        print(f"{i}. {result.get('chunk_id', 'unknown')}: {result.get('content', '')[:100]}...")
    
    system.shutdown()
    print("\n✅ Knowledge base seeded successfully!")

if __name__ == "__main__":
    asyncio.run(seed_knowledge())
