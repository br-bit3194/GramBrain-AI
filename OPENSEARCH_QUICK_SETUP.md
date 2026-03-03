# OpenSearch Quick Setup Guide

## ✅ Configuration Updated

Your `.env` file has been updated to use OpenSearch:

```env
VECTOR_DB_TYPE=opensearch
OPENSEARCH_ENDPOINT=https://search-grambraindomain-xxx.us-east-1.es.amazonaws.com
```

## 🚀 Quick Start

### 1. Seed Knowledge Base
```bash
cd backend
python seed_knowledge.py
```

This will add 10 agricultural knowledge chunks to OpenSearch covering:
- Wheat cultivation
- Rice cultivation
- Tomato cultivation
- Pest management
- Irrigation techniques
- Soil management
- Cotton cultivation
- Sugarcane cultivation
- Organic farming
- Weather management

### 2. Test Knowledge Search

#### Via API:
```bash
# Search knowledge
curl "http://localhost:8000/api/knowledge/search?query=wheat%20cultivation&top_k=3"

# Search with filters
curl "http://localhost:8000/api/knowledge/search?query=pest%20control&crop_type=tomato"
```

#### Via Python:
```python
import asyncio
from src.system import GramBrainSystem

async def test_search():
    system = GramBrainSystem(use_mock_llm=False, use_mock_rag=False)
    await system.initialize()
    
    results = await system.rag_client.search("wheat cultivation", top_k=3)
    for result in results:
        print(result['content'])
    
    system.shutdown()

asyncio.run(test_search())
```

## 📡 API Endpoints

### Add Knowledge
```bash
POST /api/knowledge
{
  "chunk_id": "custom-001",
  "content": "Your agricultural knowledge here",
  "source": "Source name",
  "topic": "crop_cultivation",
  "crop_type": "wheat",
  "region": "north_india"
}
```

### Search Knowledge
```bash
GET /api/knowledge/search?query=wheat&top_k=5&crop_type=wheat&region=north_india
```

### Bulk Add Knowledge
```bash
POST /api/knowledge/bulk
[
  {
    "chunk_id": "bulk-001",
    "content": "Knowledge 1",
    "source": "Source",
    "topic": "topic1"
  },
  {
    "chunk_id": "bulk-002",
    "content": "Knowledge 2",
    "source": "Source",
    "topic": "topic2"
  }
]
```

## 🔍 How RAG Works in Queries

When a user asks a question:

1. **Query Processing**: User query is sent to `/api/query`
2. **RAG Retrieval**: System searches OpenSearch for relevant knowledge
3. **Context Building**: Top-k results are formatted as context
4. **LLM Generation**: Context + query sent to Bedrock LLM
5. **Response**: AI-generated recommendation based on knowledge base

Example flow:
```
User: "How to grow wheat in North India?"
  ↓
RAG Search: Finds wheat cultivation knowledge
  ↓
Context: "Wheat requires well-drained loamy soil..."
  ↓
LLM: Generates personalized recommendation
  ↓
Response: Detailed wheat cultivation advice
```

## 📊 Knowledge Base Structure

Each knowledge chunk has:
- `chunk_id`: Unique identifier
- `content`: The actual knowledge text
- `source`: Where the knowledge came from
- `topic`: Category (crop_cultivation, pest_management, etc.)
- `crop_type`: Specific crop or "all"
- `region`: Geographic region or "all_india"
- `embedding`: 1536-dim vector (auto-generated)

## 🎯 Topics Available

- `crop_cultivation` - Growing specific crops
- `pest_management` - Pest and disease control
- `irrigation` - Water management
- `soil_management` - Soil health and fertility
- `organic_farming` - Organic practices
- `weather_management` - Weather-based advice
- `vegetable_cultivation` - Vegetable farming

## 🔧 Troubleshooting

### OpenSearch Connection Issues
```python
# Check if OpenSearch is accessible
from src.rag.vector_db import OpenSearchVectorDB
import os

db = OpenSearchVectorDB(
    endpoint=os.getenv("OPENSEARCH_ENDPOINT"),
    index_name="grambrain-knowledge"
)
print(f"OpenSearch available: {db.is_available()}")
```

### AWS Credentials
Make sure your AWS credentials have permissions for:
- OpenSearch domain access
- Bedrock model invocation (for embeddings)

### Index Not Found
The system auto-creates the index on first use. If you see errors:
```bash
# Manually create index
python -c "
from src.rag.vector_db import OpenSearchVectorDB
import asyncio
import os

async def create():
    db = OpenSearchVectorDB(os.getenv('OPENSEARCH_ENDPOINT'), 'grambrain-knowledge')
    await db.ensure_index_exists()
    
asyncio.run(create())
"
```

## 📈 Monitoring

Check OpenSearch health:
```bash
curl -X GET "https://your-opensearch-endpoint/_cluster/health?pretty"
```

Check index stats:
```bash
curl -X GET "https://your-opensearch-endpoint/grambrain-knowledge/_stats?pretty"
```

## 🎨 Frontend Integration

```typescript
// Search knowledge from frontend
async function searchKnowledge(query: string) {
  const response = await fetch(
    `http://localhost:8000/api/knowledge/search?query=${encodeURIComponent(query)}&top_k=5`,
    {
      headers: {
        'Authorization': `Bearer ${localStorage.getItem('access_token')}`
      }
    }
  );
  const data = await response.json();
  return data.data.results;
}

// Add knowledge from admin panel
async function addKnowledge(knowledge: any) {
  const response = await fetch('http://localhost:8000/api/knowledge', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${localStorage.getItem('access_token')}`
    },
    body: JSON.stringify(knowledge)
  });
  return response.json();
}
```

## ✅ Verification Checklist

- [ ] `.env` configured with OpenSearch endpoint
- [ ] Run `python seed_knowledge.py` successfully
- [ ] Test search via API
- [ ] Verify results are relevant
- [ ] Test in actual query flow
- [ ] Check OpenSearch dashboard for data

---

**Status**: ✅ OpenSearch configured and ready
**Knowledge Base**: 10 agricultural knowledge chunks seeded
**Next**: Use in production queries for better AI responses!
