# GramBrain AI - REST API Documentation

## Overview

GramBrain AI provides a comprehensive REST API for accessing agricultural intelligence, managing farms, processing queries, and managing marketplace products.

## Base URL

```
http://localhost:8000/api/v1
```

## Authentication

Currently, the API uses no authentication. In production, implement JWT or OAuth2.

## Response Format

All responses follow this format:

```json
{
  "status": "success|error",
  "data": {},
  "message": "Optional message"
}
```

## Error Handling

Errors return appropriate HTTP status codes:

- `200 OK`: Successful request
- `400 Bad Request`: Invalid parameters
- `404 Not Found`: Resource not found
- `500 Internal Server Error`: Server error

Error response:

```json
{
  "status": "error",
  "detail": "Error message"
}
```

---

## Endpoints

### Health Check

#### GET /health

Check system health and available agents.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2024-01-15T10:30:00",
  "agents": ["weather_agent", "soil_agent", ...]
}
```

---

## User Management

### Create User

#### POST /users

Create a new user account.

**Parameters:**
- `phone_number` (string, required): User's phone number
- `name` (string, required): User's name
- `language_preference` (string, optional): Language code (default: "en")
- `role` (string, optional): User role - "farmer", "village_leader", "policymaker", "consumer" (default: "farmer")

**Example:**
```bash
curl -X POST "http://localhost:8000/api/v1/users" \
  -H "Content-Type: application/json" \
  -d '{
    "phone_number": "9876543210",
    "name": "Ramesh Kumar",
    "language_preference": "hi",
    "role": "farmer"
  }'
```

**Response:**
```json
{
  "status": "success",
  "user": {
    "user_id": "uuid",
    "phone_number": "9876543210",
    "name": "Ramesh Kumar",
    "language_preference": "hi",
    "role": "farmer",
    "created_at": "2024-01-15T10:30:00"
  }
}
```

### Get User

#### GET /users/{user_id}

Retrieve user details.

**Parameters:**
- `user_id` (string, required): User ID

**Response:**
```json
{
  "status": "success",
  "user": { ... }
}
```

---

## Farm Management

### Create Farm

#### POST /farms

Create a new farm.

**Parameters:**
- `owner_id` (string, required): User ID of farm owner
- `latitude` (float, required): Farm latitude
- `longitude` (float, required): Farm longitude
- `area_hectares` (float, required): Farm size in hectares
- `soil_type` (string, required): Soil type (e.g., "loamy", "clay", "sandy")
- `irrigation_type` (string, optional): "drip", "flood", "sprinkler", "rainfed" (default: "drip")

**Example:**
```bash
curl -X POST "http://localhost:8000/api/v1/farms" \
  -H "Content-Type: application/json" \
  -d '{
    "owner_id": "user_001",
    "latitude": 28.5,
    "longitude": 77.0,
    "area_hectares": 2.5,
    "soil_type": "loamy",
    "irrigation_type": "drip"
  }'
```

**Response:**
```json
{
  "status": "success",
  "farm": {
    "farm_id": "uuid",
    "owner_id": "user_001",
    "location": {"lat": 28.5, "lon": 77.0},
    "area_hectares": 2.5,
    "soil_type": "loamy",
    "irrigation_type": "drip",
    "created_at": "2024-01-15T10:30:00"
  }
}
```

### Get Farm

#### GET /farms/{farm_id}

Retrieve farm details.

### List User Farms

#### GET /users/{user_id}/farms

List all farms owned by a user.

---

## Query & Recommendations

### Process Query

#### POST /query

Process a user query and get AI recommendations.

**Parameters:**
- `user_id` (string, required): User ID
- `query_text` (string, required): User's question
- `farm_id` (string, optional): Farm ID
- `latitude` (float, optional): Farm latitude
- `longitude` (float, optional): Farm longitude
- `farm_size_hectares` (float, optional): Farm size
- `crop_type` (string, optional): Current crop (e.g., "wheat", "rice", "cotton")
- `growth_stage` (string, optional): Growth stage (e.g., "germination", "tillering", "flowering")
- `soil_type` (string, optional): Soil type
- `language` (string, optional): Response language (default: "en")

**Example:**
```bash
curl -X POST "http://localhost:8000/api/v1/query" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "farmer_001",
    "query_text": "Should I irrigate my wheat field?",
    "farm_id": "farm_001",
    "latitude": 28.5,
    "longitude": 77.0,
    "farm_size_hectares": 2.0,
    "crop_type": "wheat",
    "growth_stage": "tillering",
    "soil_type": "loamy",
    "language": "en"
  }'
```

**Response:**
```json
{
  "status": "success",
  "recommendation": {
    "recommendation_id": "uuid",
    "query_id": "uuid",
    "user_id": "farmer_001",
    "farm_id": "farm_001",
    "timestamp": "2024-01-15T10:30:00",
    "recommendation_text": "Based on weather forecast and soil moisture...",
    "reasoning_chain": [
      "Analyzed weather forecast for next 3 days",
      "Expected rainfall: 15mm",
      "Soil moisture adequate",
      "Recommendation: Skip irrigation"
    ],
    "confidence": 0.85,
    "language": "en"
  }
}
```

### Get Recommendation

#### GET /recommendations/{recommendation_id}

Retrieve a specific recommendation.

### List User Recommendations

#### GET /users/{user_id}/recommendations

List all recommendations for a user.

**Query Parameters:**
- `limit` (integer, optional): Number of results (default: 10, max: 100)

---

## Marketplace

### Create Product

#### POST /products

List a product on the marketplace.

**Parameters:**
- `farmer_id` (string, required): Farmer ID
- `farm_id` (string, required): Farm ID
- `product_type` (string, required): "vegetables", "grains", "pulses", "dairy", "honey", "spices"
- `name` (string, required): Product name
- `quantity_kg` (float, required): Quantity in kg
- `price_per_kg` (float, required): Price per kg in INR
- `harvest_date` (string, required): ISO format date (e.g., "2024-01-15T10:00:00")

**Example:**
```bash
curl -X POST "http://localhost:8000/api/v1/products" \
  -H "Content-Type: application/json" \
  -d '{
    "farmer_id": "farmer_001",
    "farm_id": "farm_001",
    "product_type": "vegetables",
    "name": "Tomatoes",
    "quantity_kg": 100,
    "price_per_kg": 25,
    "harvest_date": "2024-01-15T10:00:00"
  }'
```

**Response:**
```json
{
  "status": "success",
  "product": {
    "product_id": "uuid",
    "farmer_id": "farmer_001",
    "farm_id": "farm_001",
    "product_type": "vegetables",
    "name": "Tomatoes",
    "quantity_kg": 100,
    "price_per_kg": 25,
    "harvest_date": "2024-01-15T10:00:00",
    "pure_product_score": 78,
    "status": "available",
    "created_at": "2024-01-15T10:30:00"
  }
}
```

### Get Product

#### GET /products/{product_id}

Retrieve product details.

### Search Products

#### GET /products

Search marketplace products.

**Query Parameters:**
- `product_type` (string, optional): Filter by product type
- `min_score` (float, optional): Minimum Pure Product Score (0-100)
- `max_price` (float, optional): Maximum price per kg
- `limit` (integer, optional): Number of results (default: 20, max: 100)

**Example:**
```bash
curl "http://localhost:8000/api/v1/products?product_type=vegetables&min_score=70&limit=20"
```

### List Farmer Products

#### GET /farmers/{farmer_id}/products

List all products from a specific farmer.

---

## Knowledge Management

### Add Knowledge

#### POST /knowledge

Add knowledge chunk to RAG database.

**Parameters:**
- `chunk_id` (string, required): Unique chunk identifier
- `content` (string, required): Knowledge content
- `source` (string, required): Source type ("research_paper", "best_practice", "case_study", "govt_guideline")
- `topic` (string, required): Topic/category
- `crop_type` (string, optional): Relevant crop type
- `region` (string, optional): Relevant region

**Example:**
```bash
curl -X POST "http://localhost:8000/api/v1/knowledge" \
  -H "Content-Type: application/json" \
  -d '{
    "chunk_id": "wheat_irrigation_001",
    "content": "Wheat requires 450-600mm of water during growing season. Optimal irrigation timing is at tillering and grain filling stages.",
    "source": "best_practice",
    "topic": "irrigation",
    "crop_type": "wheat",
    "region": "north_india"
  }'
```

### Search Knowledge

#### GET /knowledge/search

Search knowledge base.

**Query Parameters:**
- `query` (string, required): Search query
- `top_k` (integer, optional): Number of results (default: 5, max: 20)

**Example:**
```bash
curl "http://localhost:8000/api/v1/knowledge/search?query=wheat+irrigation&top_k=5"
```

---

## Query Examples

### Example 1: Get Irrigation Recommendation

```bash
curl -X POST "http://localhost:8000/api/v1/query" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "farmer_001",
    "query_text": "Should I irrigate today?",
    "farm_id": "farm_001",
    "latitude": 28.5,
    "longitude": 77.0,
    "farm_size_hectares": 2.0,
    "crop_type": "wheat",
    "growth_stage": "flowering",
    "soil_type": "loamy"
  }'
```

### Example 2: Get Pest Management Advice

```bash
curl -X POST "http://localhost:8000/api/v1/query" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "farmer_001",
    "query_text": "I see spots on my cotton leaves",
    "farm_id": "farm_001",
    "crop_type": "cotton",
    "growth_stage": "flowering"
  }'
```

### Example 3: Get Yield Forecast

```bash
curl -X POST "http://localhost:8000/api/v1/query" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "farmer_001",
    "query_text": "What will be my wheat yield?",
    "farm_id": "farm_001",
    "crop_type": "wheat",
    "growth_stage": "grain_filling"
  }'
```

---

## Rate Limiting

Currently no rate limiting. In production, implement:

- 100 requests per minute per user
- 1000 requests per minute per IP

---

## Pagination

For list endpoints, use:

- `limit`: Number of results (default: 10, max: 100)
- `offset`: Number of results to skip (default: 0)

---

## Versioning

API version is in the URL: `/api/v1/`

Future versions: `/api/v2/`, `/api/v3/`, etc.

---

## SDK Examples

### Python

```python
import requests

BASE_URL = "http://localhost:8000/api/v1"

# Create user
response = requests.post(
    f"{BASE_URL}/users",
    json={
        "phone_number": "9876543210",
        "name": "Ramesh Kumar",
        "role": "farmer"
    }
)
user = response.json()["user"]

# Process query
response = requests.post(
    f"{BASE_URL}/query",
    json={
        "user_id": user["user_id"],
        "query_text": "Should I irrigate?",
        "crop_type": "wheat"
    }
)
recommendation = response.json()["recommendation"]
print(recommendation["recommendation_text"])
```

### JavaScript/Node.js

```javascript
const BASE_URL = "http://localhost:8000/api/v1";

// Create user
const userResponse = await fetch(`${BASE_URL}/users`, {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    phone_number: "9876543210",
    name: "Ramesh Kumar",
    role: "farmer"
  })
});
const user = (await userResponse.json()).user;

// Process query
const queryResponse = await fetch(`${BASE_URL}/query`, {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    user_id: user.user_id,
    query_text: "Should I irrigate?",
    crop_type: "wheat"
  })
});
const recommendation = (await queryResponse.json()).recommendation;
console.log(recommendation.recommendation_text);
```

---

## Support

For API issues or questions:
- GitHub Issues: https://github.com/grambrain/grambrain-ai/issues
- Email: api-support@grambrain.ai
