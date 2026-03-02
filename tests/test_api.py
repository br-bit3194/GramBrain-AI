"""Tests for REST API."""

import pytest
from fastapi.testclient import TestClient

from src.api import app


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


class TestHealthCheck:
    """Tests for health check endpoint."""
    
    def test_health_check(self, client):
        """Test health check endpoint."""
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert "agents" in data


class TestUserEndpoints:
    """Tests for user endpoints."""
    
    def test_create_user(self, client):
        """Test user creation."""
        response = client.post(
            "/api/v1/users",
            params={
                "phone_number": "9876543210",
                "name": "Ramesh Kumar",
                "language_preference": "hi",
                "role": "farmer",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert "user" in data
    
    def test_get_user(self, client):
        """Test getting user."""
        response = client.get("/api/v1/users/user_001")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"


class TestFarmEndpoints:
    """Tests for farm endpoints."""
    
    def test_create_farm(self, client):
        """Test farm creation."""
        response = client.post(
            "/api/v1/farms",
            params={
                "owner_id": "user_001",
                "latitude": 28.5,
                "longitude": 77.0,
                "area_hectares": 2.5,
                "soil_type": "loamy",
                "irrigation_type": "drip",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert "farm" in data
    
    def test_get_farm(self, client):
        """Test getting farm."""
        response = client.get("/api/v1/farms/farm_001")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
    
    def test_list_user_farms(self, client):
        """Test listing user farms."""
        response = client.get("/api/v1/users/user_001/farms")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert "farms" in data


class TestQueryEndpoints:
    """Tests for query/recommendation endpoints."""
    
    def test_process_query(self, client):
        """Test processing a query."""
        response = client.post(
            "/api/v1/query",
            params={
                "user_id": "farmer_001",
                "query_text": "Should I irrigate my wheat field?",
                "farm_id": "farm_001",
                "latitude": 28.5,
                "longitude": 77.0,
                "farm_size_hectares": 2.0,
                "crop_type": "wheat",
                "growth_stage": "tillering",
                "soil_type": "loamy",
                "language": "en",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert "recommendation" in data
    
    def test_get_recommendation(self, client):
        """Test getting recommendation."""
        response = client.get("/api/v1/recommendations/rec_001")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
    
    def test_list_user_recommendations(self, client):
        """Test listing user recommendations."""
        response = client.get("/api/v1/users/user_001/recommendations")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert "recommendations" in data


class TestProductEndpoints:
    """Tests for product/marketplace endpoints."""
    
    def test_create_product(self, client):
        """Test creating a product."""
        response = client.post(
            "/api/v1/products",
            params={
                "farmer_id": "farmer_001",
                "farm_id": "farm_001",
                "product_type": "vegetables",
                "name": "Tomatoes",
                "quantity_kg": 100,
                "price_per_kg": 25,
                "harvest_date": "2024-01-15T10:00:00",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert "product" in data
    
    def test_get_product(self, client):
        """Test getting product."""
        response = client.get("/api/v1/products/prod_001")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
    
    def test_search_products(self, client):
        """Test searching products."""
        response = client.get(
            "/api/v1/products",
            params={
                "product_type": "vegetables",
                "min_score": 70,
                "limit": 20,
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert "products" in data
    
    def test_list_farmer_products(self, client):
        """Test listing farmer products."""
        response = client.get("/api/v1/farmers/farmer_001/products")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert "products" in data


class TestKnowledgeEndpoints:
    """Tests for knowledge/RAG endpoints."""
    
    def test_add_knowledge(self, client):
        """Test adding knowledge."""
        response = client.post(
            "/api/v1/knowledge",
            params={
                "chunk_id": "knowledge_001",
                "content": "Wheat requires 450-600mm of water",
                "source": "best_practice",
                "topic": "irrigation",
                "crop_type": "wheat",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
    
    def test_search_knowledge(self, client):
        """Test searching knowledge."""
        response = client.get(
            "/api/v1/knowledge/search",
            params={
                "query": "wheat irrigation",
                "top_k": 5,
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert "results" in data
