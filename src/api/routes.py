"""REST API routes for GramBrain system."""

from fastapi import FastAPI, HTTPException, Depends, Query as QueryParam
from fastapi.responses import JSONResponse
from typing import Optional, List
import uuid
from datetime import datetime

from ..system import GramBrainSystem
from ..data.models import User, Farm, Product, Recommendation, UserRole, ProductCategory
from ..core.agent_base import Query, UserContext


# Initialize FastAPI app
app = FastAPI(
    title="GramBrain AI API",
    description="Multi-Agent Agricultural Intelligence Platform",
    version="0.1.0",
)

# Global system instance
system: Optional[GramBrainSystem] = None


@app.on_event("startup")
async def startup_event():
    """Initialize system on startup."""
    global system
    system = GramBrainSystem(use_mock_llm=True, use_mock_rag=True)
    await system.initialize()


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    global system
    if system:
        system.shutdown()


def get_system() -> GramBrainSystem:
    """Get system instance."""
    if not system:
        raise HTTPException(status_code=500, detail="System not initialized")
    return system


# ============================================================================
# User Endpoints
# ============================================================================

@app.post("/api/v1/users")
async def create_user(
    phone_number: str,
    name: str,
    language_preference: str = "en",
    role: str = "farmer",
    sys: GramBrainSystem = Depends(get_system),
):
    """Create a new user."""
    user_id = str(uuid.uuid4())
    user = User(
        user_id=user_id,
        phone_number=phone_number,
        name=name,
        language_preference=language_preference,
        role=UserRole(role),
    )
    # TODO: Save to database
    return {"status": "success", "user": user.to_dict()}


@app.get("/api/v1/users/{user_id}")
async def get_user(user_id: str, sys: GramBrainSystem = Depends(get_system)):
    """Get user by ID."""
    # TODO: Fetch from database
    return {"status": "success", "message": f"User {user_id} not found"}


# ============================================================================
# Farm Endpoints
# ============================================================================

@app.post("/api/v1/farms")
async def create_farm(
    owner_id: str,
    latitude: float,
    longitude: float,
    area_hectares: float,
    soil_type: str,
    irrigation_type: str = "drip",
    sys: GramBrainSystem = Depends(get_system),
):
    """Create a new farm."""
    farm_id = str(uuid.uuid4())
    farm = Farm(
        farm_id=farm_id,
        owner_id=owner_id,
        location={"lat": latitude, "lon": longitude},
        area_hectares=area_hectares,
        soil_type=soil_type,
        irrigation_type=irrigation_type,
    )
    # TODO: Save to database
    return {"status": "success", "farm": farm.to_dict()}


@app.get("/api/v1/farms/{farm_id}")
async def get_farm(farm_id: str, sys: GramBrainSystem = Depends(get_system)):
    """Get farm by ID."""
    # TODO: Fetch from database
    return {"status": "success", "message": f"Farm {farm_id} not found"}


@app.get("/api/v1/users/{user_id}/farms")
async def list_user_farms(user_id: str, sys: GramBrainSystem = Depends(get_system)):
    """List all farms for a user."""
    # TODO: Fetch from database
    return {"status": "success", "farms": []}


# ============================================================================
# Query/Recommendation Endpoints
# ============================================================================

@app.post("/api/v1/query")
async def process_query(
    user_id: str,
    query_text: str,
    farm_id: Optional[str] = None,
    latitude: Optional[float] = None,
    longitude: Optional[float] = None,
    farm_size_hectares: Optional[float] = None,
    crop_type: Optional[str] = None,
    growth_stage: Optional[str] = None,
    soil_type: Optional[str] = None,
    language: str = "en",
    sys: GramBrainSystem = Depends(get_system),
):
    """Process a user query and return recommendation."""
    try:
        result = sys.process_query(
            query_text=query_text,
            user_id=user_id,
            farm_id=farm_id,
            farm_location={"lat": latitude, "lon": longitude} if latitude and longitude else None,
            farm_size_hectares=farm_size_hectares,
            crop_type=crop_type,
            growth_stage=growth_stage,
            soil_type=soil_type,
            language=language,
        )
        
        # Create recommendation record
        recommendation_id = str(uuid.uuid4())
        recommendation = Recommendation(
            recommendation_id=recommendation_id,
            query_id=result.get("query_id", ""),
            user_id=user_id,
            farm_id=farm_id,
            timestamp=datetime.now(),
            recommendation_text=result.get("recommendation", ""),
            reasoning_chain=result.get("reasoning_chain", []),
            confidence=result.get("confidence", 0.0),
            language=language,
        )
        # TODO: Save to database
        
        return {
            "status": "success",
            "recommendation": recommendation.to_dict(),
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/v1/recommendations/{recommendation_id}")
async def get_recommendation(
    recommendation_id: str,
    sys: GramBrainSystem = Depends(get_system),
):
    """Get recommendation by ID."""
    # TODO: Fetch from database
    return {"status": "success", "message": f"Recommendation {recommendation_id} not found"}


@app.get("/api/v1/users/{user_id}/recommendations")
async def list_user_recommendations(
    user_id: str,
    limit: int = QueryParam(10, ge=1, le=100),
    sys: GramBrainSystem = Depends(get_system),
):
    """List recommendations for a user."""
    # TODO: Fetch from database
    return {"status": "success", "recommendations": []}


# ============================================================================
# Product/Marketplace Endpoints
# ============================================================================

@app.post("/api/v1/products")
async def create_product(
    farmer_id: str,
    farm_id: str,
    product_type: str,
    name: str,
    quantity_kg: float,
    price_per_kg: float,
    harvest_date: str,
    sys: GramBrainSystem = Depends(get_system),
):
    """Create a new product listing."""
    try:
        product_id = str(uuid.uuid4())
        product = Product(
            product_id=product_id,
            farmer_id=farmer_id,
            farm_id=farm_id,
            product_type=ProductCategory(product_type),
            name=name,
            quantity_kg=quantity_kg,
            price_per_kg=price_per_kg,
            harvest_date=datetime.fromisoformat(harvest_date),
        )
        # TODO: Save to database
        # TODO: Calculate Pure Product Score
        
        return {"status": "success", "product": product.to_dict()}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/v1/products/{product_id}")
async def get_product(product_id: str, sys: GramBrainSystem = Depends(get_system)):
    """Get product by ID."""
    # TODO: Fetch from database
    return {"status": "success", "message": f"Product {product_id} not found"}


@app.get("/api/v1/products")
async def search_products(
    product_type: Optional[str] = None,
    min_score: float = QueryParam(0, ge=0, le=100),
    max_price: Optional[float] = None,
    limit: int = QueryParam(20, ge=1, le=100),
    sys: GramBrainSystem = Depends(get_system),
):
    """Search products in marketplace."""
    # TODO: Implement search with filters
    return {"status": "success", "products": []}


@app.get("/api/v1/farmers/{farmer_id}/products")
async def list_farmer_products(
    farmer_id: str,
    sys: GramBrainSystem = Depends(get_system),
):
    """List all products from a farmer."""
    # TODO: Fetch from database
    return {"status": "success", "products": []}


# ============================================================================
# Knowledge/RAG Endpoints
# ============================================================================

@app.post("/api/v1/knowledge")
async def add_knowledge(
    chunk_id: str,
    content: str,
    source: str,
    topic: str,
    crop_type: Optional[str] = None,
    region: Optional[str] = None,
    sys: GramBrainSystem = Depends(get_system),
):
    """Add knowledge chunk to RAG database."""
    try:
        import asyncio
        asyncio.run(sys.add_knowledge(
            chunk_id=chunk_id,
            content=content,
            source=source,
            topic=topic,
            crop_type=crop_type,
            region=region,
        ))
        return {"status": "success", "message": "Knowledge added"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/v1/knowledge/search")
async def search_knowledge(
    query: str,
    top_k: int = QueryParam(5, ge=1, le=20),
    sys: GramBrainSystem = Depends(get_system),
):
    """Search knowledge base."""
    # TODO: Implement RAG search
    return {"status": "success", "results": []}


# ============================================================================
# Health Check
# ============================================================================

@app.get("/api/v1/health")
async def health_check(sys: GramBrainSystem = Depends(get_system)):
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "agents": sys.registry.list_agents(),
    }


# ============================================================================
# Error Handlers
# ============================================================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions."""
    return JSONResponse(
        status_code=exc.status_code,
        content={"status": "error", "detail": exc.detail},
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle general exceptions."""
    return JSONResponse(
        status_code=500,
        content={"status": "error", "detail": "Internal server error"},
    )
