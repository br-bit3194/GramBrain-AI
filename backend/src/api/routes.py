"""REST API routes for GramBrain system."""

from fastapi import FastAPI, HTTPException, Depends
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from typing import Optional
import uuid
from datetime import datetime
from pydantic import BaseModel

from ..system import GramBrainSystem
from ..data.models import User, Farm, Product, Recommendation, UserRole, ProductCategory


# Initialize FastAPI app
app = FastAPI(
    title="GramBrain AI API",
    description="Multi-Agent Agricultural Intelligence Platform",
    version="0.1.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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


# Request models
class CreateUserRequest(BaseModel):
    phone_number: str
    name: str
    language_preference: str = "en"
    role: str = "farmer"


class CreateFarmRequest(BaseModel):
    owner_id: str
    latitude: float
    longitude: float
    area_hectares: float
    soil_type: str
    irrigation_type: str = "drip"


class ProcessQueryRequest(BaseModel):
    user_id: str
    query_text: str
    farm_id: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    farm_size_hectares: Optional[float] = None
    crop_type: Optional[str] = None
    growth_stage: Optional[str] = None
    soil_type: Optional[str] = None
    language: str = "en"


class CreateProductRequest(BaseModel):
    farmer_id: str
    farm_id: str
    product_type: str
    name: str
    quantity_kg: float
    price_per_kg: float
    harvest_date: str


class AddKnowledgeRequest(BaseModel):
    chunk_id: str
    content: str
    source: str
    topic: str
    crop_type: Optional[str] = None
    region: Optional[str] = None


# ============================================================================
# Health Check
# ============================================================================

@app.get("/health")
async def health_check(sys: GramBrainSystem = Depends(get_system)):
    """Health check endpoint."""
    return {
        "status": "success",
        "data": {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "agents": sys.registry.list_agents(),
        },
    }


# ============================================================================
# User Endpoints
# ============================================================================

@app.post("/api/users")
async def create_user(request: CreateUserRequest, sys: GramBrainSystem = Depends(get_system)):
    """Create a new user."""
    try:
        user_id = str(uuid.uuid4())
        user = User(
            user_id=user_id,
            phone_number=request.phone_number,
            name=request.name,
            language_preference=request.language_preference,
            role=UserRole(request.role),
        )
        # TODO: Save to database
        return {
            "status": "success",
            "data": {
                "user": user.to_dict()
            },
        }
    except Exception as e:
        return {
            "status": "error",
            "detail": str(e)
        }


@app.get("/api/users/{user_id}")
async def get_user(user_id: str):
    """Get user by ID."""
    return {
        "status": "success",
        "data": {
            "user": {
                "user_id": user_id,
                "phone_number": "+91 98765 43210",
                "name": "Test Farmer",
                "language_preference": "en",
                "role": "farmer",
                "created_at": datetime.now().isoformat(),
                "last_active": datetime.now().isoformat(),
            }
        },
    }


# ============================================================================
# Farm Endpoints
# ============================================================================

@app.post("/api/farms")
async def create_farm(request: CreateFarmRequest, sys: GramBrainSystem = Depends(get_system)):
    """Create a new farm."""
    try:
        farm_id = str(uuid.uuid4())
        farm = Farm(
            farm_id=farm_id,
            owner_id=request.owner_id,
            location={"lat": request.latitude, "lon": request.longitude},
            area_hectares=request.area_hectares,
            soil_type=request.soil_type,
            irrigation_type=request.irrigation_type,
        )
        # TODO: Save to database
        return {
            "status": "success",
            "data": {
                "farm": farm.to_dict()
            },
        }
    except Exception as e:
        return {
            "status": "error",
            "detail": str(e)
        }


@app.get("/api/farms/{farm_id}")
async def get_farm(farm_id: str):
    """Get farm by ID."""
    return {
        "status": "success",
        "data": {
            "farm": {
                "farm_id": farm_id,
                "owner_id": "owner-123",
                "location": {"lat": 28.7041, "lon": 77.1025},
                "area_hectares": 5.5,
                "soil_type": "loamy",
                "irrigation_type": "drip",
                "crops": [],
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat(),
            }
        },
    }


@app.get("/api/users/{user_id}/farms")
async def list_user_farms(user_id: str):
    """List all farms for a user."""
    return {
        "status": "success",
        "data": {
            "farms": [
                {
                    "farm_id": "farm-123",
                    "owner_id": user_id,
                    "location": {"lat": 28.7041, "lon": 77.1025},
                    "area_hectares": 5.5,
                    "soil_type": "loamy",
                    "irrigation_type": "drip",
                    "crops": [],
                    "created_at": datetime.now().isoformat(),
                    "updated_at": datetime.now().isoformat(),
                }
            ]
        },
    }


# ============================================================================
# Query/Recommendation Endpoints
# ============================================================================

@app.post("/api/query")
async def process_query(request: ProcessQueryRequest, sys: GramBrainSystem = Depends(get_system)):
    """Process a user query and return recommendation."""
    try:
        result = await sys.process_query(
            query_text=request.query_text,
            user_id=request.user_id,
            farm_id=request.farm_id,
            farm_location={"lat": request.latitude, "lon": request.longitude} if request.latitude and request.longitude else None,
            farm_size_hectares=request.farm_size_hectares,
            crop_type=request.crop_type,
            growth_stage=request.growth_stage,
            soil_type=request.soil_type,
            language=request.language,
        )
        
        # Create recommendation record
        recommendation_id = str(uuid.uuid4())
        recommendation = Recommendation(
            recommendation_id=recommendation_id,
            query_id=result.get("query_id", ""),
            user_id=request.user_id,
            farm_id=request.farm_id,
            timestamp=datetime.now(),
            recommendation_text=result.get("recommendation", ""),
            reasoning_chain=result.get("reasoning_chain", []),
            confidence=result.get("confidence", 0.0),
            language=request.language,
        )
        # TODO: Save to database
        
        return {
            "status": "success",
            "data": {
                "recommendation": recommendation.to_dict(),
            },
        }
    except Exception as e:
        return {
            "status": "error",
            "detail": str(e)
        }


@app.get("/api/recommendations/{recommendation_id}")
async def get_recommendation(recommendation_id: str):
    """Get recommendation by ID."""
    return {
        "status": "success",
        "data": {
            "recommendation": {
                "recommendation_id": recommendation_id,
                "query_id": str(uuid.uuid4()),
                "user_id": "user-123",
                "farm_id": "farm-123",
                "timestamp": datetime.now().isoformat(),
                "recommendation_text": "Sample recommendation",
                "reasoning_chain": [],
                "confidence": 0.85,
                "agent_contributions": [],
                "language": "en",
            }
        },
    }


@app.get("/api/users/{user_id}/recommendations")
async def list_user_recommendations(user_id: str, limit: int = 10):
    """List recommendations for a user."""
    return {
        "status": "success",
        "data": {
            "recommendations": []
        },
    }


# ============================================================================
# Product/Marketplace Endpoints
# ============================================================================

@app.post("/api/products")
async def create_product(request: CreateProductRequest, sys: GramBrainSystem = Depends(get_system)):
    """Create a new product listing."""
    try:
        product_id = str(uuid.uuid4())
        product = Product(
            product_id=product_id,
            farmer_id=request.farmer_id,
            farm_id=request.farm_id,
            product_type=ProductCategory(request.product_type),
            name=request.name,
            quantity_kg=request.quantity_kg,
            price_per_kg=request.price_per_kg,
            harvest_date=datetime.fromisoformat(request.harvest_date),
        )
        # TODO: Save to database
        # TODO: Calculate Pure Product Score
        
        return {
            "status": "success",
            "data": {
                "product": product.to_dict()
            },
        }
    except Exception as e:
        return {
            "status": "error",
            "detail": str(e)
        }


@app.get("/api/products/{product_id}")
async def get_product(product_id: str):
    """Get product by ID."""
    return {
        "status": "success",
        "data": {
            "product": {
                "product_id": product_id,
                "farmer_id": "farmer-123",
                "farm_id": "farm-123",
                "product_type": "vegetables",
                "name": "Tomatoes",
                "quantity_kg": 100,
                "price_per_kg": 50,
                "harvest_date": datetime.now().isoformat(),
                "images": [],
                "pure_product_score": 0.95,
                "status": "available",
                "created_at": datetime.now().isoformat(),
            }
        },
    }


@app.get("/api/products")
async def search_products(
    product_type: Optional[str] = None,
    min_score: float = 0,
    max_price: Optional[float] = None,
    limit: int = 20,
):
    """Search products in marketplace."""
    return {
        "status": "success",
        "data": {
            "products": []
        },
    }


@app.get("/api/farmers/{farmer_id}/products")
async def list_farmer_products(farmer_id: str):
    """List all products from a farmer."""
    return {
        "status": "success",
        "data": {
            "products": []
        },
    }


# ============================================================================
# Knowledge/RAG Endpoints
# ============================================================================

@app.post("/api/knowledge")
async def add_knowledge(request: AddKnowledgeRequest, sys: GramBrainSystem = Depends(get_system)):
    """Add knowledge chunk to RAG database."""
    try:
        await sys.add_knowledge(
            chunk_id=request.chunk_id,
            content=request.content,
            source=request.source,
            topic=request.topic,
            crop_type=request.crop_type,
            region=request.region,
        )
        return {
            "status": "success",
            "data": {
                "message": "Knowledge added successfully"
            },
        }
    except Exception as e:
        return {
            "status": "error",
            "detail": str(e)
        }


@app.get("/api/knowledge/search")
async def search_knowledge(query: str, top_k: int = 5):
    """Search knowledge base."""
    return {
        "status": "success",
        "data": {
            "results": []
        },
    }


# ============================================================================
# Error Handlers
# ============================================================================

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    """Handle validation errors."""
    errors = []
    for error in exc.errors():
        errors.append({
            "field": ".".join(str(x) for x in error["loc"][1:]),
            "message": error["msg"]
        })
    return JSONResponse(
        status_code=422,
        content={
            "status": "error",
            "detail": "Validation error",
            "errors": errors
        },
    )


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
