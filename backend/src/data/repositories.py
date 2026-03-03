"""Concrete repository implementations for each entity."""

from typing import Dict, Any, List, Optional
from datetime import datetime
from boto3.dynamodb.conditions import Key, Attr

from .models import (
    User, Farm, Recommendation, Product, UserRole,
    IrrigationType, ProductCategory
)
from .schemas import (
    UserSchema, FarmSchema, RecommendationSchema, ProductSchema,
    KnowledgeChunkSchema, UserCreateSchema, FarmCreateSchema,
    RecommendationCreateSchema, ProductCreateSchema, KnowledgeChunkCreateSchema
)
from .repository import DynamoDBRepository
from .dynamodb_client import DynamoDBClient


class UserRepository(DynamoDBRepository[User]):
    """Repository for User entities."""
    
    def __init__(self, dynamodb_client: DynamoDBClient, env: str = 'dev'):
        """Initialize user repository.
        
        Args:
            dynamodb_client: DynamoDB client
            env: Environment name (dev, staging, production)
        """
        table_name = f"grambrain-users-{env}"
        super().__init__(dynamodb_client, table_name)
    
    def _to_item(self, entity: User) -> Dict[str, Any]:
        """Convert User to DynamoDB item."""
        item = {
            'user_id': entity.user_id,
            'phone_number': entity.phone_number,
            'name': entity.name,
            'language_preference': entity.language_preference,
            'role': entity.role.value,
            'created_at': entity.created_at.isoformat(),
            'last_active': entity.last_active.isoformat(),
            'metadata': entity.metadata,
        }
        if entity.password_hash:
            item['password_hash'] = entity.password_hash
        return item
    
    def _from_item(self, item: Dict[str, Any]) -> User:
        """Convert DynamoDB item to User."""
        return User(
            user_id=item['user_id'],
            phone_number=item['phone_number'],
            name=item['name'],
            password_hash=item.get('password_hash'),
            language_preference=item.get('language_preference', 'en'),
            role=UserRole(item.get('role', 'farmer')),
            created_at=datetime.fromisoformat(item['created_at']),
            last_active=datetime.fromisoformat(item['last_active']),
            metadata=item.get('metadata', {}),
        )
    
    async def create_user(self, user: User) -> User:
        """Create a new user.
        
        Args:
            user: User entity
            
        Returns:
            Created user
            
        Raises:
            ValueError: If user data is invalid
        """
        # Validate using Pydantic schema
        user_dict = {
            'user_id': user.user_id,
            'phone_number': user.phone_number,
            'name': user.name,
            'language_preference': user.language_preference,
            'role': user.role,
            'created_at': user.created_at,
            'last_active': user.last_active,
            'metadata': user.metadata,
        }
        UserSchema(**user_dict)  # Validates and raises ValidationError if invalid
        
        return await self.put_item(user)
    
    async def get_user(self, user_id: str) -> Optional[User]:
        """Get user by ID.
        
        Args:
            user_id: User ID
            
        Returns:
            User if found, None otherwise
        """
        return await self.get_item({'user_id': user_id})
    
    async def get_user_by_phone(self, phone_number: str) -> Optional[User]:
        """Get user by phone number using GSI.
        
        Args:
            phone_number: Phone number
            
        Returns:
            User if found, None otherwise
        """
        result = await self.query(
            key_condition_expression=Key('phone_number').eq(phone_number),
            index_name='phone-index',
            limit=1
        )
        
        items = result.get('items', [])
        return items[0] if items else None
    
    async def update_user(self, user_id: str, updates: Dict[str, Any]) -> Dict[str, Any]:
        """Update user attributes.
        
        Args:
            user_id: User ID
            updates: Dictionary of attributes to update
            
        Returns:
            Updated attributes
        """
        # Add updated timestamp
        updates['last_active'] = datetime.now().isoformat()
        return await self.update_item({'user_id': user_id}, updates)


class FarmRepository(DynamoDBRepository[Farm]):
    """Repository for Farm entities."""
    
    def __init__(self, dynamodb_client: DynamoDBClient, env: str = 'dev'):
        """Initialize farm repository.
        
        Args:
            dynamodb_client: DynamoDB client
            env: Environment name
        """
        table_name = f"grambrain-farms-{env}"
        super().__init__(dynamodb_client, table_name)
    
    def _to_item(self, entity: Farm) -> Dict[str, Any]:
        """Convert Farm to DynamoDB item."""
        return {
            'farm_id': entity.farm_id,
            'owner_id': entity.owner_id,
            'location': entity.location,
            'area_hectares': entity.area_hectares,
            'soil_type': entity.soil_type,
            'irrigation_type': entity.irrigation_type.value,
            'crops': entity.crops,
            'created_at': entity.created_at.isoformat(),
            'updated_at': entity.updated_at.isoformat(),
            'metadata': entity.metadata,
        }
    
    def _from_item(self, item: Dict[str, Any]) -> Farm:
        """Convert DynamoDB item to Farm."""
        return Farm(
            farm_id=item['farm_id'],
            owner_id=item['owner_id'],
            location=item['location'],
            area_hectares=float(item['area_hectares']),
            soil_type=item['soil_type'],
            irrigation_type=IrrigationType(item['irrigation_type']),
            crops=item.get('crops', []),
            created_at=datetime.fromisoformat(item['created_at']),
            updated_at=datetime.fromisoformat(item['updated_at']),
            metadata=item.get('metadata', {}),
        )
    
    async def create_farm(self, farm: Farm) -> Farm:
        """Create a new farm.
        
        Args:
            farm: Farm entity
            
        Returns:
            Created farm
        """
        return await self.put_item(farm)
    
    async def get_farm(self, farm_id: str) -> Optional[Farm]:
        """Get farm by ID.
        
        Args:
            farm_id: Farm ID
            
        Returns:
            Farm if found, None otherwise
        """
        # Note: Using composite key (farm_id, owner_id)
        # This requires knowing the owner_id or querying by farm_id alone
        # For simplicity, we'll query by farm_id
        result = await self.query(
            key_condition_expression=Key('farm_id').eq(farm_id),
            limit=1
        )
        
        items = result.get('items', [])
        return items[0] if items else None
    
    async def list_user_farms(
        self,
        owner_id: str,
        limit: Optional[int] = None,
        exclusive_start_key: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """List all farms for a user using GSI.
        
        Args:
            owner_id: Owner user ID
            limit: Maximum items to return
            exclusive_start_key: Pagination token
            
        Returns:
            Dictionary with 'items' and optional 'last_evaluated_key'
        """
        return await self.query(
            key_condition_expression=Key('owner_id').eq(owner_id),
            index_name='owner-index',
            limit=limit,
            exclusive_start_key=exclusive_start_key,
            scan_index_forward=False  # Most recent first
        )
    
    async def update_farm(self, farm_id: str, owner_id: str, updates: Dict[str, Any]) -> Dict[str, Any]:
        """Update farm attributes.
        
        Args:
            farm_id: Farm ID
            owner_id: Owner ID
            updates: Dictionary of attributes to update
            
        Returns:
            Updated attributes
        """
        # Add updated timestamp
        updates['updated_at'] = datetime.now().isoformat()
        return await self.update_item(
            {'farm_id': farm_id, 'owner_id': owner_id},
            updates
        )


class RecommendationRepository(DynamoDBRepository[Recommendation]):
    """Repository for Recommendation entities."""
    
    def __init__(self, dynamodb_client: DynamoDBClient, env: str = 'dev'):
        """Initialize recommendation repository.
        
        Args:
            dynamodb_client: DynamoDB client
            env: Environment name
        """
        table_name = f"grambrain-recommendations-{env}"
        super().__init__(dynamodb_client, table_name)
    
    def _to_item(self, entity: Recommendation) -> Dict[str, Any]:
        """Convert Recommendation to DynamoDB item."""
        return {
            'user_id': entity.user_id,
            'timestamp': entity.timestamp.isoformat(),
            'recommendation_id': entity.recommendation_id,
            'query_id': entity.query_id,
            'farm_id': entity.farm_id,
            'recommendation_text': entity.recommendation_text,
            'reasoning_chain': entity.reasoning_chain,
            'confidence': entity.confidence,
            'agent_contributions': entity.agent_contributions,
            'language': entity.language,
            'voice_audio_url': entity.voice_audio_url,
            'created_at': entity.created_at.isoformat(),
        }
    
    def _from_item(self, item: Dict[str, Any]) -> Recommendation:
        """Convert DynamoDB item to Recommendation."""
        return Recommendation(
            recommendation_id=item['recommendation_id'],
            query_id=item['query_id'],
            user_id=item['user_id'],
            farm_id=item.get('farm_id'),
            timestamp=datetime.fromisoformat(item['timestamp']),
            recommendation_text=item['recommendation_text'],
            reasoning_chain=item.get('reasoning_chain', []),
            confidence=float(item.get('confidence', 0.0)),
            agent_contributions=item.get('agent_contributions', []),
            language=item.get('language', 'en'),
            voice_audio_url=item.get('voice_audio_url'),
            created_at=datetime.fromisoformat(item['created_at']),
        )
    
    async def create_recommendation(self, recommendation: Recommendation) -> Recommendation:
        """Create a new recommendation.
        
        Args:
            recommendation: Recommendation entity
            
        Returns:
            Created recommendation
        """
        return await self.put_item(recommendation)
    
    async def get_recommendation(self, user_id: str, timestamp: str) -> Optional[Recommendation]:
        """Get recommendation by composite key.
        
        Args:
            user_id: User ID
            timestamp: Timestamp (ISO format)
            
        Returns:
            Recommendation if found, None otherwise
        """
        return await self.get_item({
            'user_id': user_id,
            'timestamp': timestamp
        })
    
    async def list_user_recommendations(
        self,
        user_id: str,
        limit: Optional[int] = None,
        exclusive_start_key: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """List recommendations for a user.
        
        Args:
            user_id: User ID
            limit: Maximum items to return
            exclusive_start_key: Pagination token
            
        Returns:
            Dictionary with 'items' and optional 'last_evaluated_key'
        """
        return await self.query(
            key_condition_expression=Key('user_id').eq(user_id),
            limit=limit,
            exclusive_start_key=exclusive_start_key,
            scan_index_forward=False  # Most recent first
        )
    
    async def get_recommendation_by_query_id(self, query_id: str) -> Optional[Recommendation]:
        """Get recommendation by query ID using GSI.
        
        Args:
            query_id: Query ID
            
        Returns:
            Recommendation if found, None otherwise
        """
        result = await self.query(
            key_condition_expression=Key('query_id').eq(query_id),
            index_name='query-index',
            limit=1
        )
        
        items = result.get('items', [])
        return items[0] if items else None


class ProductRepository(DynamoDBRepository[Product]):
    """Repository for Product entities."""
    
    def __init__(self, dynamodb_client: DynamoDBClient, env: str = 'dev'):
        """Initialize product repository.
        
        Args:
            dynamodb_client: DynamoDB client
            env: Environment name
        """
        table_name = f"grambrain-products-{env}"
        super().__init__(dynamodb_client, table_name)
    
    def _to_item(self, entity: Product) -> Dict[str, Any]:
        """Convert Product to DynamoDB item."""
        return {
            'product_id': entity.product_id,
            'farmer_id': entity.farmer_id,
            'farm_id': entity.farm_id,
            'product_type': entity.product_type.value,
            'name': entity.name,
            'quantity_kg': entity.quantity_kg,
            'price_per_kg': entity.price_per_kg,
            'harvest_date': entity.harvest_date.isoformat(),
            'images': entity.images,
            'pure_product_score': entity.pure_product_score,
            'status': entity.status,
            'created_at': entity.created_at.isoformat(),
            'metadata': entity.metadata,
        }
    
    def _from_item(self, item: Dict[str, Any]) -> Product:
        """Convert DynamoDB item to Product."""
        return Product(
            product_id=item['product_id'],
            farmer_id=item['farmer_id'],
            farm_id=item['farm_id'],
            product_type=ProductCategory(item['product_type']),
            name=item['name'],
            quantity_kg=float(item['quantity_kg']),
            price_per_kg=float(item['price_per_kg']),
            harvest_date=datetime.fromisoformat(item['harvest_date']),
            images=item.get('images', []),
            pure_product_score=float(item.get('pure_product_score', 0.0)),
            status=item.get('status', 'available'),
            created_at=datetime.fromisoformat(item['created_at']),
            metadata=item.get('metadata', {}),
        )
    
    async def create_product(self, product: Product) -> Product:
        """Create a new product.
        
        Args:
            product: Product entity
            
        Returns:
            Created product
        """
        return await self.put_item(product)
    
    async def get_product(self, product_id: str) -> Optional[Product]:
        """Get product by ID.
        
        Args:
            product_id: Product ID
            
        Returns:
            Product if found, None otherwise
        """
        return await self.get_item({'product_id': product_id})
    
    async def list_farmer_products(
        self,
        farmer_id: str,
        limit: Optional[int] = None,
        exclusive_start_key: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """List products for a farmer using GSI.
        
        Args:
            farmer_id: Farmer user ID
            limit: Maximum items to return
            exclusive_start_key: Pagination token
            
        Returns:
            Dictionary with 'items' and optional 'last_evaluated_key'
        """
        return await self.query(
            key_condition_expression=Key('farmer_id').eq(farmer_id),
            index_name='farmer-index',
            limit=limit,
            exclusive_start_key=exclusive_start_key,
            scan_index_forward=False  # Most recent first
        )
    
    async def list_products_by_type(
        self,
        product_type: str,
        min_score: Optional[float] = None,
        limit: Optional[int] = None,
        exclusive_start_key: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """List products by type and score using GSI.
        
        Args:
            product_type: Product type
            min_score: Minimum pure product score
            limit: Maximum items to return
            exclusive_start_key: Pagination token
            
        Returns:
            Dictionary with 'items' and optional 'last_evaluated_key'
        """
        key_condition = Key('product_type').eq(product_type)
        
        if min_score is not None:
            key_condition = key_condition & Key('pure_product_score').gte(min_score)
        
        return await self.query(
            key_condition_expression=key_condition,
            index_name='type-score-index',
            limit=limit,
            exclusive_start_key=exclusive_start_key,
            scan_index_forward=False  # Highest score first
        )
    
    async def update_product(self, product_id: str, updates: Dict[str, Any]) -> Dict[str, Any]:
        """Update product attributes.
        
        Args:
            product_id: Product ID
            updates: Dictionary of attributes to update
            
        Returns:
            Updated attributes
        """
        return await self.update_item({'product_id': product_id}, updates)


class KnowledgeRepository(DynamoDBRepository[Dict[str, Any]]):
    """Repository for Knowledge chunks."""
    
    def __init__(self, dynamodb_client: DynamoDBClient, env: str = 'dev'):
        """Initialize knowledge repository.
        
        Args:
            dynamodb_client: DynamoDB client
            env: Environment name
        """
        table_name = f"grambrain-knowledge-{env}"
        super().__init__(dynamodb_client, table_name)
    
    def _to_item(self, entity: Dict[str, Any]) -> Dict[str, Any]:
        """Convert knowledge chunk to DynamoDB item."""
        return entity
    
    def _from_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Convert DynamoDB item to knowledge chunk."""
        return item
    
    async def create_chunk(self, chunk: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new knowledge chunk.
        
        Args:
            chunk: Knowledge chunk dictionary
            
        Returns:
            Created chunk
        """
        return await self.put_item(chunk)
    
    async def get_chunk(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        """Get knowledge chunk by ID.
        
        Args:
            chunk_id: Chunk ID
            
        Returns:
            Chunk if found, None otherwise
        """
        return await self.get_item({'chunk_id': chunk_id})
    
    async def list_chunks_by_topic(
        self,
        topic: str,
        limit: Optional[int] = None,
        exclusive_start_key: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """List knowledge chunks by topic using GSI.
        
        Args:
            topic: Topic name
            limit: Maximum items to return
            exclusive_start_key: Pagination token
            
        Returns:
            Dictionary with 'items' and optional 'last_evaluated_key'
        """
        return await self.query(
            key_condition_expression=Key('topic').eq(topic),
            index_name='topic-index',
            limit=limit,
            exclusive_start_key=exclusive_start_key,
            scan_index_forward=False  # Most recent first
        )
