"""Property-based tests for DynamoDB integration.

These tests validate correctness properties for the DynamoDB data access layer.
"""

import pytest
import asyncio
from datetime import datetime
from typing import Dict, Any
from unittest.mock import Mock, AsyncMock, patch
from hypothesis import given, strategies as st, settings, HealthCheck
from botocore.exceptions import ClientError

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend', 'src'))

from data.dynamodb_client import DynamoDBClient, RetryConfig
from data.repositories import UserRepository, FarmRepository
from data.models import User, Farm, UserRole, IrrigationType


# Hypothesis strategies for generating test data
@st.composite
def user_strategy(draw):
    """Generate random User instances."""
    return User(
        user_id=draw(st.text(min_size=1, max_size=50, alphabet=st.characters(blacklist_characters=['\x00']))),
        phone_number=draw(st.text(min_size=10, max_size=15, alphabet=st.characters(whitelist_categories=('Nd',)))),
        name=draw(st.text(min_size=1, max_size=100, alphabet=st.characters(blacklist_characters=['\x00']))),
        language_preference=draw(st.sampled_from(['en', 'hi', 'ta', 'te', 'bn'])),
        role=draw(st.sampled_from(list(UserRole))),
        created_at=datetime.now(),
        last_active=datetime.now(),
        metadata=draw(st.dictionaries(
            st.text(min_size=1, max_size=20, alphabet=st.characters(blacklist_characters=['\x00'])),
            st.text(min_size=0, max_size=50, alphabet=st.characters(blacklist_characters=['\x00'])),
            max_size=5
        ))
    )


@st.composite
def farm_strategy(draw):
    """Generate random Farm instances."""
    return Farm(
        farm_id=draw(st.text(min_size=1, max_size=50, alphabet=st.characters(blacklist_characters=['\x00']))),
        owner_id=draw(st.text(min_size=1, max_size=50, alphabet=st.characters(blacklist_characters=['\x00']))),
        location={
            'lat': draw(st.floats(min_value=-90, max_value=90, allow_nan=False, allow_infinity=False)),
            'lon': draw(st.floats(min_value=-180, max_value=180, allow_nan=False, allow_infinity=False))
        },
        area_hectares=draw(st.floats(min_value=0.1, max_value=1000, allow_nan=False, allow_infinity=False)),
        soil_type=draw(st.sampled_from(['loamy', 'clay', 'sandy', 'silt', 'peat'])),
        irrigation_type=draw(st.sampled_from(list(IrrigationType))),
        crops=draw(st.lists(st.text(min_size=1, max_size=20), max_size=5)),
        created_at=datetime.now(),
        updated_at=datetime.now(),
        metadata=draw(st.dictionaries(
            st.text(min_size=1, max_size=20, alphabet=st.characters(blacklist_characters=['\x00'])),
            st.text(min_size=0, max_size=50, alphabet=st.characters(blacklist_characters=['\x00'])),
            max_size=5
        ))
    )


class TestDynamoDBWriteKeyConsistency:
    """Test Property 1: DynamoDB write key consistency.
    
    Feature: production-readiness, Property 1: DynamoDB write key consistency
    Validates: Requirements 1.2
    
    For any data write operation to DynamoDB, the partition key and sort key 
    (if applicable) should be set according to the table schema.
    """
    
    @pytest.mark.asyncio
    @given(user=user_strategy())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
    async def test_user_write_has_correct_partition_key(self, user: User):
        """Test that User writes always include the correct partition key (user_id)."""
        # Create mock DynamoDB client
        mock_client = Mock(spec=DynamoDBClient)
        mock_table = Mock()
        mock_client.get_table.return_value = mock_table
        mock_client._execute_with_retry = AsyncMock()
        
        # Create repository
        repo = UserRepository(mock_client, env='test')
        
        # Convert user to item
        item = repo._to_item(user)
        
        # Property: Item must have partition key 'user_id'
        assert 'user_id' in item, "User item must have partition key 'user_id'"
        assert item['user_id'] == user.user_id, "Partition key must match user_id"
        assert isinstance(item['user_id'], str), "Partition key must be a string"
        assert len(item['user_id']) > 0, "Partition key must not be empty"
    
    @pytest.mark.asyncio
    @given(farm=farm_strategy())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
    async def test_farm_write_has_correct_composite_key(self, farm: Farm):
        """Test that Farm writes always include correct partition and sort keys."""
        # Create mock DynamoDB client
        mock_client = Mock(spec=DynamoDBClient)
        mock_table = Mock()
        mock_client.get_table.return_value = mock_table
        mock_client._execute_with_retry = AsyncMock()
        
        # Create repository
        repo = FarmRepository(mock_client, env='test')
        
        # Convert farm to item
        item = repo._to_item(farm)
        
        # Property: Item must have partition key 'farm_id' and sort key 'owner_id'
        assert 'farm_id' in item, "Farm item must have partition key 'farm_id'"
        assert 'owner_id' in item, "Farm item must have sort key 'owner_id'"
        assert item['farm_id'] == farm.farm_id, "Partition key must match farm_id"
        assert item['owner_id'] == farm.owner_id, "Sort key must match owner_id"
        assert isinstance(item['farm_id'], str), "Partition key must be a string"
        assert isinstance(item['owner_id'], str), "Sort key must be a string"
        assert len(item['farm_id']) > 0, "Partition key must not be empty"
        assert len(item['owner_id']) > 0, "Sort key must not be empty"


class TestDynamoDBRetryBehavior:
    """Test Property 2: DynamoDB retry with exponential backoff.
    
    Feature: production-readiness, Property 2: DynamoDB retry with exponential backoff
    Validates: Requirements 1.4
    
    For any DynamoDB operation that fails with a retryable error, the system 
    should retry with exponential backoff up to 3 attempts before failing.
    """
    
    @pytest.mark.asyncio
    @given(
        attempt=st.integers(min_value=0, max_value=10),
        base_delay=st.integers(min_value=50, max_value=500),
        max_delay=st.integers(min_value=1000, max_value=10000)
    )
    @settings(max_examples=100, deadline=None)
    async def test_exponential_backoff_calculation(self, attempt: int, base_delay: int, max_delay: int):
        """Test that delay calculation follows exponential backoff with jitter."""
        config = RetryConfig(
            base_delay_ms=base_delay,
            max_delay_ms=max_delay,
            exponential_base=2.0,
            jitter=False  # Disable jitter for deterministic testing
        )
        
        client = DynamoDBClient(retry_config=config)
        delay = client._calculate_delay(attempt)
        
        # Property: Delay should follow exponential backoff
        expected_delay_ms = min(base_delay * (2.0 ** attempt), max_delay)
        expected_delay_sec = expected_delay_ms / 1000.0
        
        assert delay == expected_delay_sec, f"Delay should be {expected_delay_sec}s"
        assert delay <= max_delay / 1000.0, "Delay should not exceed max_delay"
    
    @pytest.mark.asyncio
    @given(
        error_code=st.sampled_from([
            'ProvisionedThroughputExceededException',
            'ThrottlingException',
            'RequestLimitExceeded',
            'InternalServerError',
            'ServiceUnavailable'
        ])
    )
    @settings(max_examples=100)
    async def test_retryable_errors_are_retried(self, error_code: str):
        """Test that retryable errors trigger retry logic."""
        config = RetryConfig(max_attempts=3, base_delay_ms=10, jitter=False)
        client = DynamoDBClient(retry_config=config)
        
        # Create a mock error
        error = ClientError(
            {'Error': {'Code': error_code, 'Message': 'Test error'}},
            'TestOperation'
        )
        
        # Property: Error should be identified as retryable
        assert client._is_retryable_error(error), f"{error_code} should be retryable"
    
    @pytest.mark.asyncio
    @given(
        error_code=st.text(min_size=1, max_size=50).filter(
            lambda x: x not in {
                'ProvisionedThroughputExceededException',
                'ThrottlingException',
                'RequestLimitExceeded',
                'InternalServerError',
                'ServiceUnavailable'
            }
        )
    )
    @settings(max_examples=100)
    async def test_non_retryable_errors_fail_immediately(self, error_code: str):
        """Test that non-retryable errors are not retried."""
        config = RetryConfig(max_attempts=3)
        client = DynamoDBClient(retry_config=config)
        
        # Create a mock error
        error = ClientError(
            {'Error': {'Code': error_code, 'Message': 'Test error'}},
            'TestOperation'
        )
        
        # Property: Error should not be identified as retryable
        assert not client._is_retryable_error(error), f"{error_code} should not be retryable"
    
    @pytest.mark.asyncio
    async def test_retry_attempts_respect_max_attempts(self):
        """Test that retry logic respects max_attempts configuration."""
        config = RetryConfig(max_attempts=3, base_delay_ms=1, jitter=False)
        client = DynamoDBClient(retry_config=config)
        
        # Create a mock operation that always fails with retryable error
        attempt_count = 0
        
        def failing_operation():
            nonlocal attempt_count
            attempt_count += 1
            raise ClientError(
                {'Error': {'Code': 'ThrottlingException', 'Message': 'Throttled'}},
                'TestOperation'
            )
        
        # Property: Should retry exactly max_attempts times
        with pytest.raises(ClientError):
            await client._execute_with_retry(failing_operation)
        
        assert attempt_count == config.max_attempts, \
            f"Should attempt exactly {config.max_attempts} times, but attempted {attempt_count}"
    
    @pytest.mark.asyncio
    @given(
        base_delay=st.integers(min_value=10, max_value=100),
        max_delay=st.integers(min_value=500, max_value=2000)
    )
    @settings(max_examples=50, deadline=None)
    async def test_jitter_adds_randomness_to_delay(self, base_delay: int, max_delay: int):
        """Test that jitter adds randomness to prevent thundering herd."""
        config = RetryConfig(
            base_delay_ms=base_delay,
            max_delay_ms=max_delay,
            jitter=True
        )
        
        client = DynamoDBClient(retry_config=config)
        
        # Calculate delay multiple times for same attempt
        delays = [client._calculate_delay(1) for _ in range(10)]
        
        # Property: With jitter, delays should vary
        # (There's a tiny chance all 10 could be the same, but extremely unlikely)
        unique_delays = len(set(delays))
        
        # At least some variation should exist (allow for small chance of collision)
        assert unique_delays > 1 or len(delays) < 3, \
            "Jitter should introduce variation in delays"
        
        # All delays should be within expected range
        expected_base = base_delay * 2.0  # attempt=1, so 2^1
        min_expected = (expected_base * 0.5) / 1000.0
        max_expected = min(expected_base, max_delay) / 1000.0
        
        for delay in delays:
            assert min_expected <= delay <= max_expected, \
                f"Delay {delay} should be between {min_expected} and {max_expected}"


class TestDynamoDBPagination:
    """Test Property 43: DynamoDB pagination.
    
    Feature: production-readiness, Property 43: DynamoDB pagination
    Validates: Requirements 14.2
    
    For any list operation on DynamoDB, the system should implement pagination 
    with limit and LastEvaluatedKey.
    """
    
    @pytest.mark.asyncio
    @given(
        limit=st.integers(min_value=1, max_value=100),
        num_items=st.integers(min_value=0, max_value=200)
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture], deadline=None)
    async def test_pagination_respects_limit(self, limit: int, num_items: int):
        """Test that pagination returns at most 'limit' items per page."""
        # Create mock DynamoDB client
        mock_client = Mock(spec=DynamoDBClient)
        mock_table = Mock()
        mock_client.get_table.return_value = mock_table
        
        # Create mock items
        mock_items = [
            {
                'user_id': f'user_{i}',
                'timestamp': datetime.now().isoformat(),
                'recommendation_id': f'rec_{i}',
                'query_id': f'query_{i}',
                'recommendation_text': f'Recommendation {i}',
                'created_at': datetime.now().isoformat()
            }
            for i in range(num_items)
        ]
        
        # Mock the query response
        items_to_return = mock_items[:limit] if num_items > limit else mock_items
        has_more = num_items > limit
        
        mock_response = {
            'Items': items_to_return,
        }
        
        if has_more:
            mock_response['LastEvaluatedKey'] = {'user_id': f'user_{limit-1}', 'timestamp': datetime.now().isoformat()}
        
        mock_client._execute_with_retry = AsyncMock(return_value=mock_response)
        
        # Create repository
        from data.repositories import RecommendationRepository
        repo = RecommendationRepository(mock_client, env='test')
        
        # Query with limit
        result = await repo.list_user_recommendations(
            user_id='test_user',
            limit=limit
        )
        
        # Property: Result should contain at most 'limit' items
        assert len(result['items']) <= limit, \
            f"Result should contain at most {limit} items, got {len(result['items'])}"
        
        # Property: If there are more items than limit, LastEvaluatedKey should be present
        if num_items > limit:
            assert 'last_evaluated_key' in result, \
                "LastEvaluatedKey should be present when there are more items"
        
        # Property: If items <= limit, LastEvaluatedKey should not be present
        if num_items <= limit:
            assert 'last_evaluated_key' not in result, \
                "LastEvaluatedKey should not be present when all items fit in one page"
    
    @pytest.mark.asyncio
    @given(
        page_size=st.integers(min_value=1, max_value=50),
        total_items=st.integers(min_value=1, max_value=150)
    )
    @settings(max_examples=50, suppress_health_check=[HealthCheck.function_scoped_fixture], deadline=None)
    async def test_pagination_can_retrieve_all_items(self, page_size: int, total_items: int):
        """Test that pagination can retrieve all items across multiple pages."""
        # Create mock DynamoDB client
        mock_client = Mock(spec=DynamoDBClient)
        mock_table = Mock()
        mock_client.get_table.return_value = mock_table
        
        # Create all items
        all_items = [
            {
                'farm_id': f'farm_{i}',
                'owner_id': 'test_owner',
                'location': {'lat': 0.0, 'lon': 0.0},
                'area_hectares': 1.0,
                'soil_type': 'loamy',
                'irrigation_type': 'drip',
                'crops': [],
                'created_at': datetime.now().isoformat(),
                'updated_at': datetime.now().isoformat(),
                'metadata': {}
            }
            for i in range(total_items)
        ]
        
        # Track which page we're on
        page_num = [0]
        
        def mock_query(operation, **kwargs):
            """Mock query that returns paginated results."""
            start_idx = page_num[0] * page_size
            end_idx = min(start_idx + page_size, total_items)
            items = all_items[start_idx:end_idx]
            
            response = {'Items': items}
            
            if end_idx < total_items:
                response['LastEvaluatedKey'] = {'owner_id': 'test_owner', 'created_at': all_items[end_idx-1]['created_at']}
            
            page_num[0] += 1
            return response
        
        mock_client._execute_with_retry = AsyncMock(side_effect=mock_query)
        
        # Create repository
        from data.repositories import FarmRepository
        repo = FarmRepository(mock_client, env='test')
        
        # Retrieve all items using pagination
        retrieved_items = []
        exclusive_start_key = None
        
        while True:
            result = await repo.list_user_farms(
                owner_id='test_owner',
                limit=page_size,
                exclusive_start_key=exclusive_start_key
            )
            
            retrieved_items.extend(result['items'])
            
            if 'last_evaluated_key' not in result:
                break
            
            exclusive_start_key = result['last_evaluated_key']
        
        # Property: Should retrieve all items
        assert len(retrieved_items) == total_items, \
            f"Should retrieve all {total_items} items, got {len(retrieved_items)}"
    
    @pytest.mark.asyncio
    @given(
        limit=st.integers(min_value=1, max_value=100)
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
    async def test_pagination_with_exclusive_start_key(self, limit: int):
        """Test that pagination correctly uses exclusive_start_key."""
        # Create mock DynamoDB client
        mock_client = Mock(spec=DynamoDBClient)
        mock_table = Mock()
        mock_client.get_table.return_value = mock_table
        
        # Track if exclusive_start_key was passed
        received_start_key = [None]
        
        def mock_query(operation, **kwargs):
            """Mock query that checks for exclusive_start_key."""
            received_start_key[0] = kwargs.get('ExclusiveStartKey')
            return {'Items': []}
        
        mock_client._execute_with_retry = AsyncMock(side_effect=mock_query)
        
        # Create repository
        from data.repositories import ProductRepository
        repo = ProductRepository(mock_client, env='test')
        
        # Query with exclusive_start_key
        test_start_key = {'product_id': 'test_product'}
        await repo.list_farmer_products(
            farmer_id='test_farmer',
            limit=limit,
            exclusive_start_key=test_start_key
        )
        
        # Property: exclusive_start_key should be passed to DynamoDB query
        assert received_start_key[0] == test_start_key, \
            "exclusive_start_key should be passed to DynamoDB query"
    
    @pytest.mark.asyncio
    @given(
        limit=st.integers(min_value=1, max_value=1000)
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
    async def test_pagination_limit_validation(self, limit: int):
        """Test that pagination limit is properly validated and applied."""
        # Create mock DynamoDB client
        mock_client = Mock(spec=DynamoDBClient)
        mock_table = Mock()
        mock_client.get_table.return_value = mock_table
        
        # Track the limit passed to DynamoDB
        received_limit = [None]
        
        def mock_query(operation, **kwargs):
            """Mock query that captures the limit."""
            received_limit[0] = kwargs.get('Limit')
            return {'Items': []}
        
        mock_client._execute_with_retry = AsyncMock(side_effect=mock_query)
        
        # Create repository
        from data.repositories import UserRepository
        repo = UserRepository(mock_client, env='test')
        
        # For UserRepository, we need to use a method that supports pagination
        # Since UserRepository doesn't have a list method, let's use RecommendationRepository
        from data.repositories import RecommendationRepository
        repo = RecommendationRepository(mock_client, env='test')
        
        # Query with limit
        await repo.list_user_recommendations(
            user_id='test_user',
            limit=limit
        )
        
        # Property: Limit should be passed to DynamoDB query
        assert received_limit[0] == limit, \
            f"Limit {limit} should be passed to DynamoDB query, got {received_limit[0]}"
    
    @pytest.mark.asyncio
    async def test_pagination_empty_results(self):
        """Test that pagination handles empty results correctly."""
        # Create mock DynamoDB client
        mock_client = Mock(spec=DynamoDBClient)
        mock_table = Mock()
        mock_client.get_table.return_value = mock_table
        
        # Mock empty response
        mock_client._execute_with_retry = AsyncMock(return_value={'Items': []})
        
        # Create repository
        from data.repositories import KnowledgeRepository
        repo = KnowledgeRepository(mock_client, env='test')
        
        # Query with pagination
        result = await repo.list_chunks_by_topic(
            topic='test_topic',
            limit=10
        )
        
        # Property: Empty results should return empty items list
        assert result['items'] == [], "Empty results should return empty items list"
        
        # Property: Empty results should not have last_evaluated_key
        assert 'last_evaluated_key' not in result, \
            "Empty results should not have last_evaluated_key"


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
