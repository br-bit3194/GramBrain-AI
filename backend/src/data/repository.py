"""Base repository pattern for DynamoDB operations."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, TypeVar, Generic
from datetime import datetime
import logging

from .dynamodb_client import DynamoDBClient

logger = logging.getLogger(__name__)

T = TypeVar('T')


class DynamoDBRepository(ABC, Generic[T]):
    """Base repository for DynamoDB operations with retry logic."""
    
    def __init__(self, dynamodb_client: DynamoDBClient, table_name: str):
        """Initialize repository.
        
        Args:
            dynamodb_client: DynamoDB client with retry logic
            table_name: Name of the DynamoDB table
        """
        self.client = dynamodb_client
        self.table_name = table_name
        self.table = dynamodb_client.get_table(table_name)
    
    @abstractmethod
    def _to_item(self, entity: T) -> Dict[str, Any]:
        """Convert entity to DynamoDB item.
        
        Args:
            entity: Domain entity
            
        Returns:
            DynamoDB item dictionary
        """
        pass
    
    @abstractmethod
    def _from_item(self, item: Dict[str, Any]) -> T:
        """Convert DynamoDB item to entity.
        
        Args:
            item: DynamoDB item dictionary
            
        Returns:
            Domain entity
        """
        pass
    
    async def get_item(
        self,
        key: Dict[str, Any],
        consistent_read: bool = False
    ) -> Optional[T]:
        """Get single item by key.
        
        Args:
            key: Primary key dictionary
            consistent_read: Whether to use consistent read
            
        Returns:
            Entity if found, None otherwise
        """
        try:
            response = await self.client._execute_with_retry(
                self.table.get_item,
                Key=key,
                ConsistentRead=consistent_read
            )
            
            item = response.get('Item')
            if item:
                return self._from_item(item)
            return None
            
        except Exception as e:
            logger.error(f"Error getting item with key {key}: {e}")
            raise
    
    async def put_item(self, entity: T) -> T:
        """Put item into table.
        
        Args:
            entity: Entity to store
            
        Returns:
            Stored entity
        """
        try:
            item = self._to_item(entity)
            await self.client._execute_with_retry(
                self.table.put_item,
                Item=item
            )
            return entity
            
        except Exception as e:
            logger.error(f"Error putting item: {e}")
            raise
    
    async def update_item(
        self,
        key: Dict[str, Any],
        updates: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Update item attributes.
        
        Args:
            key: Primary key dictionary
            updates: Dictionary of attributes to update
            
        Returns:
            Updated attributes
        """
        try:
            # Build update expression
            update_expr_parts = []
            expr_attr_names = {}
            expr_attr_values = {}
            
            for i, (attr, value) in enumerate(updates.items()):
                placeholder = f"#attr{i}"
                value_placeholder = f":val{i}"
                update_expr_parts.append(f"{placeholder} = {value_placeholder}")
                expr_attr_names[placeholder] = attr
                expr_attr_values[value_placeholder] = value
            
            update_expression = "SET " + ", ".join(update_expr_parts)
            
            response = await self.client._execute_with_retry(
                self.table.update_item,
                Key=key,
                UpdateExpression=update_expression,
                ExpressionAttributeNames=expr_attr_names,
                ExpressionAttributeValues=expr_attr_values,
                ReturnValues='ALL_NEW'
            )
            
            return response.get('Attributes', {})
            
        except Exception as e:
            logger.error(f"Error updating item with key {key}: {e}")
            raise
    
    async def delete_item(self, key: Dict[str, Any]) -> None:
        """Delete item by key.
        
        Args:
            key: Primary key dictionary
        """
        try:
            await self.client._execute_with_retry(
                self.table.delete_item,
                Key=key
            )
            
        except Exception as e:
            logger.error(f"Error deleting item with key {key}: {e}")
            raise
    
    async def query(
        self,
        key_condition_expression,
        index_name: Optional[str] = None,
        filter_expression=None,
        projection_expression: Optional[str] = None,
        limit: Optional[int] = None,
        scan_index_forward: bool = True,
        exclusive_start_key: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Query items with pagination support.
        
        Args:
            key_condition_expression: Key condition for query
            index_name: GSI name if querying an index
            filter_expression: Optional filter expression
            projection_expression: Attributes to retrieve
            limit: Maximum items to return
            scan_index_forward: Sort order (True=ascending, False=descending)
            exclusive_start_key: Pagination token
            
        Returns:
            Dictionary with 'items' and optional 'last_evaluated_key'
        """
        try:
            kwargs = {
                'KeyConditionExpression': key_condition_expression,
                'ScanIndexForward': scan_index_forward,
            }
            
            if index_name:
                kwargs['IndexName'] = index_name
            if filter_expression is not None:
                kwargs['FilterExpression'] = filter_expression
            if projection_expression:
                kwargs['ProjectionExpression'] = projection_expression
            if limit:
                kwargs['Limit'] = limit
            if exclusive_start_key:
                kwargs['ExclusiveStartKey'] = exclusive_start_key
            
            response = await self.client._execute_with_retry(
                self.table.query,
                **kwargs
            )
            
            items = [self._from_item(item) for item in response.get('Items', [])]
            result = {'items': items}
            
            if 'LastEvaluatedKey' in response:
                result['last_evaluated_key'] = response['LastEvaluatedKey']
            
            return result
            
        except Exception as e:
            logger.error(f"Error querying table: {e}")
            raise
    
    async def batch_write(self, items: List[T]) -> None:
        """Batch write items with automatic chunking.
        
        Args:
            items: List of entities to write
        """
        try:
            # DynamoDB batch write limit is 25 items
            chunk_size = 25
            
            for i in range(0, len(items), chunk_size):
                chunk = items[i:i + chunk_size]
                
                with self.table.batch_writer() as batch:
                    for entity in chunk:
                        item = self._to_item(entity)
                        await self.client._execute_with_retry(
                            batch.put_item,
                            Item=item
                        )
            
            logger.info(f"Successfully batch wrote {len(items)} items")
            
        except Exception as e:
            logger.error(f"Error in batch write: {e}")
            raise
    
    async def batch_get(
        self,
        keys: List[Dict[str, Any]],
        projection_expression: Optional[str] = None
    ) -> List[T]:
        """Batch get items with automatic chunking.
        
        Args:
            keys: List of primary keys
            projection_expression: Attributes to retrieve
            
        Returns:
            List of entities
        """
        try:
            # DynamoDB batch get limit is 100 items
            chunk_size = 100
            all_items = []
            
            for i in range(0, len(keys), chunk_size):
                chunk_keys = keys[i:i + chunk_size]
                
                request_items = {
                    self.table_name: {
                        'Keys': chunk_keys
                    }
                }
                
                if projection_expression:
                    request_items[self.table_name]['ProjectionExpression'] = projection_expression
                
                response = await self.client._execute_with_retry(
                    self.client.client.batch_get_item,
                    RequestItems=request_items
                )
                
                items = response.get('Responses', {}).get(self.table_name, [])
                all_items.extend([self._from_item(item) for item in items])
            
            return all_items
            
        except Exception as e:
            logger.error(f"Error in batch get: {e}")
            raise
