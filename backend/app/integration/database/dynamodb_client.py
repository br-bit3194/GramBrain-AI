# app/aws_integration/database/dynamodb_client.py
import boto3
from boto3.dynamodb.conditions import Key, Attr
from boto3.dynamodb.types import TypeSerializer
from typing import Dict, Any, List, Optional
import logging
from datetime import datetime, timedelta
from decimal import Decimal
from ..config.aws_config import get_aws_credentials, aws_config

logger = logging.getLogger(__name__)


def convert_floats_to_decimals(obj: Any) -> Any:
    """Recursively convert floats to Decimals and datetime to ISO strings for DynamoDB"""
    if isinstance(obj, float):
        return Decimal(str(obj))
    elif isinstance(obj, datetime):
        return obj.isoformat()
    elif isinstance(obj, dict):
        return {k: convert_floats_to_decimals(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_floats_to_decimals(item) for item in obj]
    return obj


class DynamoDBClient:
    """DynamoDB client for GramBrain"""
    
    def __init__(self):
        credentials = get_aws_credentials()
        self.dynamodb = boto3.resource('dynamodb', **credentials)
        self.client = boto3.client('dynamodb', **credentials)
        
        # Table references
        self.market_prices_table = self.dynamodb.Table(aws_config.dynamodb_market_prices_table)
        self.sessions_table = self.dynamodb.Table(aws_config.dynamodb_sessions_table)
        self.analytics_table = self.dynamodb.Table(aws_config.dynamodb_analytics_table)
        
        logger.info("DynamoDB client initialized")
    
    # ============================================================================
    # Market Prices Operations
    # ============================================================================
    
    def put_market_price(self, price_data: Dict[str, Any]) -> bool:
        """Insert or update market price"""
        try:
            price_data = convert_floats_to_decimals(price_data)
            self.market_prices_table.put_item(Item=price_data)
            return True
        except Exception as e:
            logger.error(f"Error putting market price: {e}")
            return False
    
    def get_market_prices_by_commodity(
        self, 
        commodity: str, 
        days_back: int = 7
    ) -> List[Dict[str, Any]]:
        """Get market prices for a commodity"""
        try:
            # Calculate date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_back)
            
            pk = f"COMMODITY#{commodity}"
            sk_start = f"DATE#{start_date.strftime('%Y-%m-%d')}"
            sk_end = f"DATE#{end_date.strftime('%Y-%m-%d')}#ZZZZ"
            
            response = self.market_prices_table.query(
                KeyConditionExpression=Key('pk').eq(pk) & Key('sk').between(sk_start, sk_end),
                FilterExpression=Attr('is_active').eq(True),
                ScanIndexForward=False,  # False = descending order (latest first)
                Limit=100
            )
            
            return response.get('Items', [])
        except Exception as e:
            logger.error(f"Error querying market prices: {e}")
            return []
    
    def batch_write_market_prices(self, prices: List[Dict[str, Any]]) -> int:
        """Batch write market prices"""
        try:
            with self.market_prices_table.batch_writer() as batch:
                for price in prices:
                    price = convert_floats_to_decimals(price)
                    batch.put_item(Item=price)
            return len(prices)
        except Exception as e:
            logger.error(f"Error batch writing prices: {e}")
            return 0
    
    # ============================================================================
    # Session Operations
    # ============================================================================
    
    def create_session(self, session_id: str, user_id: str, initial_state: Dict[str, Any]) -> bool:
        """Create a new session"""
        try:
            # TTL: 7 days from now
            ttl = int((datetime.now() + timedelta(days=7)).timestamp())
            
            session_data = {
                'pk': f"SESSION#{session_id}",
                'sk': 'METADATA',
                'user_id': user_id,
                'session_id': session_id,
                'state': initial_state,
                'created_at': datetime.now().isoformat(),
                'updated_at': datetime.now().isoformat(),
                'ttl': ttl
            }
            
            # Convert floats to decimals
            session_data = convert_floats_to_decimals(session_data)
            
            self.sessions_table.put_item(Item=session_data)
            return True
        except Exception as e:
            logger.error(f"Error creating session: {e}")
            return False
    
    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get session by ID"""
        try:
            response = self.sessions_table.get_item(
                Key={
                    'pk': f"SESSION#{session_id}",
                    'sk': 'METADATA'
                }
            )
            return response.get('Item')
        except Exception as e:
            logger.error(f"Error getting session: {e}")
            return None
    
    def update_session_state(self, session_id: str, state_updates: Dict[str, Any]) -> bool:
        """Update session state"""
        try:
            state_updates = convert_floats_to_decimals(state_updates)
            
            self.sessions_table.update_item(
                Key={
                    'pk': f"SESSION#{session_id}",
                    'sk': 'METADATA'
                },
                UpdateExpression='SET #state = :state, updated_at = :updated_at',
                ExpressionAttributeNames={
                    '#state': 'state'
                },
                ExpressionAttributeValues={
                    ':state': state_updates,
                    ':updated_at': datetime.now().isoformat()
                }
            )
            return True
        except Exception as e:
            logger.error(f"Error updating session: {e}")
            return False
    
    # ============================================================================
    # Analytics Operations
    # ============================================================================
    
    def put_analytics(self, analytics_data: Dict[str, Any]) -> bool:
        """Insert or update analytics"""
        try:
            analytics_data = convert_floats_to_decimals(analytics_data)
            self.analytics_table.put_item(Item=analytics_data)
            return True
        except Exception as e:
            logger.error(f"Error putting analytics: {e}")
            return False
    
    def get_analytics_by_commodity(self, commodity: str, days_back: int = 30) -> List[Dict[str, Any]]:
        """Get analytics for a commodity"""
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_back)
            
            pk = f"ANALYTICS#{commodity}"
            sk_start = f"DATE#{start_date.strftime('%Y-%m-%d')}"
            sk_end = f"DATE#{end_date.strftime('%Y-%m-%d')}"
            
            response = self.analytics_table.query(
                KeyConditionExpression=Key('pk').eq(pk) & Key('sk').between(sk_start, sk_end),
                ScanIndexForward=False,  # False = descending order (latest first)
                Limit=30
            )
            
            return response.get('Items', [])
        except Exception as e:
            logger.error(f"Error querying analytics: {e}")
            return []
    
    # ============================================================================
    # Table Management
    # ============================================================================
    
    def create_tables(self):
        """Create DynamoDB tables if they don't exist"""
        try:
            # Market Prices Table
            self._create_table_if_not_exists(
                table_name=aws_config.dynamodb_market_prices_table,
                key_schema=[
                    {'AttributeName': 'pk', 'KeyType': 'HASH'},
                    {'AttributeName': 'sk', 'KeyType': 'RANGE'}
                ],
                attribute_definitions=[
                    {'AttributeName': 'pk', 'AttributeType': 'S'},
                    {'AttributeName': 'sk', 'AttributeType': 'S'}
                ]
            )
            
            # Sessions Table
            self._create_table_if_not_exists(
                table_name=aws_config.dynamodb_sessions_table,
                key_schema=[
                    {'AttributeName': 'pk', 'KeyType': 'HASH'},
                    {'AttributeName': 'sk', 'KeyType': 'RANGE'}
                ],
                attribute_definitions=[
                    {'AttributeName': 'pk', 'AttributeType': 'S'},
                    {'AttributeName': 'sk', 'AttributeType': 'S'}
                ]
            )
            
            # Analytics Table
            self._create_table_if_not_exists(
                table_name=aws_config.dynamodb_analytics_table,
                key_schema=[
                    {'AttributeName': 'pk', 'KeyType': 'HASH'},
                    {'AttributeName': 'sk', 'KeyType': 'RANGE'}
                ],
                attribute_definitions=[
                    {'AttributeName': 'pk', 'AttributeType': 'S'},
                    {'AttributeName': 'sk', 'AttributeType': 'S'}
                ]
            )
            
            logger.info("All DynamoDB tables created/verified")
        except Exception as e:
            logger.error(f"Error creating tables: {e}")
    
    def _create_table_if_not_exists(
        self, 
        table_name: str, 
        key_schema: List[Dict], 
        attribute_definitions: List[Dict]
    ):
        """Create table if it doesn't exist"""
        try:
            existing_tables = self.client.list_tables()['TableNames']
            
            if table_name not in existing_tables:
                self.client.create_table(
                    TableName=table_name,
                    KeySchema=key_schema,
                    AttributeDefinitions=attribute_definitions,
                    BillingMode='PAY_PER_REQUEST'  # On-demand pricing
                )
                logger.info(f"Created table: {table_name}")
                
                # Wait for table to be active
                waiter = self.client.get_waiter('table_exists')
                waiter.wait(TableName=table_name)
            else:
                logger.info(f"Table already exists: {table_name}")
        except Exception as e:
            logger.error(f"Error creating table {table_name}: {e}")


# Global DynamoDB client instance
dynamodb_client = DynamoDBClient()
