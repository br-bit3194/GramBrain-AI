import boto3
import json
import logging
from datetime import datetime
from typing import Dict, Any, Optional
import os

logger = logging.getLogger(__name__)

class DynamoDBSessionService:
    """AWS DynamoDB-based session management service"""
    
    def __init__(self):
        self.region = os.getenv("AWS_REGION", "us-east-1")
        self.table_name = os.getenv("DYNAMODB_TABLE_SESSIONS", "farmbot_sessions")
        
        try:
            self.dynamodb = boto3.resource('dynamodb', region_name=self.region)
            self.table = self.dynamodb.Table(self.table_name)
            logger.info(f"✅ DynamoDB session service initialized for table: {self.table_name}")
        except Exception as e:
            logger.error(f"❌ Failed to initialize DynamoDB: {e}")
            raise
    
    async def create_session(self, app_name: str, user_id: str, session_id: str, state: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new session in DynamoDB"""
        try:
            item = {
                'session_id': session_id,
                'app_name': app_name,
                'user_id': user_id,
                'state': json.dumps(state),
                'created_at': datetime.now().isoformat(),
                'updated_at': datetime.now().isoformat(),
                'ttl': int(datetime.now().timestamp()) + (86400 * 7)  # 7 days TTL
            }
            
            self.table.put_item(Item=item)
            logger.info(f"✅ Session created: {session_id}")
            return item
        except Exception as e:
            logger.error(f"❌ Error creating session: {e}")
            raise
    
    async def get_session(self, app_name: str, user_id: str, session_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a session from DynamoDB"""
        try:
            response = self.table.get_item(Key={'session_id': session_id})
            
            if 'Item' in response:
                item = response['Item']
                item['state'] = json.loads(item['state'])
                logger.info(f"✅ Session retrieved: {session_id}")
                return item
            
            logger.warning(f"Session not found: {session_id}")
            return None
        except Exception as e:
            logger.error(f"❌ Error retrieving session: {e}")
            return None
    
    async def update_session(self, session_id: str, state: Dict[str, Any]) -> bool:
        """Update session state in DynamoDB"""
        try:
            self.table.update_item(
                Key={'session_id': session_id},
                UpdateExpression='SET #state = :state, updated_at = :updated_at',
                ExpressionAttributeNames={'#state': 'state'},
                ExpressionAttributeValues={
                    ':state': json.dumps(state),
                    ':updated_at': datetime.now().isoformat()
                }
            )
            logger.info(f"✅ Session updated: {session_id}")
            return True
        except Exception as e:
            logger.error(f"❌ Error updating session: {e}")
            return False
    
    async def delete_session(self, session_id: str) -> bool:
        """Delete a session from DynamoDB"""
        try:
            self.table.delete_item(Key={'session_id': session_id})
            logger.info(f"✅ Session deleted: {session_id}")
            return True
        except Exception as e:
            logger.error(f"❌ Error deleting session: {e}")
            return False
