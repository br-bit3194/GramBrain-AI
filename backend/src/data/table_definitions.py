"""DynamoDB table definitions and creation utilities."""

from typing import Dict, Any, List
import boto3
from botocore.exceptions import ClientError
import logging

logger = logging.getLogger(__name__)


def get_user_table_definition(env: str) -> Dict[str, Any]:
    """Get Users table definition.
    
    Args:
        env: Environment name
        
    Returns:
        Table definition dictionary
    """
    return {
        'TableName': f'grambrain-users-{env}',
        'KeySchema': [
            {'AttributeName': 'user_id', 'KeyType': 'HASH'},
        ],
        'AttributeDefinitions': [
            {'AttributeName': 'user_id', 'AttributeType': 'S'},
            {'AttributeName': 'phone_number', 'AttributeType': 'S'},
        ],
        'GlobalSecondaryIndexes': [
            {
                'IndexName': 'phone-index',
                'KeySchema': [
                    {'AttributeName': 'phone_number', 'KeyType': 'HASH'},
                ],
                'Projection': {'ProjectionType': 'ALL'},
            }
        ],
        'BillingMode': 'PAY_PER_REQUEST',
        'Tags': [
            {'Key': 'Environment', 'Value': env},
            {'Key': 'Project', 'Value': 'GramBrain'},
        ],
    }


def get_farm_table_definition(env: str) -> Dict[str, Any]:
    """Get Farms table definition.
    
    Args:
        env: Environment name
        
    Returns:
        Table definition dictionary
    """
    return {
        'TableName': f'grambrain-farms-{env}',
        'KeySchema': [
            {'AttributeName': 'farm_id', 'KeyType': 'HASH'},
            {'AttributeName': 'owner_id', 'KeyType': 'RANGE'},
        ],
        'AttributeDefinitions': [
            {'AttributeName': 'farm_id', 'AttributeType': 'S'},
            {'AttributeName': 'owner_id', 'AttributeType': 'S'},
            {'AttributeName': 'created_at', 'AttributeType': 'S'},
        ],
        'GlobalSecondaryIndexes': [
            {
                'IndexName': 'owner-index',
                'KeySchema': [
                    {'AttributeName': 'owner_id', 'KeyType': 'HASH'},
                    {'AttributeName': 'created_at', 'KeyType': 'RANGE'},
                ],
                'Projection': {'ProjectionType': 'ALL'},
            }
        ],
        'BillingMode': 'PAY_PER_REQUEST',
        'Tags': [
            {'Key': 'Environment', 'Value': env},
            {'Key': 'Project', 'Value': 'GramBrain'},
        ],
    }


def get_recommendation_table_definition(env: str) -> Dict[str, Any]:
    """Get Recommendations table definition.
    
    Args:
        env: Environment name
        
    Returns:
        Table definition dictionary
    """
    return {
        'TableName': f'grambrain-recommendations-{env}',
        'KeySchema': [
            {'AttributeName': 'user_id', 'KeyType': 'HASH'},
            {'AttributeName': 'timestamp', 'KeyType': 'RANGE'},
        ],
        'AttributeDefinitions': [
            {'AttributeName': 'user_id', 'AttributeType': 'S'},
            {'AttributeName': 'timestamp', 'AttributeType': 'S'},
            {'AttributeName': 'query_id', 'AttributeType': 'S'},
        ],
        'GlobalSecondaryIndexes': [
            {
                'IndexName': 'query-index',
                'KeySchema': [
                    {'AttributeName': 'query_id', 'KeyType': 'HASH'},
                ],
                'Projection': {'ProjectionType': 'ALL'},
            }
        ],
        'BillingMode': 'PAY_PER_REQUEST',
        'Tags': [
            {'Key': 'Environment', 'Value': env},
            {'Key': 'Project', 'Value': 'GramBrain'},
        ],
    }


def get_product_table_definition(env: str) -> Dict[str, Any]:
    """Get Products table definition.
    
    Args:
        env: Environment name
        
    Returns:
        Table definition dictionary
    """
    return {
        'TableName': f'grambrain-products-{env}',
        'KeySchema': [
            {'AttributeName': 'product_id', 'KeyType': 'HASH'},
        ],
        'AttributeDefinitions': [
            {'AttributeName': 'product_id', 'AttributeType': 'S'},
            {'AttributeName': 'farmer_id', 'AttributeType': 'S'},
            {'AttributeName': 'created_at', 'AttributeType': 'S'},
            {'AttributeName': 'product_type', 'AttributeType': 'S'},
            {'AttributeName': 'pure_product_score', 'AttributeType': 'N'},
        ],
        'GlobalSecondaryIndexes': [
            {
                'IndexName': 'farmer-index',
                'KeySchema': [
                    {'AttributeName': 'farmer_id', 'KeyType': 'HASH'},
                    {'AttributeName': 'created_at', 'KeyType': 'RANGE'},
                ],
                'Projection': {'ProjectionType': 'ALL'},
            },
            {
                'IndexName': 'type-score-index',
                'KeySchema': [
                    {'AttributeName': 'product_type', 'KeyType': 'HASH'},
                    {'AttributeName': 'pure_product_score', 'KeyType': 'RANGE'},
                ],
                'Projection': {'ProjectionType': 'ALL'},
            }
        ],
        'BillingMode': 'PAY_PER_REQUEST',
        'Tags': [
            {'Key': 'Environment', 'Value': env},
            {'Key': 'Project', 'Value': 'GramBrain'},
        ],
    }


def get_knowledge_table_definition(env: str) -> Dict[str, Any]:
    """Get Knowledge chunks table definition.
    
    Args:
        env: Environment name
        
    Returns:
        Table definition dictionary
    """
    return {
        'TableName': f'grambrain-knowledge-{env}',
        'KeySchema': [
            {'AttributeName': 'chunk_id', 'KeyType': 'HASH'},
        ],
        'AttributeDefinitions': [
            {'AttributeName': 'chunk_id', 'AttributeType': 'S'},
            {'AttributeName': 'topic', 'AttributeType': 'S'},
            {'AttributeName': 'created_at', 'AttributeType': 'S'},
        ],
        'GlobalSecondaryIndexes': [
            {
                'IndexName': 'topic-index',
                'KeySchema': [
                    {'AttributeName': 'topic', 'KeyType': 'HASH'},
                    {'AttributeName': 'created_at', 'KeyType': 'RANGE'},
                ],
                'Projection': {'ProjectionType': 'ALL'},
            }
        ],
        'BillingMode': 'PAY_PER_REQUEST',
        'Tags': [
            {'Key': 'Environment', 'Value': env},
            {'Key': 'Project', 'Value': 'GramBrain'},
        ],
    }


def create_table(dynamodb_client, table_definition: Dict[str, Any]) -> bool:
    """Create a DynamoDB table.
    
    Args:
        dynamodb_client: Boto3 DynamoDB client
        table_definition: Table definition dictionary
        
    Returns:
        True if created, False if already exists
    """
    table_name = table_definition['TableName']
    
    try:
        dynamodb_client.create_table(**table_definition)
        logger.info(f"Creating table {table_name}...")
        
        # Wait for table to be created
        waiter = dynamodb_client.get_waiter('table_exists')
        waiter.wait(TableName=table_name)
        
        logger.info(f"Table {table_name} created successfully")
        return True
        
    except ClientError as e:
        if e.response['Error']['Code'] == 'ResourceInUseException':
            logger.info(f"Table {table_name} already exists")
            return False
        else:
            logger.error(f"Error creating table {table_name}: {e}")
            raise


def verify_table(dynamodb_client, table_name: str) -> bool:
    """Verify that a table exists and is active.
    
    Args:
        dynamodb_client: Boto3 DynamoDB client
        table_name: Name of the table
        
    Returns:
        True if table exists and is active
    """
    try:
        response = dynamodb_client.describe_table(TableName=table_name)
        status = response['Table']['TableStatus']
        
        if status == 'ACTIVE':
            logger.info(f"Table {table_name} is active")
            return True
        else:
            logger.warning(f"Table {table_name} status: {status}")
            return False
            
    except ClientError as e:
        if e.response['Error']['Code'] == 'ResourceNotFoundException':
            logger.warning(f"Table {table_name} does not exist")
            return False
        else:
            logger.error(f"Error verifying table {table_name}: {e}")
            raise


def initialize_tables(region_name: str = 'us-east-1', env: str = 'dev') -> Dict[str, bool]:
    """Initialize all DynamoDB tables.
    
    Args:
        region_name: AWS region
        env: Environment name
        
    Returns:
        Dictionary mapping table names to creation status
    """
    dynamodb_client = boto3.client('dynamodb', region_name=region_name)
    
    table_definitions = [
        get_user_table_definition(env),
        get_farm_table_definition(env),
        get_recommendation_table_definition(env),
        get_product_table_definition(env),
        get_knowledge_table_definition(env),
    ]
    
    results = {}
    
    for table_def in table_definitions:
        table_name = table_def['TableName']
        
        # Check if table exists
        if verify_table(dynamodb_client, table_name):
            results[table_name] = False  # Already exists
        else:
            # Create table
            created = create_table(dynamodb_client, table_def)
            results[table_name] = created
    
    return results


if __name__ == '__main__':
    # Example usage
    import sys
    
    env = sys.argv[1] if len(sys.argv) > 1 else 'dev'
    region = sys.argv[2] if len(sys.argv) > 2 else 'us-east-1'
    
    logging.basicConfig(level=logging.INFO)
    
    print(f"Initializing DynamoDB tables for environment: {env}")
    results = initialize_tables(region_name=region, env=env)
    
    print("\nResults:")
    for table_name, created in results.items():
        status = "Created" if created else "Already exists"
        print(f"  {table_name}: {status}")
