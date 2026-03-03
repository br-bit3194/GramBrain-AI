"""
Setup script to initialize DynamoDB tables for authentication
"""
import asyncio
import os
from backend.src.data.table_definitions import initialize_tables

async def main():
    print("🔧 Setting up DynamoDB tables for authentication...")
    print("=" * 60)
    
    # Get configuration from environment
    region = os.getenv("AWS_REGION", "us-east-1")
    env = os.getenv("DYNAMODB_ENV", "dev")
    
    print(f"\nConfiguration:")
    print(f"  Region: {region}")
    print(f"  Environment: {env}")
    print()
    
    # Initialize tables
    try:
        results = initialize_tables(region_name=region, env=env)
        
        print("\n✅ Table initialization complete!")
        print("\nResults:")
        for table_name, created in results.items():
            status = "✨ Created" if created else "✓ Already exists"
            print(f"  {status}: {table_name}")
        
        print("\n" + "=" * 60)
        print("✅ Setup complete! You can now use authentication.")
        print("\nNext steps:")
        print("  1. Start the backend: uvicorn backend.main:app --reload")
        print("  2. Start the frontend: cd frontend && npm run dev")
        print("  3. Test at: http://localhost:3000/auth-demo")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nMake sure:")
        print("  1. AWS credentials are configured in .env")
        print("  2. You have permissions to create DynamoDB tables")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(asyncio.run(main()))
