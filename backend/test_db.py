import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Get database URL from environment
DATABASE_URL = os.getenv("DATABASE_URI")

try:
    # Create engine and test connection
    engine = create_engine(DATABASE_URL)
    connection = engine.connect()
    print("✅ Successfully connected to the database!")
    
    # Test query
    result = connection.execute("SELECT version();")
    print("\nPostgreSQL Database Version:")
    print("-" * 30)
    print(result.fetchone()[0])
    
    # Close connection
    connection.close()
    print("\n✅ Connection closed successfully!")
    
except Exception as e:
    print("❌ Error connecting to the database:")
    print(str(e))
