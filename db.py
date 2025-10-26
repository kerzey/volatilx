from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
import os

# Database URL - Use sync SQLite only
# DATABASE_URL = "sqlite:///./data/users.db"
# DATABASE_URL = "sqlite:///./data/users.db"
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise ValueError("DATABASE_URL environment variable is required.")
# Create sync engine
engine = create_engine(
    DATABASE_URL, 
    echo=False,  # Set to False in production
    # connect_args={"check_same_thread": False}  # Needed for SQLite
)

# Session maker for sync operations
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base class for models
Base = declarative_base()

# Database dependency function
def get_db():
    """
    Dependency function to get database session
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Create data directory if it doesn't exist
def ensure_data_directory():
    """
    Ensure the data directory exists for the SQLite database
    """
    import os
    os.makedirs("data", exist_ok=True)

# Initialize database tables
def create_tables():
    """
    Create all database tables
    """
    ensure_data_directory()
    Base.metadata.create_all(bind=engine)

# Test database connection
def test_connection():
    """
    Test database connection
    """
    try:
        db = SessionLocal()
        db.execute("SELECT 1")
        db.close()
        print("✅ Database connection successful")
        return True
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        return False