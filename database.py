from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, scoped_session
from app.models import Base

# SQLite database
DATABASE_URL = 'sqlite:///stock_control.db'

engine = create_engine(DATABASE_URL, connect_args={'check_same_thread': False})
SessionLocal = scoped_session(sessionmaker(bind=engine))

def init_db():
    """Initialize the database"""
    Base.metadata.create_all(bind=engine)

def get_db():
    """Get database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
