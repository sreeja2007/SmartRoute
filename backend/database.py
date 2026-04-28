from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
import os

# Use an absolute path so all entrypoints share the same SQLite file,
# regardless of current working directory.
_DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "orders.db")
SQLALCHEMY_DATABASE_URL = f"sqlite:///{_DB_PATH}"

engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()