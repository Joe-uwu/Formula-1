import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

DATABASE_URL = os.environ.get(
    "DATABASE_URL", "postgresql+psycopg2://f1:f1@localhost:5432/f1"
)

engine = create_engine(DATABASE_URL, future=True)
Session = sessionmaker(bind=engine, future=True)
