from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from db.models import Base
from dotenv import dotenv_values

def db_connect():
    config = dotenv_values("./.env")
    username = config.get("DATABASE_USERNAME")
    password = config.get("DATABASE_PASSWORD")
    dbname = config.get("DATABASE_NAME")
    port = config.get("DATABASE_PORT")
    host = config.get("DATABASE_HOST")

    engine = create_engine(f"postgresql+psycopg2://{username}:{password}@{host}:{port}/{dbname}")
    connection = engine.connect()

    return engine, connection

def create_tables(engine):
    Base.metadata.create_all(engine, checkfirst=True)

def create_session(engine):
    Session = sessionmaker(bind=engine)
    session = Session()

    return session