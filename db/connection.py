from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

# DATABASE_URL = "postgresql://mhealthadmin:mhealth*2025@localhost:5432/mhealth_db"
DATABASE_URL = "postgresql://mhealthadmin:mhealth*2025@57.159.27.80:5432/mhealth_db"

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()
