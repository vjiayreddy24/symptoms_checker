from fastapi import FastAPI
from db.connection import Base, engine
from db import models
from routers import recommend
from fastapi.middleware.cors import CORSMiddleware
# Create DB tables
Base.metadata.create_all(bind=engine)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(recommend.router) 