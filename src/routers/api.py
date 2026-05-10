from fastapi import APIRouter
from src.routers.v1 import predict, health

api_router = APIRouter()
api_router.include_router(predict.router, prefix="/predict")
api_router.include_router(health.router, prefix="/health")
