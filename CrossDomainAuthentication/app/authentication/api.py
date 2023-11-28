# -*- encoding: utf-8 -*-
"""
@author: ztt
@time: 2023/2/15 16:08:50
@file: api.py
@desc: 
"""
from fastapi import APIRouter

from .endpoints import equipment, auth, perception

api_router = APIRouter()
api_router.include_router(equipment.router, prefix="/equipment", tags=["设备"])
api_router.include_router(auth.router, prefix="/auth", tags=["认证"])
api_router.include_router(perception.router, prefix="/perception", tags=["工业情感感知"])
