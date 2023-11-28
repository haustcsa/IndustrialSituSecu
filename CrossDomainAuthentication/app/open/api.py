# -*- encoding: utf-8 -*-
"""
@author: ztt
@time: 2023/4/22 12:48
@file: api.py
@desc: 
"""
from fastapi import APIRouter

from .endpoints import openapi

api_router = APIRouter()
api_router.include_router(openapi.router, prefix="/api", tags=["设备"])
