# -*- encoding: utf-8 -*-
"""
@author: ztt
@time: 2023/2/16 10:14:46
@file: api.py
@desc: 
"""
from fastapi import APIRouter

from .endpoints import equipment, business, perception, dh_rsa

api_router = APIRouter()
api_router.include_router(equipment.router, prefix="/equipment", tags=["设备"])
api_router.include_router(business.router, prefix="/business", tags=["业务接口"])
api_router.include_router(perception.router, prefix="/perception", tags=["持续工业感知"])
api_router.include_router(dh_rsa.router, prefix="/dh_rsa", tags=["秘钥协商"])
