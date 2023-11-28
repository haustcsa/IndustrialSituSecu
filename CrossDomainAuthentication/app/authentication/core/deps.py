# -*- encoding: utf-8 -*-
"""
@author: ztt
@time: 2023/5/28 3:14
@file: deps.py
@desc: 
"""
import time

from fastapi import Request

from app.core import logger
from app.core.handle_rsa import verify_credentials
from settings import settings


async def security_verify(request: Request) -> bool:
    credentials = request.headers.get("credentials")
    if credentials not in settings.CREDENTIALS_ALL:
        return False
    provided_credentials = request.headers.get("provided_credentials")
    logger.info(f"credentials: {credentials}")
    logger.info(f"provided_credentials: {provided_credentials}")
    return await verify_credentials(credentials, provided_credentials)


async def security_verify_with_cost(request: Request) -> dict[str, bool | float]:
    start_time = time.time()
    credentials = request.headers.get("credentials")
    if credentials not in settings.CREDENTIALS_ALL:
        return {"verify_result": False, "cost": 0}
    provided_credentials = request.headers.get("provided_credentials")
    logger.info(f"credentials: {credentials}")
    logger.info(f"provided_credentials: {provided_credentials}")
    verify_result = await verify_credentials(credentials, provided_credentials)
    cost_time = time.time() - start_time
    verify_cost = cost_time * 1000
    logger.info(f"认证服务端耗时: {verify_cost} ms")
    return {"verify_result": verify_result, "cost": verify_cost}
