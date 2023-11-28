# -*- encoding: utf-8 -*-
"""
@author: ztt
@time: 2023/5/28 3:27
@file: deps.py
@desc: 
"""
from fastapi import Request
from httpx import AsyncClient

async_client = AsyncClient(timeout=30)


async def http_client(request: Request) -> AsyncClient:
    async_client.headers.update(
        {
            "credentials": request.headers.get("credentials", ""),
            "provided_credentials": request.headers.get("provided_credentials", ""),
        }
    )
    return async_client
