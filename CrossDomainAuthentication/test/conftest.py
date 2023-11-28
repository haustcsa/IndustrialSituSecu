# -*- encoding: utf-8 -*-
"""
@author: ztt
@time: 2023/4/2 0:34
@file: conftest.py
@desc: 
"""
import asyncio
from typing import AsyncGenerator

import pytest
import pytest_asyncio
from httpx import AsyncClient

from app.core.logger import logger


@pytest.fixture(scope="session")
def event_loop():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    yield loop
    loop.close()


@pytest_asyncio.fixture(scope="session", autouse=True)
async def client() -> AsyncGenerator[AsyncClient, None]:
    async with AsyncClient() as client_instance:
        logger.debug("Client is ready")
        yield client_instance
