# -*- encoding: utf-8 -*-
"""
@author: ztt
@time: 2023/4/8 16:16
@file: test_business.py
@desc: 测试业务服务api
"""
import pytest
from httpx import AsyncClient

from app.core import logger
from settings import settings


@pytest.mark.asyncio
class TestBusinessApi:
    another_business_base_url = "http://127.0.0.1:9001"

    async def test_cross_domain_authentication_t(self, client: AsyncClient):
        headers = {"origin": settings.BUSINESS_BASE_URL}
        response = await client.post(
            url=self.another_business_base_url + "/business/cross",
            headers=headers,
            json={"msg": "hello"},
        )

        logger.debug(response.text)
