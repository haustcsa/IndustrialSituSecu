# -*- encoding: utf-8 -*-
"""
@author: ztt
@time: 2023/6/1 21:20
@file: test_credentials.py
@desc: 测试身份凭证
"""
import time

import pytest

from app.client.main import CredentialsModel
from app.core import logger, calculate_pseudo_identity
from app.core.handle_rsa import generate_credentials, verify_credentials
from app.core.random_range import generate_random_number
from settings import settings


@pytest.mark.asyncio
class TestCredentials:
    identity = calculate_pseudo_identity(b"client")[:10]
    credentials_all: dict[str, CredentialsModel] = {}

    async def test_gen_credentials(self):
        """
        测试生成身份凭证
        :return:
        """
        start_time = time.time()
        timestamp = str(int(time.time()))
        random_factor = str(generate_random_number())[:10]
        credentials = await generate_credentials(
            timestamp + self.identity + random_factor
        )
        end_time = time.time()
        cost_time = (end_time - start_time) * 1000
        logger.info(f"credentials: {credentials}")
        logger.info(f"身份凭证生成耗时：{cost_time} 毫秒")
        self.credentials_all[settings.BUSINESS_BASE_URL] = CredentialsModel(
            credentials=credentials,
            identity=self.identity,
            random_factor=random_factor,
            timestamp=timestamp,
        )
        return cost_time

    async def test_verify_credentials(self):
        """测试校验身份凭证"""
        gen_cost_time = await self.test_gen_credentials()
        credentials_item = self.credentials_all[settings.BUSINESS_BASE_URL]
        start_time = time.time()
        await verify_credentials(
            credentials_item.credentials, credentials_item.provided_credentials()
        )
        end_time = time.time()
        verify_cost_time = (end_time - start_time) * 1000
        logger.info(f"身份凭证校验耗时：{verify_cost_time} 毫秒")
        return gen_cost_time, verify_cost_time

    async def test_credentials(self, count: int = 10):
        """
        测试身份凭证的生成和校验耗时
        :param count: 实验次数
        :return:
        """
        gen_cost_time_list = []
        verify_cost_time_list = []
        for i in range(1, count + 1):
            logger.info(f"正在进行第 {i} 次测试")
            gen_cost_time, verify_cost_time = await self.test_verify_credentials()
            gen_cost_time_list.append(gen_cost_time)
            verify_cost_time_list.append(verify_cost_time)

        logger.info(f"实验次数：{count} 生成身份凭证平均耗时：{sum(gen_cost_time_list) / count} ms")
        logger.info(f"实验次数：{count} 校验身份凭证平均耗时：{sum(verify_cost_time_list) / count} ms")
