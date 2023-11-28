# -*- encoding: utf-8 -*-
"""
@author: ztt
@time: 2023/5/21 20:52
@file: test_handle_rsa.py
@desc: 
"""
import random
import time

import pytest

from app.core import (
    generate_rsa_key,
    generate_signature,
    verify_signature,
    logger,
    HandleTpm,
    calculate_pseudo_identity,
)
from app.core.handle_rsa import generate_credentials, verify_credentials
from app.core.random_range import generate_random_number
from settings import settings


@pytest.mark.asyncio
class TestHandleRsa:
    async def test_verify_signature(self):
        pcr = HandleTpm.get_pcr(settings.AUTHENTICATION_CERT_PCR_PATH)
        data = pcr

        # 生成秘钥对
        public_key, private_key = await generate_rsa_key(pcr)

        # 生成时间戳
        timestamp = str(int(time.time()))

        # 生成随机因子
        random_factor = str(random.randint(100000, 999999))

        # 将时间戳和随机因子与待签名的数据拼接在一起
        data_with_timestamp = timestamp + random_factor + data

        sign_start_time = time.time()
        # 生成签名
        signature = generate_signature(data_with_timestamp, private_key)
        logger.info(f"Signature: {signature}")
        logger.info(f"Timestamp: {timestamp}")
        logger.info(f"Random Factor: {random_factor}")

        sign_end_time = time.time()
        logger.info(f"签名耗时：{(sign_end_time - sign_start_time) * 1000} 毫秒")
        # 模拟时间过期
        # time.sleep(61)

        valid_start_time = time.time()
        # 验证签名
        is_valid = verify_signature(
            data, signature, public_key, timestamp, random_factor
        )
        logger.info(f"Signature is valid: {is_valid}")

        valid_end_time = time.time()

        logger.info(f"验签耗时：{(valid_end_time - valid_start_time) * 1000} 毫秒")
        assert is_valid

    async def test_certification_simulation(self):
        """认证模拟"""
        # pcr = HandleTpm.get_pcr(settings.AUTHENTICATION_CERT_PCR_PATH)
        # # 生成秘钥对
        # public_key, private_key = await generate_rsa_key(pcr)

        random_factor = generate_random_number()
        timestamp = str(int(time.time()))
        identity = calculate_pseudo_identity(b"client")

        start_time = time.time()

        # 1. 时间戳 + 客户端伪身份 id + 随机因子
        data_with_timestamp = timestamp + identity + str(random_factor)
        end_time = time.time()
        # 2. 生成身份凭证
        credentials = await generate_credentials(data_with_timestamp)
        logger.info(f"身份凭证生成耗时：{(end_time - start_time) * 1000} 毫秒")
        logger.info(f"data_with_timestamp: {data_with_timestamp}")
        logger.info(f"credentials: {credentials}")

        valid_start_time = time.time()
        is_verify = await verify_credentials(credentials, data_with_timestamp)
        valid_end_time = time.time()
        logger.info(f"身份凭证验签耗时：{(valid_end_time - valid_start_time) * 1000} 毫秒")
        logger.info(f"is_valid: {is_verify}")
