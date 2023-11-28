# -*- encoding: utf-8 -*-
"""
@author: ztt
@time: 2023/4/5 18:42
@file: test_client.py
@desc: 测试终端注册流程
"""
import pytest

from app.client import Client
from app.core import logger
from settings import settings


@pytest.mark.asyncio
class TestClient:
    client = Client(init=True)
    # another_business_base_url = 'http://127.0.0.1:9001'
    another_business_base_url = "http://b-domain:9001"

    async def test_client_equipment_init(self):
        """测试终端设备初始化并签名"""
        self.client.tpm_instance.sign()
        assert settings.CLIENT_CERT_EK_PRIVATE_FILE.exists()
        assert settings.CLIENT_CERT_EK_PUBLIC_FILE.exists()
        assert settings.CLIENT_CERT_AK_PRIVATE_FILE.exists()
        assert settings.CLIENT_CERT_AK_PUBLIC_FILE.exists()
        assert settings.CLIENT_CERT_AK_NAME_FILE.exists()
        assert self.client.tpm_instance.get_sig_file.exists()

    async def test_same_origin_perception(self):
        """测试同域工业情感感知"""
        assert await self.client.same_origin_perception()

    async def test_cross_domain_perception(self):
        """测试跨域工业情感感知"""
        assert await self.client.cross_domain_perception(self.another_business_base_url)

    async def test_client_register(self):
        """测试终端设备注册"""
        assert await self.client.register()

    async def test_dh_rsa_perception(self):
        """测试秘钥协商"""
        assert await self.client.dh_rsa_perception()

    async def test_client_same_origin_authentication(self, count: int = 500):
        """测试同域认证"""
        # 秘钥协商
        assert await self.client.dh_rsa_perception()
        # 这一步是同域工业情感感知，但该测试是：测试同域认证
        assert await self.client.same_origin_perception()

        client_time_list = []
        auth_time_list = []
        business_time_list = []
        total_time_list = []
        for i in range(count):
            cost_model = await self.client.same_origin_authentication()

            client_time_list.append(cost_model.client)
            auth_time_list.append(cost_model.auth_server)
            business_time_list.append(cost_model.business_server)
            total_time_list.append(cost_model.total)

        self.log_cost_time(
            count, client_time_list, auth_time_list, business_time_list, total_time_list
        )

    async def test_client_cross_domain_authentication(self, count: int = 20):
        """测试跨域认证"""
        # 秘钥协商
        assert await self.client.dh_rsa_perception(self.another_business_base_url)
        assert await self.client.cross_domain_perception(self.another_business_base_url)

        client_time_list = []
        auth_time_list = []
        business_time_list = []
        total_time_list = []
        for i in range(count):
            cost_model = await self.client.cross_domain_authentication(
                self.another_business_base_url
            )

            client_time_list.append(cost_model.client)
            auth_time_list.append(cost_model.auth_server)
            business_time_list.append(cost_model.business_server)
            total_time_list.append(cost_model.total)

        self.log_cost_time(
            count, client_time_list, auth_time_list, business_time_list, total_time_list
        )

    @staticmethod
    def log_cost_time(
        count: int,
        client_time_list: list[float],
        auth_time_list: list[float],
        business_time_list: list[float],
        total_time_list: list[float],
    ):
        logger.info(f"客户端耗时明细: {client_time_list}")
        logger.info(f"认证服务耗时明细: {auth_time_list}")
        logger.info(f"业务服务耗时明细: {business_time_list}")
        logger.info(f"总耗时明细: {total_time_list}")

        client_time_svg = sum(client_time_list) / count
        auth_time_svg = sum(auth_time_list) / count
        business_time_svg = sum(business_time_list) / count
        total_time_svg = sum(total_time_list) / count

        logger.info(f"实验次数：{count} 客户端平均耗时：{client_time_svg} ms")
        logger.info(f"实验次数：{count} 认证服务平均耗时：{auth_time_svg} ms")
        logger.info(f"实验次数：{count} 业务服务平均耗时：{business_time_svg} ms")
        logger.info(
            f"实验次数：{count} 总流程平均耗时：{client_time_svg + auth_time_svg + business_time_svg} ms"
        )
        logger.info(f"实验次数：{count} 总平均耗时：{total_time_svg} ms")

    async def test_client_quash(self):
        """测试撤销已注册的设备"""
        assert await self.client.same_origin_perception()
        assert await self.client.quash()
