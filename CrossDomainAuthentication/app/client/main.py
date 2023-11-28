# -*- encoding: utf-8 -*-
"""
@author: ztt
@time: 2023/2/10 0:25
@file: main.py
@desc: 
"""
import asyncio
import json
import time
from typing import Optional

import rsa
from cryptography.hazmat.primitives.asymmetric.dh import DHPublicKey
from httpx import AsyncClient

from app.client.schemas import CredentialsModel, DhRsaModel, CostTimeModel
from app.core import (
    logger,
    HandleTpm,
    read_bytes_file,
    write_bytes_file,
    HandleDhRsa,
    verify_signature,
)
from app.core.random_range import generate_random_number
from app.core.response import ResponseCode
from settings import settings


class Client:
    identity: str
    tpm_instance: HandleTpm
    http_client: AsyncClient = AsyncClient(timeout=30)
    credentials_all: dict[str, CredentialsModel] = {}

    # DH-RSA 算法
    dh_rsa: HandleDhRsa = HandleDhRsa()
    prime: int = dh_rsa.gen_generator_params()
    dh_rsa_key_all: dict[str, DhRsaModel] = {}

    def __init__(self, init: bool = False):
        # self.identity = calculate_pseudo_identity(b'client')[:10]
        self.identity = str(generate_random_number())[:10]
        if settings.IS_TPM and (init or not settings.CLIENT_CERT_PRIVATE_FILE.exists()):
            self.init_equipment()
        logger.debug("设备初始化完成")

    def __del__(self):
        asyncio.get_event_loop().run_until_complete(self.http_client.aclose())

    def init_equipment(self):
        """设备初始化"""
        self.tpm_instance = HandleTpm(
            project_cert_path=settings.CLIENT_CERT_PATH,
            identity=self.identity,
            ek_private_path=settings.CLIENT_CERT_EK_PRIVATE_FILE,
            ek_public_path=settings.CLIENT_CERT_EK_PUBLIC_FILE,
            ak_private_path=settings.CLIENT_CERT_AK_PRIVATE_FILE,
            ak_public_path=settings.CLIENT_CERT_AK_PUBLIC_FILE,
            ak_name_path=settings.CLIENT_CERT_AK_NAME_FILE,
            mkdir=False,
        )
        self.tpm_instance.equipment_init()
        logger.info("设备 - EK 和 AIK 证书已生成")

    async def register(self) -> bool:
        """设备注册"""
        random_factor = str(generate_random_number())[:10]
        timestamp = str(int(time.time()))
        if settings.IS_TPM:
            logger.debug("开始签名")
            self.tpm_instance.sign()
            logger.debug("签名完成")
        response = await self.http_client.post(
            url=settings.BUSINESS_BASE_URL + "/equipment",
            json={
                **self.base_verify_params,
                "random_factor": random_factor,
                "request_domain": settings.DOMAIN,
                "timestamp": timestamp,
            },
        )
        response_json = response.json()

        if response_json["code"] == ResponseCode.success:
            logger.debug("设备注册成功")
            logger.debug(response.text)
            response_data = response_json["data"]
            credentials = response_data["credentials"]
            logger.debug(f"临时身份凭证：{credentials}")

            public_key = rsa.PublicKey.load_pkcs1(response_data["public_key"])
            signature = response_data["signature"]
            is_valid = verify_signature(
                self.identity, signature, public_key, timestamp, random_factor
            )
            if not is_valid:
                logger.error("rsa 签名校验失败")
                return False

            logger.info("rsa 签名校验成功")
            self.credentials_all[settings.BUSINESS_BASE_URL] = CredentialsModel(
                credentials=credentials,
                identity=self.identity,
                random_factor=random_factor,
                timestamp=timestamp,
                public_key=public_key,
            )

            logger.info(self.credentials_all)
            return True
        else:
            logger.error(f"设备注册失败: {response.text}")
            return False

    async def same_origin_authentication(self) -> Optional[CostTimeModel]:
        """同域认证"""
        start_time = time.time()
        dh_rsa_item = self.dh_rsa_key_all[settings.BUSINESS_BASE_URL]
        source_data = "love hncsa"
        logger.info(f"原始数据: {source_data}")
        encrypt_data = await self.dh_rsa.encrypt(dh_rsa_item.shared_key, source_data)
        cost_time_1 = time.time() - start_time
        response = await self.http_client.post(
            url=settings.BUSINESS_BASE_URL + "/business/same_cross",
            headers=self.get_headers(settings.BUSINESS_BASE_URL),
            json={
                "identity": self.identity,
                "data": encrypt_data,
            },
        )
        response_json = response.json()

        start_time_2 = time.time()
        if response_json["code"] == ResponseCode.success:
            response_data: dict[str, str | float] = response_json["data"]
            # server_data = response_json["data"]
            server_data = response_data["encrypt_data"]

            # logger.info(f"同域认证成功，获取资源: {server_data}")
            server_source_data = await self.dh_rsa.decrypt(
                dh_rsa_item.shared_key, server_data
            )
            logger.info(f"解密：{server_source_data[:10]}")
            cost_time_2 = time.time() - start_time_2
            client_cost = (cost_time_1 + cost_time_2) * 1000
            total_cost = (start_time_2 - start_time) * 1000
            return CostTimeModel(
                client=client_cost,
                business_server=response_data["business_cost"],
                auth_server=response_data["auth_cost"],
                total=total_cost,
            )
        return None

    async def cross_domain_authentication(
        self, base_url: str
    ) -> Optional[CostTimeModel]:
        """跨域认证"""
        start_time = time.time()
        dh_rsa_item = self.dh_rsa_key_all[base_url]
        source_data = "like hncsa"
        logger.info(f"原始数据: {source_data}")
        headers = self.get_headers(base_url)
        headers.update({"origin": settings.BUSINESS_BASE_URL})
        encrypt_data = await self.dh_rsa.encrypt(dh_rsa_item.shared_key, source_data)
        cost_time_1 = time.time() - start_time
        data = {
            "identity": self.identity,
            "domain": "B",
            "cross_domain": True,
            "data": encrypt_data,
        }
        logger.info(f"data: {data}")
        logger.info(f"headers: {headers}")
        response = await self.http_client.post(
            url=base_url + "/business/cross?cross_domain=true",
            headers=headers,
            json=data,
        )

        response_json = response.json()
        start_time_2 = time.time()
        if response_json["code"] == ResponseCode.success:
            response_data: dict[str, str | float] = response_json["data"]
            server_data = response_data["encrypt_data"]

            credentials_item = self.credentials_all[base_url]
            is_valid = verify_signature(
                data=server_data,
                signature=response_data["signature"],
                public_key=credentials_item.public_key,
                timestamp=response_data["timestamp"],
                random_factor=response_data["random_factor"],
            )
            if not is_valid:
                logger.error("rsa 签名校验失败")
                return None

            # logger.info(f"跨域认证成功，获取资源: {server_data}")
            server_source_data = await self.dh_rsa.decrypt(
                dh_rsa_item.shared_key, server_data
            )
            logger.info(f"解密：{server_source_data[:10]}")
            cost_time_2 = time.time() - start_time_2
            client_cost = (cost_time_1 + cost_time_2) * 1000
            total_cost = (start_time_2 - start_time) * 1000
            logger.info(f"客户端耗时: {client_cost} ms")
            return CostTimeModel(
                client=client_cost,
                business_server=response_data["business_cost"],
                auth_server=response_data["auth_cost"],
                total=total_cost,
            )
        else:
            logger.error("跨域认证失败")
            return None

    async def quash(self) -> bool:
        """撤销已注册的设备"""
        response = await self.http_client.request(
            method="DELETE",
            url=settings.BUSINESS_BASE_URL + "/equipment",
            headers=self.get_headers(settings.BUSINESS_BASE_URL),
            json={"identity": self.identity, "domain": settings.DOMAIN},
        )
        response_json = response.json()
        logger.debug(response_json)
        if response_json["code"] == ResponseCode.success:
            logger.debug("服务端撤销成功")
            del self.credentials_all[settings.BUSINESS_BASE_URL]
            return True
        return False

    @property
    def base_verify_params(self):
        """
        获取基础的校验参数
        :return:
        """
        base_params = {"identity": self.identity}
        if settings.IS_TPM:
            base_params.update(
                {
                    "sign": read_bytes_file(self.tpm_instance.get_sig_file),
                    "pcr": read_bytes_file(self.tpm_instance.get_pcr_file),
                    "quote": read_bytes_file(self.tpm_instance.get_quote_file),
                    "akpub": read_bytes_file(settings.CLIENT_CERT_AK_PUBLIC_FILE),
                    "pcr_dict": self.tpm_instance.get_pcr_dict(),
                }
            )
        return base_params

    async def same_origin_perception(self) -> bool:
        """同域工业情感感知"""
        if settings.IS_TPM:
            logger.info("开始签名")
            self.tpm_instance.sign()
            logger.info("签名完成")

        timestamp = str(int(time.time()))
        random_factor = str(generate_random_number())[:10]
        data = {
            **self.base_verify_params,
            "random_factor": random_factor,
            "timestamp": timestamp,
        }
        logger.info(f"原始数据: {data}")
        dh_rsa_item = self.dh_rsa_key_all[settings.BUSINESS_BASE_URL]
        response = await self.http_client.post(
            url=settings.BUSINESS_BASE_URL + "/perception",
            json={
                "identity": data["identity"],
                "data": await self.dh_rsa.encrypt(
                    dh_rsa_item.shared_key, json.dumps(data)
                ),
            },
        )
        logger.debug(response.text)
        response_json = response.json()
        if response_json["code"] == ResponseCode.success:
            response_data_str = await self.dh_rsa.decrypt(
                dh_rsa_item.shared_key, response_json["data"]
            )
            response_data = json.loads(response_data_str)
            credentials = response_data["credentials"]
            logger.debug(f"临时身份凭证：{credentials}")

            public_key = rsa.PublicKey.load_pkcs1(response_data["public_key"])
            signature = response_data["signature"]
            is_valid = verify_signature(
                self.identity, signature, public_key, timestamp, random_factor
            )
            if not is_valid:
                logger.error("rsa 签名校验失败")
                return False

            logger.info("rsa 签名校验成功")
            self.credentials_all[settings.BUSINESS_BASE_URL] = CredentialsModel(
                credentials=credentials,
                identity=self.identity,
                random_factor=random_factor,
                timestamp=timestamp,
                public_key=public_key,
            )
            return True
        return False

    async def dh_rsa_perception(
        self, base_url: str = settings.BUSINESS_BASE_URL
    ) -> bool:
        # 生成客户端秘钥对
        private_key, public_key = await self.dh_rsa.generate_dh_keypair_with_params(
            self.dh_rsa.parameters
        )
        logger.debug("客户端秘钥对已生成")
        server_public_key = await self.dh_rsa_get_server_public_key(base_url)
        if server_public_key:
            logger.info("开始秘钥协商")
            shared_key = await self.dh_rsa.perform_dh_key_exchange(
                private_key, server_public_key
            )
            logger.debug(f"客户端秘钥协商完成：{shared_key}")
            server_gen_shared_key_result = await self.dh_rsa_server_gen_shared_key(
                base_url, public_key
            )
            if server_gen_shared_key_result:
                logger.info("服务端秘钥协商完成")
                self.dh_rsa_key_all[base_url] = DhRsaModel(
                    shared_key=shared_key,
                    private_key=private_key,
                    public_key=public_key,
                    server_public_key=server_public_key,
                )
                return True
        return False

    async def dh_rsa_get_server_public_key(
        self, base_url: str
    ) -> Optional[DHPublicKey]:
        """
        获取服务端公钥
        :param base_url
        :return:
        """
        response = await self.http_client.post(
            url=base_url + "/dh_rsa/generate_key",
            json={"identity": self.identity, "prime": self.prime},
        )
        logger.debug(f"已获取服务端公钥：{response.text}")
        response_json = response.json()
        if response_json["code"] == ResponseCode.success:
            server_public_key_pem = response_json["data"]["public_key"]
            return await self.dh_rsa.load_pem_public_key(server_public_key_pem)
        return None

    async def dh_rsa_server_gen_shared_key(
        self, base_url: str, public_key: DHPublicKey
    ):
        response = await self.http_client.post(
            url=base_url + "/dh_rsa/key_exchange",
            json={
                "identity": self.identity,
                "public_key": await self.dh_rsa.get_public_key_str(public_key),
            },
        )
        return response.json()["code"] == ResponseCode.success

    def get_headers(self, base_url: str):
        credentials_item = self.credentials_all[base_url]
        return {
            "credentials": credentials_item.credentials,
            "provided_credentials": credentials_item.provided_credentials(),
        }

    async def cross_domain_perception(self, base_url: str) -> bool:
        """跨域工业情感感知"""
        dh_rsa_item = self.dh_rsa_key_all[base_url]
        random_factor = str(generate_random_number())[:10]
        timestamp = str(int(time.time()))
        if settings.IS_TPM:
            logger.debug(f"已生成随机数: {random_factor}")
            self.tpm_instance.sign(random_factor)
            logger.debug("已完成签名")

        headers = {"origin": settings.BUSINESS_BASE_URL}
        data = {
            **self.base_verify_params,
            "random_factor": random_factor,
            "timestamp": timestamp,
            "domain": "B",
        }
        logger.info(f"原始数据: {data}")
        response = await self.http_client.post(
            url=base_url + "/perception",
            headers=headers,
            json={
                "identity": data["identity"],
                "data": await self.dh_rsa.encrypt(
                    dh_rsa_item.shared_key, json.dumps(data)
                ),
            },
        )
        logger.debug(response.text)
        response_json = response.json()
        if response_json["code"] == ResponseCode.success:
            response_data_str = await self.dh_rsa.decrypt(
                dh_rsa_item.shared_key, response_json["data"]
            )
            response_data = json.loads(response_data_str)

            credentials = response_data["credentials"]
            logger.debug(f"临时身份凭证：{credentials}")

            public_key = rsa.PublicKey.load_pkcs1(response_data["public_key"])
            signature = response_data["signature"]
            is_valid = verify_signature(
                self.identity, signature, public_key, timestamp, random_factor
            )
            if not is_valid:
                logger.error("rsa 签名校验失败")
                return False

            logger.info("rsa 签名校验成功")
            self.credentials_all[base_url] = CredentialsModel(
                credentials=credentials,
                identity=self.identity,
                random_factor=random_factor,
                timestamp=timestamp,
                public_key=public_key,
            )
            logger.info("开始认证B域身份")
            if settings.IS_TPM:
                server_data = response_json["data"]
                server_source_data = await self.dh_rsa.decrypt(
                    dh_rsa_item.shared_key, server_data
                )
                server_source_data = json.loads(server_source_data)

                checkout_result = await self.new_write_verify_params_to_file(
                    server_source_data
                )
                if checkout_result:
                    logger.debug(f"sign 校验成功: {checkout_result}")

                    await self.__cross_domain_cross_identity_perception(
                        sign_data=server_source_data,
                        auth_origin=base_url,
                        timestamp=timestamp,
                    )
                    return True
                else:
                    logger.error("跨域认证失败")
                    del self.credentials_all[base_url]
                    return True
            else:
                return True
        else:
            logger.error("B域请求失败")
            return False

    async def __cross_domain_cross_identity_perception(
        self, sign_data: dict[str, str], auth_origin: str, timestamp: str
    ):
        """验证 B 域身份环节"""
        logger.info("开始验证跨域域身份环节")
        response = await self.http_client.post(
            url=settings.BUSINESS_BASE_URL + "/perception/auth",
            json={
                "identity": self.identity,
                "timestamp": timestamp,
                "domain": "A",
                "auth_origin": auth_origin,
                "sign": sign_data["sign"],
                "pcr": sign_data["pcr"],
                "quote": sign_data["quote"],
                "akpub": sign_data["akpub"],
                "pcr_dict": sign_data["pcr_dict"],
            },
        )

        logger.debug(response.text)
        if response.json().get("code") == ResponseCode.success:
            logger.info("跨域域身份验证成功")
        else:
            logger.error("跨域域身份验证失败")

    async def new_write_verify_params_to_file(self, response_data: dict) -> bool:
        """
        写入校验参数到文件
        :param response_data:
        :return:
        """
        abs_path = settings.CLIENT_CERT_PATH / "temp-cert"
        abs_path.mkdir(parents=True, exist_ok=True)

        write_bytes_file(abs_path / "sig.out", response_data["sign"])
        write_bytes_file(abs_path / "pcrs.out", response_data["pcr"])
        write_bytes_file(abs_path / "quote.out", response_data["quote"])
        write_bytes_file(
            abs_path / settings.CERT_AK_PUBLIC_FILENAME, response_data["akpub"]
        )

        logger.debug("开始校验 sign")
        tpm_instance = HandleTpm(
            project_cert_path=abs_path,
            identity=self.identity,
            ak_public_path=abs_path / settings.CERT_AK_PUBLIC_FILENAME,
            dir_name="temp-cert",
        )
        return tpm_instance.checkquote()

    async def run(self):
        """完整流程"""
        another_business_base_url = "http://127.0.0.1:9001"
        logger.info("开始注册设备信息")
        await self.register()
        logger.info("设备注册完成")

        logger.info("开始同域认证")
        await self.same_origin_authentication()
        logger.info("同域认证完成")

        logger.info("开始跨域认证")
        await self.cross_domain_authentication(another_business_base_url)
        logger.info("跨域认证完成")

        logger.info("开始撤销已注册的设备")
        await self.quash()
        logger.info("已注册的设备撤销完成")
