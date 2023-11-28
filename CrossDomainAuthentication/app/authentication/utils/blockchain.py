# -*- encoding: utf-8 -*-
"""
@author: ztt
@time: 2023/3/16 21:38:27
@file: blockchain.py
@desc: 区块链接口

（接口目前全部采用get方式请求）
1.查询所有认证信息
   http://172.16.202.229:9999/queryAllRz

 2.根据请求的Id获取认证数据
   http://172.16.202.229:9999/queryRzById/6

 3.添加认证信息 7,M,172.168.0.1,Q,172.168.0.1,2023-1-19-22:10,30min分别对应的属性为请求Id, 请求域，发起请求的域内Ip地址，接受域，接受请求的域内Ip地址，发出请求的时间，请求的超时时间（参数用,连接）
    7,      M,     172.168.0.1,        Q,   172.168.0.1,    2023-1-19-22:10,    30min
    请求Id, 请求域，发起请求的域内Ip地址，接受域，接受请求的域内Ip地址，发出请求的时间，请求的超时时间（参数用,连接）
   http://172.16.202.229:9999/addRz/7,M,172.168.0.1,Q,172.168.0.1,2023-1-19-22:10,30min

 4.修改认证信息 7,M,172.168.0.1,Q,172.168.0.1,2023-1-19-22:10,60min分别对应的属性为请求Id, 请求域，发起请求的域内Ip地址，接受域，接受请求的域内Ip地址，发出请求的时间，请求的超时时间（参数用,连接）
   http://172.16.202.229:9999/changeRz/7,M,172.168.0.1,Q,172.168.0.1,2023-1-19-22:10,60min

 5.根据Id删除认证数据
   http://172.16.202.229:9999/deleteRzById/2

 6.查询显示对Id为7的数据的操作历史 （返回每次修改的交易哈希，修改的值，是否对数据执行删除，时间）
   http://172.16.202.229:9999/queryRzHistoryById/7
"""
import datetime

from httpx import AsyncClient, Response
from pydantic import BaseModel

from app.core import logger
from typing import Optional


class BlockchainAuthData(BaseModel):
    overtime: str
    receivedomain: str
    receiveip: str
    requestdomain: str
    requestid: str
    requestip: str
    requesttime: str
    pcr: str
    sml: str
    aik: str
    signature: str


class BlockchainApi:
    client: AsyncClient = AsyncClient()
    base_url: str = "http://172.16.202.229:9999"

    async def close(self):
        await self.client.aclose()

    async def all_auth_data(self) -> dict:
        """
        获取所有认证信息
        :return: 示例数据：
            [
                {
                    "Key" : "1" ,
                    "Record" : {
                        "overtime" : "100min" ,
                        "pcr": "PCR1",
                        "receivedomain" : "Q" ,
                        "receiveip" : "172.168.0.1" ,
                        "requestdomain" : "M" ,
                        "requestid" : "1" ,
                        "requestip" : "172.168.0.1" ,
                        "requesttime" : "2023-1-19-22:10",
                        "sml": "SML1",
                        "aik": "aik",
                        "signature": "signature",
                    }
                },
                ...
            ]
        """
        response = await self.client.get(url=self.base_url + "/queryAllRz/")
        response_json = response.json()
        logger.debug(response_json)
        return response_json

    async def get_auth_data_by_key(
        self,
        key: str,
        request_domain: str,
        accept_domain: str,
    ) -> Optional[BlockchainAuthData]:
        """
        根据 key 获取认证信息
        示例数据：
            {
                "overtime": "20min",
                "pcr": "PCR6",
                "receivedomain": "C",
                "receiveip": "192.168.101.13",
                "requestdomain": "D",
                "requestid": "6",
                "requestip": "192.168.101.10",
                "requesttime": "2023-1-14-22：33",
                "sml": "SML6",
                "aik": "aik",
                "signature": "signature",
            }

        :param key:
        :param request_domain:
        :param accept_domain:
        :return:
        """
        response = await self.client.get(
            url=self.base_url
            + "/queryRzById/"
            + self.get_current_key(key, request_domain, accept_domain)
        )
        if response.text.startswith("Could not"):
            logger.error(response.text)
            return None

        response_json = response.json()
        logger.debug(response_json)
        return BlockchainAuthData(**response_json)

    async def add_auth_data(
        self,
        key: str,
        request_domain: str,
        request_ip: str,
        accept_domain: str,
        accept_ip: str,
        pcr: str,
        sml: str,
        aik: str,
        signature: str,
        request_time: datetime.datetime = datetime.datetime.now(),
        timeout: str = "30min",
    ) -> bool:
        """
        添加认证信息
        :param key: 请求Id
        :param request_domain: 请求域
        :param request_ip: 发起请求的域内Ip地址
        :param accept_domain: 接受域
        :param accept_ip: 接受请求的域内Ip地址
        :param request_time: 发出请求的时间
        :param pcr: pcr
        :param sml: sml
        :param aik: aik
        :param signature: signature
        :param timeout: 请求的超时时间
        :return: 添加成功返回 True 否则返回 False
        """
        current_key = self.get_current_key(key, request_domain, accept_domain)
        response = await self.client.post(
            url=self.base_url + "/addRz/",
            json={
                "RequestId": current_key,
                "RequestDomain": request_domain,
                "RequestIp": request_ip,
                "ReceiveDomain": accept_domain,
                "ReceiveIp": accept_ip,
                "RequestTime": request_time.strftime("%Y-%m-%d-%H:%M"),
                "OverTime": timeout,
                "PCR": pcr,
                "SML": sml,
                "AIK": aik,
                "SIGNATURE": signature,
            },
        )
        return self._handle_response(response)

    @staticmethod
    def _handle_response(response: Response):
        """
        处理增删改的响应结果
        :param response: 响应
        :return: 成功返回 True 否则返回 False
        """
        response_text = response.text
        if response_text.startswith("Error: "):
            logger.error(response_text)
            return False

        logger.debug(response_text)
        return True

    async def update_auth_data(
        self,
        key: str,
        request_domain: str,
        request_ip: str,
        accept_domain: str,
        accept_ip: str,
        pcr: str,
        sml: str,
        aik: str,
        signature: str,
        request_time: datetime.datetime = datetime.datetime.now(),
        timeout: str = "30min",
    ) -> bool:
        """
        更新认证信息
        :param key: 请求Id
        :param request_domain: 请求域
        :param request_ip: 发起请求的域内Ip地址
        :param accept_domain: 接受域
        :param accept_ip: 接受请求的域内Ip地址
        :param request_time: 发出请求的时间
        :param pcr: pcr
        :param sml: sml
        :param aik: aik
        :param signature: signature
        :param timeout: 请求的超时时间
        :return: 更新成功返回 True  否则返回 False
        """
        current_key = self.get_current_key(key, request_domain, accept_domain)
        response = await self.client.post(
            url=self.base_url + "/changeRz",
            json={
                "RequestId": current_key,
                "RequestDomain": request_domain,
                "RequestIp": request_ip,
                "ReceiveDomain": accept_domain,
                "ReceiveIp": accept_ip,
                "RequestTime": request_time.strftime("%Y-%m-%d-%H:%M"),
                "OverTime": timeout,
                "PCR": pcr,
                "SML": sml,
                "AIK": aik,
                "SIGNATURE": signature,
            },
        )
        return self._handle_response(response)

    async def delete_auth_data(
        self,
        key: str,
        request_domain: str,
        accept_domain: str,
    ) -> bool:
        """
        删除认证数据
        注意：删除后，区块链会修改 id/key/requestId (名称有多个，但实际上是一个东西)
             示例：原 id: 948fe603f6-A-A-2
                  删除后: 948fe603f6-http://127.0.0.1:8001-A-2
        :param key: 请求Id
        :param request_domain:
        :param accept_domain:
        :return: 删除成功返回 True  否则返回 False
        """
        response = await self.client.get(
            url=self.base_url
            + "/deleteRzById/"
            + self.get_current_key(key, request_domain, accept_domain)
        )
        return self._handle_response(response)

    async def get_history_data(
        self,
        key: str,
        request_domain: str,
        accept_domain: str,
    ):
        """
        查询指定 key 的数据的操作历史
        返回每次修改的交易哈希，修改的值，是否对数据执行删除，时间

        示例数据：
        [
            {
                "Key" : "1" ,
                    "Record" : {
                        "overtime" : "100min" ,
                        "receivedomain" : "Q" ,
                        "receiveip" : "172.168.0.1" ,
                        "requestdomain" : "M" ,
                        "requestid" : "1" ,
                        "requestip" : "172.168.0.1" ,
                        "requesttime" : "2023-1-19-22:10",
                        "aik": "aik",
                        "signature": "signature",
                    }
            },
            ...
        ]

        :param key:
        :param request_domain:
        :param accept_domain:
        :return:
        """
        response = await self.client.get(
            url=self.base_url
            + "/queryRzHistoryById/"
            + self.get_current_key(key, request_domain, accept_domain)
        )
        response_json = response.json()
        logger.debug(response_json)
        return response_json

    @staticmethod
    def get_current_key(key: str, request_domain: str, accept_domain: str) -> str:
        """
        key 生成规则
        Args:
            key: 客户端的身份id 即：identity
            request_domain: 请求域，比如 设备A 则为 A
            accept_domain: 接受域，如果为 A域 则为 A，B域 则为 B

        Returns: key-request_domain-accept_domain
        """
        # 注意，最后的 -2 是我为了测试时 id 不重复随便加的，正式环境不需要，如果你测试时也遇到了 id 重复的问题，可以随便加数字
        # return f'{key}-{request_domain}-{accept_domain}-2'
        return f"{key}-{request_domain}-{accept_domain}"
