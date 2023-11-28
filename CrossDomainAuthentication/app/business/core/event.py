# -*- encoding: utf-8 -*-
"""
@author: ztt
@time: 2023/2/15 16:39:44
@file: event.py
@desc: 
"""
from typing import Any, Callable

from fastapi import FastAPI


async def startup() -> Any:
    """
    FastApi 启动完成事件
    :return: start_app
    """
    # if not settings.BUSINESS_CERT_SERVER_PRIVATE_FILE.exists():
    #     logger.info('业务服务未发现秘钥，正在重新生成秘钥')
    #     (public_key, private_key) = generate_rsa_key()
    #     with open(settings.BUSINESS_CERT_SERVER_PRIVATE_FILE, 'bw') as f:
    #         f.write(private_key.save_pkcs1())
    #
    #     with open(settings.BUSINESS_CERT_SERVER_PUBLIC_FILE, 'bw') as f:
    #         f.write(public_key.save_pkcs1())
    #     logger.info('业务服务的秘钥生成成功')
    # else:
    #     logger.info('开始读取业务服务秘钥')
    #     business_cert.get_public_key()
    #     business_cert.get_private_key()
    #     logger.info('业务服务秘钥加载完成')

    # pcr = HandleTpm.get_pcr(settings.BUSINESS_CERT_PCR_PATH)
    # settings.BUSINESS_PCR = pcr
    #
    # public_key, private_key = await generate_rsa_key(pcr)
    # settings.BUSINESS_RSA_PUBLIC_KEY = public_key
    # settings.BUSINESS_RSA_PRIVATE_KEY = private_key
    # settings.BUSINESS_RSA_PUBLIC_KEY_STR = public_key.save_pkcs1()
    # settings.BUSINESS_RSA_PRIVATE_KEY_STR = private_key.save_pkcs1()
    # logger.info("业务服务 - 基于身份信息的秘钥对已生成")


def create_start_app_handler(_: FastAPI) -> Callable:
    async def start_app() -> None:
        await startup()

    return start_app
