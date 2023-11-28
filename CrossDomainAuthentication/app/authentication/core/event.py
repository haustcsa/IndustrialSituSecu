# -*- encoding: utf-8 -*-
"""
@author: ztt
@time: 2023/5/26 22:22
@file: event.py
@desc: 
"""
from typing import Any, Callable

from fastapi import FastAPI

from app.core import logger, HandleTpm, generate_rsa_key
from settings import settings


async def startup() -> Any:
    """
    FastApi 启动完成事件
    :return: start_app
    """
    if not settings.AUTHENTICATION_CERT_EK_PRIVATE_FILE.exists() and settings.IS_TPM:
        tpm_instance = HandleTpm(
            project_cert_path=settings.AUTHENTICATION_CERT_PATH,
            identity="",
            ek_private_path=settings.AUTHENTICATION_CERT_EK_PRIVATE_FILE,
            ek_public_path=settings.AUTHENTICATION_CERT_EK_PUBLIC_FILE,
            ak_private_path=settings.AUTHENTICATION_CERT_AK_PRIVATE_FILE,
            ak_public_path=settings.AUTHENTICATION_CERT_AK_PUBLIC_FILE,
            ak_name_path=settings.AUTHENTICATION_CERT_AK_NAME_FILE,
            mkdir=False,
        )
        tpm_instance.equipment_init()
        logger.info("认证服务 - EK 和 AIK 证书已生成")

    # pcr = HandleTpm.get_pcr(settings.AUTHENTICATION_CERT_PCR_PATH)
    # settings.AUTHENTICATION_PCR = pcr

    public_key, private_key = await generate_rsa_key()
    settings.AUTHENTICATION_RSA_PUBLIC_KEY = public_key
    settings.AUTHENTICATION_RSA_PUBLIC_KEY_STR = public_key.save_pkcs1()
    settings.AUTHENTICATION_RSA_PRIVATE_KEY = private_key
    settings.AUTHENTICATION_RSA_PRIVATE_KEY_STR = private_key.save_pkcs1()
    logger.info("认证服务 - 基于身份信息的秘钥对已生成")


def create_start_app_handler(_: FastAPI) -> Callable:
    async def start_app() -> None:
        await startup()

    return start_app
