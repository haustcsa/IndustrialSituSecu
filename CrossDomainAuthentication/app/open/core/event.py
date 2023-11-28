# -*- encoding: utf-8 -*-
"""
@author: ztt
@time: 2023/4/22 16:10
@file: event.py
@desc: 
"""
from typing import Any, Callable

from fastapi import FastAPI

from app.core import logger, HandleTpm
from settings import settings


async def startup() -> Any:
    """
    FastApi 启动完成事件
    :return: start_app
    """
    if not settings.OPEN_CERT_EK_PRIVATE_FILE.exists():
        tpm_instance = HandleTpm(
            project_cert_path=settings.OPEN_CERT_DIR,
            identity="",
            ek_private_path=settings.OPEN_CERT_EK_PRIVATE_FILE,
            ek_public_path=settings.OPEN_CERT_EK_PUBLIC_FILE,
            ak_private_path=settings.OPEN_CERT_AK_PRIVATE_FILE,
            ak_public_path=settings.OPEN_CERT_AK_PUBLIC_FILE,
            ak_name_path=settings.OPEN_CERT_AK_NAME_FILE,
            mkdir=False,
        )
        tpm_instance.equipment_init()
    logger.info("EK 和 AIK 证书已生成")


def create_start_app_handler(_: FastAPI) -> Callable:
    async def start_app() -> None:
        await startup()

    return start_app
