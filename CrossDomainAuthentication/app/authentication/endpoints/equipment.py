# -*- encoding: utf-8 -*-
"""
@author: ztt
@time: 2023/5/27 18:39
@file: equipment.py
@desc: 
"""
import socket

from fastapi import APIRouter, Depends, Request

from app.authentication.core.deps import security_verify
from app.authentication.schemas.equipment_schemas import (
    AddPublicKeyRequest,
    QuashRequest,
)
from app.authentication.utils import BlockchainApi
from app.core import HandleTpm, logger, write_bytes_file
from app.core.handle_rsa import generate_credentials, generate_signature
from app.core.response import success, fail
from settings import settings

router = APIRouter()


@router.post("", summary="设备注册")
async def add_public_key(device: AddPublicKeyRequest):
    # 校验参数，记录设备信息
    abs_path = settings.AUTHENTICATION_CERT_PATH / "temp-cert" / device.identity
    abs_path.mkdir(parents=True, exist_ok=True)

    ak_public_path = abs_path / settings.CERT_AK_PUBLIC_FILENAME
    write_bytes_file(abs_path / "sig.out", device.sign)
    write_bytes_file(abs_path / "pcrs.out", device.pcr)
    write_bytes_file(abs_path / "quote.out", device.quote)
    write_bytes_file(ak_public_path, device.akpub)

    logger.debug("设备注册 开始校验 sign")
    checkout_result = HandleTpm(
        project_cert_path=abs_path,
        identity=device.identity,
        ak_public_path=ak_public_path,
        dir_name="temp-cert",
    ).checkquote()
    logger.debug(f"sign 校验完成: {checkout_result}")

    if not checkout_result:
        return fail(msg="签名校验失败")

    if settings.BLOCKCHAIN:
        add_result = await BlockchainApi().add_auth_data(
            key=device.identity,
            request_domain=device.request_domain,
            request_ip=device.request_ip,
            accept_domain=settings.DOMAIN,
            accept_ip=socket.gethostbyname(socket.gethostname()),
            pcr=device.pcr,
            sml=device.quote,
            aik=device.akpub,
            signature=device.sign,
        )
        if not add_result:
            return fail(msg="添加区块链到凭据库失败")

    credentials = await generate_credentials(
        device.timestamp + device.identity + device.random_factor
    )
    settings.CREDENTIALS_ALL.append(credentials)
    data_with_timestamp = device.timestamp + device.random_factor + device.identity
    signature = generate_signature(
        data_with_timestamp, settings.AUTHENTICATION_RSA_PRIVATE_KEY
    )
    return success({"credentials": credentials, "signature": signature})


@router.delete("", summary="撤销已注册的设备")
async def quash(
    request: Request, device: QuashRequest, is_verify: bool = Depends(security_verify)
):
    if not is_verify:
        return fail(msg="身份凭证校验失败")

    if settings.BLOCKCHAIN:
        blockchain_result = await BlockchainApi().delete_auth_data(
            key=device.identity,
            request_domain=device.domain,
            accept_domain=settings.DOMAIN,
        )
        if not blockchain_result:
            fail(msg="区块链校验失败")

    credentials = request.headers.get("credentials")
    settings.CREDENTIALS_ALL.remove(credentials)
    logger.info("撤销已注册的设备成功")
    return success()
