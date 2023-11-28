# -*- encoding: utf-8 -*-
"""
@author: ztt
@time: 2023/5/12 12:56
@file: api.py
@desc: 
"""
import time

from fastapi import APIRouter

from app.authentication.utils import BlockchainApi
from app.core import tpm_sign, read_bytes_file, save_device_cert_file, logger, HandleTpm
from app.core.response import success, fail
from app.open.endpoints.openapi.schemas import (
    SignResponse,
    CheckSignRequest,
    CheckSignResponse,
    QuashRequest,
    RegisterReqeust,
)
from settings import settings

router = APIRouter()


@router.post("/register", summary="注册新设备")
async def register(device: RegisterReqeust):
    try:
        valid = await save_device_cert_file(
            abs_path=settings.OPEN_CERT_DIR
            / "temp-cert"
            / f'tpm2-{str(time.time()).replace(".", "")}',
            sign=device.sign,
            pcr=device.pcr,
            quote=device.quote,
            akpub=device.akpub,
            identity=device.identity,
            pcr_dict=HandleTpm.get_pcr_dict(),
        )
    except Exception as e:
        logger.error(f"注册新设备失败: {e}")
        valid = False

    if not valid:
        return fail(msg="签名校验失败")

    if settings.BLOCKCHAIN:
        add_result = await BlockchainApi().add_auth_data(
            key=device.identity,
            request_domain=device.request_domain,
            request_ip=device.request_ip,
            accept_domain=device.accept_domain,
            accept_ip=device.accept_ip,
            pcr=device.pcr,
            sml=device.quote,
            aik=device.akpub,
            signature=device.sign,
        )
        if not add_result:
            return fail(msg="添加区块链到凭据库失败")

    return success(CheckSignResponse(valid=valid))


@router.get("/auth", summary="签名并获取设备pcr等信息")
async def get_device(
    identity: str,
    request_domain: str,
    request_ip: str,
    accept_domain: str,
    accept_ip: str,
):
    if len(identity) < 8:
        return fail(msg="identity 不能小于8位")

    try:
        tpm_instance = await tpm_sign(
            cert_path=settings.OPEN_CERT_DIR, identity=identity
        )
        await BlockchainApi().update_auth_data(
            key=identity,
            request_domain=request_domain,
            request_ip=request_ip,
            accept_domain=accept_domain,
            accept_ip=accept_ip,
            pcr=read_bytes_file(tpm_instance.get_pcr_file),
            sml=read_bytes_file(tpm_instance.get_quote_file),
            aik=read_bytes_file(settings.OPEN_CERT_AK_PUBLIC_FILE),
            signature=read_bytes_file(tpm_instance.get_sig_file),
        )
        return success(
            SignResponse(
                sign=read_bytes_file(tpm_instance.get_sig_file),
                pcr=read_bytes_file(tpm_instance.get_pcr_file),
                quote=read_bytes_file(tpm_instance.get_quote_file),
                akpub=read_bytes_file(settings.OPEN_CERT_AK_PUBLIC_FILE),
            )
        )
    except Exception as e:
        logger.error(f"签名失败: {e}")
        return fail(msg="签名失败")


@router.post("/auth", summary="验证签名的有效性和设备的完整性")
async def auth(device: CheckSignRequest):
    if settings.BLOCKCHAIN:
        block_result = await BlockchainApi().get_auth_data_by_key(
            key=device.identity,
            request_domain=device.request_domain,
            accept_domain=settings.DOMAIN,
        )
        if not (
            block_result
            and block_result.receivedomain == device.accept_domain
            and block_result.receiveip == device.accept_ip
            and block_result.requestip == device.request_ip
            and block_result.requestdomain == device.request_domain
        ):
            return fail(msg="区块链校验失败")

    try:
        valid = await save_device_cert_file(
            abs_path=settings.OPEN_CERT_DIR
            / "temp-cert"
            / f'tpm2-{str(time.time()).replace(".", "")}',
            sign=device.sign,
            pcr=device.pcr,
            quote=device.quote,
            akpub=device.akpub,
            identity=device.identity,
            pcr_dict=HandleTpm.get_pcr_dict(),
        )
    except Exception as e:
        logger.error(f"签名验证失败: {e}")
        valid = False
    return success(CheckSignResponse(valid=valid))


@router.delete("/auth", summary="撤销已注册的设备")
async def quash(device: QuashRequest):
    if not settings.BLOCKCHAIN:
        return success("ok")

    blockchain_api = BlockchainApi()
    block_result = await blockchain_api.get_auth_data_by_key(
        key=device.identity,
        request_domain=device.request_domain,
        accept_domain=settings.DOMAIN,
    )
    if not (
        block_result
        and block_result.receivedomain == device.accept_domain
        and block_result.receiveip == device.accept_ip
        and block_result.requestip == device.request_ip
        and block_result.requestdomain == device.request_domain
    ):
        return fail(msg="区块链校验失败")

    blockchain_result = await blockchain_api.delete_auth_data(
        key=device.identity,
        request_domain=device.request_domain,
        accept_domain=settings.DOMAIN,
    )
    if not blockchain_result:
        fail(msg="区块链校验失败")
    return success("ok")
