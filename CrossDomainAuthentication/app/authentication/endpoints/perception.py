# -*- encoding: utf-8 -*-
"""
@author: ztt
@time: 2023/5/27 18:40
@file: perception.py
@desc: 
"""
import socket

from fastapi import APIRouter

from app.authentication.schemas.perception_schemas import (
    AuthPerceptionRequest,
    AuthPerceptionCrossResponse,
    AuthPerceptionCrossRequest,
    AuthPerceptionCrossDomainCrossIdentityAuthResponse,
    AuthPerceptionResponse,
)
from app.authentication.utils import BlockchainApi
from app.core import logger, read_bytes_file, save_device_cert_file, tpm_sign, HandleTpm
from app.core.handle_rsa import generate_credentials, generate_signature
from app.core.response import success, fail
from settings import settings

router = APIRouter()


@router.post("", summary="工业感知")
async def perception(data: AuthPerceptionRequest):
    tpm_data = {}
    if settings.IS_TPM:
        checkout_result = await save_device_cert_file(
            cert_path=settings.AUTHENTICATION_CERT_PATH,
            sign=data.sign,
            pcr=data.pcr,
            quote=data.quote,
            akpub=data.akpub,
            identity=data.identity,
            random_n=data.random_factor,
            timestamp=data.timestamp,
            # pcr_dict=data.pcr_dict
            pcr_dict=HandleTpm.get_pcr_dict(),
        )
        if not checkout_result:
            return fail(msg="校验失败")

        tpm_instance_sign = await tpm_sign(
            cert_path=settings.AUTHENTICATION_CERT_PATH,
            identity=data.identity,
            timestamp=data.timestamp,
        )
        tpm_data.update(
            {
                "sign": read_bytes_file(tpm_instance_sign.get_sig_file),
                "pcr": read_bytes_file(tpm_instance_sign.get_pcr_file),
                "quote": read_bytes_file(tpm_instance_sign.get_quote_file),
                "akpub": read_bytes_file(settings.AUTHENTICATION_CERT_AK_PUBLIC_FILE),
                "pcr_dict": tpm_instance_sign.get_pcr_dict(),
            }
        )

    if settings.BLOCKCHAIN:
        block_result = await BlockchainApi().add_auth_data(
            key=data.identity,
            request_domain=data.request_domain,
            request_ip=data.request_ip,
            accept_domain=data.domain,
            accept_ip=socket.gethostbyname(socket.gethostname()),
            pcr=data.pcr,
            sml=data.quote,
            aik=data.akpub,
            signature=data.sign,
        )
        if not block_result:
            return fail(msg="添加到区块链失败")

    credentials = await generate_credentials(
        data.timestamp + data.identity + data.random_factor
    )
    settings.CREDENTIALS_ALL.append(credentials)
    data_with_timestamp = data.timestamp + data.random_factor + data.identity
    signature = generate_signature(
        data_with_timestamp, settings.AUTHENTICATION_RSA_PRIVATE_KEY
    )
    return success(
        AuthPerceptionResponse(
            **{
                **tpm_data,
                "credentials": credentials,
                "public_key": settings.AUTHENTICATION_RSA_PUBLIC_KEY_STR,
                "signature": signature,
            }
        )
    )


@router.post("/cross", summary="跨域身份信息认证")
async def cross_domain_identity_authentication(auth: AuthPerceptionCrossRequest):
    logger.info("cross 跨域身份信息认证")
    tpm_data = {}
    if settings.BLOCKCHAIN:
        block_result = await BlockchainApi().get_auth_data_by_key(
            key=auth.identity, request_domain=auth.domain, accept_domain=settings.DOMAIN
        )
        accept_ip = socket.gethostbyname(socket.gethostname())
        if not (
            block_result
            and block_result.receivedomain == settings.DOMAIN
            and block_result.receiveip == accept_ip
            and block_result.requestip == auth.request_ip
            and block_result.requestdomain == auth.domain
        ):
            return fail(msg="添加到区块链失败")

    if settings.IS_TPM:
        tpm_instance_sign = await tpm_sign(
            cert_path=settings.AUTHENTICATION_CERT_PATH,
            identity=auth.identity,
            timestamp=auth.timestamp,
        )
        tpm_data.update(
            {
                "sign": read_bytes_file(tpm_instance_sign.get_sig_file),
                "pcr": read_bytes_file(tpm_instance_sign.get_pcr_file),
                "quote": read_bytes_file(tpm_instance_sign.get_quote_file),
                "akpub": read_bytes_file(settings.AUTHENTICATION_CERT_AK_PUBLIC_FILE),
            }
        )

    return success(
        msg="跨域身份信息认证成功",
        data=AuthPerceptionCrossResponse(**{"random_n": auth.identity, **tpm_data}),
    )


@router.post("/cross_domain_auth", summary="跨域域身份信息认证")
async def cross_domain_cross_identity_perception(
    data: AuthPerceptionCrossDomainCrossIdentityAuthResponse,
):
    logger.info("cross_domain_auth 跨域域身份信息认证")
    checkout_result = await save_device_cert_file(
        cert_path=settings.AUTHENTICATION_CERT_PATH,
        sign=data.sign,
        pcr=data.pcr,
        quote=data.quote,
        akpub=data.akpub,
        identity=data.identity,
        random_n="",
        timestamp=data.timestamp,
        # pcr_dict=data.pcr_dict,
        pcr_dict=HandleTpm.get_pcr_dict(),
    )

    if checkout_result:
        return success("验证成功")

    return fail(data="验证失败 cross_domain_cross_identity_authentication111")
