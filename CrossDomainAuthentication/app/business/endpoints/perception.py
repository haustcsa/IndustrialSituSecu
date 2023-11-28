# -*- encoding: utf-8 -*-
"""
@author: ztt
@time: 2023/5/27 18:44
@file: perception.py
@desc: 
"""
import json
from typing import Optional

from fastapi import APIRouter, Request, Depends
from httpx import AsyncClient

from app.business.core.deps import http_client
from app.business.schemas.perception_schemas import (
    BusinessPerceptionAuthRequest,
    BusinessPerceptionCrossDomainCrossIdentityAuthRequest,
    PerceptionRsaRequest,
)
from app.core import logger, HandleDhRsa
from app.core.response import success, fail, ResponseCode
from settings import settings

router = APIRouter()


@router.post("", summary="工业感知")
async def perception(
    request: Request,
    device: PerceptionRsaRequest,
    cross_domain: bool = False,
    auth_origin: Optional[str] = settings.BUSINESS_BASE_URL,
    client: AsyncClient = Depends(http_client),
    dh_rsa: HandleDhRsa = Depends(HandleDhRsa),
):
    rsa_item = (
            settings.BUSINESS_DH_RSA_ALL[device.identity]
            or settings.BUSINESS_DH_RSA_ALL[f"['{device.identity}']"]
    )
    source_data = json.loads(await dh_rsa.decrypt(rsa_item.shared_key, device.data))
    source_data.update(
        {
            "identity": device.identity,
            "request_domain": auth_origin,
            "request_ip": request.client.host,
            "cross_domain": cross_domain,
            "domain": source_data.get("domain") or settings.DOMAIN,
        }
    )
    result = await client.post(
        url=settings.AUTHENTICATION_BASE_URL + "/perception", json=source_data
    )
    result_json = result.json()
    logger.debug(result_json)
    if result_json.get("code") != ResponseCode.success:
        return fail(msg=result_json["message"])

    source_data = json.dumps(result_json["data"])
    encrypt_data = await dh_rsa.encrypt(rsa_item.shared_key, source_data)
    return success(encrypt_data)


@router.post("/auth", summary="跨域身份信息认证")
async def cross_domain_identity_authentication(
    request: Request,
    auth: BusinessPerceptionAuthRequest,
    auth_origin: Optional[str] = None,
    client: AsyncClient = Depends(http_client),
):
    logger.info(f"跨域身份信息认证 {auth_origin}")
    logger.info(auth.dict())
    result = await client.post(
        url=settings.AUTHENTICATION_BASE_URL + "/perception/cross",
        json={
            **auth.dict(),
            # 'random_factor': auth.identity,
            # 'request_domain': settings.DOMAIN,
            "request_ip": request.client.host,
        },
    )

    if result.json().get("code") == ResponseCode.success:
        logger.info("已获取签名，准备跨域域身份信息认证")
        logger.info(auth.auth_origin + "/perception/cross_domain_auth")
        data = result.json()["data"]
        response = await client.post(
            url=auth.auth_origin + "/perception/cross_domain_auth",
            json={
                **data,
                "identity": auth.identity,
                "timestamp": auth.timestamp,
                "pcr_dict": auth.pcr_dict,
            },
        )

        if response.json().get("code") == ResponseCode.success:
            logger.info("跨域域身份信息认证成功")
            return success(response.json()["data"])
        else:
            logger.error("跨域域身份信息认证失败")
            logger.error(response.text)
            return fail(data="跨域域身份信息认证失败")

    return fail(data="验证失败 cross_domain_identity_authentication")


@router.post("/cross_domain_auth", summary="跨域域身份信息认证")
async def cross_domain_cross_identity_authentication(
    device: BusinessPerceptionCrossDomainCrossIdentityAuthRequest,
    client: AsyncClient = Depends(http_client),
):
    logger.info("跨域域身份信息认证")
    logger.info(device.dict())
    result = await client.post(
        url=settings.AUTHENTICATION_BASE_URL + "/perception/cross_domain_auth",
        json={
            **device.dict(),
            "random_factor": device.identity,
        },
    )

    if result.json().get("code") == ResponseCode.success:
        return success()

    logger.error(result.text)
    return fail(data="验证失败 cross_domain_cross_identity_authentication")
