# -*- encoding: utf-8 -*-
"""
@author: ztt
@time: 2023/5/27 18:43
@file: business.py
@desc: 
"""
import time
from typing import Optional

from fastapi import APIRouter, Request, Depends
from httpx import AsyncClient

from app.business.core.deps import http_client
from app.business.schemas.business_schemas import (
    SimulateBusinessRequest,
    SimulateBusinessResponse,
    CrossdomainAuthenticationRequest,
    SameCrossRequest,
)
from app.core import logger, HandleDhRsa
from app.core.response import ResponseCode, fail, success
from settings import settings

router = APIRouter()


@router.post("", summary="业务测试接口")
async def simulate_business(request: Request, auth_request: SimulateBusinessRequest):
    async with AsyncClient() as client:
        result = await client.post(
            url=settings.AUTHENTICATION_BASE_URL + "/auth",
            json=auth_request.dict().update(
                {
                    "request_domain": settings.DOMAIN,
                    "request_ip": request.client.host,
                }
            ),
        )

    if result.json().get("code") != ResponseCode.success:
        logger.error(result.text)
        return fail(msg="校验失败")

    logger.info(result.json()["data"])
    return success(SimulateBusinessResponse(**result.json()["data"]))


@router.post("/same_cross", summary="同域认证接口")
async def same_origin_authentication(
    data: SameCrossRequest,
    request: Request,
    cross_domain: bool = False,
    client: AsyncClient = Depends(http_client),
    dh_rsa: HandleDhRsa = Depends(HandleDhRsa),
):
    result = await client.post(
        url=settings.AUTHENTICATION_BASE_URL + "/auth/same_cross",
        json={
            **data.dict(),
            "request_domain": settings.DOMAIN,
            "request_ip": request.client.host,
            "cross_domain": cross_domain,
        },
    )

    result_json = result.json()
    start_time = time.time()
    if result_json["code"] != ResponseCode.success:
        # logger.error(result.text)
        return fail(msg="校验失败")

    rsa_item = settings.BUSINESS_DH_RSA_ALL.get(
        data.identity
    ) or settings.BUSINESS_DH_RSA_ALL.get(f"['{data.identity}']")
    source_data = await dh_rsa.decrypt(rsa_item.shared_key, data.data)
    # encrypt_data = await dh_rsa.encrypt(rsa_item.shared_key, source_data + ' ❤')
    encrypt_data = await dh_rsa.encrypt(
        rsa_item.shared_key, source_data + settings.MESSAGE
    )
    cost_time = time.time() - start_time
    business_cost = cost_time * 1000
    logger.info(f"业务服务耗时: {business_cost} ms")
    return success(
        {
            "encrypt_data": encrypt_data,
            "business_cost": business_cost,
            **result_json["data"],
        }
    )


@router.post("/cross", summary="跨域认证接口")
async def cross_domain_authentication(
    request: Request,
    device: CrossdomainAuthenticationRequest,
    cross_domain: bool = False,
    auth_origin: Optional[str] = None,
    client: AsyncClient = Depends(http_client),
    dh_rsa: HandleDhRsa = Depends(HandleDhRsa),
):
    if cross_domain:
        start_time = time.time()
        rsa_item = settings.BUSINESS_DH_RSA_ALL[device.identity]
        logger.debug(f"rsa_item: {rsa_item}")
        source_data = await dh_rsa.decrypt(rsa_item.shared_key, device.data)
        logger.debug(f"source_data: {source_data}")
        # encrypt_data = await dh_rsa.encrypt(rsa_item.shared_key, source_data + ' ❤')
        encrypt_data = await dh_rsa.encrypt(
            rsa_item.shared_key, source_data + settings.MESSAGE
        )
        cost_time = time.time() - start_time
        business_cost = cost_time * 1000
        logger.info(f"业务服务耗时: {business_cost} ms")

        # 跨域认证
        logger.info(f"跨域认证: {auth_origin}")
        result = await client.post(
            url=settings.AUTHENTICATION_BASE_URL + "/auth/cross",
            json={
                "encrypt_data": encrypt_data,
                # "auth_origin": auth_origin,
                "auth_origin": "B",
                "request_ip": request.client.host,
                **device.dict(),
            },
        )

        result_json = result.json()
        if result_json.get("code") == ResponseCode.success:
            # logger.debug(f'encrypt_data: {encrypt_data}')
            return success(
                {
                    "encrypt_data": encrypt_data,
                    "business_cost": business_cost,
                    **result_json["data"],
                }
            )

    return fail(data=f"校验失败 {cross_domain}")
