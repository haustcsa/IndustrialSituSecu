# -*- encoding: utf-8 -*-
"""
@author: ztt
@time: 2023/5/27 18:37
@file: auth.py
@desc: 认证
"""
import random
import socket
import time

from fastapi import APIRouter, Depends

from app.authentication.utils import BlockchainApi
from app.core import logger, generate_signature
from app.core.response import success, fail
from settings import settings
from ..core.deps import security_verify_with_cost
from ..schemas.auth_schemas import AuthenticationRequest, AuthenticationCrossRequest

router = APIRouter()


@router.post("/same_cross", summary="同域认证")
async def authentication(
    device: AuthenticationRequest,
    verify_result: dict[str, bool | float] = Depends(security_verify_with_cost),
):
    is_verify = verify_result["verify_result"]
    cost = verify_result["cost"]
    if not is_verify:
        return fail(msg="身份凭证校验失败")

    if settings.BLOCKCHAIN:
        block_result = await BlockchainApi().get_auth_data_by_key(
            key=device.identity,
            request_domain=device.request_domain,
            accept_domain=settings.DOMAIN,
        )
        accept_ip = socket.gethostbyname(socket.gethostname())
        if not (
            block_result
            and block_result.receivedomain == settings.DOMAIN
            and block_result.receiveip == accept_ip
            and block_result.requestip == device.request_ip
            and block_result.requestdomain == device.request_domain
        ):
            return fail(msg="添加到区块链失败")

    return success({"auth_cost": cost})


@router.post("/cross", summary="跨域身份信息认证")
async def cross_domain_identity_authentication(
    auth: AuthenticationCrossRequest,
    verify_result: dict[str, bool | float] = Depends(security_verify_with_cost),
):
    is_verify = verify_result["verify_result"]
    cost = verify_result["cost"]
    if not is_verify:
        return fail(msg="身份凭证校验失败")

    logger.info("跨域身份信息认证")
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

    start_time = time.time()
    timestamp = str(int(time.time()))
    random_factor = str(random.randint(100000, 999999))
    data_with_timestamp = timestamp + random_factor + auth.encrypt_data
    signature = generate_signature(
        data_with_timestamp, settings.AUTHENTICATION_RSA_PRIVATE_KEY
    )
    response = {
        "signature": signature,
        "timestamp": timestamp,
        "random_factor": random_factor,
    }
    cost_time = (time.time() - start_time) * 1000
    response["auth_cost"] = cost + cost_time
    return success(response)
