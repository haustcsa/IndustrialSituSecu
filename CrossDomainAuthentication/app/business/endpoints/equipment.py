# -*- encoding: utf-8 -*-
"""
@author: ztt
@time: 2023/5/27 18:43
@file: equipment.py
@desc: 
"""
from fastapi import APIRouter, Request, Depends
from httpx import AsyncClient

from app.core import logger
from app.core.response import fail, ResponseCode, success
from settings import settings
from ..core.deps import http_client
from ..schemas.equipment_schemas import BusinessRegisterRequest, BusinessQuashRequest

router = APIRouter()


@router.post("", summary="注册")
async def register(
    request: Request,
    device: BusinessRegisterRequest,
    client: AsyncClient = Depends(http_client),
):
    logger.info(f"设备注册 {device.dict()}")
    result = await client.post(
        url=settings.AUTHENTICATION_BASE_URL + "/equipment",
        json={**device.dict(), "request_ip": request.client.host},
    )

    result_json = result.json()
    if result_json.get("code") == ResponseCode.success:
        return success(result_json.get("data"))

    logger.error(result.text)
    return fail()


@router.delete("", summary="撤销已注册的设备")
async def quash(
    device: BusinessQuashRequest, client: AsyncClient = Depends(http_client)
):
    logger.info("撤销已注册的设备")
    result = await client.request(
        method="DELETE",
        url=settings.AUTHENTICATION_BASE_URL + "/equipment",
        json=device.dict(),
    )

    if result.json().get("code") == ResponseCode.success:
        logger.info("撤销成功")
        return success("撤销成功")

    return fail(msg="撤销失败")
