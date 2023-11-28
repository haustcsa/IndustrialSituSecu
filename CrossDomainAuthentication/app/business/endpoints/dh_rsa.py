# -*- encoding: utf-8 -*-
"""
@author: ztt
@time: 2023/6/4 16:47
@file: dh_rsa.py
@desc: 
"""
from fastapi import APIRouter, Depends

from app.business.schemas.dh_rsa_schemas import (
    GenerateKeyRequest,
    GenerateKeyResponse,
    KeyExchangeRequest,
)
from app.core import HandleDhRsa
from app.core.response import success, ResponseModel, fail
from app.core.schemas import SettingDhRsaModel
from settings import settings

router = APIRouter()


@router.post(
    "/generate_key",
    summary="生成秘钥对，并返回服务端公钥",
    response_model=ResponseModel[GenerateKeyResponse],
)
async def generate_key(
    data: GenerateKeyRequest, dh_rsa: HandleDhRsa = Depends(HandleDhRsa)
):
    private_key, public_key = await dh_rsa.generate_dh_keypair(
        data.prime, data.generator
    )
    settings.BUSINESS_DH_RSA_ALL[data.identity] = SettingDhRsaModel(
        private_key=private_key, public_key=public_key
    )
    return success(
        GenerateKeyResponse(public_key=await dh_rsa.get_public_key_str(public_key))
    )


@router.post("/key_exchange", summary="执行密钥协商", response_model=ResponseModel)
async def perform_key_exchange(
    data: KeyExchangeRequest, dh_rsa: HandleDhRsa = Depends(HandleDhRsa)
):
    client_public_key = await dh_rsa.load_pem_public_key(data.public_key)
    if data.identity not in settings.BUSINESS_DH_RSA_ALL:
        return fail()

    private_key = settings.BUSINESS_DH_RSA_ALL[data.identity].private_key
    shared_key = await dh_rsa.perform_dh_key_exchange(private_key, client_public_key)
    settings.BUSINESS_DH_RSA_ALL[data.identity].shared_key = shared_key
    settings.BUSINESS_DH_RSA_ALL[data.identity].client_public_key = client_public_key
    return success()
