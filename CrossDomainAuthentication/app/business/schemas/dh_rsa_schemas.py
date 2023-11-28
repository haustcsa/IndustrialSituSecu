# -*- encoding: utf-8 -*-
"""
@author: ztt
@time: 2023/6/4 16:52
@file: dh_rsa_schemas.py
@desc: 
"""
from pydantic import BaseModel


class GenerateKeyRequest(BaseModel):
    identity: str
    generator: int = 2
    prime: int


class GenerateKeyResponse(BaseModel):
    public_key: str


class KeyExchangeRequest(BaseModel):
    identity: str
    public_key: str
