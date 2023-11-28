# -*- encoding: utf-8 -*-
"""
@author: ztt
@time: 2023/6/4 16:05
@file: schemas.py
@desc: 
"""
from typing import Optional

from cryptography.hazmat.primitives.asymmetric.dh import DHPublicKey, DHPrivateKey
from pydantic import BaseModel
from rsa import PublicKey


class CredentialsModel(BaseModel):
    """临时身份凭证的 model"""

    credentials: str
    identity: str
    random_factor: str
    timestamp: str
    public_key: Optional[PublicKey] = None

    def provided_credentials(self):
        return self.timestamp + self.identity + self.random_factor

    class Config:
        arbitrary_types_allowed = True


class DhRsaModel(BaseModel):
    """DH-RSA 相关数据的 model"""

    shared_key: Optional[bytes] = None
    private_key: Optional[DHPrivateKey] = None
    public_key: Optional[DHPublicKey] = None
    server_public_key: Optional[DHPublicKey] = None

    class Config:
        arbitrary_types_allowed = True


class CostTimeModel(BaseModel):
    """耗时"""

    client: float = 0
    auth_server: float = 0
    business_server: float = 0
    total: float = 0
