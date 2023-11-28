# -*- encoding: utf-8 -*-
"""
@author: ztt
@time: 2023/6/4 16:42
@file: settings_schemas.py
@desc: 设置相关的 model
"""
from typing import Optional

from cryptography.hazmat.primitives.asymmetric.dh import DHPrivateKey, DHPublicKey
from pydantic import BaseModel


class SettingDhRsaModel(BaseModel):
    """DH-RSA 相关数据的 model"""

    shared_key: Optional[bytes] = None
    private_key: Optional[DHPrivateKey]
    public_key: Optional[DHPublicKey]
    client_public_key: Optional[DHPublicKey] = None

    class Config:
        arbitrary_types_allowed = True
