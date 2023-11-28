# -*- encoding: utf-8 -*-
"""
@author: ztt
@time: 2023/5/27 18:36
@file: perception_schemas.py
@desc: 
"""
from app.core import TpmCertBaseModel, TpmCertBaseModelWithDefault
from app.core.schemas.base import TpmCertBaseModelWithPcrDict


class AuthPerceptionRequest(TpmCertBaseModelWithPcrDict):
    identity: str
    random_factor: str
    timestamp: str
    request_domain: str
    request_ip: str
    cross_domain: bool = False
    domain: str


class AuthPerceptionResponse(TpmCertBaseModelWithPcrDict):
    credentials: str
    public_key: str
    signature: str


class AuthPerceptionCrossRequest(TpmCertBaseModel):
    """跨域身份信息认证请求"""

    identity: str
    auth_origin: str

    domain: str
    request_ip: str
    timestamp: str


class AuthPerceptionCrossResponse(TpmCertBaseModelWithDefault):
    random_n: str


class AuthPerceptionCrossDomainCrossIdentityAuthResponse(TpmCertBaseModelWithPcrDict):
    """跨域域身份信息认证"""

    identity: str
    random_factor: str
    timestamp: str
