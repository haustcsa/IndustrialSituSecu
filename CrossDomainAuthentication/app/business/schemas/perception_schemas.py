# -*- encoding: utf-8 -*-
"""
@author: ztt
@time: 2023/5/27 18:42
@file: perception_schemas.py
@desc: 
"""
from pydantic import BaseModel

from app.core import TpmCertBaseModel, TpmCertBaseModelWithDefault
from app.core.schemas.base import (
    TpmCertBaseModelWithPcrDictWithDefault,
    TpmCertBaseModelWithPcrDict,
)
from settings import settings


class PerceptionRequest(TpmCertBaseModelWithPcrDictWithDefault):
    identity: str
    random_factor: str
    timestamp: str
    domain: str = settings.DOMAIN


class PerceptionRsaRequest(TpmCertBaseModelWithPcrDictWithDefault):
    identity: str
    data: str


class DemoRequest(BaseModel):
    credentials: str
    provided_credentials: str


class BusinessPerceptionAuthRequest(TpmCertBaseModelWithPcrDictWithDefault):
    """跨域身份信息认证请求"""

    identity: str
    auth_origin: str
    domain: str
    timestamp: str


class BusinessPerceptionCrossDomainCrossIdentityAuthRequest(
    TpmCertBaseModelWithPcrDict
):
    """跨域域身份认证请求"""

    identity: str
    timestamp: str
