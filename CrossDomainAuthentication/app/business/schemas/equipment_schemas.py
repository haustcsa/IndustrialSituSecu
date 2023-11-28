# -*- encoding: utf-8 -*-
"""
@author: ztt
@time: 2023/5/27 18:42
@file: equipment_schemas.py
@desc: 
"""
from pydantic import BaseModel

from app.core import TpmCertBaseModel, TpmCertBaseModelWithDefault


class BusinessRegisterRequest(TpmCertBaseModelWithDefault):
    """添加设备公钥的请求"""

    # 伪身份信息
    identity: str
    random_factor: str
    timestamp: str

    request_domain: str


class BusinessQuashRequest(BaseModel):
    """撤销已注册的设备请求"""

    identity: str
    domain: str
