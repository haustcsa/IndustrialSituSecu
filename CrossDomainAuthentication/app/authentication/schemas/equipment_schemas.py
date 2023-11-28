# -*- encoding: utf-8 -*-
"""
@author: ztt
@time: 2023/5/27 18:36
@file: equipment_schemas.py
@desc: 
"""
from pydantic import BaseModel

from app.core import TpmCertBaseModel


class AddPublicKeyRequest(TpmCertBaseModel):
    """添加设备公钥的请求"""

    # 伪身份信息
    identity: str
    random_factor: str

    request_domain: str
    request_ip: str
    timestamp: str


class QuashRequest(BaseModel):
    """撤销已注册的设备"""

    identity: str
    domain: str
