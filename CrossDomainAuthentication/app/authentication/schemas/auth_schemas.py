# -*- encoding: utf-8 -*-
"""
@author: ztt
@time: 2023/5/27 18:35
@file: auth_schemas.py
@desc: 
"""
from pydantic import BaseModel


class AuthenticationRequest(BaseModel):
    identity: str
    request_domain: str
    request_ip: str

    # 是否为跨域认证
    cross_domain: bool = False


class AuthenticationCrossRequest(BaseModel):
    """跨域身份信息认证请求"""

    identity: str
    auth_origin: str
    domain: str
    request_ip: str
    encrypt_data: str
