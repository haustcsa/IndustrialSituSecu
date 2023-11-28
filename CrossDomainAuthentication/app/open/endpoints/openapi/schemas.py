# -*- encoding: utf-8 -*-
"""
@author: ztt
@time: 2023/4/22 14:40
@file: schemas.py
@desc: 
"""
from pydantic import BaseModel

from app.core import TpmCertBaseModel


class RegisterReqeust(TpmCertBaseModel):
    """注册新设备的请求"""

    identity: str
    request_ip: str
    request_domain: str
    accept_domain: str
    accept_ip: str


class SignResponse(TpmCertBaseModel):
    """签名并获取设备参数的响应"""


class CheckSignRequest(TpmCertBaseModel):
    """校验签名并获取设备参数的请求"""

    identity: str
    request_ip: str
    request_domain: str
    accept_domain: str
    accept_ip: str


class CheckSignResponse(BaseModel):
    """校验签名并获取设备参数的响应"""

    valid: bool


class QuashRequest(BaseModel):
    """撤销已注册的设备的请求"""

    identity: str
    request_ip: str
    request_domain: str
    accept_domain: str
    accept_ip: str
