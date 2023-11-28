# -*- encoding: utf-8 -*-
"""
@author: ztt
@time: 2023/5/27 18:42
@file: business_schemas.py
@desc: 
"""
from pydantic import BaseModel

from app.core import TpmCertBaseModel


class SameCrossRequest(BaseModel):
    identity: str
    data: str


class SimulateBusinessRequest(TpmCertBaseModel):
    identity: str
    random_n: str


class SimulateBusinessResponse(TpmCertBaseModel):
    msg: str


class CrossdomainAuthenticationRequest(BaseModel):
    identity: str
    domain: str
    data: str
