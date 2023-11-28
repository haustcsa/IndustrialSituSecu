# -*- encoding: utf-8 -*-
"""
@author: ztt
@time: 2023/4/7 21:56
@file: base.py
@desc: 
"""
from pydantic import BaseModel


class TpmCertBaseModel(BaseModel):
    """tpm 完整性度量的 base model"""

    sign: str
    pcr: str
    quote: str
    akpub: str


class TpmCertBaseModelWithPcrDict(TpmCertBaseModel):
    """tpm 完整性度量的 base model"""

    pcr_dict: dict[str, str]


class TpmCertBaseModelWithDefault(BaseModel):
    """tpm 完整性度量的 base model"""

    sign: str = ""
    pcr: str = ""
    quote: str = ""
    akpub: str = ""


class TpmCertBaseModelWithPcrDictWithDefault(TpmCertBaseModelWithDefault):
    """tpm 完整性度量的 base model"""

    pcr_dict: dict[str, str] = {}
