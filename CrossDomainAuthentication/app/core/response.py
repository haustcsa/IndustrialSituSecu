# -*- encoding: utf-8 -*-
"""
@author: ztt
@time: 2023/2/16 11:10:44
@file: response.py
@desc: 
"""
from enum import unique, IntEnum
from typing import Optional, TypeVar, Generic

from pydantic.generics import GenericModel

ResponseData = TypeVar("ResponseData")


@unique
class ResponseCode(IntEnum):
    """响应code"""

    success = 0
    fail = 1


class ResponseModel(GenericModel, Generic[ResponseData]):
    """响应的模型"""

    code: ResponseCode = ResponseCode.success
    message: str
    data: Optional[ResponseData] = None


def base_response(
    code: ResponseCode, msg: str, data: Optional[ResponseData] = None
) -> ResponseModel[Optional[ResponseData]]:
    """基础返回格式"""
    if data is None:
        data = []

    return ResponseModel(code=code, message=msg, data=data)


def success(
    data: Optional[ResponseData] = None, msg="success"
) -> ResponseModel[Optional[ResponseData]]:
    """成功返回格式"""
    return base_response(code=ResponseCode.success, data=data, msg=msg)


def fail(
    code: ResponseCode = ResponseCode.fail,
    msg: str = "fail",
    data: Optional[ResponseData] = None,
) -> ResponseModel[Optional[ResponseData]]:
    """失败返回格式"""
    return base_response(code=code, msg=msg, data=data)
