# -*- encoding: utf-8 -*-
"""
@author: ztt
@time: 2023/2/17 17:43:58
@file: cors_middleware.py
@desc: 
"""
from fastapi.middleware.cors import CORSMiddleware
from starlette.datastructures import Headers
from starlette.responses import JSONResponse
from starlette.types import Scope, Receive, Send

from app.core.response import ResponseModel, ResponseCode
from settings import settings


class CustomCORSMiddleware(CORSMiddleware):
    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        method = scope["method"]
        headers = Headers(scope=scope)
        origin = headers.get("origin")

        # 自定义验证逻辑，比如检查 origin 是否在白名单中
        if origin is None:
            await self.app(scope, receive, send)
            return
        elif not self.is_origin_allowed(origin):
            response = self.create_response()
            await response(scope, receive, send)
            return

        if origin is not None and origin != settings.BUSINESS_BASE_URL:
            # 添加跨域标记
            if scope["query_string"]:
                scope["query_string"] += (
                    b"&cross_domain=true&auth_origin=" + origin.encode()
                )
            else:
                scope["query_string"] = (
                    b"cross_domain=true&auth_origin=" + origin.encode()
                )

        if scope["type"] != "http":  # pragma: no cover
            await self.app(scope, receive, send)
            return

        if method == "OPTIONS" and "access-control-request-method" in headers:
            response = self.preflight_response(request_headers=headers)
            await response(scope, receive, send)
            return

        await self.simple_response(scope, receive, send, request_headers=headers)

    def is_origin_allowed(self, origin):
        return origin in self.allow_origins

    def create_response(self):
        headers = dict(self.preflight_headers)
        content = ResponseModel(code=ResponseCode.fail, message="fail")
        return JSONResponse(content=content.dict(), status_code=400, headers=headers)
