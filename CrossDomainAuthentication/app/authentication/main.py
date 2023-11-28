# -*- encoding: utf-8 -*-
"""
@author: ztt
@time: 2023/2/9 23:49
@file: main.py
@desc: 认证服务
"""
from fastapi import FastAPI

from app.authentication.api import api_router
from app.authentication.core import event
from settings import settings

app = FastAPI(
    title=settings.AUTHENTICATION_PROJECT_NAME,
    version=settings.VERSION,
    description=settings.AUTHENTICATION_DESCRIPTION,
    openapi_url="/openapi.json",
    docs_url="/",
)
app.add_event_handler("startup", event.create_start_app_handler(app))

app.include_router(api_router)
