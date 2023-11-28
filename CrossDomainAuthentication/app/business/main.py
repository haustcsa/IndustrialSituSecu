# -*- encoding: utf-8 -*-
"""
@author: ztt
@time: 2023/2/9 23:39
@file: main.py
@desc: 业务服务
"""
from fastapi import FastAPI

from app.business.api import api_router
from app.business.core import event
from app.business.middleware.cors_middleware import CustomCORSMiddleware
from settings import settings

app = FastAPI(
    title=settings.BUSINESS_PROJECT_NAME,
    version=settings.VERSION,
    description=settings.BUSINESS_DESCRIPTION,
    openapi_url="/openapi.json",
    docs_url="/",
)

app.add_event_handler("startup", event.create_start_app_handler(app))
app.add_middleware(
    CustomCORSMiddleware,
    allow_origins=[
        # settings.BUSINESS_BASE_URL,
        # settings.AUTHENTICATION_BASE_URL,
        "http://127.0.0.1:8001",
        "http://127.0.0.1:9001",
        "http://a-domain:8001",
        "http://b-domain:9001",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router)
