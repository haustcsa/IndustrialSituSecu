# -*- encoding: utf-8 -*-
"""
@author: ztt
@time: 2023/4/22 12:48
@file: main.py
@desc: 
"""
from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware

from app.open.api import api_router
from settings import settings
from .core import event

app = FastAPI(
    title=settings.OPEN_PROJECT_NAME,
    version=settings.VERSION,
    description=settings.OPEN_DESCRIPTION,
    openapi_url="/openapi.json",
    docs_url="/",
)

app.add_event_handler("startup", event.create_start_app_handler(app))
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router)
