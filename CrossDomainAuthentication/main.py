# -*- encoding: utf-8 -*-
"""
@author: ztt
@time: 2023/2/17 15:38:27
@file: main.py
@desc:
"""
import asyncio
import time

import keyboard
import typer
import uvicorn

from app.client import Client
from settings import settings, load_settings

app = typer.Typer()


@app.command("auth")
def auth_server():
    """认证服务"""
    uvicorn.run(
        app="app.authentication.main:app",
        host="0.0.0.0",
        port=settings.AUTHENTICATION_PORT,
        reload=True,
    )


@app.command("auth2")
def auth_server_2():
    """认证服务"""
    load_settings()
    uvicorn.run(
        app="app.authentication.main:app",
        host="0.0.0.0",
        port=settings.AUTHENTICATION_PORT,
        reload=True,
    )


@app.command("business")
def business_server():
    """业务服务"""
    uvicorn.run(
        app="app.business.main:app",
        host="0.0.0.0",
        port=settings.BUSINESS_PORT,
        reload=True,
    )


@app.command("business2")
def business_server_2():
    """业务服务"""
    load_settings()
    uvicorn.run(
        app="app.business.main:app",
        host="0.0.0.0",
        port=settings.BUSINESS_PORT,
        reload=True,
    )


@app.command("open")
def open_server():
    """开放服务"""
    load_settings()
    uvicorn.run(
        app="app.open.main:app",
        host=settings.OPEN_HOST,
        port=settings.OPEN_PORT,
        reload=True,
    )


def client_func(func: str):
    _client = Client()
    loop = asyncio.new_event_loop()
    loop.run_until_complete(getattr(_client, func)())
    raise typer.Exit()


@app.command("client")
def client():
    """客户端"""
    options = {
        "设备注册": "register",
        "同域认证": "same_origin_authentication",
        "跨域认证": "cross_domain_authentication",
        "跨域重认证": "re_cross_domain_authentication",
        "撤销设备": "quash",
    }
    selected = 0

    while True:
        typer.clear()
        typer.echo("使用键盘方向键选择功能:")
        for i, option in enumerate(options.keys()):
            if i == selected:
                typer.secho(f"> {option}", fg=typer.colors.GREEN)
            else:
                typer.echo(f"  {option}")

        if keyboard.is_pressed("up"):
            selected = (selected - 1) % len(options)
        elif keyboard.is_pressed("down"):
            selected = (selected + 1) % len(options)
        elif keyboard.is_pressed("enter"):
            selected_option = list(options.keys())[selected]
            # selected_function = options[selected_option]
            # selected_function()
            client_func(options[selected_option])

        time.sleep(0.08)


if __name__ == "__main__":
    app()
