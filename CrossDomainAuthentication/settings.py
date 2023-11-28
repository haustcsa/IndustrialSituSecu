# -*- encoding: utf-8 -*-
"""
@author: ztt
@time: 2023/2/15 16:14:38
@file: settings.py
@desc: 项目的配置文件
"""
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from pydantic import BaseSettings, validator
from rsa import PublicKey, PrivateKey

from app.core.random_range import generate_random_number
from app.core.schemas import SettingDhRsaModel


class Settings(BaseSettings):
    VERSION: str = "1.0.0"
    DOMAIN: str = "A"
    PROJECT_DIR = Path(__file__).parent
    CREDENTIALS_ALL: list[str] = []

    # 是否开启区块链
    BLOCKCHAIN: bool = False
    # 是否使用 tpm，因论文实验部分不需要测试 tpm，所以测试时间时关闭1
    IS_TPM: bool = True

    # tpm2 前缀
    # tpm2-toolbox.
    # tpm2_
    TPM_PREFIX: str = "tpm2_"

    DOCKER_HOST: str = "a-domain"
    IDENTITY: str = str(generate_random_number())[:10]

    # 证书名称
    CERT_EK_PRIVATE_FILENAME: str = "ek.ctx"
    CERT_EK_PUBLIC_FILENAME: str = "ekpub.pem"
    CERT_AK_PRIVATE_FILENAME: str = "ak.ctx"
    CERT_AK_PUBLIC_FILENAME: str = "akpub.pem"
    CERT_AK_NAME_FILENAME: str = "ak.name"
    CERT_PCR_FILENAME: str = "pcrs.out"

    # 认证服务
    AUTHENTICATION_PATH: Path = PROJECT_DIR / "app" / "authentication"
    AUTHENTICATION_CERT_PATH: Path = AUTHENTICATION_PATH / "cert" / IDENTITY
    AUTHENTICATION_CERT_PATH.mkdir(parents=True, exist_ok=True)

    AUTHENTICATION_PROJECT_NAME: str = "认证服务api"
    AUTHENTICATION_DESCRIPTION: str = "跨域认证 - 任务服务"
    AUTHENTICATION_HOST: str = "127.0.0.1"
    AUTHENTICATION_PORT: int = 8002
    AUTHENTICATION_BASE_URL: str = ""
    # ek
    AUTHENTICATION_CERT_EK_PATH = AUTHENTICATION_CERT_PATH / "ek"
    AUTHENTICATION_CERT_EK_PATH.mkdir(parents=True, exist_ok=True)
    AUTHENTICATION_CERT_EK_PRIVATE_FILE: Path = (
        AUTHENTICATION_CERT_EK_PATH / CERT_EK_PRIVATE_FILENAME
    )
    AUTHENTICATION_CERT_EK_PUBLIC_FILE: Path = (
        AUTHENTICATION_CERT_EK_PATH / CERT_EK_PUBLIC_FILENAME
    )
    # ak
    AUTHENTICATION_CERT_AK_PATH = AUTHENTICATION_CERT_PATH / "ak"
    AUTHENTICATION_CERT_AK_PATH.mkdir(parents=True, exist_ok=True)
    AUTHENTICATION_CERT_AK_PRIVATE_FILE: Path = (
        AUTHENTICATION_CERT_AK_PATH / CERT_AK_PRIVATE_FILENAME
    )
    AUTHENTICATION_CERT_AK_PUBLIC_FILE: Path = (
        AUTHENTICATION_CERT_AK_PATH / CERT_AK_PUBLIC_FILENAME
    )
    AUTHENTICATION_CERT_AK_NAME_FILE: Path = (
        AUTHENTICATION_CERT_AK_PATH / CERT_AK_NAME_FILENAME
    )
    # pcr
    AUTHENTICATION_CERT_PCR_PATH: Path = AUTHENTICATION_CERT_PATH / CERT_PCR_FILENAME
    AUTHENTICATION_PCR: str = ""
    # rsa key
    AUTHENTICATION_RSA_CERT_PATH: Path = AUTHENTICATION_CERT_PATH
    AUTHENTICATION_RSA_PUBLIC_KEY: Optional[PublicKey] = None
    AUTHENTICATION_RSA_PUBLIC_KEY_STR: str = ""
    AUTHENTICATION_RSA_PRIVATE_KEY: Optional[PrivateKey] = None
    AUTHENTICATION_RSA_PRIVATE_KEY_STR: str = ""

    # 业务服务
    BUSINESS_PROJECT_NAME: str = "业务服务api"
    BUSINESS_DESCRIPTION: str = "跨域认证 - 业务服务"
    BUSINESS_HOST: str = "127.0.0.1"
    BUSINESS_PORT: int = 8001
    BUSINESS_BASE_URL: str = ""
    BUSINESS_DIR: Path = PROJECT_DIR / "app" / "business"

    # 业务服务的证书路径
    BUSINESS_CERT_PATH: Path = BUSINESS_DIR / "cert" / IDENTITY
    BUSINESS_CERT_PATH.mkdir(parents=True, exist_ok=True)

    BUSINESS_CERT_SERVER_PATH: Path = BUSINESS_CERT_PATH / "server"
    BUSINESS_CERT_SERVER_PATH.mkdir(parents=True, exist_ok=True)
    BUSINESS_CERT_SERVER_PRIVATE_FILE: Path = BUSINESS_CERT_SERVER_PATH / "private.pem"
    BUSINESS_CERT_SERVER_PUBLIC_FILE: Path = BUSINESS_CERT_SERVER_PATH / "public.pem"

    BUSINESS_CERT_CLIENT_PATH: Path = BUSINESS_CERT_PATH / "client" / IDENTITY
    BUSINESS_CERT_CLIENT_PATH.mkdir(parents=True, exist_ok=True)
    # ek
    BUSINESS_CERT_EK_PATH = BUSINESS_CERT_PATH / "ek"
    BUSINESS_CERT_EK_PATH.mkdir(parents=True, exist_ok=True)
    BUSINESS_CERT_EK_PRIVATE_FILE: Path = (
        BUSINESS_CERT_EK_PATH / CERT_EK_PRIVATE_FILENAME
    )
    BUSINESS_CERT_EK_PUBLIC_FILE: Path = BUSINESS_CERT_EK_PATH / CERT_EK_PUBLIC_FILENAME
    # ak
    BUSINESS_CERT_AK_PATH = BUSINESS_CERT_PATH / "ak"
    BUSINESS_CERT_AK_PATH.mkdir(parents=True, exist_ok=True)
    BUSINESS_CERT_AK_PRIVATE_FILE: Path = (
        BUSINESS_CERT_AK_PATH / CERT_AK_PRIVATE_FILENAME
    )
    BUSINESS_CERT_AK_PUBLIC_FILE: Path = BUSINESS_CERT_AK_PATH / CERT_AK_PUBLIC_FILENAME
    BUSINESS_CERT_AK_NAME_FILE: Path = BUSINESS_CERT_AK_PATH / CERT_AK_NAME_FILENAME
    # pcr
    BUSINESS_CERT_PCR_PATH: Path = BUSINESS_CERT_PATH / CERT_PCR_FILENAME
    BUSINESS_PCR: str = ""
    # rsa key
    BUSINESS_RSA_CERT_PATH: Path = BUSINESS_CERT_PATH
    BUSINESS_RSA_PUBLIC_KEY: Optional[PublicKey] = None
    BUSINESS_RSA_PUBLIC_KEY_STR: str = ""
    BUSINESS_RSA_PRIVATE_KEY: Optional[PrivateKey] = None
    BUSINESS_RSA_PRIVATE_KEY_STR: str = ""
    # dh-rsa
    # TODO: 证书没有保存到本地，目前只是存在变量里
    BUSINESS_DH_RSA_ALL: dict[str, SettingDhRsaModel] = {}
    MESSAGE: str = ""

    # client
    CLIENT_DIR: Path = PROJECT_DIR / "app" / "client"
    CLIENT_CERT_PATH: Path = CLIENT_DIR / "cert"
    CLIENT_CERT_PATH.mkdir(parents=True, exist_ok=True)
    CLIENT_CERT_PRIVATE_FILE: Path = CLIENT_CERT_PATH / "private.pem"
    # CLIENT_CERT_PUBLIC_FILE: Path = CLIENT_CERT_PATH / 'public.pem'
    # CLIENT_CERT_SERVER_PUBLIC_FILE: Path = CLIENT_CERT_PATH / 'server_public.pem'
    # ek
    CLIENT_CERT_EK_PATH = CLIENT_CERT_PATH / "ek"
    CLIENT_CERT_EK_PATH.mkdir(parents=True, exist_ok=True)
    CLIENT_CERT_EK_PRIVATE_FILE: Path = CLIENT_CERT_EK_PATH / CERT_EK_PRIVATE_FILENAME
    CLIENT_CERT_EK_PUBLIC_FILE: Path = CLIENT_CERT_EK_PATH / CERT_EK_PUBLIC_FILENAME
    # ak
    CLIENT_CERT_AK_PATH = CLIENT_CERT_PATH / "ak"
    CLIENT_CERT_AK_PATH.mkdir(parents=True, exist_ok=True)
    CLIENT_CERT_AK_PRIVATE_FILE: Path = CLIENT_CERT_AK_PATH / CERT_AK_PRIVATE_FILENAME
    CLIENT_CERT_AK_PUBLIC_FILE: Path = CLIENT_CERT_AK_PATH / CERT_AK_PUBLIC_FILENAME
    CLIENT_CERT_AK_NAME_FILE: Path = CLIENT_CERT_AK_PATH / CERT_AK_NAME_FILENAME
    # pcr
    CLIENT_CERT_PCR_PATH: Path = CLIENT_CERT_PATH / CERT_PCR_FILENAME

    # 证书
    CERT_CA_FILE: Path = PROJECT_DIR / "app" / "handle_cert" / "ca.crt"
    CERT_SERVER_KEY_FILE: Path = PROJECT_DIR / "app" / "handle_cert" / "server.key"
    CERT_SERVER_CERT_FILE: Path = PROJECT_DIR / "app" / "handle_cert" / "server.crt"

    # LOGGER
    LOGGER_FILENAME = PROJECT_DIR / "log" / "log.log"
    LOGGER_ERROR_FILENAME = PROJECT_DIR / "log" / "error.log"

    # open 开放服务
    OPEN_DIR: Path = PROJECT_DIR / "app" / "open"
    OPEN_CERT_DIR: Path = OPEN_DIR / "cert"
    OPEN_CERT_DIR.mkdir(parents=True, exist_ok=True)

    OPEN_PROJECT_NAME: str = "开放服务api"
    OPEN_DESCRIPTION: str = "项目对接文档"
    OPEN_HOST: str = "0.0.0.0"
    OPEN_PORT: int = 8080
    # ek
    OPEN_CERT_EK_PATH = OPEN_CERT_DIR / "ek"
    OPEN_CERT_EK_PATH.mkdir(parents=True, exist_ok=True)
    OPEN_CERT_EK_PRIVATE_FILE: Path = OPEN_CERT_EK_PATH / CERT_EK_PRIVATE_FILENAME
    OPEN_CERT_EK_PUBLIC_FILE: Path = OPEN_CERT_EK_PATH / CERT_EK_PUBLIC_FILENAME
    # ak
    OPEN_CERT_AK_PATH = OPEN_CERT_DIR / "ak"
    OPEN_CERT_AK_PATH.mkdir(parents=True, exist_ok=True)
    OPEN_CERT_AK_PRIVATE_FILE: Path = OPEN_CERT_AK_PATH / CERT_AK_PRIVATE_FILENAME
    OPEN_CERT_AK_PUBLIC_FILE: Path = OPEN_CERT_AK_PATH / CERT_AK_PUBLIC_FILENAME
    OPEN_CERT_AK_NAME_FILE: Path = OPEN_CERT_AK_PATH / CERT_AK_NAME_FILENAME

    class Config:
        env_file = None
        case_sensitive = True

    @validator("AUTHENTICATION_BASE_URL")
    def _get_authentication_base_url(cls, _: str, values: dict[str, str]) -> str:
        return f'http://{values["DOCKER_HOST"]}:{values["AUTHENTICATION_PORT"]}'

    @validator("BUSINESS_BASE_URL")
    def _get_business_base_url(cls, _: str, values: dict[str, str]) -> str:
        return f'http://{values["DOCKER_HOST"]}:{values["BUSINESS_PORT"]}'


settings: Settings = Settings()
with open(settings.PROJECT_DIR / "app/static/message.txt", "r", encoding="utf-8") as f:
    settings.MESSAGE = f.read()


def load_settings():
    # 从环境变量或者文件中读取配置
    env = "./env/.another_env"
    Settings.Config.env_file = env
    load_dotenv(env)
    for key, value in Settings().dict().items():
        setattr(settings, key, value)
