# -*- encoding: utf-8 -*-
"""
@author: ztt
@time: 2023/2/17 11:50:20
@file: cert.py
@desc: 
"""
from typing import Optional

import rsa
from pydantic import BaseModel

from app.core import logger
from settings import settings


class CertModel(BaseModel):
    public_key: Optional[rsa.PublicKey] = None
    private_key: Optional[rsa.PrivateKey] = None

    client_public_key: dict[str, rsa.PublicKey] = {}

    def get_public_key(self):
        """读取公钥"""
        with open(settings.BUSINESS_CERT_SERVER_PUBLIC_FILE, "rb") as f:
            public_key_pem = f.read()
        self.public_key = rsa.PublicKey.load_pkcs1(public_key_pem)
        logger.info("公钥读取成功")

    def get_private_key(self):
        """读取私钥"""
        with open(settings.BUSINESS_CERT_SERVER_PRIVATE_FILE, "rb") as f:
            private_key_pem = f.read()
        self.private_key = rsa.PrivateKey.load_pkcs1(private_key_pem)
        logger.info("私钥读取成功")

    def get_client_public_key(self, identity: str) -> rsa.PublicKey:
        """获取客户端的公钥"""
        if identity in self.client_public_key:
            return self.client_public_key[identity]

        client_public_key_file = (
            settings.BUSINESS_CERT_CLIENT_PATH / identity / "public.pem"
        )
        if client_public_key_file.exists():
            with open(client_public_key_file, "rb") as f:
                public_key_pem = f.read()
            public_key = rsa.PublicKey.load_pkcs1(public_key_pem)
            self.client_public_key[identity] = public_key
            return public_key

    class Config:
        arbitrary_types_allowed = True


business_cert = CertModel()
