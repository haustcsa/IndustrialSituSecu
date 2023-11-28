# -*- encoding: utf-8 -*-
"""
@author: ztt
@time: 2023/6/4 14:57
@file: handle_dh_rsa.py
@desc: DH-RSA 算法
"""
import base64
from typing import Optional

from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import padding, serialization
from cryptography.hazmat.primitives.asymmetric import dh
from cryptography.hazmat.primitives.asymmetric.dh import (
    DHParameters,
    DHPrivateKey,
    DHPublicKey,
)
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes


class HandleDhRsa:
    generator = 2
    backend = default_backend()
    parameters: Optional[DHParameters] = None

    def gen_generator_params(self) -> int:
        self.parameters = dh.generate_parameters(
            generator=self.generator, key_size=1024, backend=self.backend
        )
        return self.parameters.parameter_numbers().p

    @staticmethod
    async def perform_dh_key_exchange(private_key, peer_public_key) -> bytes:
        shared_key = private_key.exchange(peer_public_key)
        return shared_key[:32]  # 从共享密钥中截取32个字节作为AES密钥

    async def encrypt(self, key: bytes, data: str) -> str:
        iv = b"\x00" * 16  # 使用固定的IV，实际应用中应使用随机生成的IV
        cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=self.backend)
        encryptor = cipher.encryptor()
        padder = padding.PKCS7(128).padder()

        padded_plaintext = padder.update(data.encode()) + padder.finalize()
        ciphertext = encryptor.update(padded_plaintext) + encryptor.finalize()
        return base64.b64encode(iv + ciphertext).decode()

    async def decrypt(self, key: bytes, data: str) -> str:
        ciphertext = base64.b64decode(data)
        iv = ciphertext[:16]
        ciphertext = ciphertext[16:]
        cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=self.backend)
        decryptor = cipher.decryptor()
        unpadder = padding.PKCS7(128).unpadder()

        padded_plaintext = decryptor.update(ciphertext) + decryptor.finalize()
        plaintext = unpadder.update(padded_plaintext) + unpadder.finalize()
        return plaintext.decode()

    async def generate_dh_keypair(
        self, prime: int, generator: int
    ) -> tuple[DHPrivateKey, DHPublicKey]:
        params = await self.get_params(prime, generator)
        return await self.generate_dh_keypair_with_params(params)

    @staticmethod
    async def generate_dh_keypair_with_params(
        params: DHParameters,
    ) -> tuple[DHPrivateKey, DHPublicKey]:
        private_key = params.generate_private_key()
        public_key = private_key.public_key()
        return private_key, public_key

    async def get_params(self, prime: int, generator: int):
        return dh.DHParameterNumbers(prime, generator).parameters(self.backend)

    @staticmethod
    def export_dh_keypair(private_key: DHPrivateKey, public_key: DHPublicKey):
        """导出公钥和私钥"""
        private_pem = private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )
        public_pem = public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )
        return private_pem, public_pem

    async def load_pem_public_key(self, public_key_pem: str) -> DHPublicKey:
        """
        从字符串加载公钥
        :param public_key_pem:
        :return:
        """
        return serialization.load_pem_public_key(
            public_key_pem.encode(), backend=self.backend
        )

    @staticmethod
    async def get_public_key_str(public_key: DHPublicKey) -> str:
        """
        获取公钥的字符串
        :param public_key:
        :return:
        """
        return public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        ).decode()
