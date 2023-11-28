# -*- encoding: utf-8 -*-
"""
@author: ztt
@time: 2023/2/10 0:37
@file: handle_rsa.py
@desc: 
"""
import base64
import hashlib
import os
import time
from typing import Optional

import rsa
from rsa import PublicKey, PrivateKey, key


async def generate_rsa_key(
    identity: Optional[str] = None, key_length: int = 512
) -> tuple[PublicKey, PrivateKey]:
    """
    生成秘钥
    :param identity: 身份标识
    :param key_length: 秘钥长度
    :return:
    """
    custom_exponent = (
        int.from_bytes(identity.encode(), "big") if identity is not None else None
    )
    return rsa.newkeys(key_length, exponent=custom_exponent or key.DEFAULT_EXPONENT)


def generate_signature(data: str, private_key: PrivateKey) -> str:
    """
    生成签名
    :param data: 自定义数据（客户端的伪身份 id）
    :param private_key: 私钥
    :return:
    """
    # 使用SHA-256哈希算法计算数据的摘要
    data_hash = hashlib.sha256(data.encode("utf-8")).digest()

    # 使用私钥对数据摘要进行签名
    signature = rsa.sign(data_hash, private_key, "SHA-256")

    # 将签名转换为字符串格式（base64编码）
    return base64.b64encode(signature).decode("utf-8")


def verify_signature(
    data: str,
    signature: str,
    public_key: PublicKey,
    timestamp: str,
    random_factor: str,
    expiration_time: int = 15,
) -> bool:
    """
    校验签名
    :param data: 自定义数据（客户端的伪身份 id）
    :param signature: 签名值（临时身份凭证）
    :param public_key: 公钥
    :param timestamp: 时间戳
    :param random_factor: 随机因子
    :param expiration_time: 过期时间，默认 15ms
    :return:
    """
    # 验证时间戳是否过期
    if int(time.time()) - int(timestamp) > expiration_time:
        return False  # 时间戳过期，验签失败

    # 将时间戳和随机因子与待验签的数据拼接在一起
    data_with_timestamp = timestamp + random_factor + data

    # 使用SHA-256哈希算法计算数据的摘要
    data_hash = hashlib.sha256(data_with_timestamp.encode("utf-8")).digest()

    # 将签名从字符串格式转换为字节类型
    signature_bytes = base64.b64decode(signature)

    # 使用公钥对签名进行验签
    try:
        rsa.verify(data_hash, signature_bytes, public_key)
        return True  # 验签通过
    except rsa.VerificationError:
        return False  # 验签失败


async def generate_credentials(
    data: str,
) -> str:
    """
    生成身份凭证
    :param data:
    :return:
    """
    # 生成随机的盐值
    salt = os.urandom(16)

    # 使用 PBKDF2 算法进行哈希处理
    hashed_data = hashlib.pbkdf2_hmac(
        "sha256",  # 使用 SHA-256 哈希算法
        data.encode("utf-8"),  # 将密码转换为字节串
        salt,
        10000,  # 迭代次数
    )

    # 将盐值和哈希值合并
    credentials = salt + hashed_data
    # base64 编码
    return base64.b64encode(credentials).decode("utf-8")


async def verify_credentials(
    credentials: str, provided_credentials: str, expiration_time: int = 600
) -> bool:
    """
    校验身份凭证
    :param credentials: 身份凭证
    :param provided_credentials: 生成身份凭证的原始字符串
    :param expiration_time: 过期时间，默认 10 分钟
    :return:
    """
    # 验证时间戳是否过期
    if int(time.time()) - int(provided_credentials[:10]) > expiration_time:
        return False  # 时间戳过期

    credentials_bytes = base64.b64decode(credentials)
    salt = credentials_bytes[:16]  # 获取盐值
    stored_hash = credentials_bytes[16:]  # 获取哈希值

    # 对提供的密码进行哈希处理
    hashed_credentials = hashlib.pbkdf2_hmac(
        "sha256",  # 使用 SHA-256 哈希算法
        provided_credentials.encode("utf-8"),  # 转换为字节串
        salt,
        10000,  # 迭代次数
    )

    # 比较哈希值是否相同
    return hashed_credentials == stored_hash


def encrypt_message(message: str, public_key) -> str:
    # 使用公钥对消息进行加密
    encrypted_message = rsa.encrypt(message.encode("utf-8"), public_key)
    return base64.b64encode(encrypted_message).decode("utf-8")


def decrypt_message(encrypted_message: str, private_key):
    # 使用私钥对加密的消息进行解密
    encrypted_message_bytes = base64.b64decode(encrypted_message)
    decrypted_message = rsa.decrypt(encrypted_message_bytes, private_key)
    return decrypted_message.decode("utf-8")


async def encode_message(message: str, public_key: rsa.PublicKey) -> str:
    """加密消息"""
    crypto = b""
    divide = int(len(message) / 117)
    divide = divide if divide > 0 else divide + 1
    line = divide if len(message) % 117 == 0 else divide + 1
    for i in range(line):
        crypto += rsa.encrypt(message[i * 117 : (i + 1) * 117].encode(), public_key)

    return base64.b64encode(crypto).decode()


async def decode_message(message: str, private_key: rsa.PrivateKey) -> str:
    """解密消息"""
    _message = base64.b64decode(message)
    length = len(_message)
    default_length = 128
    if length < default_length:
        return b"".join(rsa.decrypt(_message, private_key)).decode("utf8")

    offset = 0
    res = []
    while length - offset > 0:
        # 先判断剩余待解密消息的长度是否大于 default_length（每段解密长度）
        # 如果大于，就取出长度为 default_length 的一段进行解密；
        # 否则，直接将剩余的消息全部取出进行解密，并将解密结果添加到 res 列表中
        if length - offset > default_length:
            res.append(
                rsa.decrypt(_message[offset : offset + default_length], private_key)
            )
        else:
            res.append(rsa.decrypt(_message[offset:], private_key))
        offset += default_length

    return b"".join(res).decode("utf8")
