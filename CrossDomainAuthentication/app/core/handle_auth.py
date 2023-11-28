# -*- encoding: utf-8 -*-
"""
@author: ztt
@time: 2023/4/22 14:47
@file: handle_auth.py
@desc: 设备任务及签名校验相关功能
"""
from pathlib import Path
from typing import Optional

from settings import settings
from .handle_cert import write_bytes_file
from .handle_tpm import HandleTpm, get_sha256_hash, get_anticipate_pcr
from .logger import logger


async def save_device_cert_file(
    sign: str,
    pcr: str,
    quote: str,
    akpub: str,
    timestamp: str,
    pcr_dict: dict[str, str],
    random_n: str = "",
    identity: str = "",
    cert_path: Optional[Path] = None,
    abs_path: Optional[Path] = None,
) -> bool:
    """
    保存 tpm 相关的参数到文件中，并校验签名
    :param cert_path:
    :param sign:
    :param pcr:
    :param quote:
    :param akpub:
    :param timestamp:
    :param pcr_dict:
    :param identity:
    :param random_n:
    :param abs_path:
    :return: 签名和 pcr 全部校验通过后返回 True，否则为 False
    """
    if not abs_path:
        current_abs_path = cert_path / "temp-cert" / identity
    else:
        current_abs_path = abs_path

    current_abs_path.mkdir(parents=True, exist_ok=True)

    write_bytes_file(current_abs_path / "sig.out", sign)
    write_bytes_file(current_abs_path / "pcrs.out", pcr)
    write_bytes_file(current_abs_path / "quote.out", quote)
    write_bytes_file(current_abs_path / settings.CERT_AK_PUBLIC_FILENAME, akpub)

    logger.debug("开始校验 sign")
    tpm_instance = HandleTpm(
        project_cert_path=current_abs_path,
        identity=f"{identity}{random_n}",
        ak_public_path=current_abs_path / settings.CERT_AK_PUBLIC_FILENAME,
        dir_name="temp-cert",
    )
    checkout_result = tpm_instance.checkquote()
    logger.debug(f"sign 校验完成: {checkout_result}")
    return await pcr_verify(identity, timestamp, pcr_dict) if checkout_result else False


async def tpm_sign(cert_path: Path, identity: str, timestamp: str) -> HandleTpm:
    """
    执行 tpm 签名操作
    :param cert_path:
    :param identity:
    :param timestamp:
    :return:
    """
    ek_path = cert_path / "ek"
    ak_path = cert_path / "ak"

    tpm_instance = HandleTpm(
        project_cert_path=cert_path,
        identity=identity,
        ek_private_path=ek_path / settings.CERT_EK_PRIVATE_FILENAME,
        ek_public_path=ek_path / settings.CERT_EK_PUBLIC_FILENAME,
        ak_private_path=ak_path / settings.CERT_AK_PRIVATE_FILENAME,
        ak_public_path=ak_path / settings.CERT_AK_PUBLIC_FILENAME,
        ak_name_path=ak_path / settings.CERT_AK_NAME_FILENAME,
        mkdir=False,
    )
    # TODO: 是否需要修改
    # tpm_instance.pcr_event(14, identity + timestamp + "14")
    # tpm_instance.pcr_event(15, identity + timestamp + "15")
    # tpm_instance.pcr_event(16, identity + timestamp + "16")
    tpm_instance.gen_aik_cert()
    tpm_instance.sign()
    logger.debug("签名成功")
    return tpm_instance


async def pcr_verify(identity: str, timestamp: str, pcr_dict: dict[str, str]) -> bool:
    """
    校验 pcr
    需要拓展的值，实验里为：身份id + 时间戳 + 数字
    :param identity: 身份id
    :param timestamp: 时间戳
    :param pcr_dict: pcr 字典
    :return: 校验结果。True(成功), False(失败)
    """
    anticipate_pcr_dict = {}
    logger.info(f"pcr_dict: {pcr_dict}")

    for index in range(14, 17):
        index_str = str(index)
        data = identity + str(timestamp + index_str)
        logger.info(f"{index} 拓展的值为：{data}")
        data_hash = get_sha256_hash(data)
        logger.info(f"data_hash: {data_hash}")

        HandleTpm.pcr_extend(index, data_hash)

        old_pcr = pcr_dict.get(index_str)
        anticipate_pcr = get_anticipate_pcr(old_pcr, data_hash)
        logger.info(f"anticipate_pcr: {anticipate_pcr}")
        anticipate_pcr_dict[index_str] = anticipate_pcr

    real_pcr_dict = HandleTpm.get_pcr_dict()
    for index in range(14, 17):
        index_str = str(index)
        real_pcr = real_pcr_dict.get(index_str)
        anticipate_pcr = anticipate_pcr_dict.get(index_str)
        logger.info(f"{index} real_pcr: {real_pcr} anticipate_pcr: {anticipate_pcr}")
        if real_pcr != anticipate_pcr:
            return False

    return True
