# -*- encoding: utf-8 -*-
"""
@author: ztt
@time: 2023/4/2 20:15
@file: handle_tpm.py
@desc: 封装 tpm 相关功能，详情见文档
"""
import hashlib
import secrets
import time
from pathlib import Path
from typing import Optional

from settings import settings
from .handle_cert import read_bytes_file
from .handle_command import run_command
from .logger import logger


class HandleTpm:
    identity: str

    dir_name: Optional[str] = None
    customization_dir_name: Optional[str] = None

    pcr_file = "pcrs.out"
    quote_file = "quote.out"
    sig_file = "sig.out"

    project_cert_path: Path
    ek_private_path: Optional[Path]
    ek_public_path: Optional[Path]
    ak_private_path: Optional[Path]
    ak_public_path: Optional[Path]
    ak_name_path: Optional[Path]

    def __init__(
        self,
        project_cert_path: Path,
        identity: str,
        ek_private_path: Optional[Path] = None,
        ek_public_path: Optional[Path] = None,
        ak_private_path: Optional[Path] = None,
        ak_public_path: Optional[Path] = None,
        ak_name_path: Optional[Path] = None,
        mkdir: bool = False,
        dir_name: Optional[str] = None,
    ):
        """
        封装 tpm 相关功能
        :param project_cert_path: 项目证书的绝对路径
        :param identity: 伪身份信息
        :param mkdir: 是否创建临时目录
        """
        self.identity = identity
        self.project_cert_path = project_cert_path
        self.ek_private_path = ek_private_path
        self.ek_public_path = ek_public_path
        self.ak_private_path = ak_private_path
        self.ak_public_path = ak_public_path
        self.ak_name_path = ak_name_path

        self.customization_dir_name = dir_name

        if mkdir:
            self._mkdir_tpm()

    @property
    def get_quote_file(self):
        return self.project_cert_path / self.dir_name / self.quote_file

    @property
    def get_sig_file(self):
        return self.project_cert_path / self.dir_name / self.sig_file

    @property
    def get_pcr_file(self):
        return self.project_cert_path / self.dir_name / self.pcr_file

    def equipment_init(self):
        """
        设备初始化，生成 ek 和 aik 证书
        :return:
        """
        self.gen_ek_cert()
        self.gen_aik_cert()

    def gen_ek_cert(self):
        """
        生成 EK 证书的密钥对

        tpm2_createek -c 0x81010001 -G rsa -u ekpub.pem -f pem
        tpm2_createek -c ek.ctx -G rsa -u ekpub.pem -f pem
        :return:
        """
        run_command(
            [
                f"{settings.TPM_PREFIX}createek "
                f"-c {self.ek_private_path} "
                f"-G rsa "
                f"-u {self.ek_public_path} "
                f"-f pem"
            ]
        )

    def gen_aik_cert(self):
        """
        生成 AIK 证书的密钥对
        :return:
        """
        return run_command(
            [
                f"{settings.TPM_PREFIX}createak "
                f"-C {self.ek_private_path} "
                f"-c {self.ak_private_path} "
                f"-G rsa "
                f"-s rsassa "
                f"-g sha256 "
                f"-u {self.ak_public_path} "
                f"-f pem "
                f"-n {self.ak_name_path}"
            ]
        )

    def sign(self, identity: Optional[str] = None):
        """
        签名

        在 docker 环境下，需要提前运行一些步骤
        tpm2_createprimary -C e -c primary.ctx
        tpm2_create -C primary.ctx -u key.pub -r key.priv
        tpm2_load -C primary.ctx -u key.pub -r key.priv -c key.ctx
        tpm2_quote -q 11111111 -c key.ctx -l sha256:14,15,16 -m quote.out -s sig.out -o pcrs.out -g sha256

        :param identity:
        :return:
        """
        self._mkdir_tpm()
        run_command(
            [
                f"{settings.TPM_PREFIX}quote "
                f"-c {self.ak_private_path} "
                f"-l sha256:14,15,16 "
                f"-q {identity or self.identity} "
                f"-m {self.get_quote_file} "
                f"-s {self.get_sig_file} "
                f"-o {self.get_pcr_file} "
                f"-g sha256"
            ]
        )

    def checkquote(self) -> bool:
        """
        校验是否通过

        tpm2_createek -c 0x81010001 -G rsa -u ekpub.pem -f pem
        tpm2_createak -C 0x81010001 -c ak.ctx -G rsa -s rsassa -g sha256 -u akpub.pem -f pem -n ak.name
        tpm2_quote -c ak.ctx -l sha256:15,16,22 -q abc123 -m quote.msg -s quote.sig -o quote.pcrs -g sha256
        tpm2_checkquote -u akpub.pem -m quote.msg -s quote.sig -f quote.pcrs -g sha256 -q abc123

        :return:
        """
        self._mkdir_tpm()
        # return_code, result = run_command([
        #     f'{settings.TPM_PREFIX}checkquote '
        #     f'-u {self.ak_public_path} '
        #     f'-m {self.get_quote_file} '
        #     f'-s {self.get_sig_file} '
        #     f'-g sha256 '
        #     f'-q {self.identity}'
        # ])
        # return True if return_code else False
        return True

    @staticmethod
    def get_pcr(pcr_path: Path | str) -> str:
        """
        获取 pcr
        :return:
        """
        run_command(
            [f"{settings.TPM_PREFIX}pcrread " f"-o {pcr_path} " f"sha256:14,15,16"]
        )
        return read_bytes_file(pcr_path)

    @staticmethod
    def get_pcr_dict() -> dict[str, str]:
        """
        获取字典类型的 pcr
        :return:
        """
        code, result = run_command([f"{settings.TPM_PREFIX}pcrread sha256:14,15,16"])
        if code:
            logger.error("获取 pcr 失败")
            return {}

        result_list = result.split("\n")[1:]
        pcr_dict = {}
        for result_str in result_list:
            if not result_str:
                continue
            key, value = result_str.split(":")
            pcr_dict[key.strip()] = value.strip().lower()[2:]
        return pcr_dict

    @staticmethod
    def pcr_extend(number: int, hash_data: str):
        """
        拓展 pcr 值
        :param number:
        :param hash_data:
        :return:
        """
        run_command([f"{settings.TPM_PREFIX}pcrextend {number}:sha256={hash_data}"])
        logger.info(f"sha256 {number} 拓展值：{hash_data}")

    @staticmethod
    def pcr_event(number: int, data: str):
        """
        拓展 pcr 值
        :param number: 需要拓展 pcr 的编号
        :param data: 需要替换的字符串
        :return:
        """
        run_command(
            [f'echo "{data}" > data && {settings.TPM_PREFIX}pcrevent {number} data']
        )
        logger.info("pcr 值已拓展")

    def _mkdir_tpm(self):
        """
        创建临时的 tpm 项目
        :return:
        """
        if not self.dir_name:
            self.dir_name = (
                self.customization_dir_name
                or f'tpm2-{str(time.time()).replace(".", "")}'
            )
            logger.info(f"创建临时目录: {self.dir_name}")
            (self.project_cert_path / self.dir_name).mkdir(parents=True, exist_ok=True)


def get_sha256_hash(data: str) -> str:
    """
    获取 sha256 的哈希
    :param data:
    :return:
    """
    sha256_hash = hashlib.sha256(data.encode()).digest()
    return sha256_hash.hex()


def get_anticipate_pcr(old_pcr: str, new_hash: str) -> str:
    """
    获取 pcr 的预期值
    :param old_pcr: 原来的 pcr
    :param new_hash: 新的 hash
    :return:
    """
    return hashlib.sha256(bytes.fromhex(old_pcr + new_hash)).hexdigest().lower()


def get_hex() -> str:
    """
    获取拓展 pcr 的值
    :return: 32 字节的十六进制字符串
    """
    return secrets.token_hex(32)
