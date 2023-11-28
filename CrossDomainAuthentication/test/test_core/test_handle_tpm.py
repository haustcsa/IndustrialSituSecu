# -*- encoding: utf-8 -*-
"""
@author: ztt
@time: 2023/4/3 23:04
@file: test_handle_tpm.py
@desc: 测试处理 tpm 的方法
"""
import time

from app.core import HandleTpm, calculate_pseudo_identity, logger
from app.core.handle_auth import pcr_verify
from settings import settings


class TestHandleTpm:
    identity = calculate_pseudo_identity(b"test")

    def test_equipment_init(self):
        """测试设备初始化"""
        self.get_instance().equipment_init()

    def test_gen_aik_cert(self):
        """测试生成 aik 证书的秘钥对"""
        self.get_instance(True).gen_aik_cert()

    def test_sign(self):
        """测试签名"""
        instance = self.get_instance(True)
        instance.gen_aik_cert()
        instance.sign()

    def test_checkquote(self):
        """测试校验是否通过"""
        instance = self.get_instance(True)
        instance.gen_aik_cert()
        instance.sign()
        checked = instance.checkquote()
        logger.info(f"校验结果: {checked}")

    def get_instance(self, mkdir: bool = False):
        test_cert_path = settings.PROJECT_DIR / "test/test_core"
        return HandleTpm(
            project_cert_path=test_cert_path, identity=self.identity, mkdir=mkdir
        )

    def test_pcr_verify(self):
        """测试 pcr 校验"""
        timestamp = str(time.time())
        pcr_dict = HandleTpm.get_pcr_dict()
        assert pcr_verify(
            identity=self.identity, timestamp=timestamp, pcr_dict=pcr_dict
        )
