#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   globals.py
@Time    :   2023/10/13 04:32:07
@Author  :   Li Ruilong
@Version :   1.0
@Contact :   liruilonger@gmail.com
@Desc    :   fast API 全局对象
"""

# here put the import lib


import yaml_utils as Yaml
import logging
from AdaFaceFeature import AdaFaceFeature





class GlobalObject:
    def __init__(self):
        adaface = AdaFaceFeature()
        logging.info('🚀🚀🚀🚀 人脸特征提取相关模型加载')
        adaface.load_pretrained_model()
        logging.info('🚀🚀🚀🚀 构建上下文对象')
        self.adaface = adaface
        logging.info('🚀🚀🚀🚀🚀 \033[32m服务启动成功\033[0m')

    def do_something(self):
        # 实现全局对象的功能
        pass

