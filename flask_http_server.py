#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   flask_http_server.py
@Time    :   2023/10/11 00:20:03
@Author  :   Li Ruilong
@Version :   1.0
@Contact :   liruilonger@gmail.com
@Desc    :   flask 版本的 httpd 服务
"""

# here put the import lib
from flask import  Flask  # 导入Flask类
from flask import Flask, render_template, request,jsonify,Response
from PIL import Image
from flask import make_response
import utils
import logging
import json
import os
from flask import current_app

from functools import wraps
from flask import abort
from concurrent.futures import ThreadPoolExecutor
from AdaFaceFeature import AdaFaceFeature
import yaml_utils as Yaml


app = Flask(__name__)  # 实例化并命名为app实例

face_log = logging.basicConfig(level=logging.INFO, )

config = Yaml.get_yaml_config(file_name="config/config.yaml")
flask_config = config['flask']



class MyContextObject:
    """
    @Time    :   2023/10/11 00:22:41
    @Author  :   liruilonger@gmail.com
    @Version :   1.0
    @Desc    :   全局上下文对象
    """

    def __init__(self, adaface, ready_mark=False):
        self.adaface = adaface
        self.ready_mark = ready_mark


def init_obj():
    """
    @Time    :   2023/09/18 01:33:26
    @Author  :   liruilonger@gmail.com
    @Version :   1.0
    @Desc    :   上下文对象处理
    """

    adaface = AdaFaceFeature()
    logging.info('🚀🚀🚀🚀 人脸特征提取相关模型加载')
    adaface.load_pretrained_model()
    logging.info('🚀🚀🚀🚀 构建上下文对象')
    my_context = MyContextObject(adaface, ready_mark=True)
    app.my_context = my_context
    logging.info('🚀🚀🚀🚀🚀 \033[32m服务启动成功\033[0m')

with app.app_context():
       init_obj()

# 定义装饰器函数，用于验证 Token
def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = request.headers.get('Authorization')
        if not token:
            return jsonify({'message': 'Token is missing'}), 401

        valid_tokens = flask_config["token"]
        # 如果 Token 无效，返回未授权的响应
        if token not in valid_tokens:
             return jsonify({'message': 'Token is invalid'}), 401

        # 如果 Token 有效，继续处理请求
        return f(*args, **kwargs)

    return decorated


@app.route("/")
def index():
    """
    @Time    :   2023/09/17 22:40:15
    @Author  :   liruilonger@gmail.com
    @Version :   1.0
    @Desc    :   欢迎页
    """

    return {'result': "Hello, face"}


@app.route("/livez")
def livez():
    """
    @Time    :   2023/09/17 22:44:43
    @Author  :   liruilonger@gmail.com
    @Version :   1.0
    @Desc    :   服务存活探针接口
    """
    return {'result': " live  ^_^"}


@app.route("/token")
def token():
    return jsonify({
            "code": 200,
            "message": "获取 token",
            "token": flask_config["token"],
        })


@app.route("/readyz")
def readyz():
    """
    @Time    :   2023/09/17 22:46:20
    @Author  :   liruilonger@gmail.com
    @Version :   1.0
    @Desc    :   服务就绪探针接口
    """
    context = app.my_context
    if context.ready_mark:
        return {'result': " ready ^_^ "} 
    else:
        abort(503, "Service Unavailable")
    

@app.route("/b64_represent_byte", methods=["POST"])
def b64representBYTE():
    """
    @Time    :   2023/10/11 07:29:16
    @Author  :   liruilonger@gmail.com
    @Version :   1.0
    @Desc    :   返回二进制特征向量
    """
    
    base64_data = request.get_data(as_text=True)
    if base64_data  is not None:
        context = current_app.my_context
        feature =  context.adaface.b64_get_represent(base64_data)
        if feature  is not None:
            return Response(utils.feature2byte(feature), content_type='application/octet-stream')
        else:
            abort(400, "feature fail")
    else:
        abort(503, "Service Unavailable")
    

@app.route("/b64_represent_json", methods=["POST"])
def b64representJSON():
    """
    @Time    :   2023/09/17 22:46:20
    @Author  :   liruilonger@gmail.com
    @Version :   1.0
    @Desc    :   返回JSON 特征向量
    """
    base64_data = request.get_data(as_text=True)
    if base64_data  is not None:
        context = current_app.my_context
        feature =  context.adaface.b64_get_represent(base64_data)
        if feature  is not None:
            return Response(utils.feature2json(feature), content_type='application/json')
        else:
            abort(400, "feature fail")
    else:
        abort(503, "Service Unavailable")


@app.route("/byte_represent_byte", methods=["POST"])
def byterepresentBYTE():
    """
    @Time    :   2023/10/11 07:29:16
    @Author  :   liruilonger@gmail.com
    @Version :   1.0
    @Desc    :   返回二进制特征向量
    """
    
    base64_data = request.get_data(as_text=True)
    if base64_data  is not None:
        context = current_app.my_context
        feature =  context.adaface.byte_get_represent(base64_data)
        if feature  is not None:
            return Response(utils.feature2byte(feature), content_type='application/octet-stream')
        else:
            abort(400, "feature fail")
    else:
        abort(503, "Service Unavailable")
    

@app.route("/byte_represent_json", methods=["POST"])
def byterepresentJSON():
    """
    @Time    :   2023/09/17 22:46:20
    @Author  :   liruilonger@gmail.com
    @Version :   1.0
    @Desc    :   返回JSON 特征向量
    """
    base64_data = request.get_data(as_text=True)
    if base64_data  is not None:
        context = current_app.my_context
        feature =  context.adaface.byte_get_represent(base64_data)
        if feature  is not None:
            return Response(utils.feature2json(feature), content_type='application/json')
        else:
            abort(400, "feature fail")
    else:
        abort(503, "Service Unavailable")


if __name__ == "__main__":
    
    app.run(port=flask_config['port'], host="0.0.0.0", debug=True)  # 调用run方法，设定端口号，启动服务

