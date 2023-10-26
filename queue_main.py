#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   queue_main.py
@Time    :   2023/10/25 05:35:47
@Author  :   Li Ruilong
@Version :   1.0
@Contact :   liruilonger@gmail.com
@Desc    :   adaface 队列版本(flask 版本的 httpd 服务 改造)
"""

# here put the import lib

#!/usr/bin/env python
# -*- encoding: utf-8 -*-

# here put the import lib
from PIL import Image
import utils
import logging
import json
import os
from functools import wraps
import yaml_utils as Yaml
from AdaFaceFeature import AdaFaceFeature
from globals import GlobalObject
from redis_uits import RedisClient 
import pickle



logging.basicConfig(level=logging.INFO)
logging.getLogger("PIL.PngImagePlugin").setLevel(logging.WARNING)

config = Yaml.get_yaml_config(file_name="config/config.yaml")
queue_config = config['queue']
timeout = queue_config['timeout']
global_object = GlobalObject()


def install_redis_vector(rc,face_data,file_name):
    """
    @Time    :   2023/10/20 02:05:09
    @Author  :   liruilonger@gmail.com
    @Version :   1.0
    @Desc    :   数据入表
    """
    image_id = face_data['image_id']
    image_key = file_name
    class_code_account_period = "A0205_1" # 前端传过来
    face_key = class_code_account_period
    logging.info("当前处理的照片为：{}".format(file_name))
    pipe = rc.pipeline()
    for face in face_data['resp']:
        try:
            face_native_images_b64 = face['face_native_images_b64']
            face_id = face['face_id'] + '@@'+ json.dumps(face['face_coordinate'])
            face['face_native_images_b64'] = None
            face['face_align_images_b64'] = None
            face['face_file_name']  = file_name
            pipe.hset(image_key,face_id,face_native_images_b64)
            pipe.rpush(face_key,json.dumps(face))
            logging.info(f"redis 数据 入表 成功:  {face_id}")       
        except Exception as e:
            logging.error(e)
            continue    
    pipe.execute() 


def face_mtcnn_represent(faces):
    """
    @Time    :   2023/10/17 07:54:14
    @Author  :   liruilonger@gmail.com
    @Version :   1.0
    @Desc    :   解析 mtcnn 返回的  face 数据
    """
    if faces is not None and faces['face_efficient_total_resp'] != 0:
        resp = faces['resp']
        logging.info("||=======||当前处理的图片ID：{}".format(faces['image_id']))
        for face in resp:
            logging.info("||===||当前处理的人脸ID：{}".format(face['face_id']))
            face_b64 = face['face_align_images_b64']
            try:
                adaface = global_object.adaface
                feature =  adaface.b64_get_represent(face_b64)
                if feature  is not None:
                    face['face_vetor'] = utils.feature2json(feature)
                
            except Exception as exc:
                logging.info(f"adaface 获取人脸信息 调用异常：{exc}")
                continue
                # {"status": 400, "message": f"mtcnn 获取人脸信息 调用异常：{exc}"}
    
        logging.info("提取人脸特征成功!!!")
        return faces
    else:
        logging.info(f"{faces['image_id']} 没有符合特征提取条件的人脸")
        





if __name__ == "__main__":
    adaface_key = queue_config['single_adaface_queue_key']
    face_key = queue_config['single_face_queue_key']
    rc = RedisClient()
    logging.info("redis 建立连接成功 ！！！{},{}".format(adaface_key,face_key))
    while True:
        #try:
            logging.info("人脸照片消费, 对应 redis 队列：{}".format(face_key))
            data = rc.blpop(face_key,timeout=timeout)
            if data is not None :
                _ , face_data = data
                image_data =  pickle.loads(face_data)
                image_face = face_mtcnn_represent(image_data)
                logging.info("提取人脸信息，放入队列：  {}".format(adaface_key))
                #face_data = pickle.dumps(image_face)
                install_redis_vector(rc,image_face,image_data['face_file_name'])
                
            else:
                logging.info("队列中暂时没有元素，结束等待 ^_^ ")
                break
        #except Exception as e :
        #    logging.error(f"{e}")
        #    continue 



    

