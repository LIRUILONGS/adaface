
## 简单介绍

通过 AdaFace 提取人脸特征向量服务，项目来着：

[https://github.com/mk-minchul/AdaFace](https://github.com/mk-minchul/AdaFace)

### AdaFace 简单介绍  

`低质量人脸数据集`中的`识别`具有挑战性，因为人脸属性被模糊和降级。基于`裕量的损失函数`的进步提高了嵌入空间中人脸的可辨别性。

此外，以前的研究已经研究了`适应性损失`的影响，以更加重视`错误分类`的（硬）例子。在这项工作中，我们介绍了`损失函数自适应性`的另一个方面，即`图像质量`。我们认为，强调错误分类样本的策略应根据其图像质量进行调整。具体来说，简单和硬样品的相对重要性应基于样品的图像质量。

我们`提出了一种新的损失函数，该函数根据图像质量强调不同难度的样本。我们的方法通过用特征范数近似图像质量，以自适应裕量函数的形式实现这一点`。大量的实验表明，我们的方法`AdaFace`在四个数据集（IJB-B，IJB-C，IJB-S和TinyFace）上提高了最先进的（SoTA）的人脸识别性能。

```bash
@inproceedings{kim2022adaface,
  title={AdaFace: Quality Adaptive Margin for Face Recognition},
  author={Kim, Minchul and Jain, Anil K and Liu, Xiaoming},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2022}
}
```


实际测试中发现，`AdaFace` 确实很强大，特别适合远距离，小目标，图片质量低的人脸识别。

[https://github.com/mk-minchul/AdaFace](https://github.com/mk-minchul/AdaFace)

当前项目做了简化，只提供 输出人脸特征向量的 能力

特别说明，输入图片不管是 字节还是,b64 编码，需要符合 单个人脸照片，以做面部对齐处理,大小： 112*112


部署方式

```bash
conda env create -f /environment.yml
source activate AdaFace
pip install -r /requirements.txt  -i  https://pypi.tuna.tsinghua.edu.cn/simple
```

需要的预训练模型文件下载可以在作者的 github 项目主页找

使用方式

可以直接运行测试

```bash
python AdaFaceFeature.py
```

+ 提供了 字节，b64 编码 图片处理
+ 支持输出特征向量方式为 字节和JSON

提供了 Web 服务能力

开发

```bash
python flask_http_server.py
```

生产

```bash
gunicorn  --worker-class gevent  -b 0.0.0.0:30035  --timeout 300  flask_http_server:app
```

HTTP 调用

```bash
curl --location --request POST 'http://192.168.26.81:30035/b64_represent_json' \
--header 'Content-Type: text/plain' \
--data-raw 'iVBORw0KGgoAA.................................mCC'
```

输出 向量的 JSON 表示

```bash
[
    [
        0.054347388446331024,
        -0.031644247472286224,
   ........................
        0.022828513756394386,
        -0.03679579123854637
    ]
]
```

打包了 Docker 镜像，可以直接使用

镜像地址： [https://hub.docker.com/r/liruilong/adaface-face](https://hub.docker.com/r/liruilong/adaface-face)

```bash
docker pull liruilong/adaface-face
```


```bash
(adaface) ┌──[root@vms81.liruilongs.github.io]-[~/adaface/AdaFace_demo]
└─$ docker run --rm  -p 30035:30035  adaface-face
```

```bash
(adaface) ┌──[root@vms81.liruilongs.github.io]-[~/adaface/AdaFace_demo]
└─$ docker run -p 30035:30035 --rm adaface-face gunicorn -w 3  --worker-class gevent  -b 0.0.0.0:30035  --timeout 300  flask_http_server:app
```
