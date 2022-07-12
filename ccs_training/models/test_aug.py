#!/usr/bin/env python3
# encoding: utf-8
"""
@author:  Yan Baoming
@contact: andy.ybm@alibaba-inc.com
"""
import random
from PIL import Image, ImageFilter, ImageDraw
import time
from oss import OssProxy, OssFile
from io import BytesIO
import numpy as np
import cv2
from data import build_transform


def oss_loader(img_path):
    img = None
    for _ in range(10):  # try 10 times
        try:
            data = oss_proxy.download_to_bytes(img_path)
            temp_buffer = BytesIO()
            temp_buffer.write(data)
            temp_buffer.seek(0)
            img = np.fromstring(temp_buffer.getvalue(), np.uint8)
            img = cv2.imdecode(img, cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            temp_buffer.close()
        except Exception as err:
            print('load image error:', img_path, err)
        if img is not None:
            break
    return img


if __name__ == '__main__':
    oss_proxy = OssProxy()
    cfg = {
        # data
        'input_size': (256, 256),
        'random_crop': True,
        'random_erasing': True,
        'random_blur': True,
        'random_distortion': True,
        'random_rotate': True,
        'data_normalize': False,
        'To_pil': True
        }

    img = oss_loader('leogb/herbarium2020/rois/images/000/00/818271.jpg')
    tmp_transform = build_transform(cfg)
    print(tmp_transform)
    for tt in range(10):
        new_img = tmp_transform(img)
        new_img.show()
