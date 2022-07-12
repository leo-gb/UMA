#!/usr/bin/env python3
# encoding: utf-8
"""
@author:  Yan Baoming
@contact: andy.ybm@alibaba-inc.com
"""
import random
from PIL import Image, ImageFilter, ImageDraw
import math
import time
from oss import OssProxy, OssFile
from io import BytesIO
import numpy as np
import cv2

class RandomBlur(object):
    """randomly select a region of the image and do blur.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3)):
        if (scale[0] > scale[1]) or (ratio[0] > ratio[1]):
            warnings.warn("range should be of kind (min, max)")
        if scale[0] < 0 or scale[1] > 1:
            raise ValueError("range of scale should be between 0 and 1")
        if p < 0 or p > 1:
            raise ValueError("range of random erasing probability should be between 0 and 1")

        self.p = p
        self.scale = scale
        self.ratio = ratio
        # self.blur_type = [ImageFilter.BLUR, ImageFilter.BoxBlur(5), ImageFilter.GaussianBlur(5)]
        self.blur_type = [
        ImageFilter.BLUR, ImageFilter.BoxBlur(5), ImageFilter.GaussianBlur(5),
        ImageFilter.Kernel((5,5),(0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0)),
        ImageFilter.Kernel((5,5),(1,0,0,0,0,0,2,0,0,0,0,0,0,1,0,0,0,0,1,0,0,0,0,0,1)),
        ImageFilter.Kernel((5,5),(0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0)),
        ImageFilter.Kernel((5,5),(0,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,0))
        ]
    def __call__(self, img=None):
        if random.uniform(0, 1) < self.p:
            img = img.copy()
            width = img.width
            height = img.height
            area = width * height
            # area = 100
            this_scale = self.scale[0] + random.random() * (self.scale[1] - self.scale[0])
            this_area = this_scale * area
            this_ratio = self.ratio[0] + random.random() * (self.ratio[1] - self.ratio[0])
            this_width = min(int(math.sqrt(this_area / this_ratio)), width//2)
            this_height = min(int(this_width * this_ratio), height//2)
            x_min = int(random.randint(0, width - this_width -1))
            y_min = int(random.randint(0, height - this_height -1))
            x_max = x_min + this_width
            y_max = y_min + this_height
            croped_img = img.crop((x_min, y_min, x_max, y_max))
            this_blur_type = random.choice(self.blur_type)
            # print(this_blur_type)
            croped_img = croped_img.filter(this_blur_type)
            img.paste(croped_img, (x_min, y_min))
            draw = ImageDraw.Draw(img)
            # draw.rectangle(((x_min, y_min), (x_max, y_max)))
            # print('use blur')
        return img


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
    # img = Image.open('1000000085_i_3_038982.jpg')
    img = oss_loader('leogb/herbarium2020/rois/images/000/00/818271.jpg')
    print(img)
    width = img.width
    height = img.height
    new_img = Image.new('RGB', (width*2, height))
    blur = RandomBlur(p=1)
    for tt in range(10000000):
        print(tt)
        blured_img = blur(img)
        new_img.paste(img)
        new_img.paste(blured_img, (width, 0))
        new_img.save('blured_img.jpg')
        time.sleep(1)
    # print()
