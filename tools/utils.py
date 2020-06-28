import numpy as np

import cv2 as cv
from PIL import Image

import os


Image.MAX_IMAGE_PIXELS = 1000000000000
def image_resize_save_mask(path):
    # 1 打开图片
    img = Image.open(path)
    img = np.asarray(img)
    mask = img[:, :, -1]

    # 2 缩放
    cimg = cv.resize(img, None, fx=0.1, fy=0.1)

    # 3 保存
    ##3.1 BGR2RGB(PIL通道读取与cv2相反)
    cimg = cv.cvtColor(cimg, cv.COLOR_RGB2BGR)
    ##3.2 解析保存路径
    save_dir = r"./vis/"
    if not os.path.isdir(save_dir): os.makedirs(save_dir)
    root_path, png_name = os.path.split(path)
    filename, filetype = os.path.splitext(png_name)

    vis_mask_name = os.path.join(save_dir, filename + "_mask_vis" + filetype)
    mask_name = os.path.join(save_dir, filename + "_mask" + filetype)
    png_name = os.path.join(save_dir, filename + "_vis" + filetype)

    # cv.imwrite(png_name,cimg,[int(cv.IMWRITE_JPEG_QUALITY),100])
    cv.imwrite(png_name, cimg)
    cv.imwrite(vis_mask_name, mask)
    cv.imwrite(mask_name, mask // 255)

def crop(cnt, crop_image, save_image_dir, basename):
    image_name = os.path.join(save_image_dir, basename + "_" + str(cnt) + ".png")
    cv.imwrite(image_name, crop_image)