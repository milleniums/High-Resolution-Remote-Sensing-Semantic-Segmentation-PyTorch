import numpy as np
import pandas as pd
import cv2 as cv
from PIL import Image
from tqdm import tqdm
import os
from argparse import ArgumentParser
from multiprocessing import Pool
from utils import *
Image.MAX_IMAGE_PIXELS = 1000000000000


if __name__ == "__main__":
    parser = ArgumentParser(description="")
    parser.add_argument("-image-n", type=int)
    parser.add_argument("-image-path", type=str, default=r'E:\data_xiongan\src\GF2_PMS2_E116.0_N39.1_20170302_L1A0002214760-MSS2.jpg')
    parser.add_argument("-save-dir", type=str, default=r"./data/")
    arg = parser.parse_args()
    image_n = arg.image_n
    image_path = arg.image_path

    save_image_dir = os.path.join(arg.save_dir, "image")
    stride = 256
    target_size = (512, 512)

    if not os.path.isdir(save_image_dir): os.makedirs(save_image_dir)
    root_dir, filename = os.path.split(image_path)
    basename, filetype = os.path.splitext(filename)

    image = np.asarray(Image.open(image_path))

    cnt = 0
    csv_pos_list = []

    # 填充外边界至步长整数倍
    target_w, target_h = target_size
    h, w = image.shape[0], image.shape[1]
    new_w = (w // target_w) * target_w if (w // target_w == 0) else (w // target_w + 1) * target_w
    new_h = (h // target_h) * target_h if (h // target_h == 0) else (h // target_h + 1) * target_h
    image = cv.copyMakeBorder(image, 0, new_h - h, 0, new_w - w, cv.BORDER_CONSTANT, 0)

    # 填充1/2 stride长度的外边框
    h, w = image.shape[0], image.shape[1]
    new_w, new_h = w + stride, h + stride
    image = cv.copyMakeBorder(image, stride // 2, stride // 2, stride // 2, stride // 2, cv.BORDER_CONSTANT, 0)

    h, w = image.shape[0], image.shape[1]
    P = Pool(16)
    for i in tqdm(range(w // stride - 1)):
        for j in range(h // stride - 1):
            topleft_x = i * stride
            topleft_y = j * stride
            crop_image = image[topleft_y:topleft_y + target_h, topleft_x:topleft_x + target_w]

            if crop_image.shape[:2] != (target_h, target_h):
                print(topleft_x, topleft_y, crop_image.shape)

            if np.sum(crop_image) == 0:
                pass
            else:
                P.apply_async(crop, (cnt, crop_image, save_image_dir, basename))
                csv_pos_list.append([basename + "_" + str(cnt) + ".png", topleft_x, topleft_y, topleft_x + target_w,
                                     topleft_y + target_h])
                cnt += 1
    csv_pos_list = pd.DataFrame(csv_pos_list)
    csv_pos_list.to_csv(os.path.join(arg.save_dir, basename + ".csv"), header=None, index=None)
    P.close()
    P.join()