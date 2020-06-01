import cv2
import numpy as np
import os

def RandomCut(image_path, label_path, cutsize_h, cutsize_w, stride, filename):
    img = cv2.imread(image_path)
    label = cv2.imread(label_path)
    assert img.shape == label.shape
    print(img.shape)
    h, w = img.shape[0], img.shape[1]
    # 对大图进行padding
    h_pad_cutsize = h if (h // cutsize_h == 0) else (w // cutsize_h + 1) * cutsize_h
    w_pad_cutsize = w if (w // cutsize_w == 0) else (w // cutsize_w + 1) * cutsize_w
    img = cv2.copyMakeBorder(img, 0, h_pad_cutsize - h, 0, w_pad_cutsize - w, cv2.BORDER_CONSTANT, 0)
    label = cv2.copyMakeBorder(label, 0, h_pad_cutsize - h, 0, w_pad_cutsize - w, cv2.BORDER_CONSTANT, 0)

    h_pad_stride, w_pad_stride = h_pad_cutsize + stride, w_pad_cutsize + stride
    img = cv2.copyMakeBorder(img, stride // 2, stride // 2, stride // 2, stride // 2, cv2.BORDER_CONSTANT, 0)
    label = cv2.copyMakeBorder(label, stride // 2, stride // 2, stride // 2, stride // 2, cv2.BORDER_CONSTANT, 0)

    index = 0
    for i in range(0, h_pad_stride // cutsize_h - 1):
        for j in range(0, w_pad_stride // cutsize_w - 1):
            index = index + 1
            topleft_y = i * stride
            topleft_x = j * stride
            img_cut = img[topleft_y:topleft_y + cutsize_h, topleft_x:topleft_x + cutsize_w, :]
            label_cut = label[topleft_y:topleft_y + cutsize_h, topleft_x:topleft_x + cutsize_w, :]
            # 检查大小
            if img_cut.shape[:2] != (cutsize_h, cutsize_w):
                print(topleft_x, topleft_y, img_cut.shape)
            # 过滤掉全部是黑色图片
            if np.sum(img_cut) == 0:
                continue

            image_save_path = os.path.join(output_rgb_path, filename.replace('.tif', '_%03d.tif' % index))
            label_save_path = os.path.join(output_label_path, filename.replace('.tif', '_label_%03d.png' % index))

            cv2.imwrite(image_save_path, img_cut)
            cv2.imwrite(label_save_path, label_cut)
    print(filename)

if __name__ == "__main__":
    patch_size_list = [512, 768, 1024, 1280]
    input_dir_list = ['rgb', 'label']
    patch_size = 512
    stride = 512
    input_path = r"C:\Users\hekai\Desktop\gid_data\val"
    output_path = r"C:\Users\hekai\Desktop\gid_data\val_data\{}".format(patch_size)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    output_rgb_path = os.path.join(output_path, 'rgb')
    if not os.path.exists(output_rgb_path):
        os.mkdir(output_rgb_path)
    output_label_path = os.path.join(output_path, 'label')
    if not os.path.exists(output_label_path):
        os.mkdir(output_label_path)

    for filename in os.listdir(os.path.join(input_path, input_dir_list[0])):
        image_path = os.path.join(input_path, input_dir_list[0], filename)
        label_path = os.path.join(input_path, input_dir_list[1], filename.replace('.tif', '_label.png'))
        RandomCut(image_path, label_path, cutsize_h=patch_size, cutsize_w=patch_size, stride=stride, filename=filename)