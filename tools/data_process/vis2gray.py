import json, os
import numpy as np
from PIL import Image
'''
总体思路就是逐一读彩色的label，通过Palette.json文件找到映射关系
例如：
{ "0": [0,0,0],
  "1": [0,200,0],
  "2": [150,250,0],
  "3": [150,200,150],
  "4": [200,0,200],
  "5": [150,0,250],
  "6": [150,150,250],
  "7": [250,200,0],
  "8": [200,200,0],
  "9": [200,0,0],
  "10": [250,0,150],
  "11": [200,150,150],
  "12": [250,150,150],
  "13": [0,0,200],
  "14": [0,150,200],
  "15": [0,200,250]
}
那么，如果读的像素是[0,200,0]，则将1作为相应位置的灰度值，以此类推。
'''

def get_label_from_palette(label_img, palette_file='Palette.json'):
    with open(palette_file, 'r') as fp:
        text = json.load(fp)
        list_index = list(text.keys())
        list_value = list(text.values())
        label = np.zeros((label_img.shape[0], label_img.shape[1]), dtype=np.uint8)
        for i in range(label_img.shape[0]):
            for j in range(label_img.shape[1]):
                if not list(label_img[i, j, :]) in list(text.values()):
                    print(label_img[i, j, :])
                    print(i, j)
                    label[i, j] = 15
                    continue
                # assert list(label_img[i, j, :]) in list(text.values())
                list_color = list(label_img[i, j, :])
                label[i, j] = int(list_index[list_value.index(list_color)])
        return label

def main(path):
    for pic in os.listdir(path):
        if 'label' in pic:
            print(pic)
            label = Image.open(path + '/' + pic)
            label = np.asarray(label)
            label = get_label_from_palette(label)
            label = Image.fromarray(label)
            label.save(path + '/' + pic[:-4] + '.png')

if __name__ == '__main__':
    train_path = r'E:\yaogan\dataProcess\data\train\vis_label'
    main(train_path)
    val_path = r'E:\yaogan\dataProcess\data\val\vis_label'
    main(val_path)