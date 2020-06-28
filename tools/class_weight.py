from dataset import RSDataset
from class_names import *
from prettytable import PrettyTable
from torch.utils.data import DataLoader
import numpy as np


if __name__ == "__main__":
    # Dataset
    dataset = RSDataset(root='../../data_6_10/train', mode='train')
    num_classes = len(dataset.class_names)
    # DataLoader
    train_loader = DataLoader(dataset, batch_size=1)

    num_pixel = np.zeros((1, num_classes)).squeeze(0).astype(np.uint64)
    for idx, data in enumerate(train_loader):
        img, mask = data
        h, w = mask.shape[1], mask.shape[2]
        mask = mask.numpy().reshape(h * w * 1, 1).squeeze(1).astype(np.uint8)
        num_pixel_per_batch = np.bincount(mask, minlength=num_classes)
        num_pixel = num_pixel_per_batch + num_pixel

    proportion_pixel = num_pixel / num_pixel.sum()
    weight_temp = 1 / proportion_pixel
    max_in_weight = max(weight_temp)
    # item_index = np.argwhere(weight_1 == max_in_weight)
    # weight_1[item_index] = 0
    # max_in_weight = max(weight_1)
    weight = weight_temp / max_in_weight

    table = PrettyTable(["序号", "名称", "像素数", "惩罚权重"])
    for i in range(len(gid_classes())):
        table.add_row([i, gid_classes()[i], num_pixel[i], weight[i]])
    print(table)
    print('weight in criterion:', list(weight))