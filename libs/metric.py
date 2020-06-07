import numpy as np


def confusion_matrix(pred, label, num_classes):
    mask = (label >= 0) & (label < num_classes)
    conf_mat = np.bincount(num_classes * label[mask].astype(int) + pred[mask], minlength=num_classes**2).reshape(num_classes, num_classes)
    return conf_mat

def evaluate(conf_mat):
    acc = np.diag(conf_mat).sum() / conf_mat.sum()
    acc_per_class = np.diag(conf_mat) / conf_mat.sum(axis=1)
    acc_cls = np.nanmean(acc_per_class)

    IoU = np.diag(conf_mat) / (conf_mat.sum(axis=1) + conf_mat.sum(axis=0) - np.diag(conf_mat))
    mean_IoU = np.nanmean(IoU)

    # 求kappa
    pe = np.dot(np.sum(conf_mat, axis=0), np.sum(conf_mat, axis=1)) / (conf_mat.sum()**2)
    kappa = (acc - pe) / (1 - pe)
    return acc, acc_per_class, acc_cls, IoU, mean_IoU, kappa