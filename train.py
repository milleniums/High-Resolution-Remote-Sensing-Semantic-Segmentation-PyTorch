import argparse
import time
import os
import json
from dataset import RSDataset
import sync_transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch
from models.deeplabv3 import DeepLabV3
from libs import average_meter, lr_scheduler, metric
from torch.autograd import Variable
import numpy as np
from tqdm import tqdm
from prettytable import PrettyTable
from class_names import gid_classes


def parse_args():
    parser = argparse.ArgumentParser(description="RemoteSensingSegmentation by PyTorch")
    # dataset
    parser.add_argument('--dataset-name', type=str, default='rssrai2019_semantic_segmentation')
    parser.add_argument('--train-data-root', type=str, default='../data_6_5/train')
    parser.add_argument('--val-data-root', type=str, default='../data_6_5/val')
    parser.add_argument('--train-batch-size', type=int, default=16, metavar='N', help='batch size for training (default:16)')
    parser.add_argument('--val-batch-size', type=int, default=16, metavar='N', help='batch size for testing (default:16)')
    # output_save_path
    parser.add_argument('--experiment-start-time', type=str, default=time.strftime('%m-%d-%H:%M:%S', time.localtime(time.time())))
    # augmentation
    parser.add_argument('--base-size', type=int, default=552, help='base image size')
    parser.add_argument('--crop-size', type=int, default=512, help='crop image size')
    parser.add_argument('--flip-ratio', type=float, default=0.5)
    parser.add_argument('--resize-scale-range', type=str, default='0.5, 2.0')
    # model
    parser.add_argument('--model', type=str, default='deeplabv3', help='model name')
    parser.add_argument('--backbone', type=str, default='resnet50', help='backbone name')
    parser.add_argument('--pretrained', action='store_true', default=True)
    parser.add_argument('--n-blocks', type=str, default='3, 4, 6, 3', help='')
    parser.add_argument('--output-stride', type=int, default=16, help='')
    parser.add_argument('--multi-grids', type=str, default='1, 2, 1', help='')
    parser.add_argument('--deeplabv3-atrous-rates', type=str, default='12, 24, 36', help='')
    parser.add_argument('--deeplabv3-no-global-pooling', action='store_true', default=False)
    parser.add_argument('--deeplabv3-use-deformable-conv', action='store_true', default=False)
    parser.add_argument('--no-syncbn', action='store_true', default=False, help='using Synchronized Cross-GPU BatchNorm')
    # criterion
    parser.add_argument('--class-loss-weight', type=list, default=[0.03627257601969112,
                                                                   0.029094606950899726,
                                                                   0.03104357983254851,
                                                                   0.22757710412943985,
                                                                   0.19666243636646102,
                                                                   0.6088052968747066,
                                                                   0.15683966777104494,
                                                                   0.5288489922602664,
                                                                   0.21668940382940433,
                                                                   0.04310240828376457,
                                                                   0.18284053575941367,
                                                                   0.571096349549462,
                                                                   0.32601488184885147,
                                                                   0.45384359272537766,
                                                                   1.0,
                                                                   0.008453659908942788])
    # loss
    parser.add_argument('--loss-names', type=str, default='cross_entropy')
    parser.add_argument('--classes-weight', type=str, default=None)
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='momentum (default:0.9)')
    parser.add_argument('--weight-decay', type=float, default=0.0001, metavar='M', help='weight-decay (default:1e-4)')
    # optimizer
    parser.add_argument('--optimizer-name', type=str, default='adam')
    # learning_rate
    parser.add_argument('--base-lr', type=float, default=0.1, metavar='M', help='')
    # environment
    parser.add_argument('--use-cuda', action='store_true', default=True, help='using CUDA training')
    parser.add_argument('--num-GPUs', type=int, default=2, help='numbers of GPUs')
    parser.add_argument('--num_workers', type=int, default=2)
    # validation
    parser.add_argument('--eval', action='store_true', default=False, help='evaluation only')
    parser.add_argument('--no-val', action='store_true', default=False)

    parser.add_argument('--total-epochs', type=int, default=120, metavar='N', help='number of epochs to train (default: 120)')
    parser.add_argument('--start-epoch', type=int, default=0, metavar='N', help='start epoch (default:0)')

    args = parser.parse_args()
    directory = "work_dirs/%s/%s/%s/%s/" % (
    args.dataset_name, args.model, args.backbone, args.experiment_start_time)
    if not os.path.exists(directory):
        os.makedirs(directory)
    config_file = os.path.join(directory, 'config,json')
    with open(config_file, 'w') as file:
        json.dump(vars(args), file, indent=4)

    if args.use_cuda:
        print('Numbers of GPUs:', args.num_GPUs)
    else:
        print("Using CPU")

    return args


class Trainer(object):
    def __init__(self, args):
        self.args = args
        resize_scale_range = [float(scale) for scale in args.resize_scale_range.split(',')]
        sync_transform = sync_transforms.Compose([
            sync_transforms.RandomScale(args.base_size, args.crop_size, resize_scale_range),
            sync_transforms.RandomFlip(args.flip_ratio)
        ])
        train_dataset = RSDataset(root=args.train_data_root, mode='train', sync_transforms=sync_transform)
        self.train_loader = DataLoader(dataset=train_dataset,
                                 batch_size=args.train_batch_size,
                                 num_workers=args.num_workers,
                                 shuffle=True,
                                 drop_last=True)
        print('class names {}.'.format(train_dataset.classes()))
        print('Number samples {}.'.format(len(train_dataset)))
        if not args.no_val:
            val_data_set = RSDataset(root=args.val_data_root, mode='val', sync_transforms=None)
            self.val_loader = DataLoader(dataset=val_data_set,
                                   batch_size=args.val_batch_size,
                                   num_workers=args.num_workers,
                                   shuffle=False,
                                   drop_last=True)
        self.class_loss_weight = args.class_loss_weight
        # if not self.class_loss_weight is None:
        #     self.class_loss_weight = [float(w) for w in self.class_loss_weight.split(',')]
        # assert len(self.class_loss_weight) == len(train_dataset.classes())

        self.class_loss_weight = torch.Tensor(self.class_loss_weight)
        self.criterion = nn.CrossEntropyLoss(weight=self.class_loss_weight, reduction='mean', ignore_index=-1).cuda()

        n_blocks = args.n_blocks
        n_blocks = [int(b) for b in n_blocks.split(',')]
        atrous_rates = args.deeplabv3_atrous_rates
        atrous_rates = [int(s) for s in atrous_rates.split(',')]
        multi_grids = args.multi_grids
        multi_grids = [int(g) for g in multi_grids.split(',')]

        model = DeepLabV3(num_classes=len(train_dataset.classes()),
                          n_blocks=n_blocks,
                          atrous_rates=atrous_rates,
                          multi_grids=multi_grids,
                          output_stride=args.output_stride)
        print(model)
        if args.use_cuda:
            model = model.cuda()
            self.model = nn.DataParallel(model, device_ids=list(range(torch.cuda.device_count())))

        self.optimizer = torch.optim.SGD(params=model.parameters(),
                                         lr=args.base_lr,
                                         momentum=args.momentum,
                                         weight_decay=args.weight_decay)
        # self.lr_schedualer = lr_scheduler.PolynomialLR()
        self.max_iter = args.total_epochs * len(self.train_loader)
        self.num_classes = len(train_dataset.classes())
        print(self.num_classes)

    def training(self, epoch):
        self.model.train()

        train_loss = average_meter.AverageMeter()

        curr_iter = epoch * len(self.train_loader)
        lr = self.args.base_lr * (1 - float(curr_iter) / self.max_iter) ** 0.9
        # self.optimizer.param_group['lr'] = lr

        conf_mat = np.zeros((self.num_classes, self.num_classes)).astype(np.int64)
        tbar = tqdm(self.train_loader)
        for index, (imgs, masks) in enumerate(tbar):
            assert imgs.size()[2:] == masks.size()[1:]
            imgs = Variable(imgs)
            masks = Variable(masks)

            if self.args.use_cuda:
                imgs = imgs.cuda()
                masks = masks.cuda()
            self.optimizer.zero_grad()
            outputs = self.model(imgs)
            _, preds = torch.max(outputs, 1)
            preds = preds.data.cpu().numpy().squeeze().astype(np.uint8)

            loss = self.criterion(outputs, masks)
            train_loss.update(loss, self.args.train_batch_size)
            loss.backward()
            self.optimizer.step()

            tbar.set_description('epoch {}, training loss {}, with learning rate {}.'.format(
                epoch, train_loss, lr
            ))
            masks = masks.data.cpu().numpy().squeeze().astype(np.uint8)
            conf_mat += metric.confusion_matrix(pred=preds.flatten(), label=masks.flatten(), num_classes=self.num_classes)
        train_acc, train_acc_per_class, train_acc_cls, train_IoU, train_mean_IoU, train_kappa = metric.evaluate(conf_mat)
        table = PrettyTable(["序号", "名称", "acc", "IoU"])
        for i in range(self.num_classes):
            table.add_row([i, gid_classes()[i], train_acc_per_class[i], train_IoU[i]])
        print(table)
        print("train_acc:", train_acc)
        print("train_mean_IoU:", train_mean_IoU)
        print("kappa:", train_kappa)

    def validating(self, epoch):
        self.model.eval()

        conf_mat = np.zeros((self.num_classes, self.num_classes)).astype(np.int64)
        tbar = tqdm(self.val_loader)
        for index, (imgs, masks) in enumerate(tbar):
            assert imgs.size()[2:] == masks.size()[1:]
            imgs = Variable(imgs)
            masks = Variable(masks)

            if self.args.use_cuda:
                imgs = imgs.cuda()
                masks = masks.cuda()
            self.optimizer.zero_grad()
            outputs = self.model(imgs)
            _, preds = torch.max(outputs, 1)
            preds = preds.data.cpu().numpy().squeeze().astype(np.uint8)

            masks = masks.data.cpu().numpy().squeeze().astype(np.uint8)
            conf_mat += metric.confusion_matrix(pred=preds.flatten(), label=masks.flatten(),
                                                num_classes=self.num_classes)
        val_acc, val_acc_per_class, val_acc_cls, val_IoU, val_mean_IoU, val_kappa = metric.evaluate(
            conf_mat)
        table = PrettyTable(["序号", "名称", "acc", "IoU"])
        for i in range(self.num_classes):
            table.add_row([i, gid_classes()[i], val_acc_per_class[i], val_IoU[i]])
        print(table)
        print("val_acc:", val_acc)
        print("val_mean_IoU:", val_mean_IoU)
        print("kappa:", val_kappa)


if __name__ == "__main__":
    args = parse_args()
    trainer = Trainer(args)
    if args.eval:
        print("Evaluating model:", args.resume)
    else:
        print("Starting Epoch:", args.start_epoch)
    for epoch in range(args.start_epoch, args.total_epochs):
        trainer.training(epoch)
        if not trainer.args.no_val:
            trainer.validating(epoch)









