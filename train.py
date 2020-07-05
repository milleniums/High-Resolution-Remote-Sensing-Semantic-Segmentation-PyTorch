import argparse
import time
import os
import json
from dataset import RSDataset
import sync_transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch
from models.deeplabv3_version_1.deeplabv3 import DeepLabV3 as model1
from models.deeplabv3_version_2.deeplabv3 import DeepLabV3 as model2
from libs import average_meter, metric
from torch.autograd import Variable
import numpy as np
from tqdm import tqdm
from prettytable import PrettyTable
import torchvision
from torchvision import transforms
from palette import colorize_mask
from PIL import Image
from collections import OrderedDict
from tensorboardX import SummaryWriter

def parse_args():
    parser = argparse.ArgumentParser(description="RemoteSensingSegmentation by PyTorch")
    # dataset
    parser.add_argument('--dataset-name', type=str, default='five')
    parser.add_argument('--train-data-root', type=str, default='../data_5_6_21/train')
    parser.add_argument('--val-data-root', type=str, default='../data_5_6_21/val')
    parser.add_argument('--train-batch-size', type=int, default=32, metavar='N', help='batch size for training (default:16)')
    parser.add_argument('--val-batch-size', type=int, default=32, metavar='N', help='batch size for testing (default:16)')
    # output_save_path
    parser.add_argument('--experiment-start-time', type=str, default=time.strftime('%m-%d-%H:%M:%S', time.localtime(time.time())))
    parser.add_argument('--save-pseudo-data-path', type=str, default='/root/data/others/yaoganbisai/pseudo_data')
    # augmentation
    parser.add_argument('--base-size', type=int, default=512, help='base image size')
    parser.add_argument('--crop-size', type=int, default=512, help='crop image size')
    parser.add_argument('--flip-ratio', type=float, default=0.5)
    parser.add_argument('--resize-scale-range', type=str, default='0.5, 2.0')
    # model
    parser.add_argument('--model', type=str, default='deeplabv3_version_1', help='model name')
    parser.add_argument('--backbone', type=str, default='resnet50', help='backbone name')
    parser.add_argument('--pretrained', action='store_true', default=True)
    parser.add_argument('--n-blocks', type=str, default='3, 4, 23, 3', help='')
    parser.add_argument('--output-stride', type=int, default=16, help='')
    parser.add_argument('--multi-grids', type=str, default='1, 1, 1', help='')
    parser.add_argument('--deeplabv3-atrous-rates', type=str, default='6, 12, 18', help='')
    parser.add_argument('--deeplabv3-no-global-pooling', action='store_true', default=False)
    parser.add_argument('--deeplabv3-use-deformable-conv', action='store_true', default=False)
    parser.add_argument('--no-syncbn', action='store_true', default=False, help='using Synchronized Cross-GPU BatchNorm')
    # criterion
    parser.add_argument('--class-loss-weight', type=list, default=
    # [0.007814952234152803, 0.055862295151291756, 0.029094606950899726, 0.03104357983254851, 0.22757710412943985, 0.19666243636646102, 0.6088052968747066, 0.15683966777104494, 0.5288489922602664, 0.21668940382940433, 0.04310240828376457, 0.18284053575941367, 0.571096349549462, 0.32601488184885147, 0.45384359272537766, 1.0])
    # [0.007956167959807792, 0.05664417300631733, 0.029857031694750392, 0.03198534634969046, 0.2309102255169529,
    #  0.19627322641039702, 0.6074939752850792, 0.16196525436190998, 0.5396602408824741, 0.22346488456565283,
    #  0.04453628275090391, 0.18672995330033487, 0.5990724459491834, 0.33183887346397484, 0.47737597643193597, 1.0]
    [0.008728536232175135, 0.05870821984204281, 0.030766985878693004, 0.03295408432939304, 0.2399409412190348, 0.20305583055639448, 0.6344888568739531, 0.16440413437125656, 0.5372260524694122, 0.22310945250778813, 0.04659596810284655, 0.19246378709444723, 0.6087430986295436, 0.34431415558778183, 0.4718853977371564, 1.0])

    # loss
    parser.add_argument('--loss-names', type=str, default='cross_entropy')
    parser.add_argument('--classes-weight', type=str, default=None)
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='momentum (default:0.9)')
    parser.add_argument('--weight-decay', type=float, default=0.0001, metavar='M', help='weight-decay (default:1e-4)')
    # optimizer
    parser.add_argument('--optimizer-name', type=str, default='Adadelta')
    # learning_rate
    parser.add_argument('--base-lr', type=float, default=0.1, metavar='M', help='')
    # environment
    parser.add_argument('--use-cuda', action='store_true', default=True, help='using CUDA training')
    parser.add_argument('--num-GPUs', type=int, default=2, help='numbers of GPUs')
    parser.add_argument('--num_workers', type=int, default=4)
    # validation
    parser.add_argument('--eval', action='store_true', default=False, help='evaluation only')
    parser.add_argument('--no-val', action='store_true', default=False)

    parser.add_argument('--best-kappa', type=float, default=0)

    parser.add_argument('--total-epochs', type=int, default=12, metavar='N', help='number of epochs to train (default: 120)')
    parser.add_argument('--start-epoch', type=int, default=0, metavar='N', help='start epoch (default:0)')

    parser.add_argument('--resume-path', type=str, default=None)

    args = parser.parse_args()
    directory = "work_dirs/%s/%s/%s/%s/" % (args.dataset_name, args.model, args.backbone, args.experiment_start_time)
    args.directory = directory
    if not os.path.exists(directory):
        os.makedirs(directory)
    config_file = os.path.join(directory, 'config.json')
    with open(config_file, 'w') as file:
        json.dump(vars(args), file, indent=4)

    if args.use_cuda:
        print('Numbers of GPUs:', args.num_GPUs)
    else:
        print("Using CPU")
    return args

class DeNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor


class Trainer(object):
    def __init__(self, args):
        self.args = args
        resize_scale_range = [float(scale) for scale in args.resize_scale_range.split(',')]
        sync_transform = sync_transforms.Compose([
            sync_transforms.RandomScale(args.base_size, args.crop_size, resize_scale_range),
            sync_transforms.RandomFlip(args.flip_ratio)
        ])
        self.resore_transform = transforms.Compose([
            DeNormalize([.485, .456, .406], [.229, .224, .225]),
            transforms.ToPILImage()
        ])
        self.visualize = transforms.Compose([transforms.ToTensor()])
        
        class_name = args.dataset_name
        if class_name == 'fifteen': from class_names import fifteen_classes
        if class_name == 'five': from class_names import five_classes
        self.train_dataset = RSDataset(class_name, root=args.train_data_root, mode='train', sync_transforms=sync_transform)
        self.train_loader = DataLoader(dataset=self.train_dataset,
                                       batch_size=args.train_batch_size,
                                       num_workers=args.num_workers,
                                       shuffle=True,
                                       drop_last=True)
        print('class names {}.'.format(self.train_dataset.class_names))
        print('Number samples {}.'.format(len(self.train_dataset)))
        if not args.no_val:
            val_data_set = RSDataset(class_name, root=args.val_data_root, mode='val', sync_transforms=None)
            self.val_loader = DataLoader(dataset=val_data_set,
                                         batch_size=args.val_batch_size,
                                         num_workers=args.num_workers,
                                         shuffle=False,
                                         drop_last=True)
        self.num_classes = len(self.train_dataset.class_names)
        print("类别数：", self.num_classes)
        self.class_loss_weight = torch.Tensor(args.class_loss_weight)
        self.criterion = nn.CrossEntropyLoss(weight=None, reduction='mean', ignore_index=-1).cuda()

        n_blocks = args.n_blocks
        n_blocks = [int(b) for b in n_blocks.split(',')]
        atrous_rates = args.deeplabv3_atrous_rates
        atrous_rates = [int(s) for s in atrous_rates.split(',')]
        multi_grids = args.multi_grids
        multi_grids = [int(g) for g in multi_grids.split(',')]

        if args.model == 'deeplabv3_version_1':
            model = model1(num_classes=self.num_classes)# dilate_rate=[6,12,18]
            # resume
            if args.resume_path:
                state_dict = torch.load('/root/data/others/yaoganbisai/code_6_7/work_dirs/rssrai2019_semantic_segmentation/deeplabv3_version_1/resnet50/06-11-17:37:52/epoch_0_acc_0.42195_kappa_0.69184.pth')
                new_state_dict = OrderedDict()
                for k, v in state_dict.items():
                    name = k[7:]
                    new_state_dict[name] = v
                model.load_state_dict(new_state_dict)
        if args.model == 'deeplabv3_version_2':
            model = model2(num_classes=self.num_classes,
                           n_blocks=n_blocks,
                           atrous_rates=atrous_rates,
                           multi_grids=multi_grids,
                           output_stride=args.output_stride)
        if args.model == 'hdc':
            from models.HDC.duc_hdc import ResNetDUC
            model = ResNetDUC(num_classes=self.num_classes)
        # print(model)



        if args.use_cuda:
            model = model.cuda()
            self.model = nn.DataParallel(model, device_ids=list(range(torch.cuda.device_count())))

        # SGD不work，Adadelta出奇的好？
        if args.optimizer_name == 'Adadelta':
            self.optimizer = torch.optim.Adadelta(model.parameters(),
                                                  lr=args.base_lr,
                                                  weight_decay=args.weight_decay)
        if args.optimizer_name == 'Adam':
            self.optimizer = torch.optim.Adam(model.parameters(),
                                              lr=args.base_lr)
        if args.optimizer_name == 'SGD':
            self.optimizer = torch.optim.SGD(params=model.parameters(),
                                             lr=args.base_lr,
                                             momentum=args.momentum,
                                             weight_decay=args.weight_decay)

        self.max_iter = args.total_epochs * len(self.train_loader)
        self.save_pseudo_data_path = args.save_pseudo_data_path
        self.mixup_transform = sync_transforms.Mixup()


    def training(self, epoch):
        self.model.train()# 把module设成训练模式，对Dropout和BatchNorm有影响

        train_loss = average_meter.AverageMeter()

        curr_iter = epoch * len(self.train_loader)
        lr = self.args.base_lr * (1 - float(curr_iter) / self.max_iter) ** 0.9
        conf_mat = np.zeros((self.num_classes, self.num_classes)).astype(np.int64)
        tbar = tqdm(self.train_loader)
        for index, data in enumerate(tbar):
            # assert data[0].size()[2:] == data[1].size()[1:]
            # data = self.mixup_transform(data, epoch)
            imgs = Variable(data[0])
            masks = Variable(data[1])

            if self.args.use_cuda:
                imgs = imgs.cuda()
                masks = masks.cuda()
            self.optimizer.zero_grad()
            outputs = self.model(imgs)
            # torch.max(tensor, dim)：指定维度上最大的数，返回tensor和下标
            _, preds = torch.max(outputs, 1)
            preds = preds.data.cpu().numpy().squeeze().astype(np.uint8)


            loss = self.criterion(outputs, masks)

            train_loss.update(loss, self.args.train_batch_size)
            writer.add_scalar('train_loss', train_loss.avg, curr_iter)
            loss.backward()
            self.optimizer.step()

            tbar.set_description('epoch {}, training loss {}, with learning rate {}.'.format(
                epoch, train_loss.avg, lr
            ))
            masks = masks.data.cpu().numpy().squeeze().astype(np.uint8)
            conf_mat += metric.confusion_matrix(pred=preds.flatten(),
                                                label=masks.flatten(),
                                                num_classes=self.num_classes)
        train_acc, train_acc_per_class, train_acc_cls, train_IoU, train_mean_IoU, train_kappa = metric.evaluate(conf_mat)
        writer.add_scalar(tag='train_loss_per_epoch', scalar_value=train_loss.avg, global_step=epoch, walltime=None)
        writer.add_scalar(tag='train_acc', scalar_value=train_acc, global_step=epoch, walltime=None)
        writer.add_scalar(tag='train_kappa', scalar_value=train_kappa, global_step=epoch, walltime=None)
        table = PrettyTable(["序号", "名称", "acc", "IoU"])
        for i in range(self.num_classes):
            table.add_row([i, self.train_dataset.class_names[i], train_acc_per_class[i], train_IoU[i]])
        print(table)
        print("train_acc:", train_acc)
        print("train_mean_IoU:", train_mean_IoU)
        print("kappa:", train_kappa)

    def validating(self, epoch):
        self.model.eval()# 把module设成预测模式，对Dropout和BatchNorm有影响
        conf_mat = np.zeros((self.num_classes, self.num_classes)).astype(np.int64)
        tbar = tqdm(self.val_loader)
        for index, data in enumerate(tbar):
            # assert data[0].size()[2:] == data[1].size()[1:]
            imgs = Variable(data[0])
            masks = Variable(data[1])

            if self.args.use_cuda:
                imgs = imgs.cuda()
                masks = masks.cuda()
            self.optimizer.zero_grad()
            outputs = self.model(imgs)
            _, preds = torch.max(outputs, 1)
            preds = preds.data.cpu().numpy().squeeze().astype(np.uint8)
            masks = masks.data.cpu().numpy().squeeze().astype(np.uint8)
            score = _.data.cpu().numpy()
            val_visual = []
            for i in range(score.shape[0]):
                num_score = np.sum(score[i] > 0.9)
                if num_score > 0.9*(512*512):
                    img_pil = self.resore_transform(data[0][i])
                    preds_pil = Image.fromarray(preds[i].astype(np.uint8)).convert('L')
                    pred_vis_pil = colorize_mask(preds[i])
                    gt_vis_pil = colorize_mask(data[1][i].numpy())

                    val_visual.extend([self.visualize(img_pil.convert('RGB')),
                                       self.visualize(gt_vis_pil.convert('RGB')),
                                       self.visualize(pred_vis_pil.convert('RGB'))])

                    dir_list = ['rgb', 'label', 'vis_label', 'gt']
                    rgb_save_path = os.path.join(self.save_pseudo_data_path, dir_list[0], str(epoch))
                    label_save_path = os.path.join(self.save_pseudo_data_path, dir_list[1], str(epoch))
                    vis_save_path = os.path.join(self.save_pseudo_data_path, dir_list[2], str(epoch))
                    gt_save_path = os.path.join(self.save_pseudo_data_path, dir_list[3], str(epoch))

                    path_list = [rgb_save_path, label_save_path, vis_save_path, gt_save_path]
                    for path in range(4):
                        if not os.path.exists(path_list[path]):
                            os.makedirs(path_list[path])
                    img_pil.save(os.path.join(path_list[0], 'img_batch_%d_%d.jpg' % (index, i)))
                    preds_pil.save(os.path.join(path_list[1], 'label_%d_%d.png' % (index, i)))
                    pred_vis_pil.save(os.path.join(path_list[2], 'vis_%d_%d.png' % (index, i)))
                    gt_vis_pil.save(os.path.join(path_list[3], 'gt_%d_%d.png' % (index, i)))
            if val_visual:
                val_visual = torch.stack(val_visual, 0)
                val_visual = torchvision.utils.make_grid(tensor=val_visual,
                                                         nrow=3,
                                                         padding=5,
                                                         normalize=False,
                                                         range=None,
                                                         scale_each=False,
                                                         pad_value=0)
                writer.add_image(tag='pres&GTs', img_tensor=val_visual, global_step=None, walltime=None)
            conf_mat += metric.confusion_matrix(pred=preds.flatten(),
                                                label=masks.flatten(),
                                                num_classes=self.num_classes)
        val_acc, val_acc_per_class, val_acc_cls, val_IoU, val_mean_IoU, val_kappa = metric.evaluate(conf_mat)
        writer.add_scalars(main_tag='val_single_acc',
                           tag_scalar_dict={self.train_dataset.class_names[i]: val_acc_per_class[i] for i in range(len(self.train_dataset.class_names))},
                           global_step=epoch, walltime=None)
        writer.add_scalars(main_tag='val_single_iou',
                           tag_scalar_dict={self.train_dataset.class_names[i]: val_IoU[i] for i in range(len(self.train_dataset.class_names))},
                           global_step=epoch, walltime=None)
        writer.add_scalar('val_acc', val_acc, epoch)
        writer.add_scalar('val_acc_cls', val_acc_cls, epoch)
        writer.add_scalar('val_mean_IoU', val_mean_IoU, epoch)
        writer.add_scalar('val_kappa', val_kappa, epoch)
        model_name = 'epoch_%d_acc_%.5f_kappa_%.5f' % (epoch, val_acc, val_kappa)
        if val_kappa > self.args.best_kappa:
            torch.save(self.model.state_dict(), os.path.join(self.args.directory, model_name+'.pth'))
            self.args.best_kappa = val_kappa
        table = PrettyTable(["序号", "名称", "acc", "IoU"])
        for i in range(self.num_classes):
            table.add_row([i, self.train_dataset.class_names[i], val_acc_per_class[i], val_IoU[i]])
        print(table)
        print("val_acc:", val_acc)
        print("val_mean_IoU:", val_mean_IoU)
        print("kappa:", val_kappa)


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    args = parse_args()
    writer = SummaryWriter(args.directory)
    trainer = Trainer(args)
    if args.eval:
        # print("Evaluating model:", args.resume)
        trainer.validating(epoch=0)
    else:
        print("Starting Epoch:", args.start_epoch)
    for epoch in range(args.start_epoch, args.total_epochs):
        trainer.training(epoch)
        if not trainer.args.no_val:
            trainer.validating(epoch)