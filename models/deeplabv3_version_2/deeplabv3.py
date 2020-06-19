import torch
import torch.nn as nn
import torch.nn.functional as F
from models.deeplabv3_version_2.resnet import ResNet50
from models.deeplabv3_version_2.aspp import _ASPP
from models.deeplabv3_version_2.component import _Stem, _ResLayer, _ConvBnReLU

ch = [64 * 2 ** p for p in range(6)]
# ch = [64, 128, 256, 512, 1024, 2048]

class DeepLabV3(nn.Module):
    def __init__(self, num_classes=16, n_blocks=[3,4,6,3], atrous_rates=[6,12,18], multi_grids=[1,2,1], output_stride=16):
        super(DeepLabV3, self).__init__()
        if output_stride == 8:
            s = [1, 2, 1, 1]
            d = [1, 1, 2, 4]
        elif output_stride == 16:
            s = [1, 2, 2, 1]
            d = [1, 1, 1, 2]
        # print(ch)
        self.num_classes = num_classes
        # self.backbone = ResNet50(n_blocks, multi_grids, output_stride)
        self.add_module("aspp", _ASPP(ch[5], 256, atrous_rates))
        concat_ch = 256 * (len(atrous_rates) + 2)

        self.add_module('layer0', _Stem(ch[0]))
        self.add_module("layer1", _ResLayer(n_blocks[0], ch[0], ch[2], s[0], d[0]))
        self.add_module("layer2", _ResLayer(n_blocks[1], ch[2], ch[3], s[1], d[1]))
        self.add_module("layer3", _ResLayer(n_blocks[2], ch[3], ch[4], s[2], d[2]))
        self.add_module("layer4", _ResLayer(n_blocks[3], ch[4], ch[5], s[3], d[3], multi_grids))
        self.add_module("fc1", _ConvBnReLU(concat_ch, 256, 1, 1, 0, 1))
        self.add_module("fc2", nn.Conv2d(256, num_classes, kernel_size=1))

    def forward(self, x):
        # feature_map = self.backbone(x)
        x0 = self.layer0(x)
        x1 = self.layer1(x0)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        feature_map = self.layer4(x3)
        output = self.aspp(feature_map)
        output = self.fc1(output)
        output = self.fc2(output)
        output = F.interpolate(output, size=(x.size()[2], x.size()[3]), mode="bilinear", align_corners=False)
        return  output

if __name__ == "__main__":
    model = DeepLabV3()
    model.train()
    image = torch.randn(1, 3, 512, 512)
    print(model)
    print("input:", image.shape)
    print("output:", model(image).shape)