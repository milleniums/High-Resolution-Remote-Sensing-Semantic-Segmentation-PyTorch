import torch
import torch.nn as nn
import torch.nn.functional as F
from resnet import ResNet50
from aspp import _ASPP
from component import _ConvBnReLU

ch = [64 * 2 ** p for p in range(6)]

class DeepLabV3(nn.Module):
    def __init__(self, num_classes=16, n_blocks=[3,4,6,3], atrous_rates=[6,12,18], multi_grids=[1,2,1], output_stride=16):
        super(DeepLabV3, self).__init__()

        print(ch)
        self.num_classes = num_classes
        self.backbone = ResNet50(n_blocks, multi_grids, output_stride)
        self.add_module("aspp", _ASPP(ch[5], 256, atrous_rates))
        concat_ch = 256 * (len(atrous_rates) + 2)
        self.add_module("fc1", _ConvBnReLU(concat_ch, 256, 1, 1, 0, 1))
        self.add_module("fc2", nn.Conv2d(256, num_classes, kernel_size=1))

    def forward(self, x):
        feature_map = self.backbone(x)
        output = self.aspp(feature_map)
        output = self.fc1(output)
        output = self.fc2(output)
        output = F.interpolate(output, size=(x.size()[2], x.size()[3]), mode="bilinear", align_corners=False)
        return  output

if __name__ == "__main__":
    model = DeepLabV3()
    model.eval()
    image = torch.randn(1, 3, 512, 512)
    print(model)
    print("input:", image.shape)
    print("output:", model(image).shape)