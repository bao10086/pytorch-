import torch
from torch import nn
from torch.nn import functional as F


class ResBlk(nn.Module):
    # Resnet Block
    def __init__(self, ch_in, ch_out, stride=1):
        """

        :param ch_in:
        :param ch_out:
        """
        super(ResBlk, self).__init__()

        # convolution, 改变stride可以修改h*w
        self.conv1 = nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=stride, padding=1)
        # Batch Normalization
        self.bn1 = nn.BatchNorm2d(ch_out)
        self.conv2 = nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(ch_out)

        # [b, ch_in, h, w] => [b, ch_out, h, w]
        self.extra = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=stride),
            nn.BatchNorm2d(ch_out)
        )

    def forward(self, x):
        """

        :param x: [b, ch, h, w]
        :return:
        """
        out = F.relu((self.bn1(self.conv1(x))))
        out = self.bn2(self.conv2(out))
        # short cut
        # extra module: [b, ch_in, h, w] => [b, ch_out, h, w]
        # element-wise add:
        out = out + self.extra(x)
        return out


class Resnet18(nn.Module):
    def __init__(self):
        super(Resnet18, self).__init__()

        # 预处理层，先使其变成64
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64)
        )
        # followed 4 blocks
        # [b, 64, h, w] => [b, 128, h, w]
        self.blk1 = ResBlk(64, 128, stride=2)
        # [b, 128, h, w] => [b, 256, h, w]
        self.blk2 = ResBlk(128, 256, stride=2)
        # [b, 256, h, w] => [b, 512, h, w]
        self.blk3 = ResBlk(256, 512, stride=2)
        # [b, 512, h, w] => [b, 1024, h, w]
        self.blk4 = ResBlk(512, 512, stride=2)

        # 设置全连接层，512为输入的channel，10为输出的10个概率值
        self.outlayer = nn.Linear(512, 10)

    def forward(self, x):
        """

        :param x:
        :return:
        """
        x = F.relu(self.conv1(x))
        x = self.blk1(x)
        x = self.blk2(x)
        x = self.blk3(x)
        x = self.blk4(x)

        # print('after conv:', x.shape)
        x = F.adaptive_avg_pool2d(x, [1, 1])
        # 输出[2,512,1,1]，表示2张图片等
        # print('after pool:', x.shape)
        # 打平为二维
        x = x.view(x.size(0), -1)
        x = self.outlayer(x)

        return x


def main():
    # input: 64 channel, output:128 channel
    blk = ResBlk(64, 128, stride=2)
    # input: 64 channel
    tmp = torch.randn(2, 64, 32, 32)
    out = blk(tmp)
    print('resBlock:', out.shape)

    x = torch.randn(2, 3, 32, 32)
    model = Resnet18()
    out = model(x)
    print('resnet:', out.shape)


if __name__ == '__main__':
    main()