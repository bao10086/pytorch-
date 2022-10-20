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
        out = F.relu(out)
        return out


class Resnet18(nn.Module):
    def __init__(self, num_class):
        super(Resnet18, self).__init__()

        # 预处理层，先使其变成16
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=3, padding=1),
            nn.BatchNorm2d(16)
        )
        # followed 4 blocks
        # [b, 16, h, w] => [b, 32, h, w]
        self.blk1 = ResBlk(16, 32, stride=3)
        # [b, 32, h, w] => [b, 64, h, w]
        self.blk2 = ResBlk(32, 64, stride=3)
        # [b, 64, h, w] => [b, 128, h, w]
        self.blk3 = ResBlk(64, 128, stride=2)
        # [b, 128, h, w] => [b, 256, h, w]
        self.blk4 = ResBlk(128, 256, stride=2)

        # 设置全连接层，1024为输入的channel，num_class为输出的类别数目
        self.outlayer = nn.Linear(256*3*3, num_class)

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
        # [b, 256, 3, 3]
        # x = F.adaptive_avg_pool2d(x, [1, 1])
        # 输出[2,512,1,1]，表示2张图片等
        # print('after pool:', x.shape)
        # 打平为二维
        x = x.view(x.size(0), -1)
        x = self.outlayer(x)

        return x


def main():
    # input: 64 channel, output:128 channel
    blk = ResBlk(64, 128)
    # input: 64 channel
    tmp = torch.randn(2, 64, 224, 224)
    out = blk(tmp)
    print('resBlock:', out.shape)

    model = Resnet18(5)
    x = torch.randn(2, 3, 224, 224)
    out = model(x)
    print('resnet:', out.shape)

    p = sum(map(lambda p: p.numel(), model.parameters()))
    print('parameters size:', p)


if __name__ == '__main__':
    main()