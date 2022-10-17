import torch
from torch import nn
from torch.nn import functional as F


class Lenet5(nn.Module):
    # for cifar10 dataset
    def __init__(self):
        super(Lenet5, self).__init__()

        self.conv_unit = nn.Sequential(
            # 第一个卷积层
            # x: [b,3,32,32]  3个通道，b张照片  =>  [b,6,]
            # input_channel:3, output_channel:6
            nn.Conv2d(3, 6, kernel_size=5, stride=1, padding=0),
            nn.AvgPool2d(kernel_size=2, stride=2, padding=0),
            # 第二个卷积层
            nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
            nn.AvgPool2d(kernel_size=2, stride=2, padding=0)

        )
        # flatten
        # fc unit 打平全连接
        self.fc_unit = nn.Sequential(
            nn.Linear(16*5*5, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 10)
        )

        # b:[b,3,32,32]
        tmp = torch.randn(2, 3, 32, 32)
        out = self.conv_unit(tmp)
        # [2, 16, 5, 5]
        print('conv_out:', out.shape)

        # use cross entropy loss
        # self.criteon = nn.MSELoss()
        # self.criteon = nn.CrossEntropyLoss()  # 分类问题


    def forward(self, x):
        """

        :param x: [b, 3, 32, 32]
        :return:
        """
        batch_size = x.size(0)  # 返回b
        # [b, 3, 32, 32] => [b, 16, 5, 5]
        x = self.conv_unit(x)
        # [b, 16, 5, 5] => [b, 16*5*5]
        x = x.view(batch_size, 16*5*5)
        # [b, 16*5*5] => [b, 10]
        logits = self.fc_unit(x)

        # [b, 10]
        # pred = F.softmax(logits, dim=1)  CrossEntropyLoss已包含softmax
        # loss = self.criteon(logits, y)
        return logits



def main():
    net = Lenet5()

    # b:[b,3,32,32]
    tmp = torch.randn(2, 3, 32, 32)
    out = net(tmp)
    # [2, 16, 5, 5]
    print('conv_out:', out.shape)


if __name__ == '__main__':
    main()