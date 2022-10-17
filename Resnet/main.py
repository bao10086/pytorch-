import torch
from torch.utils.data import  DataLoader
from torchvision import datasets
from torchvision import transforms
from torch import nn
from torch import optim
from Lenet5 import Lenet5
from resnet import Resnet18


def main():
    batch_size = 32

    # 加载数据集
    cifar_train = datasets.CIFAR10('cifar', train=True, transform=transforms.Compose([
        transforms.Resize([32, 32]),
        transforms.ToTensor()
    ]), download=True)
    # 构造一次下载32个文件
    # 加载随机化
    cifar_train = DataLoader(cifar_train, batch_size=batch_size, shuffle=True)

    cifar_test = datasets.CIFAR10('cifar', train=False, transform=transforms.Compose([
        transforms.Resize([32, 32]),
        transforms.ToTensor()
    ]), download=True)
    cifar_test = DataLoader(cifar_test, batch_size=batch_size, shuffle=True)

    x, label = iter(cifar_train).next()
    print('x:', x.shape, 'label:', label.shape)

    # device = torch.device('duda')
    # model = Lenet5().to(device)
    # model = Lenet5()
    model = Resnet18()
    criteon = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    print(model)

    for epoch in range(1000):

        model.train()
        for batchidx, (x, label) in enumerate(cifar_train):
            # [b, 3, 32, 32]
            # [b]
            # x, label = x.to(device), label.to(device)
            logits = model(x)
            # logits:[b, 10]
            # label: [b]
            # loss: tensor scalar
            loss = criteon(logits, label)

            # backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(epoch, loss.item())

        model.eval()
        with torch.no_grad():
            # test
            total_correct = 0
            total_num = 0
            for x, label in cifar_test:
                # [b, 3, 32, 32]
                # [b]
                # x, label = x.to(device), label.to(device)
                # [b, 10]
                logits = model(x)
                # [b]
                pred = logits.argmax(dim=1)
                # [b] vs [b]  =>  scalar tensor
                total_correct += torch.eq(pred, label).float().sum()
                total_num += x.size(0)

            acc = total_correct / total_num
            print(epoch, acc)


if __name__ == '__main__':
    main()
