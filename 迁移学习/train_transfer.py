import torch
from torch import optim, nn
import visdom
import torchvision
from torch.utils.data import DataLoader

from pokemon import Pokemon
# from resnet import Resnet18
from torchvision.models import resnet18

from utils import Flatten


batch_size = 32
lr = 1e-3
epochs = 10
# device = torch.device('cuda')
device = torch.device('cpu')
torch.manual_seed(1234)  # 设置随机种子，复现实验结果
viz = visdom.Visdom()


train_db = Pokemon('D:\BaiduNetdiskDownload\modelsim\pokeman', 224, mode='train')
val_db = Pokemon('D:\BaiduNetdiskDownload\modelsim\pokeman', 224, mode='val')
test_db = Pokemon('D:\BaiduNetdiskDownload\modelsim\pokeman', 224, mode='test')
train_loader = DataLoader(train_db, batch_size=batch_size, shuffle=True, num_workers=4)
val_loader = DataLoader(val_db, batch_size=batch_size, num_workers=2)
test_loader = DataLoader(test_db, batch_size=batch_size, num_workers=2)


def evaluate(model, loader):
    correct = 0
    total = len(loader.dataset)
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with torch.no_grad():  # 不需要反向传播，因此不需要梯度
            logits = model(x)
            pred = logits.argmax(dim=1)
        correct += torch.eq(pred, y).sum().float().item()
    return correct / total

def main():
    # model = Resnet18(5).to(device)
    trained_model = resnet18(pretrained=True)  # 已经训练好了
    model = nn.Sequential(*list(trained_model.children())[:-1],  # 取出前17层(除最后一层), [b, 512, 1, 1]
                          Flatten(),  # [b, 512]
                          nn.Linear(512, 5)
                          ).to(device)
    x = torch.randn(2, 3, 224, 224)
    print(model(x).shape)  # 输出为[2, 512, 1, 1]

    optimizer = optim.Adam(model.parameters(), lr=lr)  # 优化器
    criteon = nn.CrossEntropyLoss()  # 损失函数，不需要使用softmax

    global_step = 0
    best_acc, best_epoch = 0, 0
    viz.line([0], [-1], win='loss', opts=dict(title='loss'))
    viz.line([0], [-1], win='val_acc', opts=dict(title='val_acc'))
    for epoch in range(epochs):
        for step, (x, y) in enumerate(train_loader):
            # x: [b, 3, 224, 224], y: [b]
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criteon(logits, y)  # 内部做one-hat
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        viz.line([loss.item()], [global_step], win='loss', update='append')
        global_step += 1

        if epoch % 2 == 0:
            val_acc = evaluate(model, val_loader)
            print(epoch, ':', val_acc)
            if val_acc > best_acc:
                best_epoch = epoch
                best_acc = val_acc

                torch.save(model.state_dict(), 'best.mdl')  # 文件后缀名任意
                viz.line([val_acc], [global_step], win='val_acc', update='append')

    print('best acc:', best_acc, 'best epoch:', best_epoch)
    model.load_state_dict(torch.load('best.mdl'))
    print('loaded from ckpt!')

    test_acc = evaluate(model, test_loader)
    print('test acc:', test_acc)


if __name__ == '__main__':
    main()