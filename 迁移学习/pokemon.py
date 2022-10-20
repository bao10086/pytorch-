# 自定义数据集
import torch
import os, glob
import random, csv

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image


class Pokemon(Dataset):
    # root: 图片存储位置
    # resize: 输出size
    # mode: 表示当前类做train 还是test
    def __init__(self, root, resize, mode):
        super(Pokemon, self).__init__()

        self.root = root
        self.resize = resize

        self.name2label = {}  # 编码映射字典，例如"squirtle": 0
        for name in os.listdir(os.path.join(root)):
            # os.path.join 连接路径
            if not os.path.isdir(os.path.join(root, name)): # 过滤掉文件
                continue
            self.name2label[name] = len(self.name2label.keys())
        # print(self.name2label)  # 调试

        # image, label
        self.images, self.labels = self.load_csv('images.csv')

        if mode == 'train': # 60%
            self.images = self.images[:int(0.6*len(self.images))]
            self.labels = self.labels[:int(0.6*len(self.labels))]
        elif mode == 'val':  # 20%
            self.images = self.images[int(0.6*len(self.images)):int(0.8*len(self.images))]
            self.labels = self.labels[int(0.6*len(self.labels)):int(0.8*len(self.labels))]
        else:  # 20%
            self.images = self.images[int(0.8*len(self.images)):]
            self.labels = self.labels[int(0.8*len(self.labels)):]
        # print(len(self.images))

    def load_csv(self, filename):
        # 如果文件不存在，存储到文件
        if not os.path.exists(os.path.join(self.root, filename)):
            images = []
            for name in self.name2label.keys():
                # 找到适配图片，例如'pokemon\\mewtwo\\00001.png'
                images += glob.glob(os.path.join(self.root, name, '*.png'))
                images += glob.glob(os.path.join(self.root, name, '*.jpg'))
                images += glob.glob(os.path.join(self.root, name, '*.jpeg'))
                images += glob.glob(os.path.join(self.root, name, '*.gif'))

            # 存储图像路径到文件
            random.shuffle(images)
            # 返回文件对象
            with open(os.path.join(self.root, filename), mode='w', newline='') as f:
                writer = csv.writer(f)
                for img in images:  # 一个文件名，'pokemon\\mewtwo\\00001.png'
                    name = img.split(os.sep)[-2]
                    label = self.name2label[name]
                    # 'pokemon\\mewtwo\\00001.png', 2
                    writer.writerow([img, label])
                print('write into csv file:', filename)

        # 否则，从文件读取图像路径
        images, labels = [], []
        with open(os.path.join(self.root, filename)) as f:
            reader = csv.reader(f)
            for row in reader:
                # 'pokemon\\mewtwo\\00001.png', 2
                img, label = row
                label = int(label)

                images.append(img)
                labels.append(label)

        assert len(images) == len(labels)
        # print(len(images))  # 1168张
        return images, labels

    def __len__(self):
        return len(self.images)

    def denormalize(self, x_hat):
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        # x_hat = (x - mean) / std
        # x = x_hat * std + mean
        # x: [c, h, w]
        # mean: [3] => [3, 1. 2]
        mean = torch.tensor(mean).unsqueeze(1).unsqueeze(1)
        std = torch.tensor(std).unsqueeze(1).unsqueeze(1)

        x = x_hat * std + mean
        return x

    def __getitem__(self, idx):
        # img: 'pokemon\\mewtwo\\00001.png'
        # label: 2
        img, label = self.images[idx], self.labels[idx]

        tf = transforms.Compose([
            lambda x: Image.open(x).convert('RGB'),  # 把路径变成image data
            transforms.Resize((int(self.resize*1.25), int(self.resize*1.25))),  # 改变尺寸
            transforms.RandomRotation(15),  # 旋转
            transforms.CenterCrop(self.resize),  # 中心裁剪
            transforms.ToTensor(),  # 将其变成pytorch的tensor
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        img = tf(img)
        label = torch.tensor(label)  # 将其转化为tensor的标量
        return img, label


def main():
    import visdom  # 可视化
    import time
    import torchvision

    viz = visdom.Visdom()
    # 第一种方式，适用于图片存储在相应目录，无额外要求
    # tf = transforms.Compose([
    #     transforms.Resize((64, 64)),  # 改变尺寸
    #     transforms.ToTensor(),  # 将其变成pytorch的tensor
    # ])
    # db = torchvision.datasets.ImageFolder(root='D:\BaiduNetdiskDownload\modelsim\pokeman', transform=tf)
    # print(db.class_to_idx)  # 查看编码方式
    # loader = DataLoader(db, batch_size=32, shuffle=True)
    # for x, y in loader:
    #     # 一行显示8个
    #     viz.images(x, nrow=8, win='batch', opts=dict(title='batch'))
    #     viz.text(str(y.numpy()), win='label', opts=dict(title='batch_y'))
    #
    #     time.sleep(10)

    # 第二种方式
    db = Pokemon('D:\BaiduNetdiskDownload\modelsim\pokeman', 224, 'train')
    x, y = next(iter(db))
    # print('sample:', x.shape, y.shape)  # 打印维度
    # win为窗口名称，opts为图像标例
    # viz.image(db.denormalize(x), win='sample_x', opts=dict(title='sample_X'))

    loader = DataLoader(db, batch_size=32, shuffle=True, num_workers=8)  # shuffle保证每次batch为随机生成,num_workers保证多线程
    for x, y in loader:
        # 一行显示8个
        viz.images(db.denormalize(x), nrow=8, win='batch', opts=dict(title='batch'))
        viz.text(str(y.numpy()), win='label', opts=dict(title='batch_y'))

        time.sleep(10)


if __name__ == '__main__':
    main()