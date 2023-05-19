import os
import time
import torch
import torchvision
from torch import nn
from mobilevit import *
from torch.utils.data import DataLoader
from torchvision import transforms,utils,datasets

if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("using {} device".format(device))

    data_transform =  transforms.Compose([
    # 然后，缩放图像以创建256*256的新图像
    transforms.RandomResizedCrop(256),
    transforms.RandomHorizontalFlip(),
    # 归一化
    torchvision.transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std= [0.229, 0.224, 0.225])
    ])
##############################################
    ata_transform = transforms.Compose([
        # 然后，缩放图像以创建256*256的新图像
        torchvision.transforms.Resize(int(256 * 1.143)),
        transforms.CenterCrop(256),
        transforms.RandomHorizontalFlip(),
        # 归一化
        torchvision.transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
#################################
    train_data = datasets.ImageFolder(root="dataset/train",transform=data_transform)
    val_data = datasets.ImageFolder(root="dataset/val",transform=ata_transform)

    train_data_size = len(train_data)
    val_data_size = len(val_data)

    print("训练数据集的长度为：{}".format(train_data_size))
    print("验证数据集的长度为：{}".format(val_data_size))

    # 利用 dataloader 来加载数据集
    train_dataloader = DataLoader(train_data,batch_size=16,shuffle=True,pin_memory =True,num_workers=2)
    val_dataloader = DataLoader(val_data,batch_size=16,shuffle=False,pin_memory =True,num_workers=2)

    #  创建网络模型
    model = mobilevit_xxs()
    model = model.to(device)

    # 损失函数
    loss_fn = nn.CrossEntropyLoss()
    loss_fn = loss_fn.to(device)

    # 优化器
    learning_rate = 0.001
    # optimizer = torch.optim.SGD(model.parameters(),lr=learning_rate)
    # optimizer = adabound.AdaBound(model.parameters(),lr=learning_rate,final_lr=0.01)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate,betas=(0.9,0.999),eps=10e-08)
    # 设置训练网络的一些参数
    # 记录训练的次数
    total_train_step = 0
    # 记录验证的次数
    total_val_step = 0
    # 训练的轮数
    epoch = 80

    # 添加tensorboard
    start_time = time.time()
    for i in range(1,epoch+1):
        print("--------第 {} 轮训练开始-------".format(i))

        # 训练步骤开始
        model.train()
        for data in train_dataloader:
            imgs,targets = data
            imgs = imgs.to(device)
            targets = targets.to(device)
            outputs =model(imgs)
            loss = loss_fn(outputs,targets)

            # 优化器优化模型
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_step = total_train_step + 1
            if total_train_step % 100 == 0:
                end_time = time.time()
                print(end_time - start_time)
                print("训练次数: {} , Loss: {} ".format(total_train_step,loss.item()))

        # 验证步骤开始
        total_val_loss = 0
        total_accuracy = 0
        with torch.no_grad():
                for data in val_dataloader:
                    imgs,targets = data
                    imgs = imgs.to(device)
                    targets = targets.to(device)
                    outputs = model(imgs)
                    loss = loss_fn(outputs,targets)
                    total_val_loss = total_val_loss + loss.item()
                    accuracy = (outputs.argmax(1) == targets).sum()   # 行最大的索引值
                    total_accuracy = total_accuracy + accuracy


        print("整体验证集上的Loss: {} ".format(total_val_loss))
        print("整体验证集上的正确率: {} ".format(total_accuracy/val_data_size))
        total_val_step = total_val_step + 1
        if i % 5 == 0:
            torch.save(model,"output/model_{}--{}.pth".format(i,total_accuracy/val_data_size))
            print("模型已保存")
