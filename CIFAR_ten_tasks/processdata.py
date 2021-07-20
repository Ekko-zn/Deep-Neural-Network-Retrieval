import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import numpy as np
from torchvision import transforms as transforms
from tqdm import tqdm
import os
os.system('mkdir data')
os.system('mkdir dataset')
trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True)

x = 100


traindata = {}
for _ in tqdm(range(x)):
    for img, label in (trainset):

        img_0 = img
        img_0 = np.array(img_0)
        label_0 = '0'
        img_90 = transforms.functional.rotate(img,90)
        img_90 = np.array(img_90)
        label_1 = '1'
        img_180 = transforms.functional.rotate(img,180)
        img_180 = np.array(img_180)
        label_2 = '2'
        img_270 = transforms.functional.rotate(img,270)
        img_270 = np.array(img_270)
        label_3 = '3'
        tmp = [[img_0,label_0],[img_90,label_1],[img_180,label_2],[img_270,label_3]]
        if label in traindata:
            if len(traindata[label]) == 5000*x:
                continue
            traindata[label].append(tmp[0])
            traindata[label].append(tmp[1])
            traindata[label].append(tmp[2])
            traindata[label].append(tmp[3])
        else:
            traindata[label] = [*tmp]

Sample_num = 100
each_task_num = (5000*x) // Sample_num

for task in traindata:
    t = []
    for i in range(Sample_num):
        t.append(traindata[task][i*each_task_num:(i+1)*each_task_num])
    traindata[task] = t

for task_num in range(10):
    for sample_num in range(Sample_num):
        data = np.asarray(traindata[task_num][sample_num])
        path = './dataset/'+str(task_num)+'/'+str(sample_num)+'/'
        if not os.path.isdir(path):
            os.makedirs(path)
        np.save('./dataset/'+str(task_num)+'/'+str(sample_num)+'/'+'data.npy',data)


