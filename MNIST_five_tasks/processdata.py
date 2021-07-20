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
trainset = torchvision.datasets.MNIST(
    root='./data', train=True, download=True)

x = 100


traindata = {}
for _ in tqdm(range(x)):
    for img, label in (trainset):
        img = transforms.RandomRotation(10)(img)
        img = np.array(img)
        label = label
        if label in traindata:
            if len(traindata[label]) == 5000*x:
                continue
            traindata[label].append(img)
        else:
            traindata[label] = [img]
multipletasks = [[], [], [], [], []]

multipletasks[0] = {0: traindata[0], 1: traindata[1]}
multipletasks[1] = {2: traindata[2], 3: traindata[3]}
multipletasks[2] = {4: traindata[4], 5: traindata[5]}
multipletasks[3] = {6: traindata[6], 7: traindata[7]}
multipletasks[4] = {8: traindata[8], 9: traindata[9]}

Sample_num = 100
each_task_num = (5000*x) // Sample_num

def tmp(x):
    keys = list(x.keys())
    t1 = x[keys[0]]
    t2 = x[keys[1]]
    result = [[] for ii in range(Sample_num)]
    for i in range(Sample_num):
        result[i] = [t1[i*each_task_num:(i+1)*each_task_num],t2[i*each_task_num:(i+1)*each_task_num]]
    return result
multipletasks[0] =tmp(multipletasks[0])
multipletasks[1] =tmp(multipletasks[1])
multipletasks[2] =tmp(multipletasks[2])
multipletasks[3] =tmp(multipletasks[3])
multipletasks[4] =tmp(multipletasks[4])

for task_num in range(5):
    for sample_num in range(100):
        data = np.asarray(multipletasks[task_num][sample_num])
        path = './dataset/'+str(task_num)+'/'+str(sample_num)+'/'
        if not os.path.isdir(path):
            os.makedirs(path)
        np.save('./dataset/'+str(task_num)+'/'+str(sample_num)+'/'+'data.npy',data)


