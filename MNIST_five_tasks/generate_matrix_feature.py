import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
arg = sys.argv
part = int(arg[1])
classifier_mode = arg[2]

import random
import numpy as np
from tqdm import tqdm
from model import PreActResNet18,Net_basic


if classifier_mode == 'basic':
    model_path = './basic/'
    net = Net_basic()
elif classifier_mode == 'PreActResNet18':
    model_path = './PreActResNet18/'
    net = PreActResNet18()

trainmodels = []
for c in range(5):
    path = model_path +str(c)+'/'
    lists = os.listdir(path)
    for model in lists:
        if '.pth' in model:
            trainmodels.append(path+model)

trainmodels = trainmodels[100*part:100*(part+1)]

criterion = torch.nn.CrossEntropyLoss()
max_step = 1000
for t in tqdm(trainmodels):
    net.load_state_dict(torch.load(t))
    net = net.cuda()
    net.eval()
    p = (torch.zeros(2,1,28,28)).cuda()
    p.requires_grad = True
    optimizer_p = torch.optim.Adam([p],lr=1e-2)
    label_zero = torch.zeros(1).to(dtype=torch.long).cuda()
    label_one = torch.ones(1).to(dtype=torch.long).cuda()
    label = torch.cat((label_zero,label_one),0)
    for step in range(max_step):
        out = net(p)
        loss = criterion(out,label)
        optimizer_p.zero_grad()
        loss.backward()
        optimizer_p.step()
    print(loss.item())
    feature = p.detach().clone().view(2,28,28)
    savepath = t.replace('pth','pkl')
    torch.save(feature,savepath)
    pass