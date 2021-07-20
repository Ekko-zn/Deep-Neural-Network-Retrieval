import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from model import PreActResNet18,Net_basic
import os
torch.backends.cudnn.benchmark = True
criterion = torch.nn.CrossEntropyLoss()


class Processdata(torch.utils.data.Dataset):
    def __init__(self,imgs):
        self.data = data
        
    def __getitem__(self,item):
        img0 = torch.from_numpy(self.data[0,item,:,:]).to(dtype=torch.float).unsqueeze(0).unsqueeze(0)
        img1 = torch.from_numpy(self.data[1,item,:,:]).to(dtype=torch.float).unsqueeze(0).unsqueeze(0)
        label0 = torch.zeros(1).to(dtype=torch.long)
        label1 = torch.ones(1).to(dtype=torch.long)
        img = torch.cat((img0,img1),0)
        label = torch.cat((label0,label1),0)
        return img,label
    def __len__(self):
        return len(self.data[1])

def Train(classifier_mode,task_num,sample_num,data,max_epoch):
    if classifier_mode == 'basic':
        classifier = Net_basic()
    elif classifier_mode == 'PreActResNet18':
        classifier = PreActResNet18()
    classifier = classifier.cuda()
    trainingdataset = Processdata(data)
    trainingdataset[0]
    trainingdataloader = torch.utils.data.DataLoader(trainingdataset,batch_size=50,shuffle=True,num_workers=0)
    optimizer = torch.optim.Adam(classifier.parameters(),lr=1e-4)
    for epoch in tqdm(range(max_epoch)):
        for imgs,labels in (trainingdataloader):
            imgs = imgs.view(-1,1,28,28).cuda()
            labels = labels.view(-1,1).cuda().squeeze()
            out = classifier(imgs)
            loss = criterion(out,labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    state = classifier.state_dict()
    if not os.path.exists('./'+classifier_mode+'/' + str(task_num) + '/'):
        os.makedirs('./'+classifier_mode+'/' + str(task_num) + '/')
    path = './'+classifier_mode+'/' + str(task_num) + '/' + str(sample_num) + '.pth'
    torch.save(state,path)
    print('Task num:{} Sample num:{} loss:{:.4f}'.format(task_num,sample_num,loss.item()))


if __name__=='__main__':
    import sys
    import os
    arg = sys.argv
    task_num = arg[1]
    classifier_mode = arg[2] # basic ; PreActResNet18;
    max_epoch = int(arg[3])
    task_num = int(task_num)
    for sample_num in range(100):
        path = './dataset/'+str(task_num)+'/'+str(sample_num)+'/data.npy'
        data = np.load(path,allow_pickle=True)
        Train(classifier_mode,task_num,sample_num,data,max_epoch)