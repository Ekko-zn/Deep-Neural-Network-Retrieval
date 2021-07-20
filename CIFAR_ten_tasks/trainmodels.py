import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from model import PreActResNet18,SimpleDLA,Net_basic,MobileNetV2
import os
torch.backends.cudnn.benchmark = True
criterion = torch.nn.CrossEntropyLoss()

class Processdata(torch.utils.data.Dataset):
    def __init__(self,imgs,labels):
        self.imgs = torch.from_numpy(imgs).to(dtype=torch.float).permute(0,3,1,2)
        self.labels = torch.tensor(labels).to(dtype=torch.long)
        pass
    def __getitem__(self,item):
        img = self.imgs[item,:,:,:]
        label = self.labels[item]
        return img,label
    def __len__(self):
        return len(self.imgs)

#-------------------------- 分batchsize

def Train(classifier_mode,task_num,sample_num,data,max_epoch):
    def dataprocess(data):
        imgs = []
        label = []
        for d in data:
            imgs.append(d[0])
            label.append(int(d[1]))
        return imgs,label
    if classifier_mode == 'basic':
        classifier = Net_basic()
    elif classifier_mode == 'PreActResNet18':
        classifier = PreActResNet18()
    classifier = classifier.cuda()
    imgs,labels = dataprocess(data)
    imgs = np.stack(imgs)
    labels = np.stack(labels)
    trainingdataset = Processdata(imgs,labels)
    trainingdataloader = torch.utils.data.DataLoader(trainingdataset,batch_size=50,shuffle=True,num_workers=0)
    optimizer = torch.optim.Adam(classifier.parameters(),lr=1e-4)
    for epoch in tqdm(range(max_epoch)):  # basic 500 PreActResNet18 10
        for imgs,labels in (trainingdataloader):
            imgs = imgs.cuda()
            labels = labels.cuda()
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