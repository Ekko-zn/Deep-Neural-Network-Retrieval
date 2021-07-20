import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import random
import numpy as np
import torch.utils.data 
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
setup_seed(1)

best_acc = 0
class Net_representation(nn.Module):
    def __init__(self):
        super(Net_representation,self).__init__()
        self.conv1 = nn.Conv2d(2, 16, 5)
        self.conv2 = nn.Conv2d(16, 32, 5)
        self.pool = nn.AvgPool2d(2)
        self.fc1 = nn.Linear(512,32)
        self.fc2 = nn.Linear(32,5)
    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool(x)
        x = x.view(-1, 512)
        feature = self.fc1(x)
        x = F.relu(feature)
        x = self.fc2(x)
        return x, feature

best_acc = 0
train_eval_rate = 0.8
batchsize = 5
classifier_mode = 'PreActResNet18'

if classifier_mode == 'basic':
    model_path = './basic/'
elif classifier_mode == 'PreActResNet18':
    model_path = './PreActResNet18/'


trainmodels = []
for c in range(5):
    path = model_path +str(c)+'/'
    lists = os.listdir(path)
    for model in lists:
        if 'pth' in model:
            continue
        trainmodels.append(path+model)

random.shuffle(trainmodels)
t_num = len(trainmodels)
train_num = int(t_num * (train_eval_rate))
traindataset = trainmodels[0:train_num]
testdataset = trainmodels[train_num::]

tripledataset = [[],[],[],[],[]]
for feature_path in traindataset:
    label = int(feature_path.split('/')[2])
    tripledataset[label].append(feature_path)

def loadtripledata(tripledataset):
    num1 = random.randint(0,4)
    neg_num1 = (num1+1) % 5
    random.shuffle(tripledataset[num1])
    random.shuffle(tripledataset[neg_num1])
    pos1 = tripledataset[num1][0:batchsize]
    pos2 = tripledataset[num1][batchsize:2*batchsize]
    neg = tripledataset[neg_num1][0:batchsize]
    d1 = []
    for p in pos1:
        d1.append(torch.load(p))
    d1 = torch.stack(d1).squeeze()
    d2 = []
    for p in pos2:
        d2.append(torch.load(p))
    d2 = torch.stack(d2).squeeze()
    d3 = []
    for p in neg:
        d3.append(torch.load(p))
    d3 = torch.stack(d3).squeeze()
    return d1,d2,d3



class processdata(torch.utils.data.Dataset):
    def __init__(self,train_flag):
        if train_flag == True:
            self.datalist = traindataset
        else:
            self.datalist = testdataset
    def __getitem__(self,item):
        path = self.datalist[item]
        label = np.array(int(path.split('/')[2]))
        feature = torch.load(path)
        feature = feature.to(dtype = torch.float).squeeze()
        label = torch.from_numpy(label).to(dtype = torch.long)
        return feature,label
    def __len__(self):
        return len(self.datalist)


traindataset = processdata(True)
traindataloader = torch.utils.data.DataLoader(traindataset,batch_size=batchsize,shuffle=True)
testdataset = processdata(False)
testdataloader = torch.utils.data.DataLoader(testdataset,batch_size=batchsize)
criterion = torch.nn.CrossEntropyLoss()
tripleloss = torch.nn.TripletMarginLoss(margin=0.0)
net = Net_representation()
net = net.cuda()
optimizer = torch.optim.Adam(net.parameters(),lr=1e-4,weight_decay=0.)

def query_eval():
    q_dataset_h = []
    q_dataset_l = []
    for models in traindataset:
        feature = models[0].unsqueeze(0)
        _, hash_value = net(feature)
        label = models[1]
        q_dataset_h.append(hash_value)
        q_dataset_l.append(label)
    q_dataset_h = torch.stack(q_dataset_h).squeeze()
    q_dataset_l = torch.stack(q_dataset_l)
    correct = 0
    LABELS = []
    PRE = []
    for models in testdataset:
        feature = models[0].unsqueeze(0)
        _, hash_value = net(feature)
        hash_value = hash_value
        res = torch.norm((q_dataset_h - hash_value),dim=1)
        pre = torch.argmin(res)
        label = models[1]
        pre_label = q_dataset_l[pre]
        correct += pre_label.eq(label).sum().item()
        LABELS.append(label.cpu().numpy())
        PRE.append(pre_label.cpu().numpy())
    LABELS = np.array(LABELS)
    PRE = np.array(PRE)
    acc = accuracy_score(LABELS,PRE)
    precision = precision_score(LABELS,PRE,average='macro')
    recall = recall_score(LABELS,PRE,average='macro')
    f1 = f1_score(LABELS,PRE,average='macro')
    with open(classifier_mode+'_matrix_log.txt','a+') as f:
        f.write('Acc:{:.4f} Precision:{:.4f} Recall:{:.4f} F1:{:.4f}\n'.format(acc,precision,recall,f1))
    return correct/len(testdataset),q_dataset_h,q_dataset_l

for epoch in range(100):
    correct = 0
    count = 0
    r_loss_cle = 0
    r_loss_triple = 0
    for feature, label in (traindataloader):
        anchor, positive, negative = loadtripledata(tripledataset)
        count += feature.shape[0]
        feature = feature.cuda()
        label = label.cuda()
        out,hashfeature = net(feature)
        _,hashfeature_anchor = net(anchor)
        _,hashfeature_positive = net(positive)
        _,hashfeature_negative = net(negative)
        losstriple = tripleloss(hashfeature_anchor, hashfeature_positive, hashfeature_negative)
        losscle = criterion(out,label)
        loss = 1*losstriple + 1*losscle
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        _, predicted = out.max(1)
        correct += predicted.eq(label).sum().item()
        r_loss_cle += losscle.item()
        r_loss_triple += losstriple.item()
    train_acc = correct/count
    t_r_loss_cle = r_loss_cle/len(traindataloader)
    t_r_loss_triple = r_loss_triple/len(traindataloader)

    correct = 0
    count = 0
    r_loss_cle = 0
    for feature, label in (testdataloader):
        count += feature.shape[0]
        feature = feature.cuda()
        label = label.cuda()
        out,_ = net(feature)
        losscle = criterion(out,label)
        _, predicted = out.max(1)
        correct += predicted.eq(label).sum().item()
        r_loss_cle += losscle.item()
    test_acc = correct/count
    e_r_loss_cle = r_loss_cle/len(testdataloader)

    query_acc,q_dataset_h,q_dataset_l = query_eval()
    if query_acc > best_acc:
        best_acc = query_acc
        q_dataset_h = q_dataset_h.cpu().detach().numpy()
        q_dataset_l = q_dataset_l.cpu().detach().numpy()
        np.save(classifier_mode+'_q_dataset_h.npy',q_dataset_h)
        np.save(classifier_mode+'_q_dataset_l.npy',q_dataset_l)
    print('epoch:{} | train_acc:{:.3f} | train_cle_loss:{:.4f} | train_cle_triple:{:.4f} | test_acc:{:.3f}| test_cle_loss:{:.4f} | query_acc:{:.4f} | best_acc:{:.4f} '.format(
        epoch,train_acc,t_r_loss_cle,t_r_loss_triple,test_acc,e_r_loss_cle,query_acc,best_acc
        )
        )
