import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import random
import numpy as np
import torch.utils.data 
from model import PreActResNet18,Net_basic,SimpleDLA,MobileNetV2
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
setup_seed(1)

best_acc = 0
classifier_mode = 'basic'
REGULARIZATION = 0
train_eval_rate = 0.8
batchsize = 10 
N =  200

if classifier_mode == 'basic':
    Net_basicmodel = Net_basic
    model_path = './basic/'
elif classifier_mode == 'PreActResNet18':
    Net_basicmodel = PreActResNet18
    model_path = './PreActResNet18/'


class Fusion_net(nn.Module):
    def __init__(self):
        super(Fusion_net,self).__init__()
        self.fc1 = nn.Linear(2*N,32)
        self.relu = nn.LeakyReLU(0.1)
        self.fc2 = nn.Linear(32,5)
    def forward(self,x):
        feature = self.fc1(x)
        out = self.fc2(self.relu(feature))
        return out, feature


trainmodels = []
for c in range(5):
    path = model_path +str(c)+'/'
    lists = os.listdir(path)
    for model in lists:
        if 'pkl' in model:
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
    d2 = []
    for p in pos2:
        d2.append(torch.load(p))
    d3 = []
    for p in neg:
        d3.append(torch.load(p))
    return d1,d2,d3



X = torch.rand((N, 3, 32, 32), requires_grad=True, device='cuda')
X.data *= 255.
F = Fusion_net()
F = F.cuda()
criterion = torch.nn.CrossEntropyLoss()
tripleloss = torch.nn.TripletMarginLoss(margin=0.0)
optimizerX = torch.optim.SGD(params=[X], lr=1e+3)  # 1e+2
optimizerWb = torch.optim.Adam(F.parameters(), lr=1e-3,weight_decay=0)  # 1e-3

def query_eval():
    q_dataset_h = []
    q_dataset_l = []
    for models in traindataset:
        cnn = Net_basicmodel()
        cnn.load_state_dict(torch.load(models))
        cnn = cnn.cuda()
        cnn.eval()
        _, hash_value= F(cnn(X).view(1, -1))
        label = torch.tensor(int(models.split('/')[2]))
        q_dataset_h.append(hash_value.detach())
        q_dataset_l.append(label)
    
    q_dataset_h = torch.stack(q_dataset_h).squeeze()
    q_dataset_l = torch.stack(q_dataset_l)
    correct = 0
    LABELS = []
    PRE = []
    for models in testdataset:
        cnn = Net_basicmodel()
        cnn.load_state_dict(torch.load(models))
        cnn = cnn.cuda()
        cnn.eval()
        _, hash_value= F(cnn(X).view(1, -1))
        hash_value = hash_value
        res = torch.norm((q_dataset_h - hash_value),dim=1)
        pre = torch.argmin(res)
        label = torch.tensor(int(models.split('/')[2]))
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
    with open(classifier_mode+'_log.txt','a+') as f:
        f.write('Acc:{:.4f} Precision:{:.4f} Recall:{:.4f} F1:{:.4f}\n'.format(acc,precision,recall,f1))
    return correct/len(testdataset),q_dataset_h,q_dataset_l

Xgrad,Fgrad = [],[]
for epoch in range(100):
    # q_acc = query_eval()
    count = 0
    randind = np.random.permutation(len(traindataset))
    train_models_labels = np.asarray(traindataset)[randind]
    for path in (train_models_labels):
        count += 1
        model_path = path
        label = torch.tensor(int(feature_path.split('/')[2])).to(dtype=torch.long).cuda().unsqueeze(0)
        anchor, positive, negative = loadtripledata(tripledataset)
        cnn,cnn_a,cnn_p,cnn_n = Net_basicmodel(),Net_basicmodel(),Net_basicmodel(),Net_basicmodel()
        cnn,cnn_a,cnn_p,cnn_n = cnn.cuda(),cnn_a.cuda(),cnn_p.cuda(),cnn_n.cuda()
        cnn.load_state_dict(torch.load(model_path))
        cnn_a.load_state_dict(anchor[0])
        cnn_p.load_state_dict(positive[0])
        cnn_n.load_state_dict(negative[0])
        cnn.eval()
        cnn_a.eval()
        cnn_p.eval()
        cnn_n.eval()
        out,_ = F(cnn(X).view(1, -1))
        _,F_a = F(cnn_a(X).view(1, -1))
        _,F_p = F(cnn_p(X).view(1, -1))
        _,F_n = F(cnn_n(X).view(1, -1))
        losscle = criterion(out,label)
        reg_loss = (torch.sum(torch.abs(X[:, :, :, :-1] - X[:, :, :, 1:])) +
                                     torch.sum(torch.abs(X[:, :, :-1, :] - X[:, :, 1:, :])))
        losstriple = tripleloss(F_a,F_p,F_n)
        loss = 1.0*losscle + 0 * reg_loss + 1*losstriple
        optimizerWb.zero_grad()
        optimizerX.zero_grad()
        loss.backward()
        if count % batchsize == 0:
            Xgrad = torch.stack(Xgrad, 0)
            X.grad.data = Xgrad.mean(0)
            optimizerX.step()
            Xgrad = []
        Xgrad.append(X.grad.data)
        optimizerWb.step()
    count,r_loss_cle,r_loss_triple,correct = 0,0,0,0
    for path in (train_models_labels):
        count += 1
        model_path = path
        label = torch.tensor(int(feature_path.split('/')[2])).to(dtype=torch.long).cuda().unsqueeze(0)
        anchor, positive, negative = loadtripledata(tripledataset)
        cnn,cnn_a,cnn_p,cnn_n = Net_basicmodel(),Net_basicmodel(),Net_basicmodel(),Net_basicmodel()
        cnn,cnn_a,cnn_p,cnn_n = cnn.cuda(),cnn_a.cuda(),cnn_p.cuda(),cnn_n.cuda()
        cnn.load_state_dict(torch.load(model_path))
        cnn_a.load_state_dict(anchor[0])
        cnn_p.load_state_dict(positive[0])
        cnn_n.load_state_dict(negative[0])
        cnn.eval()
        cnn_a.eval()
        cnn_p.eval()
        cnn_n.eval()
        out,_ = F(cnn(X).view(1, -1))
        _,F_a = F(cnn_a(X).view(1, -1))
        _,F_p = F(cnn_p(X).view(1, -1))
        _,F_n = F(cnn_n(X).view(1, -1))
        losscle = criterion(out,label)
        reg_loss = (torch.sum(torch.abs(X[:, :, :, :-1] - X[:, :, :, 1:])) +
                                     torch.sum(torch.abs(X[:, :, :-1, :] - X[:, :, 1:, :])))
        losstriple = tripleloss(F_a,F_p,F_n)
        _, predicted = out.max(1)
        correct += predicted.eq(label).sum().item()
        r_loss_cle += losscle.item()
        r_loss_triple += losstriple.item()
    t_acc = correct/count
    t_cle = r_loss_cle/count
    t_triple = r_loss_triple/count
    count,r_loss_cle,correct = 0,0,0
    for path in (testdataset):
        count += 1
        model_path = path
        label = torch.tensor(int(feature_path.split('/')[2])).to(dtype=torch.long).cuda().unsqueeze(0)
        cnn.load_state_dict(torch.load(model_path))
        cnn.eval()
        out,_ = F(cnn(X).view(1, -1))
        losscle = criterion(out,label)
        _, predicted = out.max(1)
        correct += predicted.eq(label).sum().item()
        r_loss_cle += losscle.item()
    e_acc = correct/count
    e_cle = r_loss_cle/count
    query_acc,q_dataset_h,q_dataset_l = query_eval()
    if query_acc > best_acc:
        best_acc = query_acc
        q_dataset_h = q_dataset_h.cpu().detach().numpy()
        q_dataset_l = q_dataset_l.cpu().detach().numpy()
        # np.save(classifier_mode+'_q_dataset_h_vector.npy',q_dataset_h)
        # np.save(classifier_mode+'_q_dataset_l_vector.npy',q_dataset_l)
        state = {'net':F.state_dict(),
                'X':X
        }
        torch.save(state,'./'+classifier_mode+'_vector.pth')
    print('Epoch:{} | Train acc:{:.4f} | Train cle:{:.4f} | Train triple:{:.4f} | Eval acc:{:.4f} | Eval cle:{:.4f} | query acc:{:.4f} | best acc:{:.4f}'.format(
        epoch,t_acc,t_cle,t_triple,e_acc,e_cle,query_acc,best_acc
    ))
    

