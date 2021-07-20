import sys
import time
import os
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from PIL import Image
from torchvision import transforms
import torchvision
import torch.nn as nn

Transform = transforms.Compose([
    transforms.Resize(256), 
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

class Normalize(nn.Module) :
    def __init__(self, mean, std) :
        super(Normalize, self).__init__()
        self.register_buffer('mean', torch.Tensor(mean))
        self.register_buffer('std', torch.Tensor(std))
        
    def forward(self, input):
        # Broadcasting
        mean = self.mean.reshape(1, 3, 1, 1)
        std = self.std.reshape(1, 3, 1, 1)
        return (input - mean) / std

norm_layer = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
def saveimg(ori,posioned,path):
    ori = ori.cpu()/255
    posioned = torch.round(posioned).cpu()/255
    res = torch.abs(ori-posioned)
    res = res / res.max()
    if not(os.path.exists(path)):
        os.makedirs(path)
    imgpath = path + '.bmp'
    result = torch.cat((ori,posioned,res),3)
    torchvision.utils.save_image(result,imgpath)


def Highpasscorrection(params):
    tmp = params.weight.data.detach().clone()
    m = torch.mean(tmp, dim=(2, 3)).unsqueeze(2).unsqueeze(2)
    tmp = tmp - m
    params.weight.data = tmp

class ImageNet(Dataset):
    def __init__(self, path):
        super(ImageNet, self).__init__()
        self.path = path
        self.imgslist = os.listdir(self.path)

    def __getitem__(self, index):
        # randnum = torch.round(torch.rand(1)*1) + 1

        img = Transform(Image.open(self.path + self.imgslist[index]))
        a,b,c = img.shape
        img = img * 255
        # noise = torch.round((torch.rand_like(img)-0.5)*20)
        # noise[noise>randnum] = 0
        # noise[noise<-randnum] = 0
        # label = torch.sum(torch.abs(noise))
        return img#,noise,label/(a*b*c)

    def __len__(self):
        return len(self.imgslist)


_, term_width = os.popen('stty size', 'r').read().split()
term_width = int(term_width)
TOTAL_BAR_LENGTH = 65.
last_time = time.time()
begin_time = last_time


def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width-int(TOTAL_BAR_LENGTH/2)+2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current+1, total))

    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()
