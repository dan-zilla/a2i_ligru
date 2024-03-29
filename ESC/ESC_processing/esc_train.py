from __future__ import absolute_import, division, print_function
import os, sys, pdb, pickle
from multiprocessing import Process, Pool
import math, time
from sets import Set

import numpy as np
import scipy.io.wavfile as wav
import scipy.signal as sig
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

import torch
import torch.nn.functional as F
import torch.nn as nn

import pdb

#os.environ['CUDA_VISIBLE_DEVICES']=''
device = torch.device('cuda')

model_dir = './models/'
data_folder = '../ESC_outfiles/'
data_version = 'processed-ESC50_fdfilt_fdmix'

######### Load Data ########
with open(data_folder + data_version + '_ys.pkl', 'rb') as f:
    (ytr, yva, yte, classes) = pickle.load(f)
    num_classes = len(classes)
with open(data_folder + data_version + '_Xte.npy', 'rb') as f:
    Xte = np.load(f)
with open(data_folder + data_version + '_Xva.npy', 'rb') as f:
    Xva = np.load(f)
with open(data_folder + data_version + '_Xtr.npy', 'rb') as f:
    Xtr = np.load(f)
print(Xtr.shape,Xva.shape,Xte.shape,ytr.shape,yva.shape,yte.shape)

mu = np.mean(Xtr[Xtr>-200], 0)
std = np.std(Xtr[Xtr>-200], 0)
Xtr = ((Xtr - mu) / std)
Xva = ((Xva - mu) / std)
Xte = ((Xte - mu) / std)

def to_categorical(y, num_classes):
    return np.eye(num_classes, dtype='uint8')[y]
Ytr = to_categorical(ytr, num_classes)
Yva = to_categorical(yva, num_classes)
Yte = to_categorical(yte, num_classes)

print(np.bincount(ytr))
print(np.bincount(yva))
print(np.bincount(yte))

#pdb.set_trace()

######## Helper Functions ###########
def run_eval(model, Xev, yev):
    yp = []
    cur_loss = 0
    cur_acc  = 0
    model.eval()
    for step in range((Xev.shape[0] + batch_size - 1) // batch_size):
        x = torch.from_numpy(Xev[step * batch_size : (step + 1) * batch_size]).float().to(device)
        y = torch.from_numpy(yev[step * batch_size : (step + 1) * batch_size]).to(device)
        with torch.no_grad():
            y_pred = model(x)
        yp.append(y_pred.cpu().numpy())
        cur_loss += obj(y_pred, y).item()
        cur_acc  += np.sum((y_pred.max(dim=1)[1] == y).cpu().numpy())
    return np.argmax(np.vstack(yp), -1), cur_loss / Xev.shape[0], cur_acc / Xev.shape[0]

def metrics(hist, pte):
    # https://matplotlib.org/gallery/images_contours_and_fields/image_annotated_heatmap.html
    fig, (ax1, ax2, axc) = plt.subplots(1,3,figsize=(18, 6))

    ax1.plot(range(epochs), hist['TrLoss'], 'ro-', label='Training')
    ax1.plot(range(epochs), hist['VaLoss'], 'bo-', label='Validation')
    ax1.legend()
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss')

    ax2.plot(range(epochs), hist['TrAcc'], 'ro-', label='Training')
    ax2.plot(range(epochs), hist['VaAcc'], 'bo-', label='Validation')
    ax2.legend()
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training Accuracy')

    conf_te = confusion_matrix(yte, pte)
    im = axc.imshow(np.log(1 + conf_te), cmap='jet')
    axc.set_xticks(range(num_classes))
    axc.set_yticks(range(num_classes))
    axc.set_xticklabels(classes)
    axc.set_yticklabels(classes)
    plt.setp(axc.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
    for i in range(num_classes):
        for j in range(num_classes):
            text = axc.text(j,i,conf_te[i,j], ha='center', va='center', color='w')
    axc.set_title('Confusion Matrix')
    axc.grid(False)

    fig.tight_layout()
    plt.show()

def lrf(epoch):
    ''' Cosine Annealing: https://arxiv.org/pdf/1608.03983.pdf '''
    jumps = [0, 20, 50, 100]
    for i in range(1, len(jumps)):
        if epoch < jumps[i]: return (1 + np.cos(np.pi * (epoch - jumps[i-1]) / (jumps[i] - jumps[i-1]))) / 2
    return 1e-6
plt.plot(np.arange(100), np.vectorize(lrf)(np.arange(100)))
plt.show()

####### Define Network #########
class ResLayer(nn.Module):
    def __init__(self, in_channels, out_channels, momentum=0.05, maxpool=False):
        super(ResLayer, self).__init__()
        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.maxpool      = maxpool
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=(1,1), padding=(0,0), bias=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=(3,3), padding=(1,1), bias=True)
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=(3,3), padding=(1,1), bias=True)
        self.bn = nn.BatchNorm2d(out_channels, momentum=momentum)
    
    def forward(self, x):
        x = self.conv1(x)
        xi = x
        x = F.relu(self.conv2(x))
        x = self.bn(self.conv3(x))
        x = F.relu(x + xi)
        if self.maxpool: x = F.max_pool2d(x, 2, 2)
        return x

class Net(nn.Module):
    def __init__(self, in_shape, quant_bits=(4,8,4)):
        super(Net, self).__init__()
        self.in_shape = in_shape
        self.oc = 128
        self.conv1  = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(3,3), stride=(1,1), padding=(1,1), bias=True)
        self.res1   = ResLayer(in_channels=32, out_channels=32,      momentum=0.05, maxpool=True)
        self.res2   = ResLayer(in_channels=32, out_channels=64,      momentum=0.05, maxpool=True)
        self.res3   = ResLayer(in_channels=64, out_channels=self.oc, momentum=0.05, maxpool=False)
        self.recur1 = nn.LSTM(self.oc * self.in_shape[0]//4, 256)
        self.fc1    = nn.Linear(in_features=256, out_features=256, bias=True)
        self.fc2    = nn.Linear(in_features=256, out_features=50,  bias=True)

    def forward(self, x):
        #pdb.set_trace()
        #print(self.conv1.weight)
        #print("number of nan = %9.6f; min = %9.6f; max = %9.6f" %(torch.sum(torch.isnan(x)).item(), torch.min(x).item(), torch.max(x).item()))
        x = x.unsqueeze(1)
        x = F.relu(self.conv1(x))
        x = self.res3(self.res2(self.res1(x)))
        x = x.view(-1, self.oc * self.in_shape[0]//4, self.in_shape[1]//4).permute(2, 0, 1).contiguous()
        #pdb.set_trace()
        x = self.recur1(x)[1][0].view(-1, 256)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.fc2(x)
        return x

####### Train Network ########
epochs = 100
batch_size = 25
iters  = (Xtr.shape[0] + batch_size - 1) // batch_size
v = 1
model_F_file = model_dir + 'save_%03d.pt'%v

lr = 1e-3
model = Net(Xtr.shape[1:]).to(device)
obj = nn.CrossEntropyLoss(reduction='sum')
opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.0, amsgrad=True)
scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lrf)

best_acc = 0
hist = {'TrLoss':[], 'TrAcc':[], 'VaLoss':[], 'VaAcc':[]}
for epoch in range(epochs):
    scheduler.step()
    cur_samp = 0
    cur_loss = 0
    cur_acc  = 0
    t0 = time.time()
    for step in range(iters):
        use = np.random.choice(range(Xtr.shape[0]), batch_size, replace=False)
        noise = np.random.uniform(0, 0.3) * np.random.randn(batch_size, *Xtr.shape[1:])
        noise += np.random.uniform(0, 0.3) * np.random.randn(batch_size, Xtr.shape[1], 1)
        opt.zero_grad()
        x = torch.from_numpy(Xtr[use] + noise).float().to(device)
        y = torch.from_numpy(ytr[use]).to(device)
        
        model.train()
        y_pred = model(x)
        
        loss = obj(y_pred, y)
        loss.backward()
        opt.step()
        
        cur_samp += batch_size
        cur_loss += loss.item()
        cur_acc  += np.sum((y_pred.max(dim=1)[1] == y).cpu().numpy())
        lr = scheduler.get_lr()[-1]
        print('\rEpoch %03d/%03d - Step %03d/%03d (%5.1f%%) - LR %.2e - TrLoss %.4f - TrAcc %.4f'%(
            epoch+1, epochs, step+1, iters, 100*(step+1)/iters, lr, cur_loss / cur_samp, cur_acc / cur_samp), end='')
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)
#        for p in model.parameters():
#            p.data.add_(-lr, p.grad.data)
        #pdb.set_trace()
    
    _, loss, acc = run_eval(model, Xva, yva)
    if acc > best_acc:
        best_acc = acc
        cur_best = True
        torch.save(model.state_dict(), model_F_file)
    else:
        cur_best = False
    print(' - VaLoss %.4f - VaAcc %.4f - Time %4.1fs %s'%(loss, acc, time.time() - t0, '*' if cur_best else ''))
    hist['TrLoss'].append(cur_loss / cur_samp)
    hist['TrAcc' ].append(cur_acc / cur_samp)
    hist['VaLoss'].append(loss)
    hist['VaAcc' ].append(acc)


model.load_state_dict(torch.load(model_F_file))
pte, loss, acc = run_eval(model, Xte, yte)
print('TeLoss %.4f - TeAcc %.4f'%(loss, acc))

print()
print('Data version ---- %s'%data_version)
print('Saved in -------- %s'%model_F_file)
print('Results --------- F%.2f%% - T%.2f%%'%(100*best_acc, 100*acc))
metrics(hist, pte)

