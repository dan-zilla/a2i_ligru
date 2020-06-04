from __future__ import absolute_import, division, print_function
import os, pickle
import time
#from sets import Set

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
import torch.nn as nn

os.environ['CUDA_VISIBLE_DEVICES']='1'
# from google_voice.processing.a2i_helper import run_eval
import a2i_helper

device = torch.device('cuda')
torch.set_num_threads(12);

model_dir = './models/'
data_folder = '../outfiles/'
# data_version = 'processed-mlab_chip16_12class_aug' #'processed-12_mel_noisy'
data_version = 'processed-12_mel_noiseless_8bit_quant_env_20p0rms'
# data_version = 'processed-12_mel_noisy'

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

plt.plot(np.arange(100), np.vectorize(a2i_helper.lrf)(np.arange(100)))
# from IPython import embed
# embed()
# exit()
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
        self.fc2    = nn.Linear(in_features=256, out_features=12,  bias=True)

    def forward(self, x):
        #pdb.set_trace()
        #print(self.conv1.weight)
        #print("number of nan = %9.6f; min = %9.6f; max = %9.6f" %(torch.sum(torch.isnan(x)).item(), torch.min(x).item(), torch.max(x).item()))
        x = x.unsqueeze(1)
        x = F.relu(self.conv1(x))
        x = self.res3(self.res2(self.res1(x)))
        x = x.view(-1, self.oc * self.in_shape[0]//4, self.in_shape[1]//4).permute(2, 0, 1).contiguous()
        x = self.recur1(x)[1][0].view(-1, 256)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.fc2(x)
        return x

####### Train Network ########
epochs = 100
batch_size = 256
iters  = (Xtr.shape[0] + batch_size - 1) // batch_size

v = 13
model_F_file = model_dir + 'save_%03d.pt'%v

lr = 1e-3
model = Net(Xtr.shape[1:]).to(device)
obj = nn.CrossEntropyLoss(reduction='sum')
opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.0, amsgrad=True)
scheduler = torch.optim.lr_scheduler.LambdaLR(opt, a2i_helper.lrf)

best_acc = 0
hist = {'TrLoss':[], 'TrAcc':[], 'VaLoss':[], 'VaAcc':[]}

# from IPython import embed
# embed()
# exit()

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
        x = torch.from_numpy(Xtr[use] + noise).float().to(device)
        y = torch.from_numpy(ytr[use]).to(device)
        
        model.train()
        y_pred = model(x)
        #pdb.set_trace()
        
        loss = obj(y_pred, y)
        opt.zero_grad()
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
    
    _, loss, acc = a2i_helper.run_eval(model, Xva, yva, batch_size, device, obj)
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
pte, loss, acc = a2i_helper.run_eval(model, Xte, yte, batch_size, device, obj)
print('TeLoss %.4f - TeAcc %.4f'%(loss, acc))

print()
print('Data version ---- %s'%data_version)
print('Saved in -------- %s'%model_F_file)
print('Results --------- F%.2f%% - T%.2f%%'%(100*best_acc, 100*acc))
a2i_helper.metrics(hist, pte, yte, classes, epochs)

