# Experiment 11 — CNN plant disease detection (synthetic patterns)
import numpy as np
import matplotlib.pyplot as plt
rng = np.random.default_rng(42)
import torch
torch.manual_seed(42)
from torch import nn
x=torch.rand(160,1,16,16); y=torch.arange(160)%2
x[y==0,:,3:6,:]+=.8; x[y==1,:,:,3:6]+=.8
for filters,k,lr,bs in [(4,3,.01,16)]:
    m=nn.Sequential(nn.Conv2d(1,filters,k,padding=k//2),nn.ReLU(),nn.MaxPool2d(2),nn.Conv2d(filters,8,3,padding=1),nn.ReLU(),nn.Flatten(),nn.Linear(512,2))
    opt=torch.optim.Adam(m.parameters(),lr=lr); f=nn.CrossEntropyLoss(); hist=[]; acc=[]
    for e in range(12):
        for ids in torch.randperm(120).split(bs):
            opt.zero_grad(); f(m(x[ids]),y[ids]).backward(); opt.step()
        with torch.no_grad():
            hist.append(f(m(x[120:]),y[120:]).item()); acc.append((m(x[120:]).argmax(1)==y[120:]).float().mean().item())
    print(filters,k,lr,bs,'Accuracy:',acc[-1]); plt.subplot(121); plt.plot(hist,label=str(filters)); plt.legend(); plt.subplot(122); plt.plot(acc,label=str(filters)); plt.legend()
plt.tight_layout()
plt.show()
