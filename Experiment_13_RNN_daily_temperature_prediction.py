# Experiment 13 — RNN daily temperature prediction
import numpy as np
import matplotlib.pyplot as plt
rng = np.random.default_rng(42)
import torch
torch.manual_seed(42)
from torch import nn
t=np.arange(240); series=25+5*np.sin(t/10)+rng.normal(0,.5,240)
z=(series-series[:160].mean())/series[:160].std()
x=torch.tensor(np.array([z[i:i+12] for i in range(228)]),dtype=torch.float32).unsqueeze(-1); y=torch.tensor(z[12:],dtype=torch.float32).unsqueeze(-1)
for kind,h,lr,bs in [('RNN',16,.01,16)]:
    r=getattr(nn,kind)(1,h,batch_first=True); out=nn.Linear(h,1)
    opt=torch.optim.Adam(list(r.parameters())+list(out.parameters()),lr=lr)
    def pred(a): return out(r(a)[0][:,-1])
    for e in range(25):
        for ids in torch.randperm(148).split(bs):
            opt.zero_grad(); ((pred(x[ids])-y[ids])**2).mean().backward(); opt.step()
    with torch.no_grad(): p=pred(x[148:]).numpy().ravel()*series[:160].std()+series[:160].mean()
    print(kind,h,lr,bs,'MSE:',np.mean((p-series[160:])**2)); plt.plot(p,label=kind+str(h))
plt.plot(series[160:],label='Actual'); plt.legend()
plt.tight_layout()
plt.show()
