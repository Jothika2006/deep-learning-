# Experiment 10 — L2, dropout and early stopping
import numpy as np
import matplotlib.pyplot as plt
rng = np.random.default_rng(42)
import torch, copy
x=torch.rand(200,3); y=(x.sum(1)>1.5).float().reshape(-1,1)
for name in ['Baseline','L2','Dropout','Early stopping']:
    torch.manual_seed(42); m=torch.nn.Sequential(torch.nn.Linear(3,32),torch.nn.ReLU(),torch.nn.Dropout(.4 if name=='Dropout' else 0),torch.nn.Linear(32,1))
    opt=torch.optim.Adam(m.parameters(),lr=.02,weight_decay=.01 if name=='L2' else 0); f=torch.nn.BCEWithLogitsLoss(); hist=[]; best=float('inf'); wait=0
    for e in range(100):
        m.train(); opt.zero_grad(); f(m(x[:100]),y[:100]).backward(); opt.step(); m.eval()
        with torch.no_grad(): val=f(m(x[100:]),y[100:]).item()
        hist.append(val)
        if val<best-1e-4: best=val; state=copy.deepcopy(m.state_dict()); wait=0
        else: wait+=1
        if name=='Early stopping' and wait>=8: break
    if name=='Early stopping': m.load_state_dict(state)
    print(name,'accuracy',((m(x[100:])>0)==y[100:]).float().mean().item()); plt.plot(hist,label=name)
plt.legend()
plt.tight_layout()
plt.show()
