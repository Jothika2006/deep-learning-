# Experiment 6 — Gradient descent, SGD, momentum and Adam
import numpy as np
import matplotlib.pyplot as plt
rng = np.random.default_rng(42)
import torch
torch.manual_seed(42)
x=torch.randn(120,3); y=x@torch.tensor([[2.],[3.],[1.]])
for name in ['GD','SGD','Momentum','Adam']:
    m=torch.nn.Linear(3,1); torch.nn.init.zeros_(m.weight); torch.nn.init.zeros_(m.bias)
    opt=torch.optim.Adam(m.parameters(),lr=.03) if name=='Adam' else torch.optim.SGD(m.parameters(),lr=.03,momentum=.9 if name=='Momentum' else 0)
    losses=[]
    for e in range(60):
        for ids in (torch.randperm(len(x)).split(1) if name=='SGD' else [torch.arange(len(x))]):
            opt.zero_grad(); loss=((m(x[ids])-y[ids])**2).mean(); loss.backward(); opt.step()
        losses.append(((m(x)-y)**2).mean().item())
    print(name,round(losses[-1],5)); plt.plot(losses,label=name)
plt.yscale('log'); plt.legend()
plt.tight_layout()
plt.show()
