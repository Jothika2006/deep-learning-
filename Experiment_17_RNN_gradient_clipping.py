# Experiment 17 — RNN gradient clipping
import numpy as np
import matplotlib.pyplot as plt
rng = np.random.default_rng(42)
import torch
torch.manual_seed(42)
for clip in [False,True]:
    w=torch.tensor(1.2,requires_grad=True); opt=torch.optim.SGD([w],lr=.0001); norms=[]; losses=[]
    for e in range(20):
        opt.zero_grad(); h=torch.tensor(.1)
        for i in range(20): h=w*h+.01
        loss=(h-1)**2; loss.backward(); norms.append(abs(w.grad.item())); losses.append(loss.item())
        if clip: torch.nn.utils.clip_grad_norm_([w],1.)
        opt.step()
        if not np.isfinite(loss.item()): break
    print('Clipping:',clip,'Initial gradient:',norms[0],'Final loss:',losses[-1]); plt.subplot(121); plt.semilogy(norms,label=str(clip)); plt.legend(); plt.subplot(122); plt.semilogy(losses,label=str(clip)); plt.legend()
plt.tight_layout()
plt.show()
