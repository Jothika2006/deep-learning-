# Experiment 26 — Gibbs sampling and contrastive divergence
import numpy as np
import matplotlib.pyplot as plt
rng = np.random.default_rng(42)
x=np.array([[1,1,0,0],[1,0,0,0],[0,0,1,1],[0,0,0,1]],float)
sig=lambda z:1/(1+np.exp(-z))
for k in [1,3,5]:
    w=rng.normal(0,.1,(4,2)); a=np.zeros(4); b=np.zeros(2); errors=[]
    for e in range(400):
        p=sig(x@w+b); v=x.copy(); h=p
        for j in range(k):
            hs=(rng.random(h.shape)<h).astype(float); v=(rng.random(v.shape)<sig(hs@w.T+a)).astype(float); h=sig(v@w+b)
        w+=.1*(x.T@p-v.T@h)/len(x); a+=.1*(x-v).mean(0); b+=.1*(p-h).mean(0)
        rec=sig(sig(x@w+b)@w.T+a); errors.append(np.mean((x-rec)**2))
    print('CD',k,'MSE',errors[-1],'Energy',-x[0]@w@np.array([1,0])-x[0]@a-b[0]); print(np.round(rec,2)); plt.plot(errors,label=str(k))
plt.legend()
plt.tight_layout()
plt.show()
