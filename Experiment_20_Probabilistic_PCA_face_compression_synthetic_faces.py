# Experiment 20 — Probabilistic PCA face compression (synthetic faces)
import numpy as np
import matplotlib.pyplot as plt
rng = np.random.default_rng(42)
a,b=np.mgrid[-1:1:16j,-1:1:16j]; x=np.array([np.exp(-((a-u)**2+b*b)*3)-.5*np.exp(-((a-.2)**2+(abs(b)-.3)**2)*60) for u in rng.uniform(-.3,.3,80)]).reshape(80,-1); x+=rng.normal(0,.03,x.shape)
mu=x.mean(0); _,s,v=np.linalg.svd(x-mu,full_matrices=False); eig=s*s/len(x)
plt.subplot(131); plt.imshow(x[0].reshape(16,16),cmap='gray'); plt.title('Original')
for i,k in enumerate([2,10]):
    noise=(eig[k:].sum())/(x.shape[1]-k); w=v[:k].T*np.sqrt(np.maximum(eig[:k]-noise,0)); z=(x-mu)@w@np.linalg.inv(w.T@w+noise*np.eye(k)); rec=z@w.T+mu
    print(k,'MSE:',np.mean((x-rec)**2)); plt.subplot(132+i); plt.imshow(rec[0].reshape(16,16),cmap='gray'); plt.title(str(k)+' factors')
plt.tight_layout()
plt.show()
