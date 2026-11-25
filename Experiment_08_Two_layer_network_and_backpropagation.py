# Experiment 8 — Two-layer network and backpropagation
import numpy as np
import matplotlib.pyplot as plt
rng = np.random.default_rng(42)
x=rng.random((100,2)); y=(x.sum(1)>1).reshape(-1,1)
w=rng.normal(0,.3,(2,4)); v=rng.normal(0,.3,(4,1)); b=np.zeros((1,4)); c=0
sig=lambda z:1/(1+np.exp(-z))
loss=[]
for e in range(500):
    h=sig(x@w+b); p=sig(h@v+c); d=(p-y)/len(x)
    loss.append(-np.mean(y*np.log(p)+(1-y)*np.log(1-p)))
    dh=(d@v.T)*h*(1-h); v-=.5*h.T@d; c-=.5*d.sum(); w-=.5*x.T@dh; b-=.5*dh.sum(0)
print('Initial/final loss:',loss[0],loss[-1]); plt.plot(loss)
plt.tight_layout()
plt.show()
