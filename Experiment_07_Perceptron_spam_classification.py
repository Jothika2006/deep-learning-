# Experiment 7 — Perceptron: spam classification
import numpy as np
import matplotlib.pyplot as plt
rng = np.random.default_rng(42)
x=np.array([[0,0],[0,1],[1,0],[1,1]]); y=np.array([0,0,1,1]); w=np.zeros(2); b=0
for e in range(10):
    for a,t in zip(x,y):
        err=t-int(a@w+b>0); w+=.1*err*a; b+=.1*err
p=(x@w+b>0).astype(int); print('Actual:',y,'Predicted:',p)
plt.scatter(range(4),y,label='Actual'); plt.scatter(range(4),p,marker='x',label='Predicted'); plt.legend()
plt.tight_layout()
plt.show()
