# Experiment 15 — BPTT gradient flow
import numpy as np
import matplotlib.pyplot as plt
rng = np.random.default_rng(42)
for n in [5,15,30]:
    h=[0.]
    for i in range(n): h.append(np.tanh(.2+.7*h[-1]))
    d=h[-1]-1; grads=[]; dw=0
    for i in range(n,0,-1):
        dz=d*(1-h[i]**2); dw+=dz*h[i-1]; grads.append(abs(d)); d=.7*dz
    print(n,'Loss:',.5*(h[-1]-1)**2,'dW:',dw); plt.semilogy(grads[::-1],label=str(n))
plt.legend()
plt.tight_layout()
plt.show()
