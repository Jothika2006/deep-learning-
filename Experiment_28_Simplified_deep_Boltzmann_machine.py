# Experiment 28 — Simplified deep Boltzmann machine
import numpy as np
import matplotlib.pyplot as plt
rng = np.random.default_rng(42)
x=np.array([[1,1,0,0],[0,0,1,1]],float); w=np.array([[2,-2],[2,-2],[-2,2],[-2,2]]); u=np.eye(2)*2; sig=lambda z:1/(1+np.exp(-z)); h=np.zeros((2,2)); z=h.copy()
for i in range(30):
    h=(rng.random(h.shape)<sig(x@w+z@u.T)).astype(float); z=(rng.random(z.shape)<sig(h@u)).astype(float)
r=sig(h@w.T); print('Hidden:',h,'Reconstruction:',r); plt.subplot(121); plt.imshow(x); plt.title('Original'); plt.subplot(122); plt.imshow(r,vmin=0,vmax=1); plt.title('Reconstructed')
plt.tight_layout()
plt.show()
