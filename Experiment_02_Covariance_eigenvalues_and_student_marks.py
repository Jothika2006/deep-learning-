# Experiment 2 — Covariance, eigenvalues and student marks
import numpy as np
import matplotlib.pyplot as plt
rng = np.random.default_rng(42)
x=np.array([[80,75,90],[65,70,72],[95,90,98],[70,82,76],[88,85,92]])
print('Totals:',x.sum(1),'Means:',x.mean(1))
print('Best student:',x.mean(1).argmax()+1)
print('Product:',x[:,0]*x[:,1])
c=np.cov(x,rowvar=False); v,u=np.linalg.eigh(c)
print('Covariance:',c,'Eigenvalues:',v,'Eigenvectors:',u,sep='\n')
plt.bar(['Math','Physics','CS'],x.mean(0))
plt.tight_layout()
plt.show()
