# Experiment 3 — Bayes theorem: medical diagnosis
import numpy as np
import matplotlib.pyplot as plt
rng = np.random.default_rng(42)
p=.01; s=.99; f=.05
q=s*p/(s*p+f*(1-p))
print('Posterior:',round(q*100,2),'%')
plt.bar(['Prior','Posterior'],[p,q])
plt.tight_layout()
plt.show()
