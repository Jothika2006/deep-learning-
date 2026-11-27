# Experiment 27 — Deep belief network feature learning
import numpy as np
import matplotlib.pyplot as plt
rng = np.random.default_rng(42)
from sklearn.datasets import load_digits
from sklearn.neural_network import BernoulliRBM
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
x,y=load_digits(return_X_y=True); a,b,c,d=train_test_split(x/16,y,random_state=42)
r=BernoulliRBM(n_components=32,n_iter=15,learning_rate=.05,random_state=42).fit(a); s=BernoulliRBM(n_components=16,n_iter=15,learning_rate=.05,random_state=42).fit(r.transform(a)); f=lambda z:s.transform(r.transform(z)); m=LogisticRegression(max_iter=500).fit(f(a),c)
print('Accuracy:',m.score(f(b),d)); plt.imshow(r.components_[:8],aspect='auto'); plt.colorbar()
plt.tight_layout()
plt.show()
