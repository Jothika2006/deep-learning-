# Experiment 22 — ICA audio signal separation
import numpy as np
import matplotlib.pyplot as plt
rng = np.random.default_rng(42)
from sklearn.decomposition import FastICA
from scipy.signal import sawtooth
t=np.linspace(0,8,1000); s=np.c_[np.sin(2*t),sawtooth(3*t)]; x=s@np.array([[1,.5],[.4,1]]); p=FastICA(2,random_state=42,whiten='unit-variance').fit_transform(x)
for i,(a,name) in enumerate([(s,'Original'),(x,'Mixed'),(p,'Separated')]): plt.subplot(3,1,i+1); plt.plot(a); plt.title(name)
print('Absolute correlations:',abs(np.corrcoef(s.T,p.T)[:2,2:]))
plt.tight_layout()
plt.show()
