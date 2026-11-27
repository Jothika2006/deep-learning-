# Experiment 19 — PCA and explained variance
import numpy as np
import matplotlib.pyplot as plt
rng = np.random.default_rng(42)
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
z=rng.normal(size=(100,1)); x=z@np.array([[1,2,3,4]])+rng.normal(0,.5,(100,4)); x=StandardScaler().fit_transform(x)
m=PCA().fit(x); p=m.transform(x)
print('Eigenvalues:',m.explained_variance_,'Ratios:',m.explained_variance_ratio_); print('95% components:',np.searchsorted(np.cumsum(m.explained_variance_ratio_),.95)+1)
plt.subplot(121); plt.scatter(x[:,0],x[:,1]); plt.title('Original'); plt.subplot(122); plt.scatter(p[:,0],p[:,1]); plt.title('PCA')
plt.tight_layout()
plt.show()
