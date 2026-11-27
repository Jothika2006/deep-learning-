# Experiment 21 — Factor analysis customer satisfaction
import numpy as np
import matplotlib.pyplot as plt
rng = np.random.default_rng(42)
from sklearn.decomposition import FactorAnalysis
z=rng.normal(size=(200,2)); x=z@np.array([[1,1,.1,.2,1],[.1,.2,1,1,.1]])+rng.normal(0,.2,(200,5)); m=FactorAnalysis(2,random_state=42).fit(x)
print('Loadings:',m.components_.T); plt.imshow(m.components_.T,cmap='coolwarm'); plt.yticks(range(5),['Service','Product','Price','Delivery','Support']); plt.colorbar()
plt.tight_layout()
plt.show()
