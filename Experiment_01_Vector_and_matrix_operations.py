# Experiment 1 — Vector and matrix operations
import numpy as np
import matplotlib.pyplot as plt
rng = np.random.default_rng(42)
x=np.array([[1,2],[3,4]])
w=np.array([[.1,.2,.3,.4],[.4,.3,.2,.1]])
y=w@x.ravel()+np.array([.1,.2])
print('Flattened:',x.ravel(),'Output:',y)
plt.bar(['Neuron 1','Neuron 2'],y)
plt.tight_layout()
plt.show()
