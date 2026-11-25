# Experiment 4a — Weather entropy
import numpy as np
import matplotlib.pyplot as plt
rng = np.random.default_rng(42)
from scipy.stats import entropy
p=np.array([.5,.3,.2])
print('SciPy:',entropy(p,base=2),'Manual:',-sum(p*np.log2(p)))
plt.bar(['Sunny','Rainy','Cloudy'],p)
plt.tight_layout()
plt.show()

# Experiment 4b — Email cross-entropy
import numpy as np
import matplotlib.pyplot as plt
rng = np.random.default_rng(42)
from sklearn.metrics import log_loss
y=np.array([0,0,1]); p=np.array([.2,.3,.5]); q=np.array([.1,.1,.8])
a=-sum(y*np.log(p)); b=-sum(y*np.log(q))
print('Library:',log_loss([2],[p],labels=[0,1,2]),'Manual:',a,'New:',b)
plt.bar(['Original','Improved'],[a,b])
plt.tight_layout()
plt.show()

# Experiment 4c — KL divergence
import numpy as np
import matplotlib.pyplot as plt
rng = np.random.default_rng(42)
from scipy.stats import entropy
p=np.array([.6,.3,.1]); a=[.5,.4,.1]; b=[.3,.3,.4]
k=[entropy(p,a),entropy(p,b)]
print('KL:',k,'Best:','AB'[np.argmin(k)])
plt.subplot(121); plt.plot(p,label='Actual'); plt.plot(a,label='A'); plt.plot(b,label='B'); plt.legend()
plt.subplot(122); plt.bar(['A','B'],k)
plt.tight_layout()
plt.show()

# Experiment 4d — Medical prediction entropy
import numpy as np
import matplotlib.pyplot as plt
rng = np.random.default_rng(42)
from scipy.stats import entropy
h=[entropy([.7,.2,.1],base=2),entropy([.34,.33,.33],base=2)]
print('Entropy (bits):',h)
plt.bar(['Confident','Uncertain'],h)
plt.tight_layout()
plt.show()
