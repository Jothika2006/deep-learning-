# Experiment 5 — Bias–variance analysis
import numpy as np
import matplotlib.pyplot as plt
rng = np.random.default_rng(42)
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
x=rng.uniform(0,100,(200,3)); y=x@np.array([.5,.3,.2])+rng.normal(0,5,200)
a,b,c,d=train_test_split(x,y,random_state=42)
tr=[]; te=[]
for depth in [1,4,None]:
    m=DecisionTreeRegressor(max_depth=depth,random_state=42).fit(a,c)
    tr.append(mean_squared_error(c,m.predict(a))); te.append(mean_squared_error(d,m.predict(b)))
print('Train MSE:',tr,'Test MSE:',te)
plt.plot(tr,label='Train'); plt.plot(te,label='Test'); plt.xticks(range(3),['1','4','Unlimited']); plt.legend()
plt.tight_layout()
plt.show()
