# Experiment 24 — Graphical sampling and inference
import numpy as np
import matplotlib.pyplot as plt
rng = np.random.default_rng(42)
import networkx as nx
prior=.1; rates=np.array([[.1,.2,.3],[.8,.7,.9]]); d=(rng.random(5000)<prior).astype(int); s=rng.random((5000,3))<rates[d]
for k in [1,2,3]:
    a=prior*np.prod(rates[1,:k]); b=(1-prior)*np.prod(rates[0,:k]); print(k,'symptoms: exact',a/(a+b),'sample',d[s[:,:k].all(1)].mean())
g=nx.DiGraph([('Disease',v) for v in ['Fever','Cough','Fatigue']]); nx.draw(g,nx.spring_layout(g,seed=42),with_labels=True,node_size=1600)
plt.tight_layout()
plt.show()
