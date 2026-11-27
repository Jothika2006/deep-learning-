# Experiment 23 — Bayesian network weather decisions
import numpy as np
import matplotlib.pyplot as plt
rng = np.random.default_rng(42)
import networkx as nx
g=nx.DiGraph([('Cloud','Rain'),('Humidity','Rain'),('Rain','Umbrella')]); table=np.array([[.05,.25],[.6,.9]])
for c in [0,1]:
    for h in [0,1]: print('Cloud',c,'Humidity',h,'P(Rain)',table[c,h])
print('P(Umbrella|Rain)=0.95; P(Umbrella|No rain)=0.1'); nx.draw(g,nx.spring_layout(g,seed=42),with_labels=True,node_size=1800)
plt.tight_layout()
plt.show()
