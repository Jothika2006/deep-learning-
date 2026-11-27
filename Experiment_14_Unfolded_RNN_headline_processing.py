# Experiment 14 — Unfolded RNN headline processing
import numpy as np
import matplotlib.pyplot as plt
rng = np.random.default_rng(42)
import networkx as nx
words=['Rain','hits','city']; h=0; g=nx.DiGraph(); pos={}
for i,w in enumerate(words):
    h=np.tanh(.3*(i+1)+.5*h); print(w,round(h,3))
    a,b,c=w,'h'+str(i),'y'+str(i); g.add_edges_from([(a,b),(b,c)]); pos.update({a:(i,0),b:(i,1),c:(i,2)})
    if i: g.add_edge('h'+str(i-1),b)
nx.draw(g,pos,with_labels=True,node_size=1200)
plt.tight_layout()
plt.show()
