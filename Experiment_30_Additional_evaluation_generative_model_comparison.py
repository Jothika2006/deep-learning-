# Additional evaluation — Additional evaluation: generative model comparison
import numpy as np
import matplotlib.pyplot as plt
rng = np.random.default_rng(42)
import torch
torch.manual_seed(42)
from torch import nn
from sklearn.datasets import load_digits
x=torch.tensor(load_digits().data/16,dtype=torch.float32)
enc=nn.Sequential(nn.Linear(64,32),nn.ReLU(),nn.Linear(32,8)); dec=nn.Sequential(nn.Linear(4,32),nn.ReLU(),nn.Linear(32,64),nn.Sigmoid()); opt=torch.optim.Adam(list(enc.parameters())+list(dec.parameters()),lr=.005)
g=nn.Sequential(nn.Linear(4,64),nn.ReLU(),nn.Linear(64,64),nn.Sigmoid()); d=nn.Sequential(nn.Linear(64,32),nn.LeakyReLU(.2),nn.Linear(32,1)); go=torch.optim.Adam(g.parameters(),lr=.002); do=torch.optim.Adam(d.parameters(),lr=.002); bce=nn.BCEWithLogitsLoss()
for e in range(20):
    for ids in torch.randperm(len(x)).split(64):
        a=x[ids]; mu,lv=enc(a).chunk(2,1); r=dec(mu+torch.randn_like(mu)*torch.exp(lv/2)); loss=nn.functional.binary_cross_entropy(r,a,reduction='sum')/len(a)-.5*(1+lv-mu.square()-lv.exp()).sum()/len(a); opt.zero_grad(); loss.backward(); opt.step()
        fake=g(torch.randn(len(a),4)); do.zero_grad(); dl=bce(d(a),torch.ones(len(a),1))+bce(d(fake.detach()),torch.zeros(len(a),1)); dl.backward(); do.step(); go.zero_grad(); gl=bce(d(fake),torch.ones(len(a),1)); gl.backward(); go.step()
with torch.no_grad():
    va=dec(torch.randn(8,4)).numpy(); ga=g(torch.randn(8,4)).numpy(); print('VAE loss:',loss.item(),'GAN generator loss:',gl.item())
    for i,(name,arr) in enumerate([('VAE',va),('GAN',ga)]):
        for j in range(8): plt.subplot(2,8,i*8+j+1); plt.imshow(arr[j].reshape(8,8),cmap='gray'); plt.axis('off'); plt.title(name if j==0 else '')
from sklearn.metrics import pairwise_distances
for name,arr in [('VAE',va),('GAN',ga)]:
    print(name,'Diversity:',arr.var(0).mean(),'Nearest reference MSE:',(pairwise_distances(arr,x.numpy())**2/64).min(1).mean())
plt.tight_layout()
plt.show()
