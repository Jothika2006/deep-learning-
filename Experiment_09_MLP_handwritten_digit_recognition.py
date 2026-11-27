# Experiment 9 — MLP handwritten digit recognition
import numpy as np
import matplotlib.pyplot as plt
rng = np.random.default_rng(42)
from sklearn.datasets import load_digits
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay
x,y=load_digits(return_X_y=True); a,b,c,d=train_test_split(x/16,y,random_state=42)
m=MLPClassifier(hidden_layer_sizes=(64,32),max_iter=150,random_state=42).fit(a,c)
print('Accuracy:',m.score(b,d)); print('Sample predictions:',m.predict(b[:5]))
ConfusionMatrixDisplay.from_predictions(d,m.predict(b),ax=plt.gca())
plt.tight_layout()
plt.show()
