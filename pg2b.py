import numpy as np
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score

data = {
    'AND':(np.array([[0,0],[0,1],[1,0],[1,1]]),np.array([0,0,0,1])),
    'OR':(np.array([[0,0],[0,1],[1,0],[1,1]]),np.array([0,1,1,1])),
    'XOR':(np.array([[0,0],[0,1],[1,0],[1,1]]),np.array([0,1,1,0]))
}
for gate,(x,y) in data.items():
    model = Perceptron(max_iter=10,eta0=1,random_state=42)
    model.fit(x,y)
    ypred = model.predict(x)
    acc =accuracy_score(y,ypred)
    print(f"{gate} gate accuracy: {acc:.2f}%")
    print(f"Predictions: {ypred}")  
    print(f"True Labels: {y}")
