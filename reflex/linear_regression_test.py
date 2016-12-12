import numpy as np
from sklearn.linear_model import SGDRegressor

regressor = SGDRegressor(alpha=0.05)

for i in range(1000):
    X = np.random.uniform(size=5).reshape(1, -1)
    Y = np.array(np.average(X)).reshape(1, -1)

    regressor.partial_fit(X, Y)


    
X = np.random.uniform(size=5).reshape(1, -1)
Y = np.array(np.average(X)).reshape(1, -1)
Yp = regressor.predict(X)

print Y, Yp

