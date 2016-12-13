import random
import numpy as np
from sklearn.linear_model import SGDRegressor
import pandas

regressor = SGDRegressor(alpha=0.05)

df = pandas.read_csv('../data/aapl.csv', header=None, names=['date', 'A', 'B', 'C', 'D', 'E', 'price'])
seq = df.price

hasFit = False
score = 0
for j in range(1000000):
    index = random.choice(range(0, len(seq) - 356 - 8 - 7))
    
    priorPrices = seq[index:(index + 8)].reshape(1, -1)
    targetPrice = np.array(seq[index + 8 + 7]).reshape(1, -1)
    currentPrice = seq[index + 8]

    X = (priorPrices - currentPrice) / currentPrice
    Y = (targetPrice - currentPrice) / currentPrice
    
    if hasFit:
        Yp = regressor.predict(X)[0]
        actualSign  = Y / abs(Y) if Y != 0 else 0
        predictedSign = Yp / abs(Yp) if Yp != 0 else 0
        score += actualSign * predictedSign
        if j % 1000 == 0:
            print score
        
    regressor.partial_fit(X, Y)
    hasFit = True

hasFit = False
score = 0

for index in range(len(seq) - 356, len(seq) - 8 - 7):
    priorPrices = seq[index:(index + 8)].reshape(1, -1)
    targetPrice = np.array(seq[index + 8 + 7]).reshape(1, -1)
    currentPrice = seq[index + 8]
    
    X = (priorPrices - currentPrice) / currentPrice
    Y = (targetPrice - currentPrice) / currentPrice
    
    if hasFit:
        Yp = regressor.predict(X)[0]
        actualSign  = Y / abs(Y) if Y != 0 else 0
        predictedSign = Yp / abs(Yp) if Yp != 0 else 0
        score += actualSign * predictedSign
        
    regressor.partial_fit(X, Y)
    hasFit = True
print score
