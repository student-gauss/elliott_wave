import sys
import csv
import datetime
import numpy as np
import collections
import random
import itertools
import matplotlib.pyplot as plt
import pandas as pd

df = pd.DataFrame.from_csv('rote.csv', index_col=None, header=None)
df.columns = ['i', 'Unlearned', 'Total', 'roteRatio']
df = df.apply(pd.to_numeric)
df['Iterations'] = 10 ** df.i

ax = df.plot(x='Iterations',y=['Total', 'Unlearned'], )
ax.set_xlabel('Iterations')
ax.set_xscale('log')
plt.savefig('rote.eps')
plt.close()
