import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.DataFrame.from_csv('trader_perf_aapl.csv', index_col=None, header=None)
print df
df.columns = ['symbol', 'iterations', 'result']
df[['iterations','result']] = df[['iterations','result']].apply(pd.to_numeric)

ax = df.plot(x='iterations', y='result', label='Gain')
ax.set_xlabel('iterations')
ax.set_xscale('log')
plt.savefig('trader_perf_aapl.pdf')
