import sys
import csv
import datetime
import numpy as np
import collections
import random
import itertools
import matplotlib.pyplot as plt
import pandas as pd

perfDF = None
for D in ['1', '7', '137']:
    perfFileName = 'trader_perf_100_%s_cheat.csv' % D
    df = pd.DataFrame.from_csv(perfFileName, index_col=None, header=None)
    df.columns = ['symbol', 'iteration', D]
    df[[D]] = df[[D]].apply(pd.to_numeric)
    if perfDF is not None:
        perfDF = pd.merge(perfDF, df[['symbol',D]], on='symbol')
    else:
        perfDF = df

perfDF = perfDF.append(perfDF.sum(numeric_only=True), ignore_index=True)        

print perfDF
