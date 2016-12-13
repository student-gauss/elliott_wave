import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
 
samples = []
with open('predictor_perform_dec13.csv', 'r') as dataFile:
    for row in dataFile:
        samples += [row.split()]
        
df = pd.DataFrame(samples)
df.columns = ['predictor', 'delta', 'loop', 'tp', 'fn', 'fp', 'tn', 'accuracy', 'f1']
df[['delta','loop', 'tp', 'fn', 'fp', 'tn', 'accuracy', 'f1']] = df[['delta','loop', 'tp', 'fn', 'fp', 'tn', 'accuracy', 'f1']].apply(pd.to_numeric)

for predictor in ['SimpleNNPredictor', 'LinearPredictor', 'PatternPredictor']:
    s = df[(df.predictor==predictor) & (df.delta==7)]
    ax = s.plot(x='loop', y='f1', label='F1 score')
    ax.set_xlabel('log(iterations)')
    ax.set_xscale('log')
    plt.savefig('pred-perf-%s-7-f1.eps' % predictor)

confusionMatrix=r'''
\begin{table}
  \begin{tabular}{cc}
    \begin{tabular}{cc|cc}
      & & \multicolumn{2}{c}{Predicted} \\
      & & $+ $ & $-$ \\
      \hline
      \multirow{2}{*}{Actual}
      & $+$ & %4.2f & %4.2f \\
      & $-$ & %4.2f & %4.2f \\
      \hline
    \end{tabular}
    &
    \begin{tabular}{cc}
      Accuracy & %4.2f \\
      F1 Score & %4.2f \\
    \end{tabular}
  \end{tabular}
  \caption{%s, $D=%d$, %d iterations}
  \label{pred-perf-%s-%d-%d}
\end{table}
'''

s = df[(df.loop==398)]
for index, row in s.iterrows():
    print confusionMatrix % (
        row['tp'], row['fn'], row['fp'], row['tn'],
        row['accuracy'], row['f1'],
        row['predictor'], row['delta'], row['loop'],
        row['predictor'], row['delta'], row['loop'])

    
