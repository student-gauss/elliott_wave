import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
 
samples = []
with open('predictor_performance_dec11.csv', 'r') as dataFile:
    for row in dataFile:
        samples += [row.split()]
        
df = pd.DataFrame(samples)
df.columns = ['index', 'predictor', 'delta', 'loop', 'tp', 'fn', 'fp', 'tn', 'accuracy', 'f1']
df[['delta','loop', 'tp', 'fn', 'fp', 'tn', 'accuracy', 'f1']] = df[['delta','loop', 'tp', 'fn', 'fp', 'tn', 'accuracy', 'f1']].apply(pd.to_numeric)

s = df[(df.predictor=='SimpleNNPredictor') & (df.delta==7)]
s.plot(x='loop', y='accuracy');
plt.savefig('test.png')


head=r'''
\begin{table}
  \begin{tabular}{c|cccccc}
    & TP & FN & FP & TN & Accuracy & F1 Score \\
    \hline
'''

tail=r'''
  \end{tabular}
  \caption{Predictor performance after 630 iteration}
\end{table}
'''

s = df[(df.delta==7) & (df.loop==630)]
print head
for index, row in s.iterrows():
    print '    %s & %4.2f & %4.2f & %4.2f & %4.2f & %4.2f & %4.2f \\' % (
        row['predictor'], row['tp'], row['fn'], row['fp'], row['tn'], row['accuracy'], row['f1'])
print tail
    
