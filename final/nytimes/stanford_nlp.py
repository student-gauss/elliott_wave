import json
from pycorenlp import StanfordCoreNLP
nlp = StanfordCoreNLP('http://localhost:9000')

with open('trends.json') as trends_file:    
    trends = json.load(trends_file)

for company in trends:
    data = trends[company]['data']
    for comment in data:
        date = comment['date']
        commentBody = str(''.join([i if ord(i) < 128 else ' ' for i in comment['body']]))
        res = nlp.annotate(commentBody,
                           properties={
                               'annotators': 'sentiment',
                               'outputFormat': 'json'
                           })

        total = 0.001
        score = 0
        if type(res).__name__ != 'dict':
            print type(res).__name__
            continue

        for s in res["sentences"]:
             if s['sentiment'] == 'Positive':
                 score += float(s['sentimentValue'])
                 total += float(s['sentimentValue'])
             elif s['sentiment'] == 'Negative':
                 score -= float(s['sentimentValue'])
                 total += float(s['sentimentValue'])
             else:
                 total += float(s['sentimentValue'])

        comment['sentiment'] = float(score) / total

import json
with open('trends_with_sentiment.json', 'w') as outfile:
    json.dump(trends, outfile, indent=4)
