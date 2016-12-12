import json
import numpy as np
import time
import os.path

trends = {}
trends['aapl'] = {'keywords': ['aapl', 'apple', 'iphone', 'ipad', 'mac', 'ipod', 'ios'], 'data': []}
trends['ibm'] = {'keywords': ['ibm', 'watson', 'ginni', 'rometty'], 'data': []}

print trends
for date in np.arange('2010-01-01', '2016-11-11', dtype='datetime64[D]'):
    filename = 'nyt_comment_%s.json' % str(date)
    if not os.path.isfile(filename):
        continue

    
    with open(filename, 'r') as jsonFile:
        j = json.load(jsonFile)
            
    comments = j['results']['comments']
    for comment in comments:
        commentBody = comment['commentBody']
        lowerCaseComment = commentBody.lower().split()
        
        for key in trends:
            keywords = trends[key]['keywords']
            contains = False
            for keyword in keywords:
                if keyword in lowerCaseComment:
                    contains = True
                    break

            if contains:
                print key, str(date)
                print commentBody
                
                commentEntry = {'date': str(date), 'body': commentBody}
                trends[key]['data'] += [commentEntry]

import json
with open('trends.json', 'w') as outfile:
    json.dump(trends, outfile, indent=4)
