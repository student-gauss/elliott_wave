import json
import numpy as np
import time
import os.path

trends = {}
trends['aapl'] = {'keywords': ['aapl', 'apple', 'iphone', 'ipad', 'mac', 'ipod', 'ios'], 'data': []}
trends['ibm'] = {'keywords': ['ibm', 'watson', 'ginni', 'rometty'], 'data': []}
trends['bp'] = {'keywords': ['bp', 'dudley', 'oil', 'major'], 'data': []}
trends['cop'] = {'keywords': ['conocophillips', 'lance', 'oil', 'major'], 'data': []}
trends['cost'] = {'keywords': ['costco', 'brotman', 'wholesale'], 'data': []}
trends['cvx'] = {'keywords': ['chevron', 'oil', 'gas'], 'data': []}
trends['dj'] = {'keywords': ['dow', 'jones', 'dj'], 'data': []}
trends['hd'] = {'keywords': ['home', 'depot', 'hd'], 'data': []}
trends['ko'] = {'keywords': ['coca', 'cola', 'ko'], 'data': []}
trends['low'] = {'keywords': ['lowe', 'lowe\'s'], 'data': []}
trends['nke'] = {'keywords': ['nike', 'nke'], 'data': []}
trends['qcom'] = {'keywords': ['qualcomm', 'qcom'], 'data': []}
trends['rut'] = {'keywords': ['russell', 'rut'], 'data': []}
trends['tgt'] = {'keywords': ['target', 'tgt'], 'data': []}
trends['wmt'] = {'keywords': ['wallmart', 'wmt'], 'data': []}
trends['xcom'] = {'keywords': ['xcom', 'xtera'], 'data': []}

commentCount = 0
for date in np.arange('2010-01-01', '2016-11-11', dtype='datetime64[D]'):
    filename = 'nyt_comment_%s.json' % str(date)
    if not os.path.isfile(filename):
        continue

    
    with open(filename, 'r') as jsonFile:
        j = json.load(jsonFile)
            
    comments = j['results']['comments']
    for comment in comments:
        commentCount += 1
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
                
                commentEntry = {'date': str(date), 'body': commentBody}
                trends[key]['data'] += [commentEntry]

import json
with open('trends.json', 'w') as outfile:
    json.dump(trends, outfile, indent=4)

print commentCount
for company in trends:
    data = trends[company]['data']
    print '%s %d' % (company, len(data))
    
