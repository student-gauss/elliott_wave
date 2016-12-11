import requests
import json
import numpy as np
import time

trends = {}
trends['aapl'] = {'keywords': ['aapl', 'apple', 'iphone', 'ipad', 'mac', 'ipod', 'ios'], 'data': []}
trends['ibm'] = {'keywords': ['ibm', 'watson', 'ginni', 'rometty'], 'data': []}

print trends
for date in np.arange('2010-01-01', '2010-11-11', dtype='datetime64[D]'):
    success = False
    while not success:
        time.sleep(1)
        payload = {'api-key': 'a62fe8576c644f34b6a6155e5095a158',
                   'date': str(date)}
        r = requests.get('http://api.nytimes.com/svc/community/v3/user-content/by-date.json', params=payload)
        if r.status_code == 200:
            success = True
        print r
            
    comments = r.json()['results']['comments']
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
    json.dump(trends, outfile)
