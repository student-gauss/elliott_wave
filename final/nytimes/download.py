import requests
import json
import numpy as np
import time
import json
import os.path

for date in np.arange('2012-03-19', '2016-11-11', dtype='datetime64[D]'):
    success = False
    filename = 'nyt_comment_%s.json' % str(date)
    if os.path.isfile(filename):
        print 'existed: ', filename
        continue
    
    time.sleep(1)
    payload = {'api-key': '0341df8da5964d7382b19595cf79fe37',
               'date': str(date)}
    r = requests.get('http://api.nytimes.com/svc/community/v3/user-content/by-date.json', params=payload)
    if r.status_code == 200:
        with open(filename, 'w') as outfile:
            json.dump(r.json(), outfile)
            print 'Success: ', filename
    else:
        print 'failed: ', r
