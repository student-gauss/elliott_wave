import csv
import datetime
import numpy as np

def load(key):
    dateToPrice = {}
    with open('../data/%s.csv' % key, 'rb') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            date = np.datetime64(row[0])

            ratio = 1.0
            if key.startswith('aapl') and date < np.datetime64('2014-06-09'):
                ratio = 7.0    # split
            price = float(row[1]) / ratio

            dateToPrice[date] = price
    return dateToPrice

def truth(dateToPrice, date):
    while not date in dateToPrice:
        date = date - np.timedelta64(1, 'D')
    return dateToPrice[date]

labels = {
    'dj' : [
        (np.datetime64('2016-02-11'), 15499),
	(np.datetime64('2016-02-22'), 16660),
	(np.datetime64('2016-02-24'), 16165),
	(np.datetime64('2016-03-22'), 15651),
	(np.datetime64('2016-04-07'), 17472),
	(np.datetime64('2016-04-20'), 18162),
	(np.datetime64('2016-05-19'), 17326),
	(np.datetime64('2016-06-23'), 18007),
	(np.datetime64('2016-06-27'), 17052)],

    'aapl1' : [
	(np.datetime64('2011-06-20'), 43.69), 
	(np.datetime64('2011-10-10'), 60.59),
	(np.datetime64('2011-11-21'), 51.96),
	(np.datetime64('2012-04-09'), 91.94),
	(np.datetime64('2012-05-14'), 74.63),
	(np.datetime64('2012-09-17'), 101.04)],
    
    'aapl2' : [
        (np.datetime64('2013-06-24'), 55.09),
	(np.datetime64('2013-12-09'), 81.86),
	(np.datetime64('2014-01-27'), 70.73),
	(np.datetime64('2014-11-24'), 119.75),
	(np.datetime64('2015-01-05'), 104.38),
	(np.datetime64('2015-04-27'), 134.54),
	(np.datetime64('2015-08-24'), 92.27),
	(np.datetime64('2015-11-02'), 123.88),
	(np.datetime64('2016-05-09'), 89.4)],

    'gdx' : [
	(np.datetime64('2006-10-02'), 32.3),
	(np.datetime64('2007-02-19'), 43.31),
	(np.datetime64('2007-08-13'), 32.8),
	(np.datetime64('2007-11-05'), 53.8),
	(np.datetime64('2007-12-17'), 42.3),
	(np.datetime64('2008-03-10'), 56.88)],

    'qcom' : [
        (np.datetime64('2002-08-05'), 11.61),
	(np.datetime64('2002-12-02'), 21.41),
	(np.datetime64('2003-05-12'), 14.84),
	(np.datetime64('2004-12-20'), 45),
	(np.datetime64('2005-04-18'), 32),
	(np.datetime64('2006-05-01'), 53)],

    'rut' : [
        (np.datetime64('2011-10-03'), 603.24),
	(np.datetime64('2012-03-26'), 847),
	(np.datetime64('2012-11-12'), 764),
	(np.datetime64('2014-03-03'), 1212),
	(np.datetime64('2014-10-13'), 1038),
	(np.datetime64('2015-06-22'), 1296),
	(np.datetime64('2015-08-24'), 1101),
	(np.datetime64('2015-11-30'), 1205),
	(np.datetime64('2016-02-08'), 939)],
    }


for key, label in labels.items():
    dateToPrice = load(key)

    prevDate = None
    prevPrice = None
    diff = []
    for labelDate, labelPrice in label:
        if prevDate == None:
            prevDate = labelDate
            prevPrice = labelPrice
            continue

        duration = (labelDate - prevDate) / np.timedelta64(1, 'D')
        slope = (labelPrice - prevPrice) / duration

        date = prevDate
        while date < labelDate:
            delta = (date - prevDate) / np.timedelta64(1, 'D')
            prediction = prevPrice + slope * delta
            
            diff += [prediction - truth(dateToPrice, date)]
            date += np.timedelta64(1, 'D')

    print "%s %f" % (key, sum(map(lambda x:x ** 2, diff)) / len(diff))
