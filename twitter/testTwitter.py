import twitter

api = twitter.Api(
    consumer_key='GmuQ9YLwxPHGdItj051V1mV45',
    consumer_secret='9RJNpQQalCPTB8zatDBrt8T2Whnc1pVyP0Zx6u6SZolhcmYSsJ',
    access_token_key='48260361-hvsLeEhUB1KPMSs0oa9GeGRVLdVIaUtxQhYWf9NMc',
    access_token_secret='qn8mIG8wlraZYiy5HIAj1lQXLHNAC8oIqE4chwiqMQfPg')

statuses = api.GetSearch(raw_query='q=apple%20stock%20lang%3Aen%20since%3A2015-12-10%20until%3A2016-12-10%20%3A)')
for status in statuses:
    print status.text
