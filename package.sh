#!/bin/sh
rm code.zip
rm data.zip
zip code.zip final/plotPredictorPerformance.py
zip code.zip final/plot_trader_perf.py
zip code.zip final/predictor.py
zip code.zip final/trader.py
zip code.zip final/testPredictor.py
zip code.zip final/testTrader.py
zip code.zip final/nytimes/download.py
zip code.zip final/nytimes/nytimes.py
zip code.zip final/nytimes/stanford_nlp.py
zip code.zip final/README

zip data.zip data/*.csv
zip data.zip final/predictor_perform_*.csv
zip data.zip final/trader_perf_*.csv
zip data.zip final/nytimes/nyt_comment_2014*.json
zip data.zip final/nytimes/nyt_comment_2015*.json
zip data.zip final/nytimes/nyt_comment_2016*.json
zip data.zip final/nytimes/trends.json
zip data.zip final/nytimes/trends_with_sentiment.json
