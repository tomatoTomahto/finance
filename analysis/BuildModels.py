# # BuildModels.py
# # Overview
# This script generates machine learning models used to predict the daily price change
# of NASDAQ and NYSE stocks using historical stock, news, and analyst information

# # Check to see files are available in HDFS
# Note: you should copy the data from 
#!hdfs dfs -ls finance/data

# # Imports
from pyspark.sql import SparkSession, functions as F
from pyspark import StorageLevel
from pyspark.ml import Pipeline
import os, sys
sys.path.append('finance/analysis')
import StockModel as SM, ArticleModel as AM, AnalystModel as RM

# # Spark Connection
spark = SparkSession \
  .builder \
  .appName("Finance Analysis") \
  .getOrCreate()

# # Model 1: Predict long and short positions based on historic stock information (price, volume, etc) 
# ## Load in Historical prices for all stocks on NASDAQ and NYSE exchanges
prices = SM.loadData(spark,
                     'finance/data/companylist_NASDAQ.csv',
                     'finance/data/companylist_NYSE.csv',
                     'finance/data/historicPrices.json')
prices.printSchema()

# ## Transform the data - calculate price changes, load historical price changes
priceChanges = SM.transformData(prices)
priceChanges.persist(StorageLevel.MEMORY_ONLY)
priceChanges.show(5)

# ## Build a pipeline that does all feature transformations and model generation
stages = SM.engineerFeatures()
stockPipeline = Pipeline(stages=stages)
(trainingData, testData) = priceChanges.randomSplit([0.7, 0.3])
stockRfrModel = stockPipeline.fit(trainingData)

# ## Run model against test data, evaluate our daily long and short stock rankings
stockPredictions = stockRfrModel.transform(testData)
SM.evaluatePredictions(stockPredictions, 1000)

# ## Save model in HDFS
stockRfrModel.write().overwrite().save('stockRfr.mdl')

# # Model 2: Predict long and short positions based on historic analyst ratings
# ## Load in historical analyst ratings from Yahoo Finance!
ratings = RM.loadData(spark, 'finance/data/analystRatings.json')
ratings.printSchema()
ratings.show(5)

# ## Transform the data
modelData = ratings.join(priceChanges,
                         (F.date_add(ratings.Date,1) == priceChanges.Date) & \
                          (ratings.Symbol == priceChanges.Symbol))\
  .select(priceChanges.Date, 
          priceChanges.Symbol,
          priceChanges.PriceChange,
          'Action',
          'Research Firm',
          'From',
          'To')
modelData.persist(StorageLevel.MEMORY_ONLY)
modelData.show(5)

# ## Build a pipeline that does all feature transformations and model generation
stages = RM.engineerFeatures()
ratingsPipeline = Pipeline(stages=stages)
(trainingData, testData) = modelData.randomSplit([0.7, 0.3])
ratingsRfrModel = ratingsPipeline.fit(trainingData)

# ## Run model against test data, evaluate our daily long and short stock rankings
analystPredictions = ratingsRfrModel.transform(testData)
SM.evaluatePredictions(analystPredictions, 1000)

# ## Save model in HDFS
ratingsRfrModel.write().overwrite().save('ratingsRfr.mdl')

# # Model 3: Predict long and short positions based on historic article information (keywords, topics, etc)
# ## Load in historical articles from NYTimes
articles = AM.loadData(spark, 'finance/data/historicArticles/*.json')

# ## Transform the data - extract relevant info and join articles with stock prices
articleFeatures = AM.transformData(articles)
modelData = articleFeatures.join(priceChanges,
                                 F.date_add(articleFeatures.pub_date,1) == priceChanges.Date)\
  .filter(F.year('Date')>=2016)
modelData.persist(StorageLevel.MEMORY_ONLY)
modelData.show(5)

# ## Build a pipeline that does all feature transformations and model generation
stages = AM.engineerFeatures()
articlePipeline = Pipeline(stages=stages)
(trainingData, testData) = modelData.randomSplit([0.7, 0.3])
articleRfrModel = articlePipeline.fit(trainingData)

# ## Run model against test data, evaluate our daily long and short stock rankings
articlePredictions = articleRfrModel.transform(testData)
articlePredictions.select('PriceChange','pPriceChange').describe().show()
articlePredictions.show(5)
SM.evaluatePredictions(articlePredictions, 1000)

# ## Save model in HDFS
articleRfrModel.write().overwrite().save('articlesRfr.mdl')

# # Aggregate all predictions
p1 = stockPredictions.select('Date','Symbol',
                             stockPredictions.pPriceChange.alias('spPriceChange'))
p1.show()
p2 = analystPredictions.select('Date','Symbol',
                               analystPredictions.pPriceChange.alias('rpPriceChange'))
p2.show()
p3 = articlePredictions.select('Date','Symbol',
                               articlePredictions.pPriceChange.alias('apPriceChange'))
p3.show()

p1.join(p2, ['Date','Symbol']).show()
  .join(p3, ['Date','Symbol'])