from pyspark.sql import SparkSession, functions as F
from pyspark import StorageLevel
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.regression import RandomForestRegressionModel
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

stockRfrModel = PipelineModel.load('stockRfr.mdl')

# ## Run model against test data, evaluate our daily long and short stock rankings
stockPredictions = stockRfrModel.transform(priceChanges)
SM.evaluatePredictions(stockPredictions, 1000)

# # Model 2: Predict long and short positions based on historic analyst ratings
# ## Load in historical analyst ratings from Yahoo Finance!
ratings = RM.loadData(spark, 'finance/data/analystRatings.json')
ratings.printSchema()
ratings.show(5)

# ## Transform the data
analystData = ratings.join(priceChanges,
                           (F.date_add(ratings.Date,1) == priceChanges.Date) & \
                           (ratings.Symbol == priceChanges.Symbol))\
  .select(priceChanges.Date, 
          priceChanges.Symbol,
          priceChanges.PriceChange,
          'Action',
          'Research Firm',
          'From',
          'To')
analystData.persist(StorageLevel.MEMORY_ONLY)
analystData.show(5)

analystRfrModel = PipelineModel.load('ratingsRfr.mdl')

# ## Run model against test data, evaluate our daily long and short stock rankings
analystPredictions = analystRfrModel.transform(analystData)
SM.evaluatePredictions(analystPredictions, 1000)

# # Model 3: Predict long and short positions based on historic article information (keywords, topics, etc)
# ## Load in historical articles from NYTimes
articles = AM.loadData(spark, 'finance/data/historicArticles/*.json')

# ## Transform the data - extract relevant info and join articles with stock prices
articleFeatures = AM.transformData(articles)
articleData = articleFeatures.join(priceChanges,
                                   F.date_add(articleFeatures.pub_date,1) == priceChanges.Date)
articleData.persist(StorageLevel.MEMORY_ONLY)
articleData.show(5)

articleRfrModel = PipelineModel.load('articleRfr.mdl')

# ## Run model against test data, evaluate our daily long and short stock rankings
articlePredictions = articleRfrModel.transform(articleData)
SM.evaluatePredictions(articlePredictions, 1000)


dateWindow = Window.partitionBy(priceChanges.Date)
rankedStocks = priceChanges.select('Date', 'Symbol', 'PriceChange', 
                                   F.rank().over(dateWindow.orderBy(priceChanges.PriceChange.desc())).alias('aLongRank'),
                                   F.rank().over(dateWindow.orderBy(priceChanges.PriceChange.asc())).alias('aShortRank'))
rankings = rankedStocks.groupBy('Symbol').agg(F.avg('aLongRank').alias('LongRank'), F.avg('aShortRank').alias('ShortRank'))