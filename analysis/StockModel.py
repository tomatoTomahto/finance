from pyspark.sql import functions as F, Window
from pyspark.sql.types import *
from pyspark.ml.feature import CountVectorizer, VectorAssembler, StringIndexer
from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, RegressionEvaluator

# ## Read in Historic Prices, clean up the data
def loadData(spark, companiesLoc, pricesLoc):
  # ## Read in Company Data from Exchanges
  companies = spark.read.load(companiesLoc, format="csv", header=True)
  print('Total Companies on NYSE and NASDAQ: %d' % companies.count())

  fields = [StructField('Date', DateType(), True),
            StructField('Metric', StringType(), True),
            StructField('Symbol', StringType(), True),
            StructField('Price', FloatType(), True)]
  schema = StructType(fields)
  
  prices = spark.read.load(pricesLoc, format="csv", schema=schema)\
    .groupBy('Date', 'Symbol')\
    .pivot('Metric', values=['Adj Close','Close','Open','High','Low','Volume'])\
    .sum('Price')\
    .join(companies.withColumnRenamed('industry','Industry')\
          .select('Symbol','Sector','Industry'), 'Symbol')
  
  return prices

# # Data Transformations
def transformData(prices):
  # ## Compute daily stock price changes
  priceChanges = prices.withColumn('PriceChange', (F.col('Close')-F.col('Open'))/F.col('Open')*100)\
      .withColumn('absPriceChange', F.abs(F.col('PriceChange')))\
      .withColumn('Spread', F.col('High')-F.col('Low'))\
      .select('Date','Sector','industry','Symbol','Volume','PriceChange','Spread','absPriceChange')\
      .na.fill({'PriceChange': 0.0, 'absPriceChange': 0.0})
  
  # ## Get previous day's info for each stock
  yPriceChanges = priceChanges.withColumn('Date',F.date_add(priceChanges.Date,1))\
    .withColumnRenamed('PriceChange','yPriceChange')\
    .withColumnRenamed('absPriceChange','yAbsPriceChange')\
    .withColumnRenamed('Volume','yVolume')\
    .withColumnRenamed('Spread','ySpread')
    
  priceChanges = priceChanges.join(yPriceChanges, ['Sector','Industry','Symbol','Date'])\
    .select('Date','Symbol','PriceChange','absPriceChange','Volume', 'Spread', 
            'yPriceChange', 'yAbsPriceChange','yVolume', 'ySpread','Sector','Industry')
  
  return priceChanges

# ## Feature Engineering - combine stock symbol, sector, industry, previous day's volume & price change into a feature vector
def engineerFeatures():
  stockSI = StringIndexer(inputCol="Symbol",outputCol="indexedSymbol", handleInvalid='keep')
  sectorSI = StringIndexer(inputCol='Sector', outputCol='indexedSector', handleInvalid='keep')
  industrySI = StringIndexer(inputCol='Industry', outputCol='indexedIndustry', handleInvalid='keep')
  va = VectorAssembler(inputCols=["indexedSymbol","indexedSector","indexedIndustry",
                                  "yVolume",'ySpread',"yPriceChange",'yAbsPriceChange'], outputCol="features")
  
  # ## Random Forest Regression - use the categorical and numerical features to predict today's price change
  stockRfr = RandomForestRegressor(featuresCol="features", labelCol="PriceChange", predictionCol="pPriceChange",
                                   maxBins=5700)
  
  return [stockSI, sectorSI, industrySI, va, stockRfr]

# ### Compute a window partitioned by date to get the top 3 stocks predicted and actual by price change
def evaluatePredictions(stockPredictions, topN):
  dateWindow = Window.partitionBy(stockPredictions.Date)
  rankedStocks = stockPredictions.select('Date', F.col('Symbol'), 'pPriceChange', 'PriceChange', 
                                         F.rank().over(dateWindow.orderBy(stockPredictions.pPriceChange.desc())).alias('pLongRank'),
                                         F.rank().over(dateWindow.orderBy(stockPredictions.PriceChange.desc())).alias('aLongRank'),
                                         F.rank().over(dateWindow.orderBy(stockPredictions.pPriceChange.asc())).alias('pShortRank'),
                                         F.rank().over(dateWindow.orderBy(stockPredictions.PriceChange.asc())).alias('aShortRank'))
  rankedStocks.show()
  evaluatedRanks = rankedStocks.withColumn('corrLong',(F.col('pLongRank')<topN)==(F.col('aLongRank')<topN))\
    .withColumn('corrShort',(F.col('pShortRank')<topN)==(F.col('aShortRank')<topN))
  total = evaluatedRanks.count()
  corrLong = evaluatedRanks.filter('corrLong and PriceChange>0').count()
  corrShort = evaluatedRanks.filter('corrShort and PriceChange<0').count()
  print('Stats for top %d stocks --- Long Accuracy: %.3f, Short Accuracy: %.3f' % \
      (topN, float(corrLong)/total, float(corrShort)/total))


