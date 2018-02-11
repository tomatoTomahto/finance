# # BuildModels.py
# # Overview
# This script generates machine learning models used to predict the daily price change
# of NASDAQ and NYSE stocks using historical stock, news, and analyst information

# # Data is assumed to be in AWS S3 but can also be loaded in from HDFS
#!aws s3 ls s3://sgupta-s3/finance/data/historicPrices/
#!hdfs dfs -ls finance/data

# # Imports
from pyspark.sql import SparkSession, functions as F
from pyspark import StorageLevel
from pyspark.ml import Pipeline
import os, sys
sys.path.append('analysis')
import StockModel as SM
import pandas as pd, seaborn as sb

# # Spark Connection
spark = SparkSession \
  .builder \
  .appName("Stock Analysis") \
  .getOrCreate()

companiesLoc='s3a://sgupta-s3/finance/data/companyList.csv'
pricesLoc='s3a://sgupta-s3/finance/data/historicPrices/historicPrices_*.csv'
  
# # Model 1: Predict long and short positions based on historic stock information (price, volume, etc) 
# ## Load in Historical prices for all stocks on NASDAQ and NYSE exchanges
prices = SM.loadData(spark,companiesLoc, pricesLoc)
prices.printSchema()

# ## Transform the data - calculate price changes, load historical price changes
returns = SM.transformData(prices)
returns.persist(StorageLevel.MEMORY_ONLY)
returns.show(5)

# ## Data Exploration
# ### General Correlations Between Data Attributes
returns.describe(['Return','Spread','Volume','yReturn','ySpread','yVolume']).toPandas()
returnsDF = returns.sample(False, 0.1).toPandas()\
  .astype({'Return':'float','Spread':'float','Volume':'float',
           'yReturn':'float','ySpread':'float','yVolume':'float'})
returnsDF.describe()
pd.plotting.scatter_matrix(returnsDF, figsize=(15, 15))

# Looks like there is a relationship between last and current trading day's return
# We can use Spark to calculate the actual correlation
ReturnCorr = returns.corr('yReturn','Return')
SpreadCorr = returns.corr('ySpread','Return')
VolumeCorr = returns.corr('yVolume','Return')
print("Correlation(yReturn, Return): \t\t%f \n" % ReturnCorr + \
  "Correlation(ySpread, Return): \t\t%f \n" % SpreadCorr + \
  "Correlation(yVolume, Return): \t\t%f" % VolumeCorr)

# ### Correlations by Sector and Industry
sectorCorr = pd.DataFrame(columns=['Sector','ReturnCorr','VolumeCorr','SpreadCorr'])
industryCorr = pd.DataFrame(columns=['Industry','ReturnCorr','VolumeCorr','SpreadCorr'])

sectorsPD = returns.select('Sector','Industry','Symbol')\
  .distinct().toPandas()
sectors = sectorsPD.Sector.unique()
industries = sectorsPD.Industry.unique()

for sector in sectors:
  sReturns = returns.filter(returns.Sector==sector)
  rCorr = sReturns.corr('yReturn','Return')
  sCorr = sReturns.corr('ySpread','Return')
  vCorr = sReturns.corr('yVolume','Return')
  sectorCorr = sectorCorr.append({'Sector':sector, 'ReturnCorr':rCorr, 'VolumeCorr':vCorr, 'SpreadCorr':sCorr}, ignore_index=True)
  
for industry in industries:
  iReturns = returns.filter(returns.Industry==industry)
  rCorr = iReturns.corr('yReturn','Return')
  sCorr = iReturns.corr('ySpread','Return')
  vCorr = iReturns.corr('yVolume','Return')
  industryCorr = industryCorr.append({'Industry':industry, 'ReturnCorr':rCorr, 'VolumeCorr':vCorr, 'SpreadCorr':sCorr}, ignore_index=True)

sectorCorr.sort_values(by=['ReturnCorr'], ascending=False)
industryCorr.sort_values(by=['VolumeCorr'], ascending=False)

# ## Model Development
# ### Linear Regression to predict today's return based on last trading day's return
# Let's build a linear regression model for each industry to predict today's return based on yesterday's return
from pyspark.ml.regression import LinearRegression
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler

def buildRegressionModel(data, features, labelCol):
  va = VectorAssembler(inputCols=features, outputCol='features')
  lr = LinearRegression(labelCol=labelCol,featuresCol="features", predictionCol='p'+labelCol)
  pipeline = Pipeline(stages=[va, lr])

  (trainingData, testData) = data.randomSplit([0.7, 0.3])

  lrModel = pipeline.fit(trainingData)
  return lrModel

lrModel = buildRegressionModel(data=returns, features=['yReturn','yVolume','ySpread'], labelCol='Return')
lrPredictions = lrModel.transform(testData)

def evaluateRegressionModel(lrModel):
  trainingSummary = lrModel.stages[1].summary
  print("numIterations: %d" % trainingSummary.totalIterations)
  print("objectiveHistory: %s" % str(trainingSummary.objectiveHistory))
  trainingSummary.residuals.describe().show()
  print("RMSE: %f" % trainingSummary.rootMeanSquaredError)
  print("r2: %f" % trainingSummary.r2)
  
evaluateRegressionModel(lrModel)
  
from pyspark.ml.feature import CountVectorizer, VectorAssembler, StringIndexer, OneHotEncoder, Normalizer, Bucketizer
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.regression import RandomForestRegressor, LinearRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, RegressionEvaluator

stockSI = StringIndexer(inputCol="Symbol",outputCol="indexedSymbol", handleInvalid='keep')
sectorSI = StringIndexer(inputCol='Sector', outputCol='indexedSector', handleInvalid='keep')
industrySI = StringIndexer(inputCol='Industry', outputCol='indexedIndustry', handleInvalid='keep')
stockEnc = OneHotEncoder(inputCol="indexedSymbol", outputCol="encodedSymbol")
sectorEnc = OneHotEncoder(inputCol="indexedSector", outputCol="encodedSector")
industryEnc = OneHotEncoder(inputCol="indexedIndustry", outputCol="encodedIndustry")
catVa = VectorAssembler(inputCols=["encodedSymbol","encodedSector","encodedIndustry"], outputCol="catFeatures")
numVa = VectorAssembler(inputCols["yVolume",'ySpread',"yPriceChange",], outputCol="numFeatures")

splits = [-float("inf"), -0.2, -0.15, -0.1, -0.05, -0.0, 0.05, 0.1, 0.2, float("inf")]


norm = Normalizer(inputCol="features", outputCol="normFeatures", p=1.0)
stockLr = LinearRegression(labelCol='PriceChange', featuresCol='features', predictionCol='pPriceChange')

stages = [stockSI, sectorSI, industrySI, stockEnc, sectorEnc, industryEnc, va, norm]

lrPipeline = Pipeline(stages=stages+[stockLr])

(trainingData, testData) = priceChanges.randomSplit([0.7, 0.3])

lrModel = lrPipeline.fit(trainingData)
lrPredictions = lrModel.transform(testData)

# Print the coefficients and intercept for linear regression
print("Coefficients: %s" % str(lrModel.stages[8].coefficients))
print("Intercept: %s" % str(lrModel.stages[8].intercept))

# Summarize the model over the training set and print out some metrics
trainingSummary = lrModel.stages[8].summary
print("numIterations: %d" % trainingSummary.totalIterations)
print("objectiveHistory: %s" % str(trainingSummary.objectiveHistory))
trainingSummary.residuals.show()
print("RMSE: %f" % trainingSummary.rootMeanSquaredError)
print("r2: %f" % trainingSummary.r2)

lrPredictions.select('PriceChange','pPriceChange')\
  .withColumn('correct',lrPredictions.PriceChange*lrPredictions.pPriceChange>0)\
  .groupBy('correct').count().show()

stockRfr = RandomForestRegressor(featuresCol="features", labelCol="PriceChange", 
                                 predictionCol="pPriceChange", maxBins=1000)

rfrPipeline = Pipeline(stages=stages+[stockRfr])
rfrModel = rfrPipeline.fit(trainingData)
rfrPredictions = rfrModel.transform(testData)

evaluator = RegressionEvaluator(labelCol="PriceChange", predictionCol="pPriceChange")
mae = evaluator.evaluate(stockPredictions, {evaluator.metricName: "mae"})
rmse = evaluator.evaluate(stockPredictions, {evaluator.metricName: "rmse"})
r2 = evaluator.evaluate(stockPredictions, {evaluator.metricName: "r2"})
print("MAE: %g, RMSE: %g, R2: %g" % (mae, rmse, r2))

rfrPredictions.select('PriceChange','pPriceChange')\
  .withColumn('correct',rfrPredictions.PriceChange*rfrPredictions.pPriceChange>0)\
  .groupBy('correct').count().show()

# ## Save model in HDFS
stockRfrModel.write().overwrite().save('stockRfr.mdl')

##### TO DO !!!! #####


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