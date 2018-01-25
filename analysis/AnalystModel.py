from pyspark.sql import functions as F, Window
from pyspark.sql.types import *
from pyspark.ml.feature import CountVectorizer, VectorAssembler, StringIndexer
from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, RegressionEvaluator

# ## Read in Historic Analyst Ratings
def loadData(spark, analystFile):
  analystRatings = spark.read.load(analystFile, format='json')\
    .withColumn('Date', F.from_unixtime(F.unix_timestamp(F.col('Date'),format='dd-MMM-yy'), format='yyyy-MM-dd'))

  return analystRatings

# ## Feature engineering
def engineerFeatures():
  actionSI = StringIndexer(inputCol="Action",outputCol="indexedAction", handleInvalid='keep')
  fromSI = StringIndexer(inputCol="From",outputCol="indexedFrom", handleInvalid='keep')
  toSI = StringIndexer(inputCol="To",outputCol="indexedTo", handleInvalid='keep')
  firmSI = StringIndexer(inputCol="Research Firm",outputCol="indexedFirm", handleInvalid='keep')
  symbolSI = StringIndexer(inputCol="Symbol",outputCol="indexedSymbol", handleInvalid='keep')
  va = VectorAssembler(inputCols=['indexedAction',
                                  'indexedFrom',
                                  'indexedTo',
                                  'indexedFirm',
                                  'indexedSymbol'], outputCol='features')
  analystRfr = RandomForestRegressor(featuresCol="features", labelCol="PriceChange", predictionCol="pPriceChange",
                                     maxBins=5700)
  
  stages = [actionSI,
            fromSI,
            toSI,
            firmSI,
            symbolSI,
            va,
            analystRfr]
  
  return stages