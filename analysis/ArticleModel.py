from pyspark.sql import functions as F, Window
from pyspark.sql.types import *
from pyspark.ml.feature import CountVectorizer, VectorAssembler, StringIndexer
from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.regression import RandomForestRegressor

# This module is for predicting stock movements using news article information
def loadData(spark, articlesFiles):
  # ## Read in Historic articles
  articles = spark.read.load(articlesFiles, format="json")\
      .withColumn('pub_date',F.date_format(F.col('pub_date').substr(0,10),'yyyy-MM-dd'))\
      .na.fill({'news_desk': 'unknown', 'print_page': 0, 'section_name':'unknown', 'subsection_name':'unknown', 'type_of_material':'unknown', 'word_count':0})
  articles.printSchema()
  return articles

# Extract information from articles
def transformData(articles):
  def concat(type):
    def concat_(args):
      return [item for sublist in args for item in sublist]
    return F.udf(concat_, ArrayType(type))
  
  concatStringArraysUDF = concat(StringType())

  articleFeatures = articles.withColumn('keywords', articles.keywords.value)\
    .groupBy('pub_date').agg(F.collect_list(articles.section_name).alias('sections'),
                             F.collect_list(articles.subsection_name).alias('subsections'),
                             F.collect_list(articles.news_desk).alias('newsdesks'),
                             F.collect_list(articles.type_of_material).alias('materials'),
                             F.collect_list('keywords').alias('keywords'))\
    .withColumn('keywords', concatStringArraysUDF(F.col('keywords')))
  return articleFeatures

# ## Feature Engineering - combine stock symbol and news article information into a feature vector
def engineerFeatures():
  sectionCV = CountVectorizer(inputCol='sections', outputCol="sectionVector")
  subsectionCV = CountVectorizer(inputCol='subsections', outputCol="subsectionVector")
  newsdeskCV = CountVectorizer(inputCol='newsdesks', outputCol="newsdeskVector")
  materialCV = CountVectorizer(inputCol='materials', outputCol="materialVector")
  keywordCV = CountVectorizer(inputCol='keywords', outputCol="keywordVector")
  symbolSI = StringIndexer(inputCol="Symbol",outputCol="indexedSymbol", handleInvalid='keep')
  
  va = VectorAssembler(inputCols=['sectionVector',
                                  'subsectionVector',
                                  'newsdeskVector',
                                  'materialVector',
                                  'indexedSymbol',
                                  'keywordVector'], outputCol='features')
  
  articleRfr = RandomForestRegressor(featuresCol="features", labelCol="PriceChange", predictionCol="pPriceChange",
                                     maxBins=5700)
  
  stages = [sectionCV,
            subsectionCV,
            newsdeskCV,
            materialCV,
            keywordCV,
            symbolSI,
            va,
            articleRfr]
  
  return stages