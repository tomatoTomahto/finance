# Install libraries
#!pip install -U pandas_datareader

import os, sys
sys.path.append('collect')

import Finance as f
import News as n

# Initialization
pricesStartDate = '2000-01-01'
pricesEndDate = '2018-01-31'
pricesFile = 'data/historicPrices/prices.csv'
actionsFile = 'data/historicActions/actions.csv'
articlesFile = 'data/historicArticles/'
articleStartYear = 2011
articleEndYear = 2018
articleStartMonth = 4
articleEndMonth = 2

# Read in companies listed on NYSE and NASDAQ
companies = f.loadCompanies()

f.saveHistoricalPrices(companies, pricesStartDate, pricesEndDate, pricesFile)

f.saveHistoricalActions(companies, pricesStartDate, pricesEndDate, actionsFile)

n.saveHistoricalNews(articleStartYear, articleStartMonth, articleEndYear, articleEndMonth, articlesFile)