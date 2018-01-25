import pandas as pd
from HTMLReader import HTMLReader
from yahoo_finance import Share
import datetime, time
from dateutil import relativedelta as dru, parser as dp
import json
from pandas_datareader import data
import pandas as pd

def loadCompanies():
    nasdaq = pd.read_csv('data/companylist_NASDAQ.csv')
    nasdaq['Exchange'] = 'NASDAQ'
    nyse = pd.read_csv('data/companylist_NYSE.csv')
    nyse['Exchange'] = 'NYSE'
    companies = pd.concat([nasdaq, nyse], ignore_index=True)
    return companies

def saveHistoricalPrices(companies, start, end, file):
  for idx in range(len(companies)):
    ticker = companies['Symbol'][idx]
    print('Retrieving stock data for %s' % ticker)
    try:
      historicData = data.DataReader([ticker], 'yahoo', start, end)
      with open(file, 'a') as outfile:
        outfile.write(historicData.transpose(2,1,0).fillna(-1).to_frame().stack().to_csv())
        outfile.close()
    except Exception as e:
      print("Error getting stock data for %s : %s" % (ticker, str(e)))
      
def saveHistoricalActions(companies, start, end, file):
  for idx in range(len(companies)):
    ticker = companies['Symbol'][idx]
    print('Retrieving stock data for %s' % ticker)
    try:
      historicData = data.DataReader([ticker], 'yahoo-actions', start, end)
      with open(file, 'a') as outfile:
        outfile.write(historicData.fillna(-1).to_frame().stack().to_csv())
        outfile.close()
    except Exception as e:
      print("Error getting stock data for %s : %s" % (ticker, str(e)))