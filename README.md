# finance
This repository contains Python code for retrieving and analyzing historic stock prices
from NYSE and NASDAQ exchanges. 

## Setup
Install pandas and pandas_datareader for retrieving stock prices and other information

## Data Collection
Run ```collect/Main.py``` to collect historic stock information using Yahoo Finance APIs

```collect/Finance.py``` contains functions for retrieving past stock prices for a specified
list of stocks, all written into CSV files in the data directory. This includes:
* volume, open, close, high, low and adjusted close prices for each stock
* company actions such as dividends, splits and merges for each stock

```collect/News.py``` contains functions for retrieving news articles from NY Times and
saving them to json files organized by year and month in the data directory.

```moveToS3.py``` can be optionally used to copy data into an S3 bucket

## Data Analysis
Under construction :)