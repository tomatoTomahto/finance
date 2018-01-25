import urllib2
import json
from pandas import DataFrame

def saveHistoricalNews(startYear, startMonth, endYear, endMonth, file):
  for year in range(startYear, endYear+1):
    print('Year: %d' % year)
    if year == endYear:
        finalMonth = endMonth
    else:
        finalMonth = 12

    if year != startYear:
      startMonth=1

    for month in range(startMonth,finalMonth+1):
        print(' Month: %d' % month)
        url = 'https://api.nytimes.com/svc/archive/v1/%d/%d.json?api-key=96561f5310e549b090fcc0ee4c863985' % (year, month)

        data = json.load(urllib2.urlopen(url))
        docs = data['response']['docs']
        data = DataFrame(docs)

        with open(file + '%d%d.json' % (year, month), 'w') as outfile:
            outfile.write(data.to_json(orient='records', lines=True))
