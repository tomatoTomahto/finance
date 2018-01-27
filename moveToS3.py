# Move financial data collected to S3 bucket
!aws s3 mv data/historicPrices/ s3://sgupta-s3/finance/data/historicPrices/ --recursive --exclude "*" --include "*.csv"
!aws s3 mv data/historicArticles/ s3://sgupta-s3/finance/data/historicArticles/ --recursive --exclude "*" --include "*.json"
!aws s3 mv data/historicActions/ s3://sgupta-s3/finance/data/historicActions/ --recursive --exclude "*" --include "*.csv"