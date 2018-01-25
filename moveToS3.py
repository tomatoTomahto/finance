# Move financial data collected to S3 bucket
!aws s3 mv data/historicPrices/ s3://sgupta-s3/finance/data/historicPrices/ --recursive
!aws s3 mv data/historicArticles/ s3://sgupta-s3/finance/data/historicArticles/ --recursive
!aws s3 mv data/historicActions/ s3://sgupta-s3/finance/data/historicActions/ --recursive