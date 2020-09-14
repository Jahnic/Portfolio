import scraper 
import pandas as pd 
path = "/usr/bin/google-chrome-stable"

# Data Science
df = scraper.get_jobs("Data Scientist", 1000, False, path, 10)
df.to_csv('data/data_scientist.csv')

# Data Engineering
df = scraper.get_jobs("Data Engineer", 1000, False, path, 10)
df.to_csv('data/data_engineer.csv')

# Data Analyst
df = scraper.get_jobs("Data Analyst", 1000, False, path, 10)
df.to_csv('data/data_analyst.csv')

# Machine learning engineer
df = scraper.get_jobs("Machine Learning Engineer", 1000, False, path, 10)
df.to_csv('data/ml_engineer.csv')

