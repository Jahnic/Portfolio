import numpy
import pandas as pd
import numpy as np
import streamlit as st
from tensorflow import keras
from PIL import Image
# visualizations 
import plotly.express as px
import seaborn as sns 
import matplotlib.pyplot as plt
# import data 
data = pd.read_csv('../data/data_clean.csv', index_col=[0])
# Adjusting data
data.dropna(inplace=True)
data.seniority = data.seniority.apply(lambda x: 'yes' if x == 'senior' else 'no')
new_columns = ['Job Title', 'Job Description', 'Rating', 'Location', 'Size',
       'Type of ownership', 'industry', 'Sector', 'Revenue', 'Title',
       'Seniority', 'company_txt', 'location', 'age', 'python', 'R', 'spark',
       'aws', 'excel', 'sql', 'min_salary', 'max_salary', 'avg_salary',
       'employer_provided', 'desc_len', 'locational_salary_quantiles']
data.columns = new_columns
data.location
st.title("""
         Making sense of data science!!!
         """)
st.subheader("""
        Data science jobs in California were scraped from Glassdoor for this application.
        """)
# Banner
img = Image.open('banner.png')
st.image(img, width=700)

st.write("""
         **Questions you may want to explore on this app:**  
        - Which data skills provide the highest ROI?  
        - What are the strongest differentiating factors between data science, 
        analytics, engineering and ML engineering?  
        - What would your current data job pay in California?
         """)

if st.checkbox('Data sample'):
    st.write(data.sample(10))

st.subheader("""
             Where are most job opportunities for data scientists in California?
             """)

# Top locations    
top_locations = data['location'].value_counts()[:20]
st.write(
    px.bar(top_locations, x=top_locations.index, y=top_locations.values,
           color=top_locations.values,
           title="Top 20 locations",
           labels={'index': "locations", "y": "Jobs", "color": "jobs"}
))

# Top Industries    
top_industries = data['Industry'].value_counts()[:20]
# Remove missing values
top_industries.drop("-1", inplace=True)
st.write(
    px.bar(top_industries, x=top_industries.index, y=top_industries.values,
           color=top_industries.values,
           title="Top 20 industires",
           labels={'index': "Industries", "y": "Jobs", "color": "jobs"},
           height=500
))

# Top locations    
top_companies = data['company_txt'].value_counts()[:20]
st.write(
    px.bar(top_companies, x=top_companies.index, y=top_companies.values,
           color=top_companies.values,
           title="Top 20 companies",
           labels={'index': "Companies", "y": "Jobs", "color": "jobs"}
))

st.subheader("""
             Which skills pay best for which job roles?
             """)

def roi_on_skills(feature, skill):
    with_skill = data[skill] == 1
    without_skill = data[skill] == 0
    percent = 100 * (feature[with_skill].mean() - feature[without_skill].mean())\
                / feature[without_skill].mean()
    return percent

new_column_name = ['Median percent salary change']

excel = pd.pivot_table(data, index=['seniority'],
                values = ['avg_salary'], 
                aggfunc=lambda x: roi_on_skills(x, 'excel'))

python = pd.pivot_table(data, index=['seniority'],
                values = ['avg_salary'], 
                aggfunc=lambda x: roi_on_skills(x, 'python'))

R = pd.pivot_table(data, index=['seniority'],
                values = ['avg_salary'], 
                aggfunc=lambda x: roi_on_skills(x, 'R'))


sql = pd.pivot_table(data, index=['Title', 'Seniority'],
                values = ['avg_salary'], 
                aggfunc=lambda x: round(roi_on_skills(x, 'sql')))
sql.columns = new_column_name
sql

pd.pivot_table(data, index=['seniority'],
                values = ['avg_salary'], 
                aggfunc=lambda x: roi_on_skills(x, 'aws'))

pd.pivot_table(data, index=['seniority'],
                values = ['avg_salary'], 
                aggfunc=lambda x: roi_on_skills(x, 'spark'))

pd.pivot_table()

