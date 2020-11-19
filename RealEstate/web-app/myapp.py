import numpy
import pandas as pd
import streamlit as st
from tensorflow import keras
from PIL import Image
# Geomapping
import plotly.express as px

st.write("""
         # Montreal Real Estate Application
        **Identify up and comming Montreal neighborhoods and undervalued condos**
         
         """)
# Banner
img = Image.open('banner.jpg')
st.image(img)
# Outlier
outlier_index = 2736
# Load data
data = pd.read_csv("../data/data_with_prediction_differences.csv").iloc[:, 4:]
feature_columns = ['restaurants', 'shopping', 'vibrant', 'cycling_friendly',
       'car_friendly', 'historic', 'quiet', 'elementary_schools',
       'high_schools', 'parks', 'nightlife', 'groceries', 'daycares',
       'pedestrian_friendly', 'cafes', 'transit_friendly', 'greenery',
       'year_built', 'population_2016_',
       'population_variation_between_2011_2016_', 'population_density_',
       'unemployment_rate_2016_', 'walk_score', 'rooms', 'bedrooms',
       'bathrooms', 'powder_rooms', 'total_area', 'river_proximity',
       'has_pool', 'n_parking', 'has_garage', 'is_devided', 'mr_distance']
feature_data = data[feature_columns]
cluster_data = pd.read_csv("../data/cluster_data.csv")

# Load keras model
model = keras.models.load_model('tf_linear_model_2')

# Cluster interpretation
st.write("""
    ## K-Means neighborhood clustering
    Dot size corresponds to differences between predicted and actual values.
    Larger dots are overpredictions and may potentially refer to undervalued condos.  
    ### Cluster interpretation
    **Cluster 0:** Vibrant city life with students and affluent professionals.  
    **Cluster 1:** Uneventful with predominantely lower income residents.  
    **Cluster 2:** Mostly uneventful city life, with majority middle and some lower income demographics.  
    **Cluster 3:** Moderately vibrant with students and low income residents.  
    **Cluster 4:** Suburb-like, upper middle class neighborhood.  
    **Cluster 5:** Family friendly, with highly educated residents.  
    **Cluster 6:** Mix of vibrant and family friendly neighborhood with a mix of low and high income earners.  
    **Cluster 7:** Very Family friendly and uneventful with mostly middle class families.
        """)  

# Additional cluster Interpretation
st.subheader("""
        Additional neighborhood attributes for each cluster
    """)  

mean_pivot = round(cluster_data.pivot_table(index='clusters', 
                            values=['PC_neighborhood_1', 'PC_neighborhood_2',
                                'PC_demographics_1', 'PC_demographics_2',
                                'growth', 'population_density', 'price']))
# Rename columns
new_columns = ['PC_neighborhood_1', 'PC_neighborhood_2',
                'PC_demographics_1', 'PC_demographics_2',
                'Neighborhood growth', 'Population density', 
                'Mean condo value']
mean_pivot.columns = new_columns
additional_attributes = mean_pivot[['Neighborhood growth', 
                                    'Population density',
                                    'Mean condo value']]
additional_attributes

# Further insights on clusters    
additional_insights = pd.concat([cluster_data['clusters'].astype('str'), 
                                 data[['price', 'diff', 'walk_score', 
                                 'total_area','mr_distance']]], axis=1)

pivot_table = round(pd.pivot_table(additional_insights, 
               index=['clusters'], 
               values=['diff', 'walk_score', 'mr_distance']))
pivot_table.columns = ['*Prediction diff.', 'Walk score', '*DWTN distance']
pivot_table

st.write("""
        \*DWTN distance: distance from downtown in km    
        \*Prediction diff.: predicted - actual condo price
         """)

# Geomapping
px.set_mapbox_access_token('pk.eyJ1IjoiamFobmljIiwiYSI6ImNrZ3dtbWRxNTBia3MzMW4wN2VudXZtcTUifQ.BVPxkX1DH75NahJvzt-f2Q')
st.write(
    px.scatter_mapbox(
        cluster_data, lat="lat", lon="long", color="clusters",
        color_continuous_scale=px.colors.sequential.Rainbow, 
        hover_data=['index', 'price', 'growth', 'normalized_difference'], 
        size='positive_differences',
        size_max=7, zoom=10, height=700, width=950)
)

st.subheader("Obtain records for condos of interest")

# User search for condos based on index
user_input = st.number_input(label="Type index of condo to look up:",
                             value=0
                        )

if user_input in list(data.index):
    data.iloc[user_input, :]
else:
    st.text("ValueError: Index out of range.")
 

"""

---

**Additional information on PCs and clusters**
"""

# PC Interpretation
pc_explanations = pd.DataFrame({
    "Type": ['Neighborhood', 'Neighborhood', 'Neighborhood', 'Neighborhood',
             'Demographics', 'Demographics', 'Demographics', 'Demographics'],
    "Component": ['PC1', 'PC1', 'PC2', 'PC2', 'PC1', 'PC1', 'PC2', 'PC2'],
    "Sign of value": ['Positive', 'Negative', 'Positive', 'Negative',
                   'Positive', 'Negative','Positive', 'Negative'],
    "Interpretation": ["Uneventful suburbs", "Vibrant city-life",
                       "City-life", "Family friendly", "Middle-class families",
                       "Students", "Educated and Affluent", "Poor"]
})

pc_explanations.set_index(['Type', 'Component', 'Sign of value'], inplace=True)

# Breakdown of PCs
if st.checkbox('Interpretation of principle components (PCs)'):
    st.write(
        """
        The following PCs indicate which neighborhood and demographic attributes 
        differ the most accross varying locations in Montreal. 
        The following table guides the interpretation of PC scores for negative and 
        positive values.  
          
        Example: a strong negative score of the neighborhood PC1 corresponds to 
        vibrant areas typical of big-cities. Nightlife establishments, shopping oppertunities,
        restaurant and noise pollution would all be expected in areas with such
        a PC score. 
        
        """
    )
    pc_explanations

if st.checkbox('Breakdown of clusters'):
    st.write("""
        **The following table contains mean PC values for each of the 8 clusters.**
    """)
    mean_pivot[['PC_neighborhood_1', 'PC_neighborhood_2',
                'PC_demographics_1', 'PC_demographics_2']]
    