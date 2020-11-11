import numpy
import pandas as pd
import streamlit as st
from tensorflow import keras
from PIL import Image
# Geomapping
import plotly.express as px

st.write("""
         # Montreal Real Estate
        **Identify up and comming Montreal neighborhoods and undervalued condos**
         
         """)
# Banner
img = Image.open('banner.jpg')
st.image(img)

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
model = keras.models.load_model('tf_linear_model')

    
# Geomapping
px.set_mapbox_access_token('pk.eyJ1IjoiamFobmljIiwiYSI6ImNrZ3dtbWRxNTBia3MzMW4wN2VudXZtcTUifQ.BVPxkX1DH75NahJvzt-f2Q')
st.write(
    px.scatter_mapbox(cluster_data, lat="lat", lon="long", color="clusters",
                        color_continuous_scale=px.colors.sequential.Rainbow, 
                        hover_data=['index', 'price', 'growth', 'normalized_difference'], 
                        size='positive_differences',
                        size_max=7, zoom=10, title='K-Means neighbourhood clusters')
)
# PC Interpretation
pc_explanations = pd.DataFrame({
    "Type": ['Neighborhood', 'Neighborhood', 'Neighborhood', 'Neighborhood',
             'Demographics', 'Demographics', 'Demographics', 'Demographics'],
    "Component": ['PC1', 'PC1', 'PC2', 'PC2', 'PC1', 'PC1', 'PC2', 'PC2'],
    "Sign of value": ['Positive', 'Negative', 'Positive', 'Negative',
                   'Positive', 'Negative','Positive', 'Negative'],
    "Interpretation": ["Doll suburbs", "Vibrant city-life",
                       "City-life", "Family friendly", "Middle-class families",
                       "Students", "Educated and Affluent", "Poor"]
})
pc_explanations.set_index(['Type', 'Component', 'Sign of value'], inplace=True)
if st.checkbox('Interpretation of principle components'):
    st.write(
        """
        The following principle components (PCs) indicate which neighborhood and demographic attributes 
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
    
# Cluster Interpretation
mean_pivot = round(cluster_data.pivot_table(index='clusters', 
                            values=['PC_neighborhood_1', 'PC_neighborhood_2',
                                'PC_demographics_1', 'PC_demographics_2',
                                'growth', 'population_density', 'price'],
                            aggfunc='mean'), 2)
# Rename columns
new_columns = ['PC_neighborhood_1', 'PC_neighborhood_2',
                'PC_demographics_1', 'PC_demographics_2',
                'Neighborhood growth', 'Population density', 'Condo price']
mean_pivot.columns = new_columns
# Rearrange column order

mean_pivot = mean_pivot[['PC_neighborhood_1', 'PC_neighborhood_2',
                'PC_demographics_1', 'PC_demographics_2',
                'Neighborhood growth', 'Population density', 'Condo price']]

if st.checkbox('Cluster interpretation'):
    st.write("""
        The following table contains mean values for each of the 8 clusters.
    """)
    mean_pivot
    
    st.write("""
        In the following paragraph, the most relevant details on what each of the clusters 
        represent are listed.  
          
        **Cluster identity**  
            0: ....  
            1: ....
    """)  
if st.checkbox('show data'):
    st.subheader('Sample of input data')
    st.write(feature_data.head(10))
    cluster_data.price