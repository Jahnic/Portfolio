import numpy as np
import pandas as pd
import math
import streamlit as st
from tensorflow import keras
from PIL import Image
# Geomapping
import plotly.express as px
# # Feature importance
# import eli5
# from eli5.sklearn import PermutationImportance

st.write("""
         # Montreal Real Estate Application
        **Explore Montreal neighborhoods and identify undervalued condos**
         """)

# Banner
img = Image.open('banner.jpg')
st.image(img)
# Outlier
outlier_index = 2736
# Load data
data = pd.read_csv("data/data_with_prediction_differences.csv").iloc[:, 4:]
model_columns = ['price', 'restaurants', 'vibrant', 'cycling_friendly',
       'car_friendly', 'historic', 'quiet', 'parks', 'groceries', 'cafes',
       'transit_friendly', 'greenery', 'year_built', 'walk_score', 'bedrooms',
       'bathrooms', 'powder_rooms', 'total_area', 'has_pool', 'has_garage',
       'is_devided', 'mr_distance']

model_data = data[model_columns]
cluster_data = pd.read_csv("data/cluster_data.csv")

# Load keras model
model = keras.models.load_model('tf_linear_model_2')

# Sidebar for parameter input
st.sidebar.header('Adjust Parameters for Price Prediction')

def get_mean(feat):
    return int(data[feat].mean())

def get_min(feat):
    return int(data[feat].min())

def get_max(feat):
    return int(data[feat].max())

def distance(destination, origin = (45.504557, -73.598104)):
    """Calculates distances in km from latitudinal/longitudinal data using
    the haversine formula
    
    Args:
    origin - starting point of distance calculation
    destination - end point of distance calculation

    Return
    d - distance between origin and destination in km
    """
    
    lat1, lon1 = origin
    lat2, lon2 = destination
    radius = 6371 # km
    
    #Convert from degrees to radians
    dlat = math.radians(lat2-lat1)
    dlon = math.radians(lon2-lon1)
    
    # Haversine formula
    a = math.sin(dlat/2) * math.sin(dlat/2) + math.cos(math.radians(lat1)) \
        * math.cos(math.radians(lat2)) * math.sin(dlon/2) * math.sin(dlon/2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    d = radius * c

    return d

def user_input_features():
    # Neighborhood ratings
    restaurant = st.sidebar.slider('Restaurants', 0, 10, 9, key=1)
    # shopping = st.sidebar.slider('Shopping', 0, 10, 9, key=2)
    vibrant = st.sidebar.slider('Vibrant', 0, 10, 9, key=3)
    cycling_friendly = st.sidebar.slider('Cycling Friendly', 0, 10, 9, key=4)
    car_friendly = st.sidebar.slider('Car Friendly', 0, 10, 9, key=5)
    historic = st.sidebar.slider('Historic', 0, 10, 9, key=6)
    quiet = st.sidebar.slider('Quiet', 0, 10, 9, key=7)
    # elementary_schools = st.sidebar.slider('Elementary Schools', 0, 10, 9, key=8)
    # high_schools = st.sidebar.slider('High-Schools', 0, 10, 9, key=9)
    parks = st.sidebar.slider('Parks', 0, 10, 9, key=10)
    # nightlife = st.sidebar.slider('Nightlife', 0, 10, 9, key=11)
    groceries = st.sidebar.slider('Groceries', 0, 10, 9, key=12)
    # daycares = st.sidebar.slider('Daycares', 0, 10, 9, key=13)
    # pedestrian_friendly = st.sidebar.slider('Pedestrian Friendly', 0, 10, 9, key=14)
    cafes = st.sidebar.slider('Cafes', 0, 10, 9, key=15)
    transit_friendly = st.sidebar.slider('Transit Friendly', 0, 10, 9, key=16)
    greenery = st.sidebar.slider('Greenery', 0, 10, 9, key=17)
    
    # Other parameters
    year_built = st.sidebar.slider('Year Built', get_min('year_built'), 
                                   get_max('year_built'), 
                                   get_mean('year_built'), key=18)
    # population_variation_between_2011_2016_ = st.sidebar.slider('2011-2016 Population Variation',
    #                                             get_min('population_variation_between_2011_2016_'), 
    #                                             get_max('population_variation_between_2011_2016_'), 
    #                                             get_mean('population_variation_between_2011_2016_')
    #                                             , key=19)
    # unemployment_rate_2016_ = st.sidebar.slider('2016 Unemployment Rate', 
    #                                             get_min('unemployment_rate_2016_'), 
    #                                             get_max('unemployment_rate_2016_'), 
    #                                             get_mean('unemployment_rate_2016_')
    #                                             , key=20)
    walk_score = st.sidebar.slider('Walk Score', get_min('walk_score'), 
                                   get_max('walk_score'), 
                                   get_mean('walk_score'), key=21)
    # rooms = st.sidebar.slider('Rooms', get_min('rooms'), 
    #                                get_max('rooms'), 
    #                                get_mean('rooms'), key=22)
    bedrooms = st.sidebar.slider('Bedrooms', get_min('bedrooms'), 
                                   get_max('bedrooms'), 
                                   get_mean('bedrooms'), key=23)
    bathrooms = st.sidebar.slider('Bathrooms', get_min('bathrooms'), 
                                   get_max('bathrooms'), 
                                   get_mean('bathrooms'), key=24)
    powder_rooms = st.sidebar.slider('Powder Rooms', get_min('powder_rooms'), 
                                   get_max('powder_rooms'), 
                                   get_mean('powder_rooms'), key=25)
    total_area = st.sidebar.slider('Total Area (sqft)', get_min('total_area'), 
                                   get_max('total_area'), 
                                   get_mean('total_area'), key=26)
    # river_proximity = st.sidebar.slider('River proximity', get_min('river_proximity'), 
    #                                get_max('river_proximity'), 
    #                                get_mean('river_proximity'), key=27)
    # has_pool = st.sidebar.slider('Has Pool', get_min('has_pool'), 
    #                                get_max('has_pool'), 
    #                                get_mean('has_pool'), key=28)
    # has_garage = st.sidebar.slider('Has Garage', get_min('has_garage'), 
    #                                get_max('has_garage'), 
    #                                get_mean('has_garage'), key=29)
    # is_devided = st.sidebar.slider('Divided Condo', get_min('is_devided'), 
    #                                get_max('is_devided'), 
    #                                get_mean('is_devided'), key=30)
    lat = st.sidebar.number_input('Latitude', 45.504557, format="%.6f")
    lon = st.sidebar.number_input('Longitude', -73.598104, format="%.6f")
    
    data_in = {
        'restaurants': restaurant,
        # 'shopping': shopping,
        'vibrant': vibrant,
        'cycling_friendly': cycling_friendly,
        'car_friendly': car_friendly,
        'historic': historic,
        'quiet': quiet,
        # 'elementary_schools': elementary_schools,
        # 'high_schools': high_schools,
        'parks': parks,
        # 'nightlife': nightlife,
        'groceries': groceries,
        # 'daycares': daycares,
        # 'pedestrian_friendly': pedestrian_friendly,
        'cafes': cafes, 
        'transit_friendly': transit_friendly,
        'greenery': greenery,
        'year_built': year_built,
        # 'population_variation_between_2011_2016_': population_variation_between_2011_2016_,
        # 'unemployment_rate_2016_': unemployment_rate_2016_,
        'walk_score': walk_score,
        # 'rooms': rooms,
        'bedrooms': bedrooms,
        'bathrooms': bathrooms,
        'powder_rooms': powder_rooms,
        'total_area': total_area,
        # 'river_proximity': river_proximity,
        # 'has_pool': has_pool,
        # 'has_garage': has_garage,
        # 'is_devided': is_devided,
        'mr_distance': distance(destination=((float(lat), float(lon))))
    }
    print(distance(destination=((float(lat), float(lon)))))
    return pd.DataFrame(data_in, index=[0])

def standardize_input(data_in):
    """Standardizes values inside a DataFrame (data_in)"""
    # data_in_array = data_in.to_numpy().reshape(-1, 1)
    feat_means = pd.read_csv('data/parameter_means.csv', index_col=[0])
    feat_stds = pd.read_csv('data/parameter_stds.csv', index_col=[0])
    boolean_columns = ['river_proximity', 'has_pool',
                    'has_garage', 'is_devided']
    for col in data_in.columns:
        if col not in boolean_columns:
            minus_mean = data_in[col] - feat_means.loc[col,][0]
            standardized = minus_mean / feat_stds.loc[col,][0]
            data_in[col] = standardized
        
    return data_in.to_numpy().reshape(-1, 1)   
    
# Cluster interpretation
st.write("""
    ## K-Means neighborhood clustering
    Neighborhoods have been aggregated into 8 clusters according to rated
    attributes such as proximity to parks or groceries and neighborhood demographics.
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
st.subheader("Neighbourhood clusters")
px.set_mapbox_access_token('pk.eyJ1IjoiamFobmljIiwiYSI6ImNrZ3dtbWRxNTBia3MzMW4wN2VudXZtcTUifQ.BVPxkX1DH75NahJvzt-f2Q')
st.write(
    px.scatter_mapbox(
        cluster_data, lat="lat", lon="long", color="clusters",
        color_continuous_scale=px.colors.sequential.Rainbow, 
        hover_data=['index', 'price', 'growth', 'normalized_difference'], 
        size='positive_differences',
        size_max=7, zoom=10, height=700, width=950)
)

st.text("""
    *Dot sizes corresponds to differences between predicted and actual condo price values.
    Larger dots are overpredictions and may potentially refer to undervalued condos.
    """)

st.subheader("Obtain records for condos of interest")

# User search for condos based on index
user_input = st.number_input(label="Type index of condo to look up:",
                             value=0
                        )

if user_input in list(data.index):
    data.iloc[user_input, :]
else:
    st.text("ValueError: Index out of range.")
    

st.subheader('Condo valuation')
st.write("""
        Adjust input values at the sidebar to predict prices for specific listings.
        The required parameters can all be found on [centris.ca](https://www.centris.ca/en/properties~for-sale~montreal-island?view=Thumbnail) and other Montreal
        real estate websites. Predictions work best for condos below $1.5 Million.  
        
        Intended usage: look up condos of interest and adjust the required parameters at the side bar. 
        If the predicted price is substantially higher than the actual (~$50,000) this means that 
        condos with similar attributes tend to be more expensive. This could imply that the condo is undervalued 
        and might be worth looking at in more detail.
        """)

# st.write("**Current input parameters**")
# # Display prediction input    
# prediction_params = user_input_features()     
# st.dataframe(prediction_params, height=2000)
# st.write('---')

# Predict price based on input
prediction_params = user_input_features() 
standardized_params = standardize_input(prediction_params)
prediction = model.predict(standardized_params.reshape(1,-1))
# Transform prediction back to $$$
price_std = 205882.9
price_mean = 512060.5
prediction_CAD = prediction * price_std + price_mean
prediction_CAD = pd.DataFrame({'Prediction (CAD)': prediction_CAD[0][0]}, index=[0])
# Print prediction
st.write("**Price estimation based on current input paramaters** (Adjust parameters in sidebar)")
st.write(prediction_CAD)
st.write('---')

# # Feature importance
# features_and_target = feature_data[feature_data.price < 1500000]
# standard_features = ((features_and_target - features_and_target.mean())
#                         / features_and_target.std())
# standard_target = standard_features.pop('price')
# X = standard_features.to_numpy()
# y = standard_target.to_numpy()

# @st.cache # improve performance
# def feature_importance():
#     perm = PermutationImportance(model, 
#                              scoring='neg_mean_squared_error',
#                              random_state=1).fit(X, y)
#     weights = eli5.explain_weights_df(perm,   
#                             feature_names=standard_features.columns.tolist(),
#                             top = 31)
#     return weights

# weights = feature_importance()
# weights.weight = 100*(weights.weight / weights.weight.sum())
# top_10 = weights.iloc[: 10, :]
# top_10.sort_values(by='weight', inplace=True)
# top_10.to_csv('data/top_10_features.csv')
# top_10.feature = np.array(['Restaurants', '2016 unemployment rate', 'Vibrant', 'Historic',
#        'Cafes', 'Groceries', 'Bathrooms', 'Has a garage',
#        'Distance from downtown', 'Total condo area'])
# Top 10 features

top_11 = pd.read_csv('data/top_11_features.csv', index_col='Unnamed: 0')

st.subheader('Top attributes that increase condo prices')

st.write(
    px.bar(top_11, 
    x=top_11.weight,
    y=top_11.feature,
    orientation='h',
    labels=dict(weight='Attribute importance (%)', feature="Attributes"),
    height=600,
    width=800,
    hover_data=['std']
    )
)

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
    
