import numpy as np
import pandas as pd
import math
import streamlit as st
from PIL import Image
# Geomapping
import plotly.express as px
# Boost model
import xgboost as xgb
# Load models
import pickle

st.write("""
         # Montreal Real Estate Application
        **Explore Montreal neighborhoods and identify undervalued condos**
         """)

# Banner
img_1 = Image.open('banner.jpg')
st.image(img_1)
# # Outlier
# outlier_index = 2736
# Load data
data = pd.read_csv("data/data_with_prediction_differences.csv").iloc[:, 1:]
model_columns = ['price', 'restaurants', 'vibrant', 'cycling_friendly',
       'car_friendly', 'historic', 'quiet', 'parks', 'groceries', 'cafes',
       'transit_friendly', 'greenery', 'year_built', 'walk_score', 'bedrooms',
       'bathrooms', 'powder_rooms', 'total_area', 'mr_distance']

model_data = data[model_columns]
cluster_data = pd.read_csv("data/cluster_data.csv")

# Load models 
# model = keras.models.load_model('tf_linear_model_2')
pca = pickle.load(open('pca.pkl', 'rb'))
boost_model = pickle.load(open('boost_model.dat', 'rb')) 
standard_scaler = pickle.load(open('scaler.dat', 'rb'))

# Sidebar for parameter input
st.sidebar.header('Adjust Parameters for Price Prediction')

def get_mean(feat):
    if feat not in ['lat', 'long']:
        return int(data[feat].mean())
    else:
        return float(round(data[feat].mean(), 6))

def get_min(feat):
    if feat not in ['lat', 'long']:
        return int(data[feat].min())
    else:
        return float(data[feat].min())

def get_max(feat):
    if feat not in ['lat', 'long']:
        return int(data[feat].max())
    else:
        return float(data[feat].max())

def get_median(feat):
    return int(data[feat].median())

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
    # Final order of columns
    column_order = ['year_built', 'percent_population_variation',
       'population_density', 'unemployment_rate', 'walk_score',
       'total_area', 'river_proximity', 'has_pool', 'has_garage', 'is_devided',
       'mr_distance', 'PC_neighborhood_1', 'PC_neighborhood_2',
       'PC_neighborhood_3', 'PC_neighborhood_4', 'PC_neighborhood_5',
       'PC_neighborhood_6', 'PC_neighborhood_7', 'PC_neighborhood_8',
       'bedrooms_0.0', 'bedrooms_1.0', 'bedrooms_2.0', 'bedrooms_3.0',
       'bedrooms_4.0', 'bedrooms_5.0', 'bedrooms_6.0', 'bathrooms_0.0',
       'bathrooms_1.0', 'bathrooms_2.0', 'bathrooms_3.0', 'bathrooms_4.0',
       'powder_rooms_0.0', 'powder_rooms_1.0', 'powder_rooms_2.0']
    
    # Neighborhood ratings
    restaurant = st.sidebar.slider('Restaurants', 0, 10, get_median('restaurants'), key=1)
    shopping = st.sidebar.slider('Shopping', 0, 10, get_median('shopping'), key=2)
    vibrant = st.sidebar.slider('Vibrant', 0, 10, get_median('vibrant'), key=3)
    cycling_friendly = st.sidebar.slider('Cycling Friendly', 0, 10, get_median('cycling_friendly'), key=4)
    car_friendly = st.sidebar.slider('Car Friendly', 0, 10, get_median('car_friendly'), key=5)
    historic = st.sidebar.slider('Historic', 0, 10, get_median('historic'), key=6)
    quiet = st.sidebar.slider('Quiet', 0, 10, get_median('quiet'), key=7)
    elementary_schools = st.sidebar.slider('Elementary Schools', 0, 10, get_median('elementary_schools'), key=8)
    high_schools = st.sidebar.slider('High-Schools', 0, 10, get_median('high_schools'), key=9)
    parks = st.sidebar.slider('Parks', 0, 10, get_median('parks'), key=10)
    nightlife = st.sidebar.slider('Nightlife', 0, 10, get_median('nightlife'), key=11)
    groceries = st.sidebar.slider('Groceries', 0, 10, get_median('groceries'), key=12)
    daycares = st.sidebar.slider('Daycares', 0, 10, get_median('daycares'), key=13)
    pedestrian_friendly = st.sidebar.slider('Pedestrian Friendly', 0, 10, get_median('pedestrian_friendly'), key=14)
    cafes = st.sidebar.slider('Cafes', 0, 10, get_median('cafes'), key=15)
    transit_friendly = st.sidebar.slider('Transit Friendly', 0, 10, get_median('transit_friendly'), key=16)
    greenery = st.sidebar.slider('Greenery', 0, 10, get_median('greenery'), key=17)
    
    # Other parameters
    year_built = st.sidebar.slider('Year Built', get_min('year_built'), 
                                   get_max('year_built'), 
                                   get_mean('year_built'), key=18)
    percent_population_variation = st.sidebar.slider('2011-2016 Population Variation',
                                                get_min('percent_population_variation'), 
                                                get_max('percent_population_variation'), 
                                                get_mean('percent_population_variation')
                                                , key=19)
    unemployment_rate = st.sidebar.slider('2016 Unemployment Rate', 
                                                get_min('unemployment_rate'), 
                                                get_max('unemployment_rate'), 
                                                get_mean('unemployment_rate')
                                                , key=20)
    population_density = st.sidebar.slider('2016 Population Density', 
                                                get_min('population_density'), 
                                                get_max('population_density'), 
                                                get_mean('population_density')
                                                , key=31)
    walk_score = st.sidebar.slider('Walk Score', get_min('walk_score'), 
                                   get_max('walk_score'), 
                                   get_mean('walk_score'), key=21)
    # rooms = st.sidebar.slider('Rooms', get_min('rooms'), 
    #                                get_max('rooms'), 
    #                                get_mean('rooms'), key=22)
    bedrooms = st.sidebar.slider('Bedrooms', 0, 6, 
                                   get_mean('bedrooms'), key=23)
    bathrooms = st.sidebar.slider('Bathrooms', 0, 4, 
                                   get_mean('bathrooms'), key=24)
    powder_rooms = st.sidebar.slider('Powder Rooms', 0, 2, 
                                   get_mean('powder_rooms'), key=25)
    river_proximity = st.sidebar.checkbox('River proximity', value=False, key=27)
    has_pool = st.sidebar.checkbox('Has Pool', value=False, key=28)
    has_garage = st.sidebar.checkbox('Has Garage', value=False, key=29)
    is_devided = st.sidebar.checkbox('Divided Condo', value=False, key=30)
    total_area = st.sidebar.number_input('Total Area (sqft)', min_value=get_min('total_area'), 
                                   max_value=get_max('total_area'), 
                                   value=get_mean('total_area'))
    lat = st.sidebar.number_input('Latitude', min_value=get_min('lat'), 
                                   max_value=get_max('lat'), value=45.456079,
                                    format="%.6f")
    lon = st.sidebar.number_input('Longitude', min_value=get_min('long'), 
                                   max_value=get_max('long'), value=-73.575949,
                                    format="%.6f")
    
    # Neighborhood data for PCA transformation
    neighborhoods = pd.DataFrame(
        {
        'restaurants': restaurant,
        'shopping': shopping,
        'vibrant': vibrant,
        'cycling_friendly': cycling_friendly,
        'car_friendly': car_friendly,
        'historic': historic,
        'quiet': quiet,
        'elementary_schools': elementary_schools,
        'high_schools': high_schools,
        'parks': parks,
        'nightlife': nightlife,
        'groceries': groceries,
        'daycares': daycares,
        'pedestrian_friendly': pedestrian_friendly,
        'cafes': cafes, 
        'transit_friendly': transit_friendly,
        'greenery': greenery}, index=[0]
    )
    
    # PCA transform 
    x = standard_scaler.transform(neighborhoods)
    transformed_neighborhood = pca.transform(x)
     
    data_in = {
        'year_built': year_built,
        'percent_population_variation': percent_population_variation,
        'unemployment_rate': unemployment_rate,
        'population_density': population_density,
        'walk_score': walk_score,
        'total_area': total_area,
        'river_proximity': int(river_proximity),
        'has_pool': int(has_pool),
        'has_garage': int(has_garage),
        'is_devided': int(is_devided),
        'mr_distance': distance(destination=((float(lat), float(lon)))) # compute distance from downtown
    }
    
    # Rooms into categorical
    bed = {0: 'bedrooms_0.0', 1: 'bedrooms_1.0', 2: 'bedrooms_2.0', 3: 'bedrooms_3.0',
       4: 'bedrooms_4.0', 5: 'bedrooms_5.0', 6: 'bedrooms_6.0'}
    bath = {0: 'bathrooms_0.0',
       1: 'bathrooms_1.0', 2: 'bathrooms_2.0', 3: 'bathrooms_3.0', 4: 'bathrooms_4.0'}
    powder = {0: 'powder_rooms_0.0', 1: 'powder_rooms_1.0', 2: 'powder_rooms_2.0'}
    
    # Bedrooms
    for rooms in bed.values():
        data_in[rooms] = 0
    n_bedrooms = bed[bedrooms]
    data_in[n_bedrooms] = 1
    
    # Bathrooms
    for rooms in bath.values():
        data_in[rooms] = 0
    n_bathrooms = bath[bathrooms]
    data_in[n_bathrooms] = 1
    
    # Powderrooms
    for rooms in powder.values():
        data_in[rooms] = 0
    n_powder_rooms = powder[powder_rooms]
    data_in[n_powder_rooms] = 1
    
    # Add principle components
    component = 1
    for pc in transformed_neighborhood[0]:
        pc_name = "PC_neighborhood_" + str(component)
        data_in[pc_name] = pc
        component += 1 
    
    # Convert data in to dataframe    
    input_result = pd.DataFrame(data_in, index=[0])
    
    # Transform skewed features
    input_result[['PC_neighborhood_1']] = np.sqrt(
                                4 + input_result[['PC_neighborhood_1']]
                                            )
    log_transform_features = ['total_area', 'mr_distance',
                 'year_built', 'walk_score', 'population_density',
                 'percent_population_variation']
    input_result[log_transform_features] = np.log1p(
                                input_result[log_transform_features]
                                                )
    # Correct order of results
    input_result = input_result[column_order]
    return input_result

# Predict price based on input
prediction_params = user_input_features() 
prediction = boost_model.predict(prediction_params)
# Transform prediction back to $$$
prediction = np.expm1(prediction)
prediction_CAD = pd.DataFrame({'Prediction (CAD)': prediction}, index=[0])

# Price prediction
st.header('Condo valuation')
st.write("""
        Adjust input values at the sidebar to predict prices for specific listings.
        The required parameters can all be found on [centris.ca](https://www.centris.ca/en/properties~for-sale~montreal-island?view=Thumbnail) and other Montreal
        real estate websites. Predictions work best for condos valued below $1.2 Million which corresponds to about 95% of all condos.  
        
        Intended usage: look up condos of interest and adjust the required parameters at the side bar. 
        If the predicted price is substantially higher than the actual (~15%) this means that 
        condos with similar attributes tend to be more expensive. This could imply that the condo is undervalued 
        and might be worth looking at in more detail.
        """)
st.write("**Price estimation based on current input paramaters** (Adjust parameters in sidebar)")
st.write(prediction_CAD)
st.write('---')
    
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
    **Cluster 6:** Mix of vibrant and family friendly neighborhood with both low and high income earners.  
    **Cluster 7:** Very Family friendly and uneventful with mostly middle class families.
        """)  

# Additional cluster Interpretation
st.subheader("""
        Additional attributes averaged over each cluster
    """)  

mean_pivot = round(cluster_data.pivot_table(index='Cluster', 
                            values=['PC_neighborhood_1', 'PC_neighborhood_2',
                                'PC_demographics_1', 'PC_demographics_2',
                                'Neighborhood growth (%)', 'Population density',
                                'Unemployment rate',
                                'Price', 'Percent difference', 'Condo age'],
                            aggfunc='mean'))

additional_attributes = mean_pivot[['Neighborhood growth (%)', 
                                    'Population density',
                                    'Condo age',
                                    'Price'
                                    ]]
st.table(
    additional_attributes
)

# Geomapping
st.subheader("Neighbourhood clusters")
px.set_mapbox_access_token('pk.eyJ1IjoiamFobmljIiwiYSI6ImNrZ3dtbWRxNTBia3MzMW4wN2VudXZtcTUifQ.BVPxkX1DH75NahJvzt-f2Q')
st.write(
    px.scatter_mapbox(
        cluster_data, lat="lat", lon="long", color="Cluster",
        color_continuous_scale=px.colors.sequential.Rainbow, 
        hover_data=['Index', 'Neighborhood growth (%)', 
                                    'Population density',
                                    'Unemployment rate',
                                    'Condo age',
                                    'Price',
                                    'Percent difference'], 
        size='Exponential difference',  
        size_max=7, zoom=10, height=700, width=950)
)

st.text("""
    *Dot sizes corresponds to differences between predicted and actual condo price values.
    Larger dots are overpredictions and may potentially refer to undervalued condos.
    """)


st.subheader("Obtain records for condos of interest")
# User search for condos based on index
user_input = st.number_input(label="Type the index of condos found on the map:",
                             value=0
                        )

if user_input in list(data.index):
    st.table(
        data.iloc[user_input, :]
    )
else:
    st.text("ValueError: Index out of range.")

# Hunt for undervalued condos
st.header("Uncover undervalued condos")
# Top 11 features of linear model
st.subheader('Top linear attributes that increase value')
st.write("""
    Linear attributes with large positive weights have a strong impact on price, regardless of 
    other factors. The results should be taken with a grain of salt, since the data is not well
    represented by the linear model.
         """)
top_11 = pd.read_csv('data/top_11_features.csv', index_col='Unnamed: 0')
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

# Top non linear features of boost model
st.subheader('Non-linear attributes that influence value')
st.write("""
    The effect of non-linear attributes on price depends on the context supplied by the values
    that the remaining attributes exhibit. The following
    figure lists non-linear attributes according to how strongly they influence condo prices. The data
    is better represented by the non-linear model, however, the circumstances under which prices increase
    or decrease are not known.
         """)
img_2 = Image.open('non_linear_feat_importance.png')
st.image(img_2, width=800)

# Overpredictions
st.subheader("Top overpredicted prices")
filter_in = st.radio('Select filter', ['Absolute difference', 'Percent difference'])
pred = data[['price', 'predicted', 'prediction_difference']]
pred = pred.astype('int')
pred['Percent difference'] = cluster_data['Percent difference'].astype('int')
pred['Neighborhood growth'] = cluster_data['Neighborhood growth (%)']
pred['cluster'] = cluster_data.Cluster

# More readable columns names
new_cols = ['Price (CAD)', 'Predicted', 'Absolute difference', 
            'Percent difference', 'Neighborhood growth', 'Neighborhood cluster']
pred.columns = new_cols
if filter_in == 'Absolute difference':
    top_pred = pred.sort_values(by='Absolute difference', ascending=False).iloc[: 200, :]
elif filter_in == 'Percent difference':
    top_pred = pred.sort_values(by='Percent difference', ascending=False).iloc[: 200, :]
# Display top predictions
for col in top_pred:
    if col in ['Price (CAD)', 'Predicted', 'Absolute difference']:
        top_pred[col] = top_pred[col].apply(lambda x: f'${x:,}')
    elif col in ['Percent difference', 'Neighborhood growth']:
        top_pred[col] = top_pred[col].apply(lambda x: f'{x:,}%')
st.dataframe(
    top_pred
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
    
