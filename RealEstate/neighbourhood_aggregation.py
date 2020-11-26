import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
import numpy as np
# Cluster classifications
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score, davies_bouldin_score,v_measure_score
from sklearn.mixture import GaussianMixture
from itertools import combinations
# Geomapping
import plotly.express as px
# Model saving
import pickle


# Data including feature on prediction differences
# diff = actual - predicted
data = pd.read_csv("web-app/data/data_with_prediction_differences.csv")
"""
Neighborhood Clustering
"""

neighbourhood_data = ['restaurants',
       'shopping', 'vibrant', 'cycling_friendly', 'car_friendly', 'historic',
       'quiet', 'elementary_schools', 'high_schools', 'parks', 'nightlife',
       'groceries', 'daycares', 'pedestrian_friendly', 'cafes',
       'transit_friendly', 'greenery']
# Slice neighbourhood and demographic data 
neighbourhoods = data[neighbourhood_data]

# Save standardization scaler
scaler = StandardScaler().fit(neighbourhoods)
pickle.dump(scaler, open('web-app/scaler.dat', 'wb'))
# Test and used loaded scaler model
x = pickle.load(open('web-app/scaler.dat', 'rb')).transform(neighbourhoods)

# PCA
pca = PCA(n_components=8)
pca_model = pca.fit(x) # save
pickle.dump(pca_model,open('web-app/pca.pkl', 'wb'))
# Transform data
X_neighborhood = pickle.load(open('web-app/pca.pkl', 'rb')).transform(x)
X_neighborhood
# # t-SNE
# tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
# tsne_results = tsne.fit_transform(x)
# print('t-SNE done!')

def visualize_components(data, component_1, component_2):
       '''Visualization of PC or t-SNE components'''
       plt.figure(figsize=(16,10))
       sns.scatterplot(
              x=component_1, y=component_2,
              palette=sns.color_palette("hls", 10),
              data=data,
              legend="full",
              alpha=0.3
       )

# Data for visualization of PC and t-SNE componenets 
reduced_dimensions = data.copy()
# reduced_dimensions['tsne-one'] = tsne_results[:,0]
# reduced_dimensions['tsne-two'] = tsne_results[:,1]
reduced_dimensions['pc-one'] = X_neighborhood[:,0]
reduced_dimensions['pc-two'] = X_neighborhood[:,1]
# Visualize reduced dimensions
visualize_components(reduced_dimensions, 'pc-one', 'pc-two')
# visualize_components(reduced_dimensions, 'tsne-one', 'tsne-two')

def silhouette_analysis(n_cluster_range, X):
       for n_clusters in n_cluster_range:
              # subplot with 1 row and 2 columns
              fig, (ax1, ax2) = plt.subplots(1, 2)
              fig.set_size_inches(18, 7)
              
              # First plot is sihouette plot [-1, 1]
              ax1.set_xlim([-0.25, 1])
              ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])
              
              # Initialize kmeans
              k_means = KMeans(n_clusters=n_clusters, random_state=42)
              cluster_labels = k_means.fit_predict(X)
              
              # Silhouette score gives perspective on density and separation
              silhouette_avg = silhouette_score(X, cluster_labels)
              print("For n_clusters =", n_clusters,
                     "The average silhouette_score is :", silhouette_avg)
              # Score for each sample
              sample_silhouette_values = silhouette_samples(X, cluster_labels)
              
              y_lower = 10
              for i in range(n_clusters):
                     # Aggregate silhouette scores for samples in cluster i and sort
                     ith_cluster_silhouette_values = \
                            sample_silhouette_values[cluster_labels == i]
                     ith_cluster_silhouette_values.sort()
                     size_cluster_i = ith_cluster_silhouette_values.shape[0]
                     y_upper = y_lower + size_cluster_i
                     
                     color = cm.nipy_spectral(float(i) / n_clusters)
                     ax1.fill_betweenx(np.arange(y_lower, y_upper),
                                   0, ith_cluster_silhouette_values,
                                   facecolor=color, edgecolor=color,
                                   alpha=0.7)
                     # Label silhouette plots with cluster numbers
                     ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
                     # New y_lower for next plot
                     y_lower = y_upper + 10
                     
                     ax1.set_title("The silhouette plot for the various clusters.")
                     ax1.set_xlabel("The silhouette coefficient values")
                     ax1.set_ylabel("Cluster label")   

                     # Vertical line for average score of all values
                     ax1.axvline(x=silhouette_avg, color="red", linestyle="--")
                     ax1.set_yticks([]) # Clear the yaxis labels / ticks
                     ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
                     
                     # 2nd Plot showing actual clusters formed
                     colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
                     ax2.scatter(X[:, 0], X[:, 1], marker='.', s=30, lw=0, alpha=0.7,
                                   c=colors, edgecolor='k')
                     
                     # Labeling clusters
                     centers = k_means.cluster_centers_
                     # Draw white circles at cluster centers
                     ax2.scatter(centers[:, 0], centers[:, 1], marker='o',
                                   c="white", alpha=1, s=200, edgecolor='k')
                     
                     for i, c in enumerate(centers):
                            ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1,
                                          s=50, edgecolor='k')
                     ax2.set_title("The visualization of the clustered data.")
                     ax2.set_xlabel("Feature space for the 1st feature")
                     ax2.set_ylabel("Feature space for the 2nd feature")
                     plt.suptitle(("Silhouette analysis for KMeans clustering on \
                     sample data " + "with n_clusters = %d" % n_clusters),
                     fontsize=14, fontweight='bold')
                     plt.show()
         
def k_optimization_plots(n_ks, sum_squared_distances, km_silhouette, gm_bic):
       K = range(2, n_ks + 1)
       # Elbow plot
       plt.figure(figsize=(13, 10))
       plt.plot(K, sum_squared_distances, 'bx-')
       plt.xlabel('K')
       plt.ylabel('Sum of squared distances')
       plt.xticks(K, fontsize=14)
       plt.yticks(fontsize=15)
       plt.title('Elbow Method for determining optimal K\n', fontsize=16)
       plt.show()
       
       # Silhouette plot
       plt.figure(figsize=(13, 10))
       plt.title("The silhouette coefficient method \nfor determining number of clusters\n",fontsize=16)
       plt.scatter(K, y=km_silhouette, s=150, edgecolor='k')
       plt.grid(True)
       plt.xlabel("Number of clusters",fontsize=14)
       plt.ylabel("Silhouette score",fontsize=15)
       plt.xticks(K, fontsize=14)
       plt.yticks(fontsize=15)
       plt.show()
       
       #  Bayesian Information Criterion (BIC) score plot
       plt.figure(figsize=(13, 10))
       plt.title("The Gaussian Mixture model BIC \nfor determining number of clusters\n",fontsize=16)
       plt.scatter(K,y=np.log(gm_bic),s=150,edgecolor='k')
       plt.grid(True)
       plt.xlabel("Number of clusters",fontsize=14)
       plt.ylabel("Log of Gaussian mixture BIC score",fontsize=15)
       plt.xticks(K, fontsize=14)
       plt.yticks(fontsize=15)
       plt.show()
      
"""
Demographics Clustering
"""
demographic_data = ['less_than_$50,000_(%)',
       'between_$50,000_and_$80,000_(%)', 'between_$80,000_and_$100,000_(%)',
       'between_$100,000_and_$150,000_(%)', 'more_than_$150,000_(%)',
       'couples_without_children_at_home_(%)',
       'couples_with_children_at_home_(%)', 'single-parent_families_(%)',
       'owners_(%)', 'renters_(%)', 'university_(%)', 'college_(%)', 'secondary_(high)_school_(%)',
       'apprentice_or_trade_school_diploma_(%)', 'no_diploma_(%)']
"""
PC1 (-): less_than_$50,000, couples_without_children_at_home, renters, university

PC1 (+): between_$50,000_and_$80,000, between_$80,000_and_$100,000, 
       between_$100,000_and_$150,000, couples_with_children_at_home, single-parent_families,
       owners, college, secondary,apprentice_or_trade_school, no_diploma

PC2 (-): less_than_$50,000, between_$50,000_and_$80,000, single-parent_families, 
       renter, secondary_(high)_school, apprentice_or_trade_school_diploma, no_diploma

PC2 (+): between_$100,000_and_$150,000, more_than_$150,000, owners, university
"""

# Slice neighbourhood and demographic data 
demographics = data[demographic_data]
demographics.columns
# Standardize data
x = StandardScaler().fit_transform(demographics)

# Initial PCA
pca = PCA()
pca.fit(x)
explained_variance = pca.explained_variance_
pca.components_
plt.plot(range(1, 16), explained_variance/sum(explained_variance))

# PCA
pca = PCA(n_components=2)
X_demo = pca.fit_transform(x)

# t-SNE
# tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
# tsne_results = tsne.fit_transform(x)
# print('t-SNE done!')
# test =  tsne.fit(x)

# Data for visualization of PC and t-SNE componenets 
# reduced_dimensions['demo-tsne-one'] = tsne_results[:,0]
# reduced_dimensions['demo-tsne-two'] = tsne_results[:,1]
# reduced_dimensions['demo-pc-one'] = X_demo[:,0]
# reduced_dimensions['demo-pc-two'] = X_demo[:,1]
# Visualize reduced dimensions
# visualize_components(reduced_dimensions, 'demo-pc-one', 'demo-pc-two')
# visualize_components(reduced_dimensions, 'demo-tsne-one', 'demo-tsne-two')

# Neighborhood and demographic data merged
X_merged = np.concatenate((X_neighborhood, X_demo), axis=1)

# silhouette_analysis(n_cluster_range, X_demo)
# n_cluster_range = [10, 11, 12, 13, 14, 15, 16, 17]
# silhouette_analysis(n_cluster_range, X_merged)

# Create dataframe with neighbourhood features of interest
neighbourhood_column_names = [('PC_neighborhood_' + str(i)) for i in range(1,9)]
demo_column_names = [('PC_demographics_' + str(i)) for i in range(1,3)]
column_names = neighbourhood_column_names + demo_column_names
column_names
new_data = pd.DataFrame(X_merged, columns=column_names)
new_data['price'] = data.price
new_data['walk_score'] = data.walk_score
new_data['condo_age'] = 2020 - data.year_built
# Normalize prediction differences (predicted - actual)
new_data['normalized_difference'] = 100 * (data['prediction_difference'] / data['price'])

"""
Best results using PC-1, PC-2 and neighborhood growth
------------------------------------------------------------
PC1: -> vibrant, urban life (Negative values)
PC2: -> family friendly, green, moderately vibrant and quiet (Negative Values)
"""
new_data.drop(['price'], axis=1).corr()
cluster_features = ['PC_neighborhood_1', 'PC_neighborhood_2', 
              'PC_demographics_1', 'PC_demographics_2']
x = new_data[cluster_features]
# Scale features
X_scaled = StandardScaler().fit_transform(x)
pd.DataFrame(X_scaled).corr()

# # Hyperparameter tuning
# sum_squared_distances= []
# km_silhouette = []
# db_score = []
# gm_bic= []
# for i in range(2,15):
#     km = KMeans(n_clusters=i, random_state=0).fit(X_scaled)
#     preds = km.predict(X_scaled)
    
#     # Variance 
#     print("Score for number of cluster(s) {}: {}".format(i,km.score(X_scaled)))
#     sum_squared_distances.append(-km.score(X_scaled))
    
#     # Silhouette
#     silhouette = silhouette_score(X_scaled,preds)
#     km_silhouette.append(silhouette)
#     print("Silhouette score for number of cluster(s) {}: {}".format(i,silhouette))
    
#     # Davies Bouldin
#     db = davies_bouldin_score(X_scaled,preds)
#     db_score.append(db)
#     print("Davies Bouldin score for number of cluster(s) {}: {}".format(i,db))
    
#     # Expectation-maximization (Gaussian mixture model)
#     gm = GaussianMixture(n_components=i,n_init=10,tol=1e-3,max_iter=1000).fit(X_scaled)
#     print("BIC for number of cluster(s) {}: {}".format(i,gm.bic(X_scaled)))
#     print("Log-likelihood score for number of cluster(s) {}: {}".format(i,gm.score(X_scaled)))
#     print("-"*100)
#     gm_bic.append(-gm.bic(X_scaled))

# k_optimization_plots(14, sum_squared_distances, km_silhouette, gm_bic)


# Final clustering
k_means = KMeans(n_clusters=8, random_state=0)
cluster_labels = k_means.fit_predict(X_scaled)
labels = cluster_labels.flatten()

# 2D visualization of cluster separation
for i,c in enumerate(cluster_features):
    plt.figure(figsize=(15,20))
    plt.subplot(9,1,i+1)
    sns.boxplot(y=x[c],x=labels)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.xlabel("Class",fontsize=15)
    plt.ylabel(c,fontsize=15)
    plt.show()

# Add clusters and indices
new_data['clusters'] = labels
new_data['index'] = new_data.index

# # -----> Skip clustering and load saved cluster_data table
# new_data = pd.read_csv('data/cluster_data.csv')

# Pivot tables
# print('Median condo price and growth per cluster:\n')
# print(round(new_data.pivot_table(index='clusters', 
#                            values=['price', 'growth', 'population_density',
#                                   'PC_neighborhood_1', 'PC_neighborhood_2',
#                                   'PC_demographics_1', 'PC_demographics_2',
#                                   'normalized_difference'],
#                            aggfunc='median'), 2))
# print('-'*50)
# print('Mean condo price and growth per cluster:\n')
# print(round(new_data.pivot_table(index='clusters', 
#                                  values=['price', 'growth', 'population_density',
#                                         'PC_neighborhood_1', 'PC_neighborhood_2',
#                                         'PC_demographics_1', 'PC_demographics_2',
#                                         'normalized_difference'],
#                                  aggfunc='mean'), 2))


# Need positive values for 'size' parameter of plotly scatter_mapbox
# Square to stretch extreme points further apart and better differentiate data points on the plot
new_data['positive_differences'] = ((new_data['normalized_difference'] - # -- = + 
                                   new_data['normalized_difference'].min()))**2
# Change cluster number for better color seperation on map
# Avoid light blue next to dark blue...
cluster_transform = {3:0, 1:1, 4:2, 6:3, 2:4, 5:5, 0:6, 7:7}
new_data.clusters = new_data.clusters.apply(lambda x: cluster_transform[x])


# Mapbox public access token
px.set_mapbox_access_token('pk.eyJ1IjoiamFobmljIiwiYSI6ImNrZ3dtbWRxNTBia3MzMW4wN2VudXZtcTUifQ.BVPxkX1DH75NahJvzt-f2Q')
# Additional columns for plotting
df = pd.concat([new_data, data[['lat', 'long', 'percent_population_variation',
                                'population_density', 'unemployment_rate']]], axis=1)
# Rename normalized to percent difference 
mapper = {'normalized_difference': 'Percent difference',
          'index': 'Index',
          'price': 'Price',
          'percent_population_variation': 'Neighborhood growth (%)',
          'clusters': 'Cluster',
          'positive_differences': 'Exponential difference',
          'population_density': 'Population density',
          'unemployment_rate': 'Unemployment rate',
          'condo_age': 'Condo age'}
df.rename(columns=mapper, inplace=True)
# Plot
fig = px.scatter_mapbox(df, lat="lat", lon="long", color="Cluster",
                        color_continuous_scale=px.colors.sequential.Rainbow, 
                        hover_data=['Index', 'Price', 'Condo age',
                                    'Percent difference', 'Neighborhood growth (%)'], 
                        size='Exponential difference',
                        size_max=7, zoom=10, title='K-Means neighbourhood clusters')
fig.show()

# Save cluster data
# df.to_csv('web-app/data/cluster_data.csv', index=False)
# pd.read_csv('web-app/data/cluster_data.csv')
df.to_csv('data/cluster_data.csv')
# Test
test_data = pd.read_csv('data/cluster_data.csv')
test_data['Percent difference'].std()
