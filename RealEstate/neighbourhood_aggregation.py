import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
# Cluster classifications
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score

data = pd.read_csv("data/complete_data.csv")

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
# Standardize data only if using neighborhood and demographics
# x = StandardScaler().fit_transform(neighbourhoods)
x = neighbourhoods

# PCA
pca = PCA(n_components=6)
X_neighborhood = pca.fit_transform(x)

def silhouette_analysis(n_cluster_range, X):
       for n_clusters in n_cluster_range:
              # subplot with 1 row and 2 columns
              fig, (ax1, ax2) = plt.subplots(1, 2)
              fig.set_size_inches(18, 7)
              
              # First plot is sihouette plot [-1, 1]
              ax1.set_xlim([-0.25, 1])
              ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])
              
              # Initialize kmeans
              k_means = KMeans(n_clusters=n_clusters, random_state=10)
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
         
def elbow_plot(data):
       sum_squared_distances = []
       K = range(1, 15)
       for k in K:
              km = KMeans(n_clusters=k)
              km = km.fit(data)
              sum_squared_distances.append(km.inertia_)
       # Plotting
       plt.figure(figsize=(13, 10))
       plt.plot(K, sum_squared_distances, 'bx-')
       plt.xlabel('K')
       plt.ylabel('Sum of squared distances')
       plt.title('Elbow Method for Optimal K')
       plt.show()
       
# """
# Demographics Clustering
# """

# demographic_data = ['less_than_$50,000_(%)',
#        'between_$50,000_and_$80,000_(%)', 'between_$80,000_and_$100,000_(%)',
#        'between_$100,000_and_$150,000_(%)', 'more_than_$150,000_(%)',
#        '1-person_households_(%)', '2-person_households_(%)',
#        '3-person_households_(%)', '4-person_households_(%)',
#        '5-person_or_more_households_(%)',
#        'couples_without_children_at_home_(%)',
#        'couples_with_children_at_home_(%)', 'single-parent_families_(%)',
#        'owners_(%)', 'renters_(%)', 'before_1960_(%)',
#        'between_1961_and_1980_(%)', 'between_1981_and_1990_(%)',
#        'between_1991_and_2000_(%)', 'between_2001_and_2010_(%)',
#        'between_2011_and_2016_(%)', 'single-family_homes_(%)',
#        'semi-detached_or_row_houses_(%)',
#        'buildings_with_less_than_5_floors_(%)',
#        'buildings_with_5_or_more_floors_(%)', 'mobile_homes_(%)',
#        'university_(%)', 'college_(%)', 'secondary_(high)_school_(%)',
#        'apprentice_or_trade_school_diploma_(%)', 'no_diploma_(%)',
#        'non-immigrant_population_(%)', 'immigrant_population_(%)',
#        'french_(%)', 'english_(%)', 'others_languages_(%)']

# # Slice neighbourhood and demographic data 
# demographics = data[demographic_data]
# # Standardize data
# x = StandardScaler().fit_transform(demographics)

# # PCA
# pca = PCA()
# pca.fit(x)
# explained_variance = pca.explained_variance_
# print(explained_variance)

# # PCA
# pca = PCA(n_components=6)
# X_demo = pca.fit_transform(x)

# Neighborhood and demographic data merged
# X_merged = np.concatenate((X_neighborhood, X_demo), axis=1)

# Silhouette analysis on neighborhood and demographic PCs
n_cluster_range = [2, 3, 4, 5, 6]
silhouette_analysis(n_cluster_range, X_neighborhood)
# Elbow method
elbow_plot(X_neighborhood)

# silhouette_analysis(n_cluster_range, X_demo)
# n_cluster_range = [10, 11, 12, 13, 14, 15, 16, 17]
# silhouette_analysis(n_cluster_range, X_merged)

# Final clustering
k_means = KMeans(n_clusters=3, random_state=10)
cluster_labels = k_means.fit_predict(x)
labels = cluster_labels.flatten()

# Create dataframe of neighbourhood PCs, price, and clusters
column_names = [('PC_' + str(i)) for i in range(1,7)]
new_data = pd.DataFrame(X_neighborhood, columns=column_names)
new_data['price'] = data.price
new_data['clusters'] = labels
new_data['growth'] = data.population_variation_between_2011_2016_
# Mean price pivot table
new_data.pivot_table(index='clusters', values=['price'], aggfunc='mean')
# Mean neighborhood growth pivot table
new_data.pivot_table(index='clusters', values=['growth'], aggfunc='mean')
