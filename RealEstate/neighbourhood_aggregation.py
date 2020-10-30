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
# Standardize data
x = StandardScaler().fit_transform(neighbourhoods)

# PCA
pca = PCA(n_components=6)
X = pca.fit_transform(x)

n_cluster_range = [2, 3, 4, 5, 6, 7, 8]

for n_clusters in n_cluster_range:
       # subplot with 1 row and 2 columns
       fig, (ax1, ax2) = plt.subplots(1, 2)
       fig.set_size_inches(18, 7)
       
       # First plot is sihouette plot [-1, 1]
       ax1.set_xlim([-1, 1])
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
              ax1.fill_betweenx()   
       

"""
Demographics Clustering
"""

demographic_data = ['less_than_$50,000_(%)',
       'between_$50,000_and_$80,000_(%)', 'between_$80,000_and_$100,000_(%)',
       'between_$100,000_and_$150,000_(%)', 'more_than_$150,000_(%)',
       '1-person_households_(%)', '2-person_households_(%)',
       '3-person_households_(%)', '4-person_households_(%)',
       '5-person_or_more_households_(%)',
       'couples_without_children_at_home_(%)',
       'couples_with_children_at_home_(%)', 'single-parent_families_(%)',
       'owners_(%)', 'renters_(%)', 'before_1960_(%)',
       'between_1961_and_1980_(%)', 'between_1981_and_1990_(%)',
       'between_1991_and_2000_(%)', 'between_2001_and_2010_(%)',
       'between_2011_and_2016_(%)', 'single-family_homes_(%)',
       'semi-detached_or_row_houses_(%)',
       'buildings_with_less_than_5_floors_(%)',
       'buildings_with_5_or_more_floors_(%)', 'mobile_homes_(%)',
       'university_(%)', 'college_(%)', 'secondary_(high)_school_(%)',
       'apprentice_or_trade_school_diploma_(%)', 'no_diploma_(%)',
       'non-immigrant_population_(%)', 'immigrant_population_(%)',
       'french_(%)', 'english_(%)', 'others_languages_(%)']

# Slice neighbourhood and demographic data 
demographics = data[demographic_data]
# Standardize data
x = StandardScaler().fit_transform(demographics)

# PCA
pca = PCA()
pca.fit(x)
explained_variance = pca.explained_variance_
print(explained_variance)

# PCA
pca = PCA(n_components=6)
pca_transform = pca.fit_transform(x)