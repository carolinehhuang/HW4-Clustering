# Write your k-means unit tests here
from operator import index

import pytest
import numpy as np
from cluster.utils import make_clusters
from cluster.kmeans import KMeans
from sklearn.cluster import KMeans as KMeans_check

def test_kmeans():
    mat, labels = make_clusters(k = 7, scale = 0.3)

    test_km = KMeans(k=7)
    test_km.fit(mat)
    predict = test_km.predict(mat)

    centroids = test_km.get_centroids()
    centroids_sorted = np.sort(centroids[:,0]) #sort the centroids by x value
    centroids_sorted_idx = np.argsort(centroids[:, 0]) #sort cluster number by centroid value

    check_km = KMeans_check(n_clusters = 7) #use sklearn builtin function to initialize kmeans
    check_predict = check_km.fit_predict(mat) #fit to matrix

    check_centroids = check_km.cluster_centers_
    check_centroids_sorted = np.sort(check_centroids[:,0]) #sort the centroids by x value
    check_centroids_sorted_idx = np.argsort(check_centroids[:, 0]) #sort cluster number by centroid value

    mapping = {a:b for a,b in zip(centroids_sorted_idx,check_centroids_sorted_idx)} #create mapping of cluster number from implementation to scikit learn solution
    translate = np.array([mapping[i] for i in predict]) #translate the cluster predictions from implementation clusters to scikit clusters

    assert np.allclose(centroids_sorted,check_centroids_sorted)
    assert np.allclose(check_predict, translate)

def test_multidim():
    multi_features = np.array([[1, 1, 1, 1, 2], [5, 5, 5, 5, 6], [4, 3, 75, 2, 3], [7, 14, 134, 3, 123], [12, 3, 324, 3, 98]])
    test_km = KMeans(2)
    test_km.fit(multi_features)
    predict = test_km.predict(multi_features)

    assert len(test_km.get_centroids()) == 2
    assert test_km.get_centroids().shape[1] == 5

def test_edge_cases():

    #parameter is not an integer > 2
    with pytest.raises(ValueError):
        KMeans(-4)

    #parameter is not the correct type
    with pytest.raises(ValueError):
        KMeans("BMI203")

    #number of clusters is greater than number of observations
    basket, _ = make_clusters(k = 3, scale = 0.3) #store matrix of only 500 observations
    elephant = KMeans(1000)
    with pytest.raises(ValueError):
        elephant.fit(basket) #hard to fit an elephant in a basket

    #try to predict with an array of predictions the wrong dimension
    cheetah = KMeans(3)
    pants = np.array([[1,1,1,1,2],[5,5,5,5,6]])
    cheetah.fit(basket)
    with pytest.raises(ValueError):
        cheetah.predict(pants) #how would cheetahs wear pants anyway??











