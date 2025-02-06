# write your silhouette scoreREADME.md unit tests here
import pytest
import numpy as np
from sklearn.metrics import silhouette_score
from cluster.utils import make_clusters
from cluster.kmeans import KMeans
from cluster.silhouette import Silhouette

def test_silhouette():
    #check with 20 clusters
    mat, labels = make_clusters(n=1000, k=20) #make matrix of 1000 points assigned to 20 clusters stored in array labels
    test_kmeans = KMeans(20) #run Kmeans algorithm with 20 clusters
    test_kmeans.fit(mat) #fit the matrix of observations to 20 clusters
    predict = test_kmeans.predict(mat) #return a 1D array of predictions for which cluster each point is assigned to

    sil = Silhouette() #create a silhouette object
    test_scores = sil.score(mat, predict) #using the matrix generated from make_clusters and the prediction array generated from the user function written in the homework, score the kmeans clustering
    test_av_scores = test_scores.mean() #compute the average of the silhouette scores of each point

    check_scores = silhouette_score(mat, predict, metric = 'euclidean') #get the average silhouette scores of the matrix with scikit learn built in function
    assert abs(test_av_scores-check_scores) <= 1e-3 #assert that the result are within 0.001 of each other

def test_edge_cases():
    mat = np.array([[2,3], [5,6], [17,40],[2,6],[8,10]])
    sil = Silhouette()

    # check where k > n observed
    pred_error = np.array([1, 0, 2, 5, 3, 4, 2, 6, 2, 1, 0])
    with pytest.raises(ValueError):
        sil.score(mat, pred_error)

    # check where 1 cluster
    pred_one = np.array([1,1,1,1,1])
    with pytest.raises(ValueError):
        sil.score(mat, pred_one)

    # check where observations are missing cluster assignments
    pred_missing = np.array([1, 0, 1, 2])
    with pytest.raises(ValueError):
        sil.score(mat, pred_missing)






