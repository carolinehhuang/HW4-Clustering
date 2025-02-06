import numpy as np
from scipy.spatial.distance import cdist


class Silhouette:
    def __init__(self):
        """
        inputs:
            none
        """
    def score(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        calculates the silhouette score for each of the observations

        inputs:
            X: np.ndarray
                A 2D matrix where the rows are observations and columns are features.

            y: np.ndarray
                a 1D array representing the cluster labels for each of the observations in `X`

        outputs:
            np.ndarray
                a 1D array with the silhouette scores for each of the observations in `X`
        """
        #check that the inputs have the right amount of data
        if len(X) != len(y):
            raise ValueError("Number of observations and number of cluster assignments don't match")
        #check if there is more than 1 cluster
        if len(np.unique(y)) < 2:
            raise ValueError("Not enough clusters")
        if len(np.unique(y)) > len(X):
            raise ValueError("More clusters than observations")
        if not isinstance(X, np.ndarray) or X.ndim != 2:
            raise ValueError("X must be a 2D array")
        if not isinstance(y, np.ndarray) or y.ndim != 1:
            raise ValueError("y must be a 1D array")


        silhouette = [] #initialize array for silhouette scores for each observation
        for i in range(len(X)): #iterate through all the observations
            intra = self._intra_score(X, y, i) #calculate the mean distances between the observation and all other points in the same cluster
            inter = self._inter_score(X, y, i) #calculate the mean distance between the observation and all other points in different clusters
            silhouette.append((inter-intra)/max(intra, inter)) #append the silhouette score

        return np.array(silhouette) #return the array for the silhouette scores of each point


    def _intra_score(self, X: np.ndarray, y: np.ndarray, point):
        cluster = y[point] #get the cluster of the observation being evaluated
        coordinates = X[point].reshape(1,-1) #get coordinates of the observation
        #X[y == cluster] creates a matrix with a mask of the rows where the cluster matches the cluster of the given point
        distances = np.array(cdist(coordinates, X[y == cluster],'euclidean')) #evaluate the distances between the observation point and all other points that are assigned to same cluster
        return np.mean(distances[distances != 0]) #return the mean of the distances between point and all other points in cluster

    def _inter_score(self, X: np.ndarray, y: np.ndarray, point):
        cluster = y[point] #get the cluster of the observation being evaluated
        coordinates = X[point].reshape(1,-1) #get coordinates of the observation
        cluster_dist = [] #initialize empty array of cluster distances
        for cluster_id in np.unique(y): #iterate through all the unique clusters
            if cluster_id != cluster: #ensure the cluster is not the same cluster the point is in
                distances = np.array(cdist(coordinates, X[y == cluster_id], 'euclidean')) #compute distances between observation and all points in each cluster that meets the previous criteria
                cluster_dist.append(np.mean(distances)) #add the mean of the distances of each cluster to the array
        return min(cluster_dist) #return the minimum mean distance of the point to a different cluster





