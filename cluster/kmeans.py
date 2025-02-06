import numpy as np
from scipy.spatial.distance import cdist
import random as rd


class KMeans:
    def __init__(self, k: int, tol: float = 1e-6, max_iter: int = 100):
        """
        In this method you should initialize whatever attributes will be required for the class.

        You can also do some basic error handling.

        What should happen if the user provides the wrong input or wrong type of input for the
        argument k?

        inputs:
            k: int
                the number of centroids to use in cluster fitting
            tol: float
                the minimum error tolerance from previous error during optimization to quit the model fit
            max_iter: int
                the maximum number of iterations before quitting model fit
        """
        self.k = k
        self.tol = tol
        self.max_iter = max_iter
        self.centroids = None
        self.error = 0

        if not isinstance(k, int) or not isinstance(tol, float) or not isinstance(max_iter, int):
            raise ValueError("Parameters are incorrect types")
        if k <2:
            raise ValueError("Number of clusters must be an integer greater than 2")


    def fit(self, mat: np.ndarray):
        """
        Fits the kmeans algorithm onto a provided 2D matrix.
        As a bit of background, this method should not return anything.
        The intent here is to have this method find the k cluster centers from the data
        with the tolerance, then you will use .predict() to identify the
        clusters that best match some data that is provided.

        In sklearn there is also a fit_predict() method that combines these
        functions, but for now we will have you implement them both separately.

        inputs:
            mat: np.ndarray
                A 2D matrix where the rows are observations and columns are features
        """

        if not isinstance(mat, np.ndarray):
            raise ValueError("Matrix needs to be of type np.ndarray")
        if mat.shape[1] <1:
            raise ValueError("Observations must have at least one feature")
        if self.centroids is not None and mat.shape[1] != self.centroids.shape[1]:
            raise ValueError("Observations must have same features as centroids")
        if self.k > len(mat):
            raise ValueError("k must be smaller than the number of observations")

        #self.centroids = mat[np.random.choice(len(mat), size=self.k, replace=False)] if random kmeans

        #KMeans++ initialization
        obs, _ = mat.shape
        centroids = [mat[np.random.choice(obs)]] #initialize first centroid to random point in matrix

        for _ in range(1,self.k): #iterate through all the clusters to assign centroids
            sq_dist = cdist(mat, np.array(centroids), metric = 'sqeuclidean') #compute squared distances to centroid
            min_dist = np.min(sq_dist, axis =1) #for each point, find the minimum distance to nearest centroid

            prob = min_dist/np.sum(min_dist) #create weighted probability proportional to minimum distance^2
            next_centroid = np.random.choice(obs, p = prob) #choose next centroid based on probability
            centroids.append(mat[next_centroid]) #append centroid to centroid array

        self.centroids = np.array(centroids) #set centroids to the kmeans++ identified centroids

        #kmeans alg
        iteration = 0
        change_sse = 1
        while change_sse > self.tol and iteration < self.max_iter:
            cluster_id = np.argmin(cdist(mat, self.centroids), axis = 1) #compute pairwise distances between points in the two matrices and assign cluster to each point based on the argument number of the centroid that has the minimum distance
            sum_squared_error = 0
            for j in range(self.k):
                cluster = mat[cluster_id == j] #apply mask to only include the values in the matrix that belong to cluster j
                self.centroids[j] = np.mean(cluster, axis = 0) #calculate new centroids based on the mean of the points assigned to that cluster
                sum_squared_error += np.sum(cdist(cluster,self.centroids[j].reshape(1,-1)) ** 2) #compute sum squared error of cluster
            change_sse = abs(self.error - sum_squared_error) #compute the change in sum squared error between previous run and current run
            self.error = sum_squared_error #set new error to sum squared error
            iteration += 1



    def predict(self, mat: np.ndarray) -> np.ndarray:
        """
        Predicts the cluster labels for a provided matrix of data points--
            question: what sorts of data inputs here would prevent the code from running?
            How would you catch these sorts of end-user related errors?
            What if, for example, the matrix is of a different number of features than
            the data that the clusters were fit on?

        inputs:
            mat: np.ndarray
                A 2D matrix where the rows are observations and columns are features

        outputs:
            np.ndarray
                a 1D array with the cluster label for each of the observations in `mat`
        """
        if self.centroids is not None and mat.shape[1] != self.centroids.shape[1]:
            raise ValueError("Observations must have same number of features as centroids")

        if self.centroids is None:
            raise ValueError("Centroids must be fit before predicting cluster labels")

        cluster_id = np.argmin(cdist(mat, self.centroids), axis = 1)  #compute pairwise distances between points in the two matrices and assign cluster to each point based on the argument number of the centroid that has the minimum distance
        return cluster_id #return the ids of the centroids

    def get_error(self) -> float:
        """
        Returns the final squared-mean error of the fit model. You can either do this by storing the
        original dataset or recording it following the end of model fitting.

        outputs:
            float
                the squared-mean error of the fit model
        """

        if self.centroids is None:
            raise ValueError("Centroids must be fit before calling get_error()")
        else:
            return self.error


    def get_centroids(self) -> np.ndarray:
        """
        Returns the centroid locations of the fit model.

        outputs:
            np.ndarray
                a `k x m` 2D matrix representing the cluster centroids of the fit model
        """
        if self.centroids is None:
            raise ValueError("Centroids must be fit before calling get_centroids()")
        else:
            return self.centroids
