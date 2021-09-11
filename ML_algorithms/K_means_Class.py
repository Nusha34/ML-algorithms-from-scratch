#  /******************************************************************************
#   * Copyright (c) - 2021 - Anna Chechulina                                  *
#   * The code in K_means_Class.py  is proprietary and confidential.                  *
#   * Unauthorized copying of the file and any parts of it                       *
#   * as well as the project itself is strictly prohibited.                      *
#   * Written by Anna Chechulina  <chechulinaan17@gmail.com>,   2021                 *
#   ******************************************************************************/

import numpy as np
import random

class K_means:

    
    def euclidean_distance(self, a,b):
        #make a square for every string and then sum up it for example 1*1+2*2, then we'will get 5, 13,25 in one row therefore we make reshape
        A_sq=np.reshape(np.sum(a * a, axis=1), (a.shape[0], 1))
        B_sq = np.reshape(np.sum(b * b, axis=1), (1, b.shape[0]))
        AB=np.dot(a, b.transpose())
        #AB = a@b.T    
        C = -2 * AB + B_sq + A_sq
        return np.sqrt(C)
    
    def get_initial_centroids(self, X, k):
        """
        Description:
        Picks k random unique points from dataset X. Selected points can be used as intial centroids.
    
        Args:
        X (numpy.ndarray) : dataset points array, size N:D
        k (int): number of centroids

        Returns: 
        (numpy.ndarray): array of k unique initial centroids, size K:D

        """
        # count num samples
        num_samples = X.shape[0]
        # sample k points in range(num_samples)
        sample_pt_idx = random.sample(range(0, num_samples), k)
        # assign k points as centroids (choosing one random element from dataset)
        centroids = [tuple(X[i]) for i in sample_pt_idx]
        unique_centroids = list(set(centroids))
        num_unique_centroids = len(unique_centroids)

        return np.array(unique_centroids)
    
    
    def compute_clusters(self, X, centroids):
        """
        Description:
        Function finds k centroids and assigns each of the N points of array X to one centroid
        Args:
        X (numpy.ndarray): array of sample points, size N:D
        centroids (numpy.ndarray): array of centroids, size K:D
        distance_mesuring_method (function): function taking 2 Matrices A (N1:D) and B (N2:D) and returning distance
        between all points from matrix A and all points from matrix B, size N1:N2

        Returns:
        dict {cluster_number: list_of_points_in_cluster}
        """
        # k?
        k = centroids.shape[0]
        # new clusters dict
        clusters = {}
        # compute distance to centroids
        distance_mat = self.euclidean_distance(X, centroids)
        # Assign pt to closest cluster
        closest_cluster_ids = np.argmin(distance_mat, axis=1)


        #create necessary number of clusters 
        for i in range(k):
            clusters[i] = []


        for i, cluster_id in enumerate(closest_cluster_ids):
            clusters[cluster_id].append(X[i])

        return clusters
    
    
    def check_convergence(self, previous_centroids, new_centroids, movement_threshold_delta):
        """
        Description:
        Function checks if any of centroids moved more than the MOVEMENT_THRESHOLD_DELTA if not we assume the centroids were found
    
        Args:
        previous_centroids (numpy.ndarray): array of k old centroids, size K:D
        new_centroids (numpy.ndarray): array of k new centroids, size K:D
        distance_mesuring_method (function): function taking 2 Matrices A (N1:D) and B (N2:D) and returning distance
        movement_threshold_delta (float): threshold value, if centroids move less we assume that algorithm covered


        Returns: 
        boolean True if centroids coverd False if not

        """
        #use euclidean_distance
        distances_between_old_and_new_centroids = self.euclidean_distance(previous_centroids, new_centroids)
        #if we don't move then we found our centroids
        converged = np.max(distances_between_old_and_new_centroids.diagonal()) <= movement_threshold_delta

        return converged
    
    
    
    def do_k_means(self, X, k, movement_threshold_delta=0):
        """
        Description:
        Performs k-means algorithm on a given dataset, finds and returns k centroids
    
        Args:
        X (numpy.ndarray) : dataset points array, size N:D
        distance_mesuring_method (function): function taking 2 Matrices A (N1:D) and B (N2:D) and returning distance
        between all points from matrix A and all points from matrix B, size N1:N2.
        k (int): number of centroids
        movement_threshold_delta (float): threshold value, if centroids move less we assume that algorithm covered

        Returns:
        (numpy.ndarray): array of k centroids, size K:D
        """
        new_centroids = self.get_initial_centroids(X=X, k=k)

        converged = False

        while not converged:
            previous_centroids = new_centroids
            clusters = self.compute_clusters(X, previous_centroids)
            #create next centoids via mean 
            new_centroids = np.array([np.mean(clusters[key], axis=0, dtype=X.dtype) for key in sorted(clusters.keys())])
            converged = self.check_convergence(previous_centroids, new_centroids, movement_threshold_delta)

        return new_centroids