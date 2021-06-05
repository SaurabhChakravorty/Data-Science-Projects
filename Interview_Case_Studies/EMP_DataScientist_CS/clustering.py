## import libraries
from sklearn.preprocessing import StandardScaler as std
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.preprocessing import MinMaxScaler, power_transform
from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_score,davies_bouldin_score
from sklearn.cluster import AgglomerativeClustering,DBSCAN,KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import scipy.cluster.hierarchy as sch
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import collections
import numpy as np
import pandas as pd
from IPython.display import display, HTML
from termcolor import colored
import matplotlib.pyplot as plt

#### Clustering 

# Clustering is done here with the help of following set of algorithms:
#  - KMeans
#  - Agglomerative clustering
#     - Hierarchial clustering
#     - DB scan clustering


# The following library has all the validation tests and algorithms to perform clustering

class clustering:
    """
    This class has all the tests of clustering criterion to find the optimal number of clusters
    and all the clustering methods to do clustering

    Clustering methods dicussed over here are:

    1.  Agglomerative clustering
         1.1) Hierarchial clustering
         1.2) DB scan clustering

    2.  K-means clustering

    """

    def __init__(self, X):
        self.X = X

    def cluster_plot(self):
        """
        Cluster plotting for different cluster algorithms
        """
        train          = StandardScaler().fit_transform(self.X)
        pca            = PCA(n_components=3)
        pca_component  = pca.fit_transform(self.X)
        fig = plt.figure(figsize=(10,8))
        sns.set_palette(sns.color_palette("cubehelix", 8))
        ax = Axes3D(fig)
        ax.scatter(pca_component[:,0].tolist(),pca_component[:,1].tolist(),pca_component[:,2].tolist(),c=self.labels,marker='v')
        ax.legend()
        plt.show()

    def dendogram(self):
        """
        This method plots dendogram for hierarchial clustering
        """
      
        plt.figure(figsize=(20, 7))
        dendrogram = sch.dendrogram(sch.linkage(self.X, method='ward'))
        plt.title("Dendograms")
        plt.axhline(linestyle='--', y=5) 
        plt.show()
  
    def silhouette_scores(self):
        """
        This method plots silhouette_scores for k-means clustering to find optimal number of clusters

        """
        kmeans_models = [KMeans(n_clusters=k, random_state=42).fit(self.X) for k in range(1, 10)]
        silhouette_scores = [silhouette_score(self.X, model.labels_) for model in kmeans_models[1:]]
        print(colored("The maximum silhouette score is %0.02f at the cluster number %d\n" % (np.max(silhouette_scores),(silhouette_scores.index(np.max(silhouette_scores))+2)),color = 'blue', attrs=['bold']))
        plt.figure(figsize=(16, 8))
        plt.plot(range(2, 10), silhouette_scores, "bo-")
        plt.xlabel("$k$", fontsize=14)
        plt.ylabel("Silhouette score", fontsize=14)
        plt.show()


    def davies_bouldin_score(self):
     
        """
        Validation test to check score after clustering

        """
        print(colored("The davies bouldin score of the clustering is %0.002f\n" %(davies_bouldin_score(self.X, self.labels)),color = 'red', attrs=['bold']))
        print()
        print(colored("The points in each cluster are : ",color = 'yellow', attrs=['bold']))
        print(collections.Counter(self.labels))


    def kmeans_clustering(self,k):

        """
        Performs k-means algorithm with given clusters 'k'

        Input  : The input to this algorithm is clusters
        Output : Output is clustering labels
        """
        
        print(colored("Performing K-means clustering with %d clusters\n"%k,color = 'yellow', attrs=['bold']))
        kmeans = KMeans(n_clusters=k, random_state=0, n_init=10, max_iter=100, n_jobs=-1, ).fit(self.X)
        self.labels = kmeans.labels_
        self.davies_bouldin_score()
        print()
        print(colored("The k-means inertia is %0.002f\n" %(kmeans.inertia_),color = 'red', attrs=['bold']))
        self.cluster_plot()
        return self.labels , kmeans.cluster_centers_,kmeans


    def hierarchial_clustering(self,k):

        """
        Performs hierarchial clustering with given clusters'k'

        Input  : The input to this algorithm are clusters
        Output : Output is clustering labels
        """

        print(colored("Performing hierarchial clustering",color = 'yellow', attrs=['bold']))
        self.clustering = AgglomerativeClustering(affinity='euclidean', linkage='ward').fit(self.X)
        self.labels = self.clustering.labels_
        self.davies_bouldin_score()
        print()
        print(colored("The number of cluster centers formed are %d\n" %(self.clustering.n_clusters_),color = 'red', attrs=['bold']))
        self.cluster_plot()
        return self.labels


    def DBscan_clustering(self,d,s):

        """
        Performs DBscan clustering with given distance 'd' and 'sample size 's'

        Input  : The input to this algorithm is clustering distance and samples
        Output : Output is clustering labels
        """
        print(colored("Performing agglomerative clustering",color = 'yellow', attrs=['bold']))
        self.clustering = DBSCAN(eps=d,min_samples=s,metric = 'euclidean').fit(self.X)
        self.labels = self.clustering.labels_
        self.davies_bouldin_score()
        print()
        print(colored("The number of cluster centers formed are %d\n"%len(np.unique(self.labels)),color = 'red', attrs=['bold']))
        self.cluster_plot()
        return self.labels