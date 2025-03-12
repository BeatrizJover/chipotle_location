import pandas as pd
import numpy as np
import geopandas as gpd
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from shapely.geometry import Point


class ClusterAnalyzer:
    """Class to handle clustering analysis for geographic data."""

    def __init__(self, data_path, geo_data_path):
        """Initialize with paths to data and geodata files."""
        self.df = pd.read_csv(data_path)
        self.geometry = [Point(xy) for xy in zip(self.df['longitude'], self.df['latitude'])]
        self.gdf = gpd.GeoDataFrame(self.df, geometry=self.geometry, crs="EPSG:4326")
        self.usa = gpd.read_file(geo_data_path)
        self.usa = self.usa[self.usa['ADMIN'] == 'United States of America']
        self.X = self.df[['latitude', 'longitude']].values

    def compute_elbow_method(self, k_range=range(1, 20)):
        """Compute inertia for different k values to determine optimal clusters."""
        inertias = []
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(self.X)
            inertias.append(kmeans.inertia_)
        return list(k_range), inertias

    def run_clustering_algorithms(self, optimal_k=10):
        """Run and compare multiple clustering algorithms."""
        clustering_results = {}

        # K-means
        kmeans = KMeans(n_clusters=optimal_k, random_state=42)
        kmeans_labels = kmeans.fit_predict(self.X)
        kmeans_silhouette = silhouette_score(self.X, kmeans_labels)
        clustering_results['K-means'] = {'labels': kmeans_labels, 'score': kmeans_silhouette}
        print(f"K-means Silhouette Score: {kmeans_silhouette:.3f}")

        # DBSCAN
        gdf_utm = self.gdf.to_crs(epsg=32610)
        X_km = np.array([[point.x, point.y] for point in gdf_utm.geometry]) / 1000
        best_silhouette, best_labels = -1, None
        best_eps, best_min_samples = None, None

        for eps_km, min_samples in [(15, 5), (25, 5), (50, 5), (100, 10)]:
            dbscan = DBSCAN(eps=eps_km, min_samples=min_samples, metric='euclidean')
            dbscan_labels = dbscan.fit_predict(X_km)
            n_clusters = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)

            if n_clusters > 1:
                dbscan_silhouette = silhouette_score(X_km, dbscan_labels)
                print(f"DBSCAN (eps={eps_km} km, min_samples={min_samples}): {dbscan_silhouette:.3f}, Clusters: {n_clusters}")
                if dbscan_silhouette > best_silhouette:
                    best_silhouette = dbscan_silhouette
                    best_labels = dbscan_labels
                    best_eps = eps_km
                    best_min_samples = min_samples
            else:
                print(f"DBSCAN (eps={eps_km} km, min_samples={min_samples}): No clusters or all noise")

        if best_silhouette > -1:
            clustering_results['DBSCAN'] = {'labels': best_labels, 'score': best_silhouette}
            print(f"Best DBSCAN Silhouette Score: {best_silhouette:.3f} (eps={best_eps}, min_samples={best_min_samples})")

        # Agglomerative Clustering
        agglo = AgglomerativeClustering(n_clusters=optimal_k, linkage='ward')
        agglo_labels = agglo.fit_predict(self.X)
        agglo_silhouette = silhouette_score(self.X, agglo_labels)
        clustering_results['Agglomerative'] = {'labels': agglo_labels, 'score': agglo_silhouette}
        print(f"Agglomerative Clustering Silhouette Score: {agglo_silhouette:.3f}")

        # Gaussian Mixture
        gmm = GaussianMixture(n_components=optimal_k, covariance_type='spherical', random_state=42)
        gmm_labels = gmm.fit_predict(self.X)
        gmm_silhouette = silhouette_score(self.X, gmm_labels)
        clustering_results['Gaussian Mixtures'] = {'labels': gmm_labels, 'score': gmm_silhouette}
        print(f"Gaussian Mixtures Silhouette Score: {gmm_silhouette:.3f}")

        # Determine best method
        best_method = max(clustering_results, key=lambda x: clustering_results[x]['score'])
        print(f"\nBest Clustering Method: {best_method} with Silhouette Score: {clustering_results[best_method]['score']:.3f}")
        return best_method, clustering_results[best_method]['labels']

    def analyze_clusters(self, labels):
        """Analyze clusters and compute centroids."""
        self.gdf['cluster'] = labels
        clusters = self.gdf[self.gdf['cluster'] != -1]
        noise = self.gdf[self.gdf['cluster'] == -1]

        centroids = clusters.groupby('cluster').agg({'latitude': 'mean', 'longitude': 'mean'}).reset_index()
        cluster_sizes = clusters['cluster'].value_counts()
        centroids['size'] = centroids['cluster'].map(cluster_sizes)

        best_centroid = centroids.loc[centroids['size'].idxmax()]
        best_cluster_id = best_centroid['cluster']

        print(f"Best Centroid: Lat {best_centroid['latitude']:.3f}, Lon {best_centroid['longitude']:.3f}")
        print(f"Cluster Size (Density): {best_centroid['size']}")

        centroid_geometry = [Point(xy) for xy in zip(centroids['longitude'], centroids['latitude'])]
        centroids_gdf = gpd.GeoDataFrame(centroids, geometry=centroid_geometry, crs="EPSG:4326")

        return clusters, noise, centroids_gdf, best_cluster_id