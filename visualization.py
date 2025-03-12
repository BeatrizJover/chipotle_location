import matplotlib.pyplot as plt


class Visualizer:
    """Class to handle visualization of geographic and clustering data."""

    @staticmethod
    def plot_locations(usa, gdf):
        """Plot raw Chipotle locations on a USA map."""
        fig, ax = plt.subplots(figsize=(12, 8))
        usa.plot(ax=ax, color='lightgrey', edgecolor='black')
        gdf.plot(ax=ax, color='red', markersize=5, label='Chipotle Locations')
        plt.title('Chipotle Locations in the USA')
        plt.legend()
        plt.show()

    @staticmethod
    def plot_elbow_method(k_range, inertias):
        """Plot the Elbow Method results."""
        plt.figure(figsize=(20, 10))
        plt.plot(k_range, inertias, marker='o')
        plt.title('Elbow Method for Optimal Number of Clusters')
        plt.xlabel('Number of Clusters (k)')
        plt.ylabel('Inertia (WCSS)')
        plt.grid(True)
        plt.show()

    @staticmethod
    def plot_clusters(usa, clusters, noise, centroids_gdf, best_cluster_id, best_method):
        """Plot clusters, noise, centroids, and the best centroid."""
        fig, ax = plt.subplots(figsize=(12, 8))
        usa.plot(ax=ax, color='lightgrey', edgecolor='black')

        if len(noise) > 0:
            noise.plot(ax=ax, color='grey', markersize=5, label='Noise', alpha=0.5)

        for cluster_id in clusters['cluster'].unique():
            cluster_data = clusters[clusters['cluster'] == cluster_id]
            cluster_data.plot(ax=ax, markersize=10, label=f'Cluster {cluster_id}', alpha=0.7)

        centroids_gdf.plot(ax=ax, color='red', marker='x', markersize=100, label='Centroids')
        best_centroid_gdf = centroids_gdf[centroids_gdf['cluster'] == best_cluster_id]
        best_centroid_gdf.plot(ax=ax, color='blue', marker='*', markersize=200, label='Best Centroid')

        plt.title(f'Chipotle Clusters using {best_method}')
        plt.legend()
        plt.show()

    @staticmethod
    def print_centroid_insights(best_cluster_id, best_centroid):
        """Print insights about the best centroid."""
        print("\nWhy This Centroid?")
        print(f"- Highest Density: Cluster {best_cluster_id} has {best_centroid['size']} locations.")
        print(f"- Proximity: Central to {best_centroid['size']} Chipotle stores.")
        print("- Connectivity: Ideal starting point for a road trip visiting dense Chipotle locations.")