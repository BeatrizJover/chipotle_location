from clustering_methods import ClusterAnalyzer
from visualization import Visualizer


def main():
    # Initialize data paths
    data_path = 'data/chipotle_stores.csv'
    geo_data_path = 'data/geodata/ne_110m_admin_0_countries.shp'

    # Create analyzer and visualizer instances
    analyzer = ClusterAnalyzer(data_path, geo_data_path)
    visualizer = Visualizer()

    # Step 1: Visualize raw locations
    visualizer.plot_locations(analyzer.usa, analyzer.gdf)

    # Step 2: Compute and visualize Elbow Method
    k_range, inertias = analyzer.compute_elbow_method()
    visualizer.plot_elbow_method(k_range, inertias)

    # Step 3: Run clustering algorithms and get best results
    optimal_k = 10  # Could be dynamically set from elbow method in a more advanced version
    best_method, best_labels = analyzer.run_clustering_algorithms(optimal_k)

    # Step 4: Analyze clusters and centroids
    clusters, noise, centroids_gdf, best_cluster_id = analyzer.analyze_clusters(best_labels)

    # Step 5: Visualize clusters and print insights
    visualizer.plot_clusters(analyzer.usa, clusters, noise, centroids_gdf, best_cluster_id, best_method)
    best_centroid = centroids_gdf.loc[centroids_gdf['cluster'] == best_cluster_id].iloc[0]
    visualizer.print_centroid_insights(best_cluster_id, best_centroid)


if __name__ == "__main__":
    main()