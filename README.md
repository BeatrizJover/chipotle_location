# Finding the Best Location for Your Ideal Chipotle Lifestyle

In this project, the goal is to find the optimal location(s) in the USA to live for enjoying the ideal Chipotle lifestyle by clustering Chipotle restaurant locations.  It uses multiple clustering algorithms (K-means, DBSCAN, Agglomerative Clustering, and Gaussian Mixture Models) to group stores based on their latitude and longitude, evaluates their performance, and visualizes the results on a map. Then select the best centroid based on specific criteria.

## Table of Contents

1. [Project Structure](#project-structure)

2. [Description](#description)

3. [Installation](#installation)

4. [Usage](#usage)

5. [Data Source](#data-source)

6. [Contributors](#contributors)

## Project Structure

```bash
chipotle-clustering/
├── data/
│   ├── chipotle_stores.csv           
│   └── geodata/
│       └── U.S. shapefile
├── visualization/
│   ├── chiplote_locations_USA.png        
│   ├── elbow_method.png            
│   └── best_location_with_K-means.png
├── clustering_analysis.ipynb
├── clustering_methods.py  
├── main.py                
├── visualization.py                 
├── requirements.txt                 
└── README.md    
```

## Description

`main.py`: Orchestrates the workflow, calling methods from other modules.

`clustering_methods.py`: Contains the ClusterAnalyzer class for data loading, clustering, and analysis.

`visualization.py`: Contains the Visualizer class for generating plots and insights.

`clustering_analysis.ipynb`: Jupyter notebook detailing the analysis process.

`visualization/`: Stores PNG outputs of visualizations.

## Installation

Execute the `requirements.txt` to install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

Run the analysis from the command line:

```bash
python main.py
```

### Outputs

- Initial Map: Displays all Chipotle locations in the U.S.

- Elbow Plot: Shows inertia vs. number of clusters to justify k=10.

- Clustering Results: Prints silhouette scores for each algorithm and identifies the best method.

- Cluster Map: Visualizes clusters, noise (if any), centroids, and the densest cluster’s centroid.

- Centroid Insights: Explains why the densest centroid is significant.

## Data Source

The Chipotle store location data (chipotle_stores.csv) is sourced from Kaggle:
[Chipotle Locations Dataset by Jeffrey Braun](https://www.kaggle.com/datasets/jeffreybraun/chipotle-locations)

The U.S. shapefile (ne_110m_admin_0_countries.shp) is available from [Natural Earth](https://www.naturalearthdata.com/downloads/110m-cultural-vectors/).

## Contributors

[BeatrizJover](https://github.com/BeatrizJover)