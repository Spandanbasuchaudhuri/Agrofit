import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from joblib import dump, load
import os
import numpy as np


class AgroFit:
    def __init__(self, data_path, model_path=None):
        # Load the dataset
        self.data = pd.read_csv(data_path)

        # Manually encode categorical features
        self.soil_type_mapping = {soil: i for i, soil in enumerate(self.data['Soil Type'].unique())}
        self.planting_season_mapping = {season: i for i, season in enumerate(self.data['Planting Season'].unique())}
        self.harvesting_season_mapping = {season: i for i, season in enumerate(self.data['Harvesting Season'].unique())}

        self.data['Soil Type'] = self.data['Soil Type'].map(self.soil_type_mapping)
        self.data['Planting Season'] = self.data['Planting Season'].map(self.planting_season_mapping)
        self.data['Harvesting Season'] = self.data['Harvesting Season'].map(self.harvesting_season_mapping)

        # Select features
        self.features = self.data[['Soil Type', 'pH Level', 'Nitrogen Content (ppm)', 'Phosphorus Content (ppm)',
                                   'Potassium Content (ppm)', 'Rainfall (mm)', 'Temperature (°C)', 'Humidity (%)',
                                   'Sunlight Hours (per day)', 'Altitude (m)', 'Planting Season', 'Harvesting Season',
                                   'Growing Period (days)']].copy()

        # Scale the features
        self.scaler = StandardScaler()
        self.scaled_features = self.scaler.fit_transform(self.features)

        # Apply PCA for dimensionality reduction
        self.pca = PCA(n_components=8)  # Adjust number of components to retain enough variance
        self.pca_features = self.pca.fit_transform(self.scaled_features)

        # Check if model_path is provided and model exists
        if model_path and os.path.exists(model_path):
            # Load the KMeans model and mappings
            self.load_model(model_path)
            print(f"Loaded KMeans model from {model_path}")
        else:
            # Set number of clusters to 100
            k = 100
            self.kmeans = KMeans(n_clusters=k, random_state=0)
            self.kmeans.fit(self.pca_features)
            self.features['Cluster'] = self.kmeans.labels_
            # Assign clusters to data
            self.data['Cluster'] = self.features['Cluster']
            # Evaluation: Silhouette Score
            self.silhouette_avg = silhouette_score(self.pca_features, self.kmeans.labels_)
            print(f"\nSilhouette Score with {k} clusters: {self.silhouette_avg}")
            if model_path:
                self.save_model(model_path)
                print(f"KMeans model saved to {model_path}")

    def save_model(self, model_path):
        """Saves the trained KMeans model and encoders to the specified path."""
        model_data = {
            'kmeans': self.kmeans,
            'scaler': self.scaler,
            'pca': self.pca,
            'soil_type_mapping': self.soil_type_mapping,
            'planting_season_mapping': self.planting_season_mapping,
            'harvesting_season_mapping': self.harvesting_season_mapping,
            'data': self.data,  # Save data with clusters assigned
        }
        dump(model_data, model_path)

    def load_model(self, model_path):
        """Loads the KMeans model and encoders from the specified path."""
        model_data = load(model_path)
        self.kmeans = model_data['kmeans']
        self.scaler = model_data['scaler']
        self.pca = model_data['pca']
        self.soil_type_mapping = model_data['soil_type_mapping']
        self.planting_season_mapping = model_data['planting_season_mapping']
        self.harvesting_season_mapping = model_data['harvesting_season_mapping']
        self.data = model_data['data']

    def recommend_conditions(self, subtype, variety):
        # Find all data entries for the specific subtype and variety
        crop_data = self.data[(self.data['Subtype'] == subtype) & (self.data['Varieties'] == variety)]
        if crop_data.empty:
            return "No data available for the specified subtype and variety."

        # Get the most common cluster for this crop subtype and variety
        cluster_label = crop_data['Cluster'].mode()[0]

        # Get all data points in that cluster
        cluster_data = self.data[self.data['Cluster'] == cluster_label]

        # Calculate average conditions
        avg_conditions = cluster_data.mean(numeric_only=True)

        # Get modes for categorical features
        soil_type_mode = int(cluster_data['Soil Type'].mode()[0])
        planting_season_mode = int(cluster_data['Planting Season'].mode()[0])
        harvesting_season_mode = int(cluster_data['Harvesting Season'].mode()[0])

        # Map back to original categories
        soil_type = [key for key, value in self.soil_type_mapping.items() if value == soil_type_mode][0]
        planting_season = [key for key, value in self.planting_season_mapping.items() if value == planting_season_mode][
            0]
        harvesting_season = \
        [key for key, value in self.harvesting_season_mapping.items() if value == harvesting_season_mode][0]

        # Prepare recommended conditions
        recommended_conditions = avg_conditions.to_dict()
        recommended_conditions.update({
            'Soil Type': soil_type,
            'Planting Season': planting_season,
            'Harvesting Season': harvesting_season
        })

        return recommended_conditions