# agrofit.py

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from joblib import dump, load
import os


class AgroFit:
    def __init__(self, data_path, model_path=None):
        # Load the dataset
        self.data = pd.read_csv(data_path)

        # Manually encode categorical features to ensure all categories are accounted for
        self.soil_type_mapping = {soil: i for i, soil in enumerate(self.data['Soil Type'].unique())}
        self.planting_season_mapping = {season: i for i, season in enumerate(self.data['Planting Season'].unique())}
        self.harvesting_season_mapping = {season: i for i, season in enumerate(self.data['Harvesting Season'].unique())}

        # Apply these mappings directly to the DataFrame
        self.data['Soil Type'] = self.data['Soil Type'].map(self.soil_type_mapping)
        self.data['Planting Season'] = self.data['Planting Season'].map(self.planting_season_mapping)
        self.data['Harvesting Season'] = self.data['Harvesting Season'].map(self.harvesting_season_mapping)

        # Selecting features for clustering
        self.features = self.data[['Soil Type', 'pH Level', 'Nitrogen Content (ppm)', 'Phosphorus Content (ppm)',
                                   'Potassium Content (ppm)', 'Rainfall (mm)', 'Temperature (°C)', 'Humidity (%)',
                                   'Sunlight Hours (per day)', 'Altitude (m)', 'Planting Season', 'Harvesting Season',
                                   'Growing Period (days)']].copy()

        # Scaling the numerical features for clustering
        self.scaler = StandardScaler()
        self.scaled_features = self.scaler.fit_transform(self.features)

        # Check if model_path is provided and model exists
        if model_path and os.path.exists(model_path):
            # Load the KMeans model and mappings
            self.load_model(model_path)
            print(f"Loaded KMeans model from {model_path}")
        else:
            # Applying KMeans clustering to group similar conditions
            self.kmeans = KMeans(n_clusters=10, random_state=0)
            self.features['Cluster'] = self.kmeans.fit_predict(self.scaled_features)
            # Adding the clusters to the original data to facilitate recommendations
            self.data['Cluster'] = self.features['Cluster']
            # Evaluation: Silhouette Score for Clustering
            self.silhouette_avg = silhouette_score(self.scaled_features, self.features['Cluster'])
            print("\nSilhouette Score of the clustering:", self.silhouette_avg)
            if model_path:
                self.save_model(model_path)
                print(f"KMeans model saved to {model_path}")

    def save_model(self, model_path):
        """Saves the trained KMeans model and encoders to the specified path."""
        # Save the KMeans model and encoders as a dictionary
        model_data = {
            'kmeans': self.kmeans,
            'scaler': self.scaler,
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
        self.soil_type_mapping = model_data['soil_type_mapping']
        self.planting_season_mapping = model_data['planting_season_mapping']
        self.harvesting_season_mapping = model_data['harvesting_season_mapping']
        self.data = model_data['data']

    def recommend_conditions(self, subtype, variety):
        # Find all data entries for the specific subtype and variety
        crop_data = self.data[(self.data['Subtype'] == subtype) & (self.data['Varieties'] == variety)]

        # Check if any entries exist for the specified subtype and variety
        if crop_data.empty:
            return "No data available for the specified subtype and variety."

        # Get the most common cluster for this crop subtype and variety
        cluster_label = crop_data['Cluster'].mode()[0]

        # Calculate the average conditions within this cluster
        cluster_data = self.data[self.data['Cluster'] == cluster_label]
        avg_conditions = cluster_data.mean(numeric_only=True)

        # Convert encoded values back to their original categories
        soil_type = \
        [key for key, value in self.soil_type_mapping.items() if value == int(cluster_data['Soil Type'].mode()[0])][0]
        planting_season = [key for key, value in self.planting_season_mapping.items() if
                           value == int(cluster_data['Planting Season'].mode()[0])][0]
        harvesting_season = [key for key, value in self.harvesting_season_mapping.items() if
                             value == int(cluster_data['Harvesting Season'].mode()[0])][0]

        # Format the conditions for display
        recommended_conditions = {col: avg_conditions[col] for col in avg_conditions.index}
        recommended_conditions.update({
            'Soil Type': soil_type,
            'Planting Season': planting_season,
            'Harvesting Season': harvesting_season
        })

        return recommended_conditions
