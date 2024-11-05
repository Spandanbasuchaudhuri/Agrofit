import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from joblib import load, dump
import os


class CropMatch:
    def __init__(self, data_path, model_path=None):
        # Load the dataset and preprocess it
        self.data = pd.read_csv(data_path)
        self.label_encoders = {}
        self.scaler = StandardScaler()

        # Prepare data by removing 'Varieties' and separating target
        X = self.data.drop(columns=['Subtype', 'Varieties'])
        y = self.data['Subtype']

        # Identify categorical and numerical features
        self.categorical_features = X.select_dtypes(include=['object']).columns
        self.numerical_features = X.select_dtypes(include=['float64', 'int']).columns

        # Encode features
        X_encoded = self._encode_features(X)

        # Train-test split and model training
        X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

        # Check if model_path is provided and model exists
        if model_path and os.path.exists(model_path):
            # Load the Random Forest model and encoders
            self.load_model(model_path)
            print(f"Loaded Random Forest model from {model_path}")
        else:
            self.model = RandomForestClassifier(n_estimators=20, max_depth=4, random_state=42)
            self.model.fit(X_train, y_train)
            if model_path:
                self.save_model(model_path)
                print(f"Random Forest model saved to {model_path}")

        # Initialize label encoder for 'Subtype' if not already
        if 'Subtype' not in self.label_encoders:
            self.label_encoders['Subtype'] = LabelEncoder()
            self.label_encoders['Subtype'].fit(self.data['Subtype'])

    def _encode_features(self, X):
        X_encoded = X.copy()
        for col in self.categorical_features:
            self.label_encoders[col] = LabelEncoder()
            X_encoded[col] = self.label_encoders[col].fit_transform(X[col])
        X_encoded[self.numerical_features] = self.scaler.fit_transform(X[self.numerical_features])
        return X_encoded

    def save_model(self, model_path):
        """Saves the trained model and encoders to the specified path."""
        model_data = {
            'model': self.model,
            'label_encoders': self.label_encoders,
            'scaler': self.scaler,
            'categorical_features': self.categorical_features,
            'numerical_features': self.numerical_features,
            'data_columns': self.data.drop(columns=['Subtype', 'Varieties']).columns
        }
        dump(model_data, model_path)

    def load_model(self, model_path):
        """Loads the trained model and encoders from the specified path."""
        model_data = load(model_path)
        self.model = model_data['model']
        self.label_encoders = model_data['label_encoders']
        self.scaler = model_data['scaler']
        self.categorical_features = model_data['categorical_features']
        self.numerical_features = model_data['numerical_features']
        self.data_columns = model_data['data_columns']

    def predict_subtype(self, user_input):
        """Predicts the subtype based on user input."""
        # Encode categorical features
        for col in self.categorical_features:
            user_input[col] = self.label_encoders[col].transform([user_input[col]])[0]

        # Convert input into a DataFrame and scale numerical features
        user_df = pd.DataFrame([user_input])
        user_df[self.numerical_features] = self.scaler.transform(user_df[self.numerical_features])

        # Ensure user_df columns match the order of training data columns
        user_df = user_df[self.data_columns]

        # Predict probabilities for the top 3 subtypes
        probabilities = self.model.predict_proba(user_df)
        top_3_indices = np.argsort(probabilities[0])[::-1][:3]
        top_3_subtypes = self.label_encoders['Subtype'].inverse_transform(top_3_indices)
        top_3_probs = probabilities[0][top_3_indices]

        # Return the top 3 predictions
        return {top_3_subtypes[i]: top_3_probs[i] for i in range(3)}