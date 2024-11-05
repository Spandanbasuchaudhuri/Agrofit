import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score
from joblib import dump, load
import os


class YieldPredictor:
    def __init__(self, data_path, model_path=None):
        # Load the dataset
        self.data = pd.read_csv(data_path)
        self.label_encoders = {}

        # Prepare the data
        self._encode_categorical_variables()

        # Define features and target variable
        X = self.data.drop("Yield (kg/ha)", axis=1)
        y = self.data["Yield (kg/ha)"]

        # Split the dataset into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Check if model_path is provided and model exists
        if model_path and os.path.exists(model_path):
            # Load the Random Forest Regressor model
            self.load_model(model_path)
            print(f"Loaded Yield Predictor model from {model_path}")
        else:
            # Initialize and train the Random Forest Regressor with specified parameters
            self.model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
            self.model.fit(X_train, y_train)

            # Evaluate the model (metrics are calculated but not printed)
            y_pred = self.model.predict(X_test)
            self.mse = mean_squared_error(y_test, y_pred)
            self.r2 = r2_score(y_test, y_pred)
            self.accuracy_percentage = self.r2 * 100

            # Commented out the print statements to suppress accuracy output
            # print("Mean Squared Error (MSE):", self.mse)
            # print("R^2 Score:", self.r2)
            # print("Accuracy (%):", self.accuracy_percentage)

            if model_path:
                self.save_model(model_path)
                print(f"Yield Predictor model saved to {model_path}")

    def _encode_categorical_variables(self):
        """Encodes categorical variables using Label Encoding."""
        categorical_columns = ["Soil Type", "Planting Season", "Harvesting Season", "Subtype", "Crop Type", "Varieties"]
        for col in categorical_columns:
            le = LabelEncoder()
            self.data[col] = le.fit_transform(self.data[col])
            self.label_encoders[col] = le

    def save_model(self, model_path):
        """Saves the trained model and encoders to the specified path."""
        model_data = {
            'model': self.model,
            'label_encoders': self.label_encoders
        }
        dump(model_data, model_path)

    def load_model(self, model_path):
        """Loads the trained model and encoders from the specified path."""
        model_data = load(model_path)
        self.model = model_data['model']
        self.label_encoders = model_data['label_encoders']

    def predict_yield(self, user_input):
        """Predicts the yield based on user input."""
        # Encode categorical variables
        for col, le in self.label_encoders.items():
            if col in user_input:
                user_input[col] = le.transform([user_input[col]])[0]
            else:
                print(f"Missing value for {col}")
                return None

        # Convert input into a DataFrame
        user_df = pd.DataFrame([user_input])

        # Ensure all columns are present
        missing_cols = set(self.data.drop("Yield (kg/ha)", axis=1).columns) - set(user_df.columns)
        if missing_cols:
            print(f"Missing columns in input: {missing_cols}")
            return None

        # Reorder columns to match training data
        user_df = user_df[self.data.drop("Yield (kg/ha)", axis=1).columns]

        # Predict yield
        predicted_yield = self.model.predict(user_df)[0]
        return predicted_yield
