# AgroFit: Intelligent Agricultural Assistant

**AgroFit** is a sophisticated, AI-driven chatbot designed to empower farmers, agronomists, and agricultural enthusiasts with actionable insights and data-driven recommendations. By leveraging machine learning models and natural language processing, AgroFit assists users in making informed decisions related to crop selection, optimal growing conditions, yield predictions, and strategic adjustments to various agricultural parameters. The chatbot supports both text and voice interactions, ensuring accessibility and ease of use across different user preferences.

## Table of Contents

1. [Features](#features)
2. [Architecture](#architecture)
3. [Directory Structure](#directory-structure)
4. [Installation](#installation)
5. [Configuration](#configuration)
6. [Usage](#usage)
7. [Modules](#modules)
    - [main.py](#mainpy)
    - [cropmatch.py](#cropmatchpy)
    - [agrofit.py](#agrofitpy)
    - [yield_predictor.py](#yield_predictorpy)
    - [transcriber.py](#transcriberpy)
    - [kartik_chatbot.py](#kartik_chatbotpy)
8. [Model Training](#model-training)
    - [CropMatch Model](#cropmatch-model)
    - [AgroFit Model](#agrofit-model)
    - [YieldPredictor Model](#yieldpredictor-model)
9. [Data Requirements](#data-requirements)
10. [Dependencies](#dependencies)
11. [Running the Application](#running-the-application)
12. [Troubleshooting](#troubleshooting)
13. [Contributing](#contributing)
14. [License](#license)
15. [Acknowledgements](#acknowledgements)
16. [Contact](#contact)

---

## Features

- **Crop Subtype Recommendation**: Suggests optimal crop subtypes based on user-provided agricultural data.
- **Ideal Conditions Provision**: Offers recommended growing conditions tailored to specific crop subtypes and varieties.
- **Yield Prediction**: Estimates expected crop yields using advanced regression models.
- **Adjustment Strategies**: Provides actionable strategies to adjust agricultural parameters such as pH levels, nutrient content, rainfall, temperature, and more.
- **Conversational Interface**: Engages users through intuitive text and voice-based interactions.
- **Model Insights**: Displays detailed information about the underlying machine learning models powering the chatbot.
- **Session Management**: Allows users to start, manage, and end conversation sessions seamlessly.

## Architecture

AgroFit is built with a modular architecture, ensuring scalability, maintainability, and ease of integration. The core components include:

- **Machine Learning Models**: Responsible for predictions and recommendations.
- **Natural Language Processing (NLP)**: Handles user interactions and intent recognition.
- **Voice Processing**: Facilitates voice-based interactions using speech recognition and text-to-speech technologies.
- **Data Processing**: Manages data ingestion, preprocessing, and feature engineering.

![Architecture Diagram](https://via.placeholder.com/800x400?text=Architecture+Diagram)

*Note: Replace the placeholder image with an actual architecture diagram for better visualization.*

## Directory Structure

```
agri_chatbot/
├── main.py
├── cropmatch.py
├── agrofit.py
├── yield_predictor.py
├── transcriber.py
├── kartik_chatbot.py
├── requirements.txt
├── README.md
└── models/
    ├── crop_match_model.pkl
    ├── agrofit_model.pkl
    └── yield_predictor_model.pkl
```

- **main.py**: Entry point of the application. Initializes models and handles user interactions.
- **cropmatch.py**: Contains the `CropMatch` class responsible for crop subtype recommendations.
- **agrofit.py**: Contains the `AgroFit` class that provides ideal agricultural conditions.
- **yield_predictor.py**: Contains the `YieldPredictor` class for predicting crop yields.
- **transcriber.py**: Contains the `Transcriber` class for voice-to-text functionality.
- **kartik_chatbot.py**: Contains the `KartikChatbot` class that manages user interactions and integrates all functionalities.
- **requirements.txt**: Lists all Python dependencies required for the project.
- **models/**: Directory to store trained machine learning model files (`.pkl` files).
- **README.md**: This documentation file.

## Installation

### Prerequisites

- **Python 3.7 or higher**: Ensure you have Python installed. Download it from [Python's official website](https://www.python.org/downloads/).
- **Git**: For cloning the repository. Download it from [Git's official website](https://git-scm.com/downloads).
- **Virtual Environment (Optional but Recommended)**: To manage project dependencies separately.

### Clone the Repository

```bash
git clone https://github.com/yourusername/agri_chatbot.git
cd agri_chatbot
```

### Set Up a Virtual Environment

Using `venv`:

```bash
python -m venv venv
```

Activate the virtual environment:

- **Windows**:

    ```bash
    venv\Scripts\activate
    ```

- **macOS/Linux**:

    ```bash
    source venv/bin/activate
    ```

### Install Dependencies

Ensure you have `pip` installed. Then, install the required Python packages:

```bash
pip install -r requirements.txt
```

**Note**: Some packages like `torch` may require specific installation commands based on your system and whether you want GPU support. Refer to [PyTorch's official installation guide](https://pytorch.org/get-started/locally/) for detailed instructions.

## Configuration

1. **Dataset Preparation**:

    - Ensure your dataset (`enlarged_agriculture_dataset.csv`) is placed in a known directory.
    - Update the `data_path` variable in `main.py` with the correct path to your dataset.

2. **Model Storage**:

    - The application expects a `models/` directory in the project root to store trained model files (`.pkl`).
    - If the `models/` directory does not exist, create it:

        ```bash
        mkdir models
        ```

3. **Microphone Setup**:

    - For voice interactions, ensure your system has a functional microphone.
    - Adjust microphone settings if necessary.

## Usage

1. **Run the Application**

    ```bash
    python main.py
    ```

2. **Interact with the Chatbot**

    Upon running, you'll be presented with a list of options:

    ```
    Options:
    [1] Start New Session
    [2] Talk to Kartik (Text Input)
    [3] Talk to Kartik (Voice Input)
    [4] Get Crop Subtype Recommendation
    [5] Get Recommended Conditions for Subtype and Variety
    [6] Predict Yield for Your Crop
    [7] End Session
    [8] Exit
    Type '99' to see the options again.
    ```

    - **Start New Session**: Initializes a new conversation session.
    - **Talk to Kartik**: Engage in a conversation either via text or voice.
    - **Get Crop Subtype Recommendation**: Receive crop subtype suggestions based on your input.
    - **Get Recommended Conditions**: Obtain ideal growing conditions for a specific crop subtype and variety.
    - **Predict Yield**: Estimate the yield for your crop based on provided parameters.
    - **End Session**: Ends the current conversation session.
    - **Exit**: Closes the application.
    - **Type '99'**: Redisplay the options menu.

3. **Voice Interaction**

    - Select option `[3]` to interact via voice.
    - Ensure your microphone is active and properly configured.
    - Speak clearly into the microphone when prompted.

## Modules

### main.py

**Description**: Serves as the entry point of the application. It initializes all necessary models (`CropMatch`, `AgroFit`, `YieldPredictor`, `Transcriber`) and the `KartikChatbot`. It also handles the user interface by presenting options and routing user choices to the appropriate functionalities.

**Key Functions**:
- Initializes and saves models.
- Presents interactive menu to the user.
- Handles user input and directs actions accordingly.

### cropmatch.py

**Description**: Contains the `CropMatch` class, which utilizes a Random Forest classifier to recommend crop subtypes based on user-provided agricultural data.

**Key Functions**:
- **Data Preprocessing and Encoding**: Handles encoding of categorical and scaling of numerical features.
- **Model Training and Persistence**: Trains the model if not already trained and handles saving/loading of the model.
- **Prediction**: Predicts the top 3 crop subtypes with associated probabilities based on user input.

**Usage Example**:

```python
from cropmatch import CropMatch

crop_match = CropMatch(data_path='path/to/dataset.csv', model_path='models/crop_match_model.pkl')
user_input = {
    'Soil Type': 'Loamy',
    'pH Level': 6.5,
    'Nitrogen Content (ppm)': 30,
    'Phosphorus Content (ppm)': 20,
    'Potassium Content (ppm)': 25,
    # Add other required features
}
recommendations = crop_match.predict_subtype(user_input)
print(recommendations)
```

### agrofit.py

**Description**: Contains the `AgroFit` class, which uses KMeans clustering to provide ideal agricultural conditions for specific crop subtypes and varieties.

**Key Functions**:
- **Data Preprocessing and Encoding**: Manually encodes categorical features to ensure consistency.
- **Clustering**: Applies KMeans clustering to group similar agricultural conditions.
- **Condition Recommendation**: Recommends average conditions based on crop subtype and variety.

**Usage Example**:

```python
from agrofit import AgroFit

agrofit = AgroFit(data_path='path/to/dataset.csv', model_path='models/agrofit_model.pkl')
recommendation = agrofit.recommend_conditions(subtype='Wheat', variety='VarietyA')
print(recommendation)
```

### yield_predictor.py

**Description**: Contains the `YieldPredictor` class, which employs a Random Forest Regressor to predict crop yields based on input parameters.

**Key Functions**:
- **Data Preprocessing and Encoding**: Handles encoding of categorical variables.
- **Model Training and Persistence**: Trains the model if not already trained and handles saving/loading of the model.
- **Yield Prediction**: Predicts yield based on user input.

**Usage Example**:

```python
from yield_predictor import YieldPredictor

yield_predictor = YieldPredictor(data_path='path/to/dataset.csv', model_path='models/yield_predictor_model.pkl')
user_input = {
    'Soil Type': 'Sandy',
    'pH Level': 7.0,
    'Nitrogen Content (ppm)': 25,
    'Phosphorus Content (ppm)': 15,
    'Potassium Content (ppm)': 20,
    # Add other required features
}
predicted_yield = yield_predictor.predict_yield(user_input)
print(f"Predicted Yield: {predicted_yield} kg/ha")
```

### transcriber.py

**Description**: Contains the `Transcriber` class, which handles voice-to-text transcription using the `speech_recognition` library.

**Key Functions**:
- **Listening**: Captures audio input from the microphone.
- **Transcription**: Converts captured audio into text using Google's Speech Recognition API.

**Usage Example**:

```python
from transcriber import Transcriber

transcriber = Transcriber()
user_speech = transcriber.listen()
print(f"You said: {user_speech}")
```

### kartik_chatbot.py

**Description**: Contains the `KartikChatbot` class, which manages user interactions, intent recognition, and integrates functionalities from all other modules. It leverages the `transformers` library's DialoGPT model for conversational responses and `pyttsx3` for text-to-speech.

**Key Functions**:
- **Session Management**: Starts and ends conversation sessions.
- **Chat Handling**: Manages both text and voice-based chats.
- **Intent Recognition**: Identifies user intents to route requests appropriately.
- **Recommendation & Prediction**: Interfaces with `CropMatch`, `AgroFit`, and `YieldPredictor` for providing insights.
- **Adjustment Strategies**: Offers strategies to adjust agricultural parameters.
- **Text-to-Speech**: Converts chatbot responses into audible speech.

**Usage Example**:

```python
from cropmatch import CropMatch
from agrofit import AgroFit
from yield_predictor import YieldPredictor
from transcriber import Transcriber
from kartik_chatbot import KartikChatbot

# Initialize models
crop_match = CropMatch(data_path='path/to/dataset.csv', model_path='models/crop_match_model.pkl')
agrofit = AgroFit(data_path='path/to/dataset.csv', model_path='models/agrofit_model.pkl')
yield_predictor = YieldPredictor(data_path='path/to/dataset.csv', model_path='models/yield_predictor_model.pkl')
transcriber = Transcriber()

# Initialize chatbot
chatbot = KartikChatbot(crop_match, agrofit, transcriber, yield_predictor)

# Start chatting
chatbot.chat()
```

## Model Training

### CropMatch Model

**Algorithm**: Random Forest Classifier

**Purpose**: Recommends crop subtypes based on user-provided agricultural data.

**Training Process**:
1. **Data Preprocessing**:
    - Encodes categorical variables using `LabelEncoder`.
    - Scales numerical features using `StandardScaler`.
2. **Model Training**:
    - Splits data into training and testing sets (80-20 split).
    - Trains a Random Forest Classifier with 20 trees and a maximum depth of 4.
3. **Evaluation**:
    - Assesses model performance using accuracy, precision, recall, and F1-score.
4. **Persistence**:
    - Saves the trained model and encoders using `joblib` for future use.

**Customization**:
- Adjust hyperparameters like `n_estimators` and `max_depth` in `cropmatch.py` to optimize performance.

### AgroFit Model

**Algorithm**: KMeans Clustering

**Purpose**: Provides ideal agricultural conditions by clustering similar environmental factors.

**Training Process**:
1. **Data Preprocessing**:
    - Manually encodes categorical features to ensure consistency.
    - Scales numerical features using `StandardScaler`.
2. **Clustering**:
    - Applies KMeans clustering with 10 clusters.
    - Evaluates clustering performance using Silhouette Score.
3. **Persistence**:
    - Saves the trained KMeans model and encoders using `joblib` for future use.

**Customization**:
- Adjust the number of clusters (`n_clusters`) in `agrofit.py` based on dataset diversity.

### YieldPredictor Model

**Algorithm**: Random Forest Regressor

**Purpose**: Predicts crop yields based on various input parameters.

**Training Process**:
1. **Data Preprocessing**:
    - Encodes categorical variables using `LabelEncoder`.
    - Handles missing values and outliers as needed.
2. **Model Training**:
    - Splits data into training and testing sets (80-20 split).
    - Trains a Random Forest Regressor with 100 trees and a maximum depth of 10.
3. **Evaluation**:
    - Assesses model performance using Mean Squared Error (MSE) and R² Score.
4. **Persistence**:
    - Saves the trained model and encoders using `joblib` for future use.

**Customization**:
- Modify hyperparameters like `n_estimators` and `max_depth` in `yield_predictor.py` to enhance prediction accuracy.

## Data Requirements

- **Dataset**: `enlarged_agriculture_dataset.csv`
- **Format**: CSV file containing agricultural data with relevant features.
- **Essential Columns**:
    - `Subtype`
    - `Varieties`
    - `Soil Type`
    - `pH Level`
    - `Nitrogen Content (ppm)`
    - `Phosphorus Content (ppm)`
    - `Potassium Content (ppm)`
    - `Rainfall (mm)`
    - `Temperature (°C)`
    - `Humidity (%)`
    - `Sunlight Hours (per day)`
    - `Altitude (m)`
    - `Planting Season`
    - `Harvesting Season`
    - `Growing Period (days)`
    - `Yield (kg/ha)` (for yield prediction)

**Data Quality**:
- Ensure no missing values in essential columns.
- Handle outliers and inconsistencies during preprocessing.
- Maintain consistency in categorical variables (e.g., naming conventions).

## Dependencies

All necessary Python packages are listed in `requirements.txt`. Key dependencies include:

- **pandas**: Data manipulation and analysis.
- **scikit-learn**: Machine learning algorithms and preprocessing tools.
- **joblib**: Model serialization.
- **speechrecognition**: Voice recognition functionality.
- **pyttsx3**: Text-to-speech conversion.
- **transformers**: Pre-trained language models for conversational AI.
- **torch**: PyTorch library for deep learning models.
- **numpy**: Numerical operations.
- **re**: Regular expressions for intent recognition.

### Installing Dependencies

To install all dependencies, run:

```bash
pip install -r requirements.txt
```

**Note**: Some packages, especially `torch`, may have specific installation instructions based on your system and whether you require GPU support.

## Running the Application

1. **Ensure Models are Trained and Saved**

    - On the first run, the application will train the necessary models if `.pkl` files are not found in the `models/` directory.
    - Training may take some time depending on dataset size and system performance.

2. **Start the Chatbot**

    ```bash
    python main.py
    ```

3. **Follow On-Screen Instructions**

    - Choose from the presented options to interact with the chatbot.
    - For voice interactions, ensure your microphone is active and properly configured.

## Troubleshooting

- **Microphone Issues**:
    - Ensure your microphone is connected and not muted.
    - Check system settings to verify the correct microphone is selected.
    - Install necessary drivers or troubleshoot hardware issues if the microphone isn't recognized.

- **Speech Recognition Errors**:
    - Verify internet connectivity as Google's Speech Recognition API requires it.
    - If transcription fails, try speaking clearly and avoiding background noise.
    - Handle API rate limits by implementing retries or upgrading your API usage plan.

- **Model Loading Errors**:
    - Ensure that the `.pkl` files exist in the `models/` directory.
    - Verify that the `data_path` in `main.py` points to the correct dataset.
    - Check for compatibility issues between `joblib` versions used during saving and loading.

- **Dependency Conflicts**:
    - Use a virtual environment to isolate project dependencies.
    - Ensure all packages are up-to-date or match the versions specified in `requirements.txt`.

- **Performance Issues**:
    - For large datasets, consider optimizing data preprocessing steps.
    - Utilize GPU acceleration by installing CUDA-enabled versions of `torch` if available.

## Contributing

Contributions are highly encouraged! Whether it's reporting bugs, suggesting features, or improving documentation, your input is valuable.

### Steps to Contribute

1. **Fork the Repository**

    Click the "Fork" button on the repository's GitHub page to create a personal copy.

2. **Clone Your Fork**

    ```bash
    git clone https://github.com/yourusername/agri_chatbot.git
    cd agri_chatbot
    ```

3. **Create a Feature Branch**

    ```bash
    git checkout -b feature/YourFeatureName
    ```

4. **Make Your Changes**

    Implement your feature or fix in the appropriate module.

5. **Commit Your Changes**

    ```bash
    git commit -m "Add some feature"
    ```

6. **Push to Your Fork**

    ```bash
    git push origin feature/YourFeatureName
    ```

7. **Open a Pull Request**

    Navigate to the original repository and open a pull request describing your changes.

### Guidelines

- **Code Quality**: Ensure your code adheres to PEP 8 standards for Python.
- **Documentation**: Update or add documentation as necessary.
- **Testing**: Include unit tests for new features or bug fixes.
- **Commit Messages**: Write clear and descriptive commit messages.

## License

This project is licensed under the [MIT License](LICENSE). You are free to use, modify, and distribute the software as per the license terms.

## Acknowledgements

- **[Transformers by Hugging Face](https://github.com/huggingface/transformers)**: For providing state-of-the-art natural language processing models.
- **[PyTorch](https://pytorch.org/)**: For the deep learning framework used in model training.
- **[SpeechRecognition](https://pypi.org/project/SpeechRecognition/)**: For enabling voice-to-text functionality.
- **[pyttsx3](https://pyttsx3.readthedocs.io/en/latest/)**: For text-to-speech conversion.
- **[Scikit-learn](https://scikit-learn.org/stable/)**: For machine learning algorithms and tools.
- **[Joblib](https://joblib.readthedocs.io/en/latest/)**: For efficient model serialization.
- **[GitHub](https://github.com/)**: For hosting the repository and facilitating collaboration.

## Contact

For any questions, suggestions, or support, please contact:

- **Name**: [Spandan Basu Chaudhuri]
- **Email**: [Spandanbasu139@gmail.com]
- **GitHub**: [Spandanbasuchaudhuri](https://github.com/Spandanbasuchaudhuri)
- **LinkedIn**: [Your LinkedIn Profile](https://www.linkedin.com/in/spandan-basu-chaudhuri-2327b324b/)

---

*This README was generated to provide comprehensive guidance on setting up, using, and contributing to AgroFit. For further assistance, refer to the documentation within each module or reach out through the contact channels provided.*
