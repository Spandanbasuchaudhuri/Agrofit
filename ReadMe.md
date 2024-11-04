# AgriChatbot: Intelligent Agricultural Assistant

**AgriChatbot** is an intelligent chatbot designed to assist farmers and agricultural professionals by providing crop subtype recommendations, ideal growing conditions, yield predictions, and strategies to adjust various agricultural parameters. Leveraging machine learning models and natural language processing, AgriChatbot offers both text and voice interactions to enhance user experience and support informed decision-making in agriculture.

## Table of Contents

- [Features](#features)
- [Directory Structure](#directory-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Modules](#modules)
  - [main.py](#mainpy)
  - [cropmatch.py](#cropmatchpy)
  - [agrofit.py](#agrofitpy)
  - [yield_predictor.py](#yield_predictorpy)
  - [transcriber.py](#transcriberpy)
  - [kartik_chatbot.py](#kartik_chatbotpy)
- [Models](#models)
- [Dependencies](#dependencies)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Features

- **Crop Subtype Recommendation**: Suggests the top 3 crop subtypes based on user-provided agricultural data.
- **Ideal Conditions Provision**: Offers recommended growing conditions for specific crop subtypes and varieties.
- **Yield Prediction**: Estimates crop yield based on input parameters.
- **Adjustment Strategies**: Provides strategies to adjust various agricultural parameters such as pH level, nutrient content, rainfall, temperature, and more.
- **Text and Voice Interaction**: Users can interact with the chatbot via text input or voice commands.
- **Model Information**: Displays information about the underlying machine learning models used.

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

- **main.py**: The entry point of the application. Initializes models and handles user interactions.
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

- **Python 3.7 or higher**: Ensure you have Python installed. You can download it from [Python's official website](https://www.python.org/downloads/).

### Clone the Repository

```bash
git clone https://github.com/yourusername/agri_chatbot.git
cd agri_chatbot
```

### Create a Virtual Environment (Optional but Recommended)

Using `venv`:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Install Dependencies

Ensure you have `pip` installed. Then, install the required Python packages:

```bash
pip install -r requirements.txt
```

**Note**: Some packages like `torch` may require specific installation commands based on your system and whether you want GPU support. Refer to [PyTorch's official installation guide](https://pytorch.org/get-started/locally/) for detailed instructions.

## Usage

1. **Prepare Your Dataset**

   - Ensure your dataset (`enlarged_agriculture_dataset.csv`) is placed in a known directory.
   - Update the `data_path` variable in `main.py` with the correct path to your dataset.

2. **Run the Application**

   ```bash
   python main.py
   ```

3. **Interact with the Chatbot**

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
- Data preprocessing and encoding.
- Model training and saving/loading.
- Predicting top 3 crop subtypes with probabilities.

### agrofit.py

**Description**: Contains the `AgroFit` class, which uses KMeans clustering to provide ideal agricultural conditions for specific crop subtypes and varieties.

**Key Functions**:
- Data preprocessing and encoding.
- Clustering similar agricultural conditions.
- Recommending average conditions based on crop subtype and variety.

### yield_predictor.py

**Description**: Contains the `YieldPredictor` class, which employs a Random Forest Regressor to predict crop yields based on input parameters.

**Key Functions**:
- Data preprocessing and encoding.
- Model training and saving/loading.
- Predicting yield based on user input.

### transcriber.py

**Description**: Contains the `Transcriber` class, which handles voice-to-text transcription using the `speech_recognition` library.

**Key Functions**:
- Capturing audio from the microphone.
- Transcribing speech to text using Google's Speech Recognition API.

### kartik_chatbot.py

**Description**: Contains the `KartikChatbot` class, which manages user interactions, intent recognition, and integrates functionalities from all other modules. It leverages the `transformers` library's DialoGPT model for conversational responses and `pyttsx3` for text-to-speech.

**Key Functions**:
- Starting and ending conversation sessions.
- Handling text and voice-based chats.
- Recognizing user intents and routing to appropriate functions.
- Providing recommendations, yield predictions, and adjustment strategies.
- Converting text responses to speech.

## Models

All trained machine learning models are stored in the `models/` directory:

- `crop_match_model.pkl`: Trained Random Forest model for crop subtype recommendations.
- `agrofit_model.pkl`: Trained KMeans model for clustering agricultural conditions.
- `yield_predictor_model.pkl`: Trained Random Forest Regressor model for yield prediction.

**Note**: Ensure that the `models/` directory exists in the project root. The application will automatically train and save models if they do not exist.

## Dependencies

All necessary Python packages are listed in `requirements.txt`. Key dependencies include:

- **pandas**: Data manipulation and analysis.
- **scikit-learn**: Machine learning algorithms and preprocessing tools.
- **joblib**: Model serialization.
- **speechrecognition**: Voice recognition functionality.
- **pyttsx3**: Text-to-speech conversion.
- **transformers**: Pre-trained language models for conversational AI.
- **torch**: PyTorch library for deep learning models.

To install all dependencies, run:

```bash
pip install -r requirements.txt
```

**Note**: Some packages, especially `torch`, may have specific installation instructions based on your system and whether you require GPU support.

## Contributing

Contributions are welcome! If you have suggestions for improvements or encounter any issues, feel free to open an issue or submit a pull request.

### Steps to Contribute

1. **Fork the Repository**

2. **Create a Feature Branch**

   ```bash
   git checkout -b feature/YourFeatureName
   ```

3. **Commit Your Changes**

   ```bash
   git commit -m "Add some feature"
   ```

4. **Push to the Branch**

   ```bash
   git push origin feature/YourFeatureName
   ```

5. **Open a Pull Request**

## License

This project is licensed under the [MIT License](LICENSE).

## Acknowledgements

- **[Transformers by Hugging Face](https://github.com/huggingface/transformers)**: For providing state-of-the-art natural language processing models.
- **[PyTorch](https://pytorch.org/)**: For the deep learning framework used in model training.
- **[SpeechRecognition](https://pypi.org/project/SpeechRecognition/)**: For enabling voice-to-text functionality.
- **[pyttsx3](https://pyttsx3.readthedocs.io/en/latest/)**: For text-to-speech conversion.
- **[Scikit-learn](https://scikit-learn.org/stable/)**: For machine learning algorithms and tools.

---

Feel free to reach out for any questions or support regarding AgriChatbot!
