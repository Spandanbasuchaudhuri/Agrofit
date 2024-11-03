# **Kartik: An Interactive Agricultural Chatbot**

## **Table of Contents**

- [Project Overview](#project-overview)
- [Features](#features)
- [Architecture](#architecture)
- [Installation Instructions](#installation-instructions)
  - [Prerequisites](#prerequisites)
  - [Step-by-Step Installation](#step-by-step-installation)
- [Usage Instructions](#usage-instructions)
  - [Running the Application](#running-the-application)
  - [Interacting with Kartik](#interacting-with-kartik)
  - [Options Explained](#options-explained)
- [Dataset Preparation](#dataset-preparation)
- [Project Structure](#project-structure)
- [Code Explanation](#code-explanation)
  - [`cropmatch.py`](#1-cropmatchpy)
  - [`agrofit.py`](#2-agrofitpy)
  - [`transcriber.py`](#3-transcriberpy)
  - [`kartik_chatbot.py`](#4-kartik_chatbotpy)
  - [`main.py`](#5-mainpy)
- [Dependencies](#dependencies)
- [Troubleshooting](#troubleshooting)
- [Frequently Asked Questions (FAQs)](#frequently-asked-questions-faqs)
- [Future Enhancements](#future-enhancements)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)
- [Contact Information](#contact-information)

---

## **Project Overview**

**Kartik** is an advanced, interactive chatbot designed to assist users with agricultural insights and recommendations. Leveraging machine learning models and natural language processing, Kartik can help farmers, agronomists, and agricultural enthusiasts make informed decisions about crop selection and cultivation practices.

### **Key Objectives**

- Provide personalized crop subtype recommendations based on environmental and soil conditions.
- Offer ideal cultivation conditions for specific crop subtypes and varieties.
- Deliver strategies to adjust agricultural parameters for optimal crop growth.
- Enable seamless interaction through text and voice, enhancing user experience.

---

## **Features**

### **1. Crop Subtype Recommendation**

- **Model Used**: `CropMatch`
- **Functionality**: Recommends the top crop subtypes that are most suitable for the user's provided environmental and soil conditions.
- **User Input**: Numerical and categorical environmental parameters.
- **Output**: Top 3 crop subtypes with associated probabilities.

### **2. Ideal Conditions Provision**

- **Model Used**: `AgroFit`
- **Functionality**: Provides recommended agricultural conditions for specific crop subtypes and varieties.
- **User Input**: Crop subtype and variety.
- **Output**: Detailed ideal conditions (e.g., soil type, pH level, nutrient content).

### **3. Adjustment Strategies**

- **Functionality**: Offers practical strategies to adjust various agricultural parameters, such as pH levels, nutrient content, and environmental factors.
- **User Input**: Specific parameter for which the user needs adjustment strategies.
- **Output**: Strategies for increasing or decreasing the parameter to reach optimal levels.

### **4. Conversational AI with Voice Interaction**

- **Voice Input**: Users can interact with Kartik using voice commands.
- **Text-to-Speech (TTS)**: Kartik responds using TTS, providing a more engaging experience.
- **Natural Language Understanding**: Kartik can understand and respond to various user intents, thanks to intent recognition patterns.

### **5. User-Friendly Interface**

- **Menu-Driven Interaction**: Clear options are provided for users to select desired actions.
- **Error Handling**: Provides informative messages for invalid inputs or errors.

---

## **Architecture**

The application is modular, comprising several components that work together:

- **Data Processing Modules**: `cropmatch.py` and `agrofit.py` handle data loading, preprocessing, model training, and predictions.
- **Chatbot Module**: `kartik_chatbot.py` manages the conversation flow, intent recognition, and integration with models.
- **Voice Interaction Module**: `transcriber.py` handles voice input using speech recognition.
- **Main Application**: `main.py` serves as the entry point, initializing models and handling user interactions.

---

## **Installation Instructions**

### **Prerequisites**

- **Operating System**: Windows, macOS, or Linux.
- **Python Version**: Python 3.7 or higher.
- **Microphone**: For voice input functionality.
- **Internet Connection**: Required for speech recognition API and downloading pre-trained models.

### **Step-by-Step Installation**

#### **1. Clone the Repository**

```bash
git clone https://github.com/Spandanbasuchaudhuri/kartik-chatbot.git
cd kartik-chatbot
```

#### **2. Create a Virtual Environment (Optional but Recommended)**

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

#### **3. Install Required Libraries**

##### **Install Dependencies**

```bash
pip install -r requirements.txt
```

If `requirements.txt` is not available, install dependencies manually:

```bash
pip install pandas scikit-learn numpy torch transformers joblib regex speechrecognition pyttsx3
```

##### **Install PyAudio**

**Note**: PyAudio is required for voice input functionality.

- **Windows Users**:

  - Download the appropriate PyAudio wheel file from [PyAudio Windows Wheels](https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyaudio).
  - Install it using:

    ```bash
    pip install PyAudio‑0.2.11‑cp39‑cp39‑win_amd64.whl
    ```

- **Linux/Mac Users**:

  - Install `portaudio`:

    ```bash
    # For Debian/Ubuntu
    sudo apt-get install portaudio19-dev
    # For macOS using Homebrew
    brew install portaudio
    ```

  - Install `PyAudio`:

    ```bash
    pip install PyAudio
    ```

---

## **Usage Instructions**

### **Running the Application**

Ensure your dataset is prepared and the `data_path` variable in `main.py` is correctly set.

```bash
python main.py
```

### **Interacting with Kartik**

Upon running the application, you will see a menu with several options.

#### **Options Menu**

```
Options:
[1] Start New Session
[2] Talk to Kartik (Text Input)
[3] Talk to Kartik (Voice Input)
[4] Get Crop Subtype Recommendation
[5] Get Recommended Conditions for Subtype and Variety
[6] End Session
[7] Exit
Type '99' to see the options again.
```

### **Options Explained**

#### **[1] Start New Session**

- **Purpose**: Initializes a new conversation session with Kartik, resetting any previous chat history.
- **Usage**:

  ```
  Choose an option: 1
  Kartik: Starting a new session...
  ```

#### **[2] Talk to Kartik (Text Input)**

- **Purpose**: Engage in a conversation with Kartik using text input.
- **Features**:
  - Ask for crop recommendations.
  - Request ideal conditions for a crop.
  - Seek strategies to adjust parameters.
  - General conversation.

- **Usage**:

  ```
  Choose an option: 2
  Kartik: Starting a new session automatically.
  Kartik: Starting a new session...
  Kartik: You can now chat with me! Say 'exit' or type 'exit' to end the chat.
  You: What can you do?
  Kartik: I can recommend crop subtypes, provide ideal conditions, and offer strategies to adjust parameters. Just ask me!
  ```

#### **[3] Talk to Kartik (Voice Input)**

- **Purpose**: Interact with Kartik using voice commands. Kartik will also respond using TTS.
- **Requirements**: Microphone and internet connection.
- **Usage**:

  ```
  Choose an option: 3
  Kartik: Starting a new session automatically.
  Kartik: Starting a new session...
  Kartik: You can now chat with me! Say 'exit' or type 'exit' to end the chat.
  [Kartik speaks the message aloud]
  Listening...
  You said: what can you do
  Kartik: I can recommend crop subtypes, provide ideal conditions, and offer strategies to adjust parameters. Just ask me!
  [Kartik speaks the response aloud]
  ```

#### **[4] Get Crop Subtype Recommendation**

- **Purpose**: Receive crop subtype recommendations based on environmental and soil conditions you provide.
- **Process**:
  - Input numerical features (e.g., pH level, temperature).
  - Input categorical features (e.g., soil type).
  - Receive top 3 recommended crop subtypes with probabilities.
- **Usage**:

  ```
  Choose an option: 4
  Please enter the following numerical features:
  pH Level: 6.5
  Nitrogen Content (ppm): 30
  Phosphorus Content (ppm): 20
  Potassium Content (ppm): 15
  Rainfall (mm): 500
  Temperature (°C): 25
  Humidity (%): 60
  Sunlight Hours (per day): 8
  Altitude (m): 200
  Growing Period (days): 120
  Yield (kg/ha): 5000

  Thank you! Now, please enter the following categorical features:
  Soil Type options: ['Clay', 'Loamy', 'Sandy']
  Soil Type: Loamy
  Planting Season options: ['Spring', 'Summer', 'Autumn']
  Planting Season: Spring
  Harvesting Season options: ['Summer', 'Autumn', 'Winter']
  Harvesting Season: Summer

  Based on your input, here are the top 3 recommended subtypes:
  1. Wheat with probability 0.85
  2. Barley with probability 0.10
  3. Oats with probability 0.05
  [Kartik speaks the recommendations]
  ```

#### **[5] Get Recommended Conditions for Subtype and Variety**

- **Purpose**: Obtain ideal agricultural conditions for a specific crop subtype and variety.
- **Process**:
  - Input the crop subtype.
  - Input the variety.
  - Receive detailed recommended conditions.
- **Usage**:

  ```
  Choose an option: 5
  Enter the crop subtype: Rice
  Enter the variety: Basmati

  Recommended Conditions for Rice - Basmati:
  pH Level: 6.00
  Nitrogen Content (ppm): 25.00
  Phosphorus Content (ppm): 20.00
  Potassium Content (ppm): 15.00
  Rainfall (mm): 1200.00
  Temperature (°C): 28.00
  Humidity (%): 80.00
  Sunlight Hours (per day): 7.00
  Altitude (m): 150.00
  Growing Period (days): 150.00
  Soil Type: Clay
  Planting Season: Summer
  Harvesting Season: Autumn
  [Kartik speaks the recommended conditions]
  ```

#### **[6] End Session**

- **Purpose**: Ends the current conversation session and clears chat history.
- **Usage**:

  ```
  Choose an option: 6
  Kartik: Ending session. Conversation history cleared.
  ```

#### **[7] Exit**

- **Purpose**: Exits the application.
- **Usage**:

  ```
  Choose an option: 7
  Exiting Kartik. Goodbye!
  ```

---

## **Dataset Preparation**

### **Dataset Requirements**

- **Format**: CSV file.
- **Columns**: The dataset should include the following columns (ensure exact naming):

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
  - `Yield (kg/ha)`

### **Preparing Your Dataset**

1. **Collect Data**: Gather data relevant to the crops you wish to include.
2. **Format Data**: Ensure all columns are correctly named and data types are appropriate (e.g., numerical values for numerical features).
3. **Handle Missing Values**: Fill in or remove any missing data points to prevent errors during model training.
4. **Save Dataset**: Place the dataset in the project directory or specify the correct path in `main.py`.

### **Updating `main.py`**

Ensure the `data_path` variable in `main.py` points to your dataset:

```python
data_path = 'path/to/your/enlarged_agriculture_dataset.csv'
```

---

## **Project Structure**

```
kartik-chatbot/
├── cropmatch.py
├── agrofit.py
├── transcriber.py
├── kartik_chatbot.py
├── main.py
├── requirements.txt
├── enlarged_agriculture_dataset.csv
└── README.md
```

---

## **Code Explanation**

### **1. `cropmatch.py`**

- **Purpose**: Implements the `CropMatch` class, which handles data preprocessing, model training, and predicting crop subtypes.
- **Key Components**:
  - **Data Loading**: Reads the dataset and separates features and target variable.
  - **Preprocessing**: Encodes categorical features and scales numerical features.
  - **Model Training**: Trains a Random Forest Classifier.
  - **Model Saving/Loading**: Saves and loads the model for future use.
  - **Prediction**: Provides top 3 crop subtype recommendations based on user input.

### **2. `agrofit.py`**

- **Purpose**: Implements the `AgroFit` class, which provides ideal conditions for specific crop subtypes and varieties.
- **Key Components**:
  - **Data Loading**: Reads the dataset and processes it for clustering.
  - **Clustering**: Uses KMeans clustering to group similar agricultural conditions.
  - **Recommendation**: Calculates average conditions within a cluster to recommend ideal conditions.
  - **Model Saving/Loading**: Saves and loads the clustering model.

### **3. `transcriber.py`**

- **Purpose**: Handles voice input using the `speech_recognition` library.
- **Key Components**:
  - **Transcriber Class**: Captures audio from the microphone and transcribes it to text.
  - **Error Handling**: Manages exceptions related to speech recognition.

### **4. `kartik_chatbot.py`**

- **Purpose**: Manages the conversation flow, intent recognition, and integrates with `CropMatch` and `AgroFit`.
- **Key Components**:
  - **KartikChatbot Class**: The main chatbot class.
  - **Initialization**: Loads pre-trained models and initializes TTS engine.
  - **Conversation Management**: Handles user input, maintains chat history, and generates responses.
  - **Intent Recognition**: Identifies user intents to provide appropriate responses.
  - **Voice Interaction**: Incorporates voice input and TTS for responses.
  - **Parameter Adjustment Strategies**: Provides strategies to adjust agricultural parameters.

### **5. `main.py`**

- **Purpose**: Serves as the entry point of the application, initializing models and handling user interactions.
- **Key Components**:
  - **Model Initialization**: Loads or trains `CropMatch` and `AgroFit` models.
  - **Transcriber Initialization**: Sets up the `Transcriber` for voice input.
  - **Menu Loop**: Presents options to the user and handles their choices.
  - **Session Management**: Manages conversation sessions with Kartik.

---

## **Dependencies**

- **Python Libraries**:
  - `pandas`
  - `scikit-learn`
  - `numpy`
  - `torch`
  - `transformers`
  - `joblib`
  - `regex`
  - `speechrecognition`
  - `pyttsx3`
  - `pyaudio` (for voice input)
- **External Requirements**:
  - **Microphone**: For voice input functionality.
  - **Internet Connection**: Required for speech recognition API and downloading pre-trained models.

---

## **Troubleshooting**

### **Common Issues and Solutions**

#### **1. PyAudio Installation Issues**

- **Problem**: Unable to install `PyAudio` using `pip install pyaudio`.
- **Solution**:
  - **Windows Users**:
    - Download the appropriate wheel file from [PyAudio Windows Wheels](https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyaudio).
    - Install using `pip install PyAudio‑...whl`.
  - **Linux/Mac Users**:
    - Install `portaudio` development headers using your package manager.
    - Then install `PyAudio` using `pip install pyaudio`.

#### **2. Microphone Not Recognized**

- **Problem**: The application cannot access the microphone.
- **Solution**:
  - Check if your microphone is properly connected and recognized by the operating system.
  - Ensure that the application has permission to access the microphone.
  - Test the microphone using another application to confirm it's working.

#### **3. TTS Not Functioning**

- **Problem**: Kartik does not speak responses aloud.
- **Solution**:
  - Verify that `pyttsx3` is installed.
  - On Linux, ensure that `espeak` or another speech engine is installed.
  - Check your system's audio settings.

#### **4. Import Errors**

- **Problem**: ImportError for missing modules.
- **Solution**:
  - Ensure all dependencies are installed.
  - Use `pip list` to verify installed packages.
  - Install any missing packages.

#### **5. Incorrect Data Path**

- **Problem**: Application cannot find the dataset.
- **Solution**:
  - Verify the `data_path` variable in `main.py` points to the correct file location.
  - Ensure the dataset file exists at the specified path.

#### **6. ValueError or TypeError During Execution**

- **Problem**: Errors occur when providing inputs.
- **Solution**:
  - Ensure that numerical inputs are valid numbers.
  - For categorical inputs, select from the provided options.
  - Follow the prompts carefully.

---

## **Frequently Asked Questions (FAQs)**

### **1. Can I use my own dataset?**

Yes, you can use your own dataset. Ensure it follows the required format and includes all necessary columns as specified in the [Dataset Preparation](#dataset-preparation) section.

### **2. How do I update the models with new data?**

Replace the dataset with your updated data and rerun the application. The models will retrain using the new dataset.

### **3. Can Kartik work offline?**

Most functionalities can work offline, except for voice input which relies on the Google Speech Recognition API. The TTS functionality provided by `pyttsx3` works offline.

### **4. How can I change the voice of Kartik's TTS?**

In `kartik_chatbot.py`, within the `__init__` method, you can select different voices:

```python
voices = self.engine.getProperty('voices')
self.engine.setProperty('voice', voices[0].id)  # Change index to select different voices
```

### **5. Is there a way to improve the speech recognition accuracy?**

Ensure you are speaking clearly and in a quiet environment. You can also explore using different speech recognition engines or APIs that might offer better accuracy.

---

## **Future Enhancements**

- **Graphical User Interface (GUI)**: Develop a user-friendly GUI for easier interaction.
- **Offline Speech Recognition**: Implement offline speech recognition using libraries like `vosk` to remove dependency on internet connection.
- **Multilingual Support**: Extend support for multiple languages in both input and TTS.
- **Enhanced Machine Learning Models**: Incorporate deep learning models for better predictions and recommendations.
- **Data Visualization**: Provide graphical representations of data and recommendations.
- **Integration with IoT Devices**: Connect with sensors for real-time data input and recommendations.

---

## **Contributing**

Contributions are welcome! To contribute:

1. **Fork the Repository**

   ```bash
   git clone https://github.com/Spandanbasuchaudhuri/kartik-chatbot.git
   ```

2. **Create a New Branch**

   ```bash
   git checkout -b feature/YourFeature
   ```

3. **Make Changes**

   - Add your feature or fix bugs.
   - Write clear and concise commit messages.

4. **Commit Changes**

   ```bash
   git commit -am 'Add your feature'
   ```

5. **Push to the Branch**

   ```bash
   git push origin feature/YourFeature
   ```

6. **Open a Pull Request**

   - Go to the repository on GitHub.
   - Click on "Compare & pull request".
   - Provide a descriptive title and comment.

---

## **License**

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.

---

## **Acknowledgments**

- **Hugging Face**: For the `transformers` library and pre-trained models.
- **SpeechRecognition Library**: For enabling voice input functionality.
- **pyttsx3 Library**: For offline Text-to-Speech capabilities.
- **Scikit-learn**: For machine learning algorithms.
- **OpenAI's GPT-3**: For language generation capabilities.
- **Community**: Thanks to all contributors and the open-source community for their invaluable resources.

---

## **Contact Information**

For questions, suggestions, or assistance, please contact:

- **Name**: Spandan Basu Chaudhuri
- **Email**: Spandanbasu139@gmail.com
- **GitHub**: [Spandanbasuchaudhuri](https://github.com/Spandanbasuchaudhuri)

---

**Thank you for choosing Kartik! We hope this tool enhances your agricultural practices and decision-making.**
