from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import pandas as pd
import re
import pyttsx3  # For Text-to-Speech
from yield_predictor import YieldPredictor  # Import YieldPredictor


class KartikChatbot:
    def __init__(self, crop_match, agrofit, transcriber=None, yield_predictor=None):
        # Load the pre-trained model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
        self.model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")
        self.chat_history_ids = None  # Stores conversation history within a session
        self.crop_match = crop_match  # Store the CropMatch model
        self.agrofit = agrofit  # Store the AgroFit model
        self.transcriber = transcriber  # Store the Transcriber module (for voice input)
        self.yield_predictor = yield_predictor  # Store the YieldPredictor model

        # Load the adjustment strategies into the chatbot
        self.adjustment_strategies = self._load_adjustment_strategies()

        # Initialize the TTS engine
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', 150)  # Adjust speech rate if needed
        self.engine.setProperty('volume', 1.0)  # Set volume level (0.0 to 1.0)
        # Optionally, set the voice
        # voices = self.engine.getProperty('voices')
        # self.engine.setProperty('voice', voices[0].id)  # Change index to select different voices

    def start_new_session(self):
        """Starts a new conversation session and clears chat history."""
        print("Kartik: Starting a new session...")
        self.chat_history_ids = None  # Reset the conversation history

    def get_response(self, user_input):
        """Generates a response based on user input and maintains session history."""
        # Encode user input and append to the conversation history
        new_user_input_ids = self.tokenizer.encode(user_input + self.tokenizer.eos_token, return_tensors="pt")
        bot_input_ids = new_user_input_ids if self.chat_history_ids is None else torch.cat([self.chat_history_ids, new_user_input_ids], dim=-1)

        # Generate a response, updating chat history
        self.chat_history_ids = self.model.generate(
            bot_input_ids,
            max_length=1000,
            pad_token_id=self.tokenizer.eos_token_id,
            attention_mask=self._get_attention_mask(bot_input_ids)  # Add attention mask
        )
        response = self.tokenizer.decode(
            self.chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True
        )
        return response

    def _get_attention_mask(self, input_ids):
        """Creates an attention mask from the input IDs."""
        return torch.ones(input_ids.shape, dtype=torch.long)

    def end_session(self):
        """Ends the session and clears chat history."""
        print("Kartik: Ending session. Conversation history cleared.")
        self.chat_history_ids = None  # Clear conversation history

    def collect_features(self):
        """Collects crop features from the user."""
        numerical_features = {}
        print("Please enter the following numerical features:")
        for feature in self.crop_match.numerical_features:
            while True:
                try:
                    value = input(f"{feature}: ").strip()
                    if not value:
                        print("Input cannot be empty. Please enter a numeric value.")
                        continue
                    value = float(value)
                    numerical_features[feature] = value
                    break
                except ValueError:
                    print("Invalid input. Please enter a numeric value.")

        categorical_features = {}
        print("\nThank you! Now, please enter the following categorical features:")
        for feature in self.crop_match.categorical_features:
            options = list(self.crop_match.label_encoders[feature].classes_)
            print(f"{feature} options: {options}")
            while True:
                value = input(f"{feature}: ").strip()
                if not value:
                    print("Input cannot be empty. Please select from the options.")
                    continue
                if value in options:
                    categorical_features[feature] = value
                    break
                else:
                    print(f"Invalid choice. Please select from {options}.")

        # Combine numerical and categorical features
        user_input = {**numerical_features, **categorical_features}
        return user_input

    def make_recommendation(self):
        """Makes a recommendation based on collected inputs."""
        user_input = self.collect_features()
        try:
            recommendations = self.crop_match.predict_subtype(user_input)
            self.display_recommendations(recommendations)
        except Exception as e:
            print(f"An error occurred during recommendation: {e}")

    def display_recommendations(self, recommendations):
        print("\nBased on your input, here are the top 3 recommended subtypes:")
        for i, (subtype, probability) in enumerate(recommendations.items(), 1):
            print(f"{i}. {subtype} with probability {probability:.2f}")
        # Speak the recommendations
        recommendations_text = "Based on your input, here are the top recommended subtypes: " + ", ".join([f"{subtype}" for subtype in recommendations.keys()])
        self.speak(recommendations_text)

    def get_conditions_recommendation(self):
        """Gets recommended conditions based on subtype and variety."""
        subtype = input("Enter the crop subtype: ").strip()
        variety = input("Enter the variety: ").strip()
        recommendation = self.agrofit.recommend_conditions(subtype, variety)
        if isinstance(recommendation, dict):
            print(f"\nRecommended Conditions for {subtype} - {variety}:")
            conditions_text = f"Recommended conditions for {subtype} - {variety} are as follows: "
            for key, value in recommendation.items():
                if isinstance(value, (int, float)):
                    formatted_value = f"{value:.2f}"
                else:
                    formatted_value = value
                print(f"{key}: {formatted_value}")
                conditions_text += f"{key} is {formatted_value}. "
            self.speak(conditions_text)
        else:
            print(recommendation)
            self.speak(recommendation)

    def predict_yield(self):
        """Predicts the yield based on user inputs."""
        print("Let's predict the yield for your crop.")
        # Collect inputs
        user_input = {}
        features = self.yield_predictor.data.drop("Yield (kg/ha)", axis=1).columns
        categorical_columns = list(self.yield_predictor.label_encoders.keys())
        numerical_columns = [col for col in features if col not in categorical_columns]

        print("Please enter the following numerical features:")
        for feature in numerical_columns:
            while True:
                try:
                    value = input(f"{feature}: ").strip()
                    if not value:
                        print("Input cannot be empty. Please enter a numeric value.")
                        continue
                    value = float(value)
                    user_input[feature] = value
                    break
                except ValueError:
                    print("Invalid input. Please enter a numeric value.")

        print("\nNow, please enter the following categorical features:")
        for feature in categorical_columns:
            options = list(self.yield_predictor.label_encoders[feature].classes_)
            print(f"{feature} options: {options}")
            while True:
                value = input(f"{feature}: ").strip()
                if not value:
                    print("Input cannot be empty. Please select from the options.")
                    continue
                if value in options:
                    user_input[feature] = value
                    break
                else:
                    print(f"Invalid choice. Please select from {options}.")

        # Predict yield
        predicted_yield = self.yield_predictor.predict_yield(user_input)
        if predicted_yield is not None:
            print(f"\nThe predicted yield is: {predicted_yield:.2f} kg/ha")
            self.speak(f"The predicted yield is {predicted_yield:.2f} kilograms per hectare.")
        else:
            print("Could not predict yield due to missing or invalid inputs.")
            self.speak("Could not predict yield due to missing or invalid inputs.")

    def provide_adjustment_strategies(self):
        """Provides strategies to adjust parameters when requested."""
        print("Please specify the parameter you'd like strategies for (e.g., 'pH Level'):")
        parameter = input("Parameter: ").strip()
        # Normalize and tokenize the user's input
        user_tokens = set(parameter.lower().split())
        # Get the list of parameters from the adjustment strategies
        available_parameters = list(self.adjustment_strategies.keys())
        matches = []
        for param in available_parameters:
            # Normalize and tokenize the parameter name
            param_tokens = set(param.lower().split())
            # Check if any token in the user's input matches any token in the parameter name
            if user_tokens & param_tokens:
                matches.append(param)
        if matches:
            if len(matches) == 1:
                actual_parameter = matches[0]
                strategies = self.adjustment_strategies[actual_parameter]
                response = f"\nAdjustment Strategies for {actual_parameter}:\n"
                response += f"**If higher than required:** {strategies['higher']}\n"
                response += f"**If lower than required:** {strategies['lower']}"
                print(response)
                self.speak(f"Here are the adjustment strategies for {actual_parameter}. If higher than required: {strategies['higher']}. If lower than required: {strategies['lower']}.")
            else:
                print("I found multiple parameters that might match your input:")
                for i, match in enumerate(matches, 1):
                    print(f"[{i}] {match}")
                choice = input("Please enter the number corresponding to the parameter: ").strip()
                try:
                    choice_index = int(choice) - 1
                    if 0 <= choice_index < len(matches):
                        actual_parameter = matches[choice_index]
                        strategies = self.adjustment_strategies[actual_parameter]
                        response = f"\nAdjustment Strategies for {actual_parameter}:\n"
                        response += f"**If higher than required:** {strategies['higher']}\n"
                        response += f"**If lower than required:** {strategies['lower']}"
                        print(response)
                        self.speak(f"Here are the adjustment strategies for {actual_parameter}. If higher than required: {strategies['higher']}. If lower than required: {strategies['lower']}.")
                    else:
                        print("Invalid selection.")
                        self.speak("Invalid selection.")
                except ValueError:
                    print("Invalid input.")
                    self.speak("Invalid input.")
        else:
            print(f"Sorry, I don't have strategies for '{parameter}'.")
            self.speak(f"Sorry, I don't have strategies for {parameter}.")

    def chat(self, use_voice=False):
        """Enables conversation with the user, including intent recognition."""
        print("Kartik: You can now chat with me! Say 'exit' or type 'exit' to end the chat.")
        self.speak("You can now chat with me! Say 'exit' or type 'exit' to end the chat.")
        while True:
            if use_voice and self.transcriber:
                user_input = self.transcriber.listen()
                if not user_input:
                    continue  # Skip to next iteration if no input
            else:
                user_input = input("You: ").strip()

            if user_input.lower() in ['exit', 'quit', 'bye']:
                response = "It was nice talking to you. Goodbye!"
                print(f"Kartik: {response}")
                self.speak(response)
                break

            # Intent recognition for adjustment strategies request
            if self._is_adjustment_strategies_request(user_input):
                response = "Sure, I can provide strategies to help with varying parameters."
                print(f"Kartik: {response}")
                self.speak(response)
                self.provide_adjustment_strategies()
                continue  # Continue the chat

            # Intent recognition for model info request
            elif self._is_model_info_request(user_input):
                self.display_model_info()
                continue  # Continue the chat

            # Intent recognition for conditions recommendation
            elif self._is_conditions_request(user_input):
                response = "Sure, I can provide recommended conditions for a crop subtype and variety."
                print(f"Kartik: {response}")
                self.speak(response)
                self.get_conditions_recommendation()
                continue  # Continue the chat after recommendation

            # Intent recognition for crop recommendation
            elif self._is_recommendation_request(user_input):
                response = "Sure, I can help you with recommending a crop subtype."
                print(f"Kartik: {response}")
                self.speak(response)
                self.make_recommendation()
                continue  # Continue the chat after recommendation

            # Intent recognition for yield prediction
            elif self._is_yield_prediction_request(user_input):
                response = "Sure, I can help you predict the yield for your crop."
                print(f"Kartik: {response}")
                self.speak(response)
                self.predict_yield()
                continue  # Continue the chat after prediction

            # Intent recognition for name query
            elif self._is_name_query(user_input):
                response = "My name is Kartik."
                print(f"Kartik: {response}")
                self.speak(response)
                continue  # Continue the chat

            # Intent recognition for capability query
            elif self._is_capability_query(user_input):
                response = "I can recommend crop subtypes, provide ideal conditions, predict yield, and offer strategies to adjust parameters. Just ask me!"
                print(f"Kartik: {response}")
                self.speak(response)
                continue  # Continue the chat

            # Generate a response using DialoGPT
            response = self.get_response(user_input)
            print(f"Kartik: {response}")
            self.speak(response)

    def speak(self, text):
        """Converts text to speech."""
        self.engine.say(text)
        self.engine.runAndWait()

    def _is_adjustment_strategies_request(self, user_input):
        """Checks if the user is asking for strategies to adjust parameters."""
        strategies_patterns = [
            r'\b(methods|strategies|tips|ways)\b.*\b(help\b.*\bwith|for|to adjust)\b.*\b(parameters|conditions|factors)\b',
            r'\b(how to|ways to|help me)\b.*\b(adjust|manage|control|improve)\b.*\b(parameters|conditions|factors)\b',
            r'\b(provide|give)\b.*\b(strategies|methods|tips)\b.*\b(parameters|conditions|factors)\b',
            r'\b(need|want)\b.*\b(help|advice)\b.*\b(parameters|conditions|factors)\b'
        ]
        user_input_lower = user_input.lower()
        for pattern in strategies_patterns:
            if re.search(pattern, user_input_lower):
                return True
        return False

    def _load_adjustment_strategies(self):
        """Loads the adjustment strategies into the chatbot."""
        return {
            'pH Level': {
                'higher': "Apply elemental sulfur or aluminum sulfate to lower pH. Organic matter, such as peat moss, can also acidify the soil gradually.",
                'lower': "Add lime (calcium carbonate or dolomite) to raise pH levels. Properly monitoring application rates is important to avoid over-application."
            },
            'Nitrogen Content (ppm)': {
                'higher': "Reduce nitrogen fertilizer applications, and plant cover crops (e.g., legumes) that utilize excess nitrogen. Using nitrogen-absorbing plants can also help.",
                'lower': "Apply nitrogen-rich fertilizers like urea, ammonium nitrate, or organic options (compost, manure). Slow-release nitrogen sources provide steady nutrient availability."
            },
            'Phosphorus Content (ppm)': {
                'higher': "Avoid using phosphorus fertilizers for some time, as excess can lead to imbalances with other nutrients.",
                'lower': "Use phosphorus fertilizers (such as superphosphate or bone meal) and add organic matter. Phosphorus availability increases in slightly acidic soil, so adjust pH if needed."
            },
            'Potassium Content (ppm)': {
                'higher': "Reduce potassium fertilization and monitor soil tests frequently. Excess potassium can interfere with other nutrient uptake.",
                'lower': "Apply potassium fertilizers like potassium sulfate or potassium chloride. Compost and wood ash (sparingly) can also increase potassium levels."
            },
            'Rainfall (mm)': {
                'higher': "Ensure proper drainage systems (such as raised beds or ditches) to prevent waterlogging. Plant cover crops that help absorb excess water.",
                'lower': "Use efficient irrigation techniques (e.g., drip irrigation), mulch to retain soil moisture, and choose drought-tolerant crop varieties."
            },
            'Temperature (°C)': {
                'higher': "Use shade nets or plant crops that tolerate heat. Adjust planting schedules to avoid peak temperatures and employ mulching to cool soil.",
                'lower': "Utilize greenhouses or row covers to trap heat. Select crop varieties suited to lower temperatures, or time planting to warmer months."
            },
            'Humidity (%)': {
                'higher': "Improve ventilation in greenhouses and use moisture-absorbing mulch. Disease-resistant crops and fungicides help combat mold and mildew in high humidity.",
                'lower': "Irrigate more frequently to increase local humidity. Protect plants from dry winds and use humidity trays in greenhouse setups."
            },
            'Sunlight Hours (per day)': {
                'higher': "Use shade cloth to reduce light exposure or select crop varieties that are tolerant to high light intensity.",
                'lower': "Use supplemental grow lights in greenhouses. Opt for varieties that grow in partial sunlight or plant in locations with maximal exposure."
            },
            'Altitude (m)': {
                'higher': "Select altitude-tolerant crop varieties suited to cooler and windier conditions typical at high altitudes.",
                'lower': "Grow varieties suited to low-altitude climates, where temperature and oxygen levels are more stable."
            },
            'Growing Period (days)': {
                'higher': "Choose faster-maturing crop varieties to reduce growing time. Optimize nutrient levels and manage pests to prevent delays in growth.",
                'lower': "Use varieties with extended growing periods. Use growth regulators or manage planting schedules to align with seasonal conditions."
            },
            'Yield (kg/ha)': {
                'higher': "Optimize all environmental and nutrient factors to support growth, prevent diseases, and increase efficiency in irrigation and fertilization.",
                'lower': "Implement crop rotation and soil health management practices to avoid soil depletion and sustain productivity."
            }
            # Add more parameters and strategies as needed
        }

    def display_model_info(self):
        """Displays information about the models in a tabular format."""
        print("Kartik: Here are the models I have available:\n")
        self.speak("Here are the models I have available.")
        model_info = [
            ["Model Name", "Purpose"],
            ["CropMatch", "Recommends crop subtypes based on your input data."],
            ["AgroFit", "Provides ideal agricultural conditions for specific crop subtypes and varieties."],
            ["YieldPredictor", "Predicts the expected yield based on your crop and conditions."]
        ]

        # Calculate column widths
        col_widths = [max(len(str(row[i])) for row in model_info) + 2 for i in range(len(model_info[0]))]

        # Build the table
        table_lines = []
        for i, row in enumerate(model_info):
            line = ""
            for j, cell in enumerate(row):
                line += str(cell).ljust(col_widths[j])
            table_lines.append(line)
            if i == 0:
                # Add a separator after the header
                separator = ''.join(['-' * w for w in col_widths])
                table_lines.append(separator)

        # Print the table
        for line in table_lines:
            print(line)
        print()  # Add an extra newline for spacing

    def _is_model_info_request(self, user_input):
        """Checks if the user is asking about the models available."""
        model_info_patterns = [
            r'\bwhat models do you have\b',
            r'\bwhat are the models (you|we) (have|use)\b',
            r'\btell me about the models\b',
            r'\bwhat models are available\b',
            r'\bhow many models do you have\b',
            r'\bwhat (are|do) the models (do|you have)\b',
            r'\bwhich models are available\b',
            r'\bwhat models do i have\b',
            r'\bwhat are the models i have\b'
        ]
        user_input_lower = user_input.lower()
        for pattern in model_info_patterns:
            if re.search(pattern, user_input_lower):
                return True
        return False

    def _is_recommendation_request(self, user_input):
        """Checks if the user wants help with recommending a crop subtype."""
        recommendation_keywords = [
            r'\b(recommend|suggest|recommendation|suggestion)\b.*\b(crop subtype|subtype|crop type)\b',
            r'\b(help me with|assist me with|can you help me with|need help with)\b.*\b(recommend|suggest)\b.*\b(crop subtype|subtype|crop type)\b',
            r'\brecommend a crop subtype\b',
        ]
        user_input_lower = user_input.lower()
        for pattern in recommendation_keywords:
            if re.search(pattern, user_input_lower):
                return True
        return False

    def _is_conditions_request(self, user_input):
        """Checks if the user is asking for recommended conditions for a crop subtype and variety."""
        conditions_keywords = [
            r'\b(recommend|suggest|provide|give|tell me)\b.*\b(ideal conditions|best conditions|conditions)\b.*\b(for|of)\b.*\b(crop|subtype|variety)\b',
            r'\b(need|want)\b.*\b(ideal conditions|best conditions|conditions)\b.*\b(for|of)\b.*\b(crop|subtype|variety)\b',
            r'\b(get|find out)\b.*\b(ideal conditions|best conditions|conditions)\b',
            r'\b(i would like to|get me)\b.*\b(ideal conditions|best conditions|conditions)\b.*\b(crop|subtype|variety)\b',
        ]
        user_input_lower = user_input.lower()
        for pattern in conditions_keywords:
            if re.search(pattern, user_input_lower):
                return True
        return False

    def _is_yield_prediction_request(self, user_input):
        """Checks if the user wants to predict yield."""
        yield_prediction_keywords = [
            r'\b(predict|calculate|estimate)\b.*\b(yield)\b',
            r'\b(what is|tell me|give me)\b.*\b(yield)\b',
            r'\b(yield prediction|yield estimate)\b',
            r'\b(how much)\b.*\b(yield)\b'
        ]
        user_input_lower = user_input.lower()
        for pattern in yield_prediction_keywords:
            if re.search(pattern, user_input_lower):
                return True
        return False

    def _is_name_query(self, user_input):
        """Checks if the user is asking for the chatbot's name."""
        name_query_patterns = [
            r'\bwhat is your name\b',
            r'\bwho are you\b',
            r'\bdo you have a name\b',
            r'\btell me your name\b',
            r'\byour name\b',
            r'\bmay i know your name\b',
            r'\bcan you tell me your name\b',
            r'\bwhat should i call you\b'
        ]
        user_input_lower = user_input.lower()
        for pattern in name_query_patterns:
            if re.search(pattern, user_input_lower):
                return True
        return False

    def _is_capability_query(self, user_input):
        """Checks if the user is asking about the chatbot's capabilities."""
        capability_query_patterns = [
            r'\bwhat can you do\b',
            r'\bwhat do you do\b',
            r'\bhow can you help me\b',
            r'\bwhat services do you offer\b',
            r'\bwhat are your capabilities\b',
            r'\bwhat features do you have\b',
            r'\bhow can you assist me\b',
            r'\bwhat support do you provide\b'
        ]
        user_input_lower = user_input.lower()
        for pattern in capability_query_patterns:
            if re.search(pattern, user_input_lower):
                return True
        return False