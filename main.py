# main.py

from cropmatch import CropMatch
from kartik_chatbot import KartikChatbot
from agrofit import AgroFit
from transcriber import Transcriber  # Import the transcriber module
import warnings


def main():
    # Suppress OpenMP warnings if any
    warnings.filterwarnings("ignore", message=".*libiomp.*libomp.*")

    data_path = 'C:/Users/spand/enlarged_agriculture_dataset.csv'  # Update the path to your dataset

    # Initialize the CropMatch model with the dataset path
    crop_match = CropMatch(
        data_path=data_path,
        model_path='crop_match_model.pkl'
    )
    # Save the model (if you want to use it later)
    crop_match.save_model('crop_match_model.pkl')
    print("Model saved to crop_match_model.pkl")

    # Initialize AgroFit
    agrofit = AgroFit(
        data_path=data_path,
        model_path='agrofit_model.pkl'
    )

    # Initialize Transcriber
    transcriber = Transcriber()

    # The models are saved during initialization if not already saved

    chatbot = KartikChatbot(crop_match, agrofit, transcriber)

    # Display options once at the beginning
    print("\nOptions:")
    print("[1] Start New Session")
    print("[2] Talk to Kartik (Text Input)")
    print("[3] Talk to Kartik (Voice Input)")
    print("[4] Get Crop Subtype Recommendation")
    print("[5] Get Recommended Conditions for Subtype and Variety")
    print("[6] End Session")
    print("[7] Exit")
    print("Type '99' to see the options again.")

    while True:
        choice = input("Choose an option: ").strip()

        if choice == "1":
            chatbot.start_new_session()
        elif choice == "2":
            if chatbot.chat_history_ids is None:
                print("Kartik: Starting a new session automatically.")
                chatbot.start_new_session()
            chatbot.chat()
        elif choice == "3":
            if chatbot.chat_history_ids is None:
                print("Kartik: Starting a new session automatically.")
                chatbot.start_new_session()
            chatbot.chat(use_voice=True)  # Enable voice input
        elif choice == "4":
            chatbot.make_recommendation()
        elif choice == "5":
            # Get recommended conditions
            chatbot.get_conditions_recommendation()
        elif choice == "6":
            chatbot.end_session()
        elif choice == "7":
            print("Exiting Kartik. Goodbye!")
            break
        elif choice == "99":
            print("\nOptions:")
            print("[1] Start New Session")
            print("[2] Talk to Kartik (Text Input)")
            print("[3] Talk to Kartik (Voice Input)")
            print("[4] Get Crop Subtype Recommendation")
            print("[5] Get Recommended Conditions for Subtype and Variety")
            print("[6] End Session")
            print("[7] Exit")
        else:
            print("Invalid option. Please try again.")


if __name__ == "__main__":
    main()
