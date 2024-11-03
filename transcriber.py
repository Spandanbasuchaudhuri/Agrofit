# transcriber.py

import speech_recognition as sr


class Transcriber:
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()

    def listen(self):
        """Captures audio from the microphone and transcribes it to text."""
        with self.microphone as source:
            print("Listening...")
            self.recognizer.adjust_for_ambient_noise(source)  # Adjust for ambient noise
            audio = self.recognizer.listen(source)
        try:
            # Use Google's speech recognition API
            text = self.recognizer.recognize_google(audio)
            print(f"You said: {text}")
            return text
        except sr.UnknownValueError:
            print("Sorry, I could not understand the audio.")
            return ""
        except sr.RequestError as e:
            print(f"Could not request results from the speech recognition service; {e}")
            return ""