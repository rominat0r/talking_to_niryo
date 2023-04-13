import speech_recognition as sr
import requests

API_URL = "https://api-inference.huggingface.co/models/deepset/roberta-base-squad2"
headers = {"Authorization": "Bearer ********************************"}

#Speech recognition 
def query(payload):
	response = requests.post(API_URL, headers=headers, json=payload)
	return response.json()

def user_request():
# Create a recognizer instance
    r = sr.Recognizer()

# Use the microphone as the audio source
    while True:
        with sr.Microphone() as source:
            print("Say something!")
            audio = r.listen(source, phrase_time_limit=5)
        try:
            text = r.recognize_google(audio)
            break
        except sr.UnknownValueError:
            print("Google Speech Recognition could not understand audio, please try again.")
        except sr.RequestError as e:
            print("Could not request results from Google Speech Recognition service; {0}".format(e))
            print("Google Speech Recognition thinks you said: " + text)

    try:
        output = query({
            "inputs": {
            "question": "Tell me what I want you to give me",
            "context": text
        },
        })
        answer = output['answer'].split()  # split the string into a list of words
        user_request = answer[-1] 
        return user_request
    except:
        print("Something went wrong with the query.")
