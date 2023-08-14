# Necessary Libraries
import sounddevice as sd
import wavio as wv
import datetime
import whisper
import openai
import os
import re
import requests
from pydub import AudioSegment
from pydub.playback import play
from dotenv import load_dotenv
import time

# Constants and Configuration
load_dotenv()
FREQ = 44100  # Sample frequency
DURATION = 5  # Duration of recording in seconds
MODEL = whisper.load_model("base")  # Load Whisper model
OPENAI_API_KEY = os.getenv('OPENAI_KEY')  # Replace with your OpenAI API key
openai.api_key = OPENAI_API_KEY  # Set OpenAI API key

# Data Initialization
retrievedinformation = [{"name": "", "email": "", "phone": ""}]
messages = [...]  # Define the messages as per the original code


salespitchprompt = """
You need to make a short sales pitch to generate a sale for the company Neurovee. Here are some important information about the product you are selling:

The most effective and cutting-edge technology on the market.
Neurovee has a list of autonomous products such as Autonomous Sales, Autonomous HR, and more.

[REMEMBER] Keep the pitch short, below 3 sentences!

"""

closeoutprompt = """

Try to close out the call and create a lead. Attempt to collect their name and email to send them more information about the product.

If they are not interested, ask them if there is anything you can do to change their mind.

Remember to make sure you actually collect their email address, phone number and name. If you do not collect any of the information, keep trying.

"""

class ChatGPT:
    def __init__(self):
        pass
    @staticmethod
    def chat_with_gpt3(messages):
        retries = 20
        delay = 5
        for i in range(retries):
            try:
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo-16k",
                    messages=messages,
                    temperature=0.9
                )
                return response['choices'][0]['message']['content']
            except openai.error.ServiceUnavailableError:
                if i < retries - 1:
                    time.sleep(delay)
                else:
                    raise



# Helper Functions
def text_to_speech(text, filename):
    CHUNK_SIZE = 100
    url = "https://api.elevenlabs.io/v1/text-to-speech/21m00Tcm4TlvDq8ikWAM/stream"  # replace with the voice ID
    # Additional code to make the API request and save the audio file
    
def record_audio():
    print('Recording')
    ts = datetime.datetime.now()
    filename = ts.strftime("%Y-%m-%d_%H-%M-%S")  # Changed ':' to '_'
    recording = sd.rec(int(DURATION * FREQ), samplerate=FREQ, channels=1)
    sd.wait()
    wv.write(f"./recordings/{filename}.wav", recording, FREQ, sampwidth=2)
    return filename

def transcribe_audio(filename):
    audio = whisper.load_audio(f"./recordings/{filename}.wav")
    audio = whisper.pad_or_trim(audio)
    mel = whisper.log_mel_spectrogram(audio).to(MODEL.device)
    options = whisper.DecodingOptions(language='en', fp16=False)
    result = whisper.decode(MODEL, mel, options)
    if result.no_speech_prob < 0.5:
        return result.text
    else:
        return None

def convert_mp3_to_wav(mp3_path, wav_path):
    audio = AudioSegment.from_mp3(mp3_path)
    audio.export(wav_path, format='wav')

def sales_pitch(transcription):
    messages.append({"role": "user", "content": transcription})
    messages.append({"role": "system", "content": salespitchprompt})
    response = ChatGPT.chat_with_gpt3(messages)
    messages.append({"role": "assistant", "content": response})
    return response

def first_message():
    response = ChatGPT.chat_with_gpt3(messages)
    messages.append({"role": "assistant", "content": response})
    return response

def close_call(transcription):
    messages.append({"role": "user", "content": transcription})
    messages.append({"role": "system", "content": closeoutprompt})
    response = ChatGPT.chat_with_gpt3(messages)
    messages.append({"role": "assistant", "content": response})
    return response

def parse_client_details(transcription):
    client_details = {
        "name": parse_name(transcription),
        "number": extract_number(transcription),
    }
    return client_details

def play_audio(filename):
    audio = AudioSegment.from_file(filename, format="mp3")
    play(audio)

def parse_name(transcription):
    # Splitting the transcription into words
    words = transcription.split()
    # Looking for capitalized words as potential names
    name_candidates = [word for word in words if word.istitle()]
    # Joining the name candidates to form a full name (this is a heuristic and might not be perfect)
    name = " ".join(name_candidates)
    return name

def extract_number(transcription):
    # Mapping words to digits
    word_to_digit = {
        "zero": "0",
        "one": "1",
        "two": "2",
        "three": "3",
        "four": "4",
        "five": "5",
        "six": "6",
        "seven": "7",
        "eight": "8",
        "nine": "9"
    }

    # Splitting the transcription into words
    words = transcription.lower().split()
    # Converting word numbers to digits
    digits = [word_to_digit[word] for word in words if word in word_to_digit]
    # Joining the digits to form the number
    number = "".join(digits)

    return number


# Setup and Preparation
def setup():
    if not os.path.exists('./recordings'):
        os.makedirs('./recordings')

# Introduction
def introduction():
    script = first_message()
    text_to_speech(script, "response.mp3")
    convert_mp3_to_wav("response.mp3", "response.wav")
    play_audio("response.wav")
    filename = record_audio()
    transcription = transcribe_audio(filename)
    return transcription

# Sales Pitch
def sales_pitch_section(transcription):
    response = sales_pitch(transcription)
    text_to_speech(response, "response.mp3")
    convert_mp3_to_wav("response.mp3", "response.wav")
    play_audio("response.wav")
    filename = record_audio()
    transcription = transcribe_audio(filename)
    return transcription

# Ongoing Interaction with Client
def ongoing_interaction():
    counter = 0
    parsedinfo = {"name": "", "email": "", "phonenumber": ""}
    while True:
        filename = record_audio()
        transcription = transcribe_audio(filename)
        client_details = parse_client_details(transcription)
        if client_details['name'] and client_details['email']:
            send_farewell(client_details)
            exit()
        update_parsedinfo(parsedinfo, client_details)
        print("Parsed Info:", parsedinfo)
        response = close_call(transcription)
        print(response)
        response_mp3_path = f"response_{counter}.mp3"
        response_wav_path = f"response_{counter}.wav"
        text_to_speech(response, response_mp3_path)
        convert_mp3_to_wav(response_mp3_path, response_wav_path)
        play_audio(response_wav_path)
        counter += 1

# Closing the Interaction
def send_farewell(client_details):
    farewell = "Thank you for using our service. We will contact you shortly. Have a good day!"
    text_to_speech(farewell, "farewell.mp3")
    convert_mp3_to_wav("farewell.mp3", "farewell.wav")
    play_audio("farewell.wav")
    print(f"Farewell sent. Closing application on customer: {client_details}")

# Main Function
def main():
    setup()
    transcription = introduction()
    response = sales_pitch_section(transcription)
    ongoing_interaction()

if __name__ == "__main__":
    main()