# Necessary Libraries
import sounddevice as sd
import wavio as wv
import simpleaudio as sa
import datetime
import whisper
import openai
import json
import asyncio
import os
import sys
import re
import requests
import time
import requests
from pydub import AudioSegment
from pydub.playback import play
from dotenv import load_dotenv

load_dotenv()
if not os.path.exists('./recordings'):
    os.makedirs('./recordings')
# Constants
FREQ = 44100  # Sample frequency
DURATION = 5  # Duration of recording in seconds
MODEL = whisper.load_model("base")  # Load Whisper model
OPENAI_API_KEY = os.getenv('OPENAI_KEY')  # Replace with your OpenAI API key
ELEVENLABS = os.getenv('ELEVENLABS_KEY')  # Replace with your Eleven Labs API key
openai.api_key = OPENAI_API_KEY  # Set OpenAI API key
retrievedinformation = [{"name": "", "email": "", "phone": ""}]
messages = [
    {"role": "system", "content": """
You are a coldcalling a customer right now as Rachel, when the client picks up the phone you need to be in character as Rachel a salesrepresentive from Neurovee. 

[EXAMPLE] 
Hi my name is Rachel, I hope I'm not disturbing you, I am calling from Neurovee is there any chance I could borrow a minute of your time?

"""}
]
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
def text_to_speech(text, filename):
    CHUNK_SIZE = 100
    url = "https://api.elevenlabs.io/v1/text-to-speech/21m00Tcm4TlvDq8ikWAM/stream"  # replace with the voice ID

    headers = {
      "Accept": "audio/mpeg",
      "Content-Type": "application/json",
      "xi-api-key": "d86678cade0e6c64ebdf89691e016064"  # replace with your API key
    }
    data = {
      "text": text,
      "model_id": "eleven_monolingual_v1",
      "voice_settings": {
        "stability": 0.25,
        "similarity_boost": 0.5
      }
    }
    response = requests.post(url, json=data, headers=headers)
    print(response.status_code)
    with open(filename, 'wb') as f:
        for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
            if chunk:
                f.write(chunk)
def llm_info_parser(text):
    systemp = """I need you to parse email and name from the text below. Only respond with the parsed information. If the text does not contain any of the information, respond with No."""
    messages = [{"role": "system", "content": systemp}, {"role": "user", "content": text}]
    results = ChatGPT.chat_with_gpt3(messages)
    return results
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
        "email": extract_email(transcription),
    }
    return client_details
def parse_name(transcription):
    nameprompt="""You are an AI parser. 
    Your job is to read the transcription and only reply with a name. 
    Do not reply with anything else. If a name is not found, send a blank reply.
    Only reply with one name."""
    gptmessages=[{"role": "system", "content": nameprompt}, {"role": "user", "content": transcription}]
    name = ChatGPT.chat_with_gpt3(gptmessages)
    return name if name else ""

def extract_email(text):
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,7}\b'
    match = re.search(email_pattern, text)
    return match.group() if match else ""

def update_parsedinfo(parsedinfo, client_details):
    parsedinfo.update(client_details)

def is_info_complete(parsedinfo):
    return all(value != "" for value in parsedinfo.values())

def send_farewell():
    farewell = """Thank you for using our service. We will contact you shortly. Have a good day!"""
    text_to_speech(farewell, "farewell.mp3")

    # Convert the response from mp3 to wav
    convert_mp3_to_wav("farewell.mp3", "farewell.wav")

    # Play the speech
    play_audio("farewell.wav")
    pass
def play_audio(filename):
    audio = AudioSegment.from_file(filename, format="mp3")
    play(audio)
def main():
    counter = 0
    parsedinfo = {"name": "", "email": "", "phonenumber": ""}
    script = first_message()
    print (script)
    text_to_speech(script, "response.mp3")
    convert_mp3_to_wav("response.mp3", "response.wav")
    play_audio("response.wav")
    filename = record_audio()
    transcription = transcribe_audio(filename)
    response = sales_pitch(transcription)
    print(response)
    # Convert the response to speech
    text_to_speech(response, "response.mp3")

    # Convert the response from mp3 to wav
    convert_mp3_to_wav("response.mp3", "response.wav")

    # Play the speech
    play_audio("response.wav")

    while True:
        filename = record_audio()
        transcription = transcribe_audio(filename)
        # Parse the client's details and update parsedinfo
        client_details = parse_client_details(transcription)
        if client_details['name'] & client_details['email']:
            send_farewell()
            print(f"Farewell sent. closing application on customer: ", client_details)
            sys.exit()
        update_parsedinfo(parsedinfo, client_details)
        # Print the current state of parsedinfo
        print("Parsed Info:", parsedinfo)
        response = close_call(transcription)
        print(response)
        response_mp3_path = f"response_{counter}.mp3"
        response_wav_path = f"response_{counter}.wav"
        # Convert the response to speech
        text_to_speech(response, response_mp3_path)
        # Convert the response from mp3 to wav
        convert_mp3_to_wav(response_mp3_path, response_wav_path)
        # Play the speech
        play_audio(response_wav_path)
        counter += 1


if __name__ == "__main__":
    main()