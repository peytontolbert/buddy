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

openai.api_key = OPENAI_API_KEY  # Set OpenAI API key
messages = [
    {"role": "system", "content": "You are a helpful sales agent. Your task is to sell a product Neurovee which is an AI sales agent. You must stay on task in trying to sell the sales agent to the customer, stay focused and do not let the customer know they are talking to an AI."}
]

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
    CHUNK_SIZE = 1024
    url = "https://api.elevenlabs.io/v1/text-to-speech/21m00Tcm4TlvDq8ikWAM"  # replace with the voice ID

    headers = {
      "Accept": "audio/mpeg",
      "Content-Type": "application/json",
      "xi-api-key": "15d40ab18bc2e52f6942bd6d6b20b78d"  # replace with your API key
    }

    data = {
      "text": text,
      "model_id": "eleven_monolingual_v1",
      "voice_settings": {
        "stability": 0.5,
        "similarity_boost": 0.5
      }
    }

    response = requests.post(url, json=data, headers=headers)
    print(response.status_code)
    with open(filename, 'wb') as f:
        for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
            if chunk:
                f.write(chunk)

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

def interpret_command(transcription):
    messages.append({"role": "user", "content": transcription})
    response = ChatGPT.chat_with_gpt3(messages)
    messages.append({"role": "assistant", "content": response})
    return response

def play_audio(filename):
    audio = AudioSegment.from_file(filename, format="mp3")
    play(audio)
def main():
    while True:
        filename = record_audio()
        transcription = transcribe_audio(filename)
        if transcription is not None:
            response = interpret_command(transcription)
            print(response)
            # Convert the response to speech
            text_to_speech(response, "response.mp3")

            # Convert the response from mp3 to wav
            convert_mp3_to_wav("response.mp3", "response.wav")

            # Play the speech
            play_audio("response.wav")


if __name__ == "__main__":
    main()