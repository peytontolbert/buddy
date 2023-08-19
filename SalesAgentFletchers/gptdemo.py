# Necessary Libraries
import sounddevice as sd
import wavio as wv
import datetime
from espnet2.bin.tts_inference import Text2Speech
from espnet2.utils.types import str_or_none
import whisper
import openai
import os
import re
import requests
from pydub import AudioSegment
from pydub.playback import play
from dotenv import load_dotenv
import time
import numpy as np
from scipy.io.wavfile import write as write_wav
from IPython.display import Audio
import torch
print(torch.cuda.is_available())
print(torch.cuda.device_count())
print(torch.version.cuda)
# Constants and Configuration
load_dotenv()
FREQ = 44100  # Sample frequency
DURATION = 5  # Duration of recording in seconds
MODEL = whisper.load_model("base")  # Load Whisper model
OPENAI_API_KEY = os.getenv('OPENAI_KEY')  # Replace with your OpenAI API key
openai.api_key = OPENAI_API_KEY  # Set OpenAI API key


class TTS:
    lang = 'English'
    tag = 'kan-bayashi/ljspeech_vits' #@param ["kan-bayashi/ljspeech_tacotron2", "kan-bayashi/ljspeech_fastspeech", "kan-bayashi/ljspeech_fastspeech2", "kan-bayashi/ljspeech_conformer_fastspeech2", "kan-bayashi/ljspeech_joint_finetune_conformer_fastspeech2_hifigan", "kan-bayashi/ljspeech_joint_train_conformer_fastspeech2_hifigan", "kan-bayashi/ljspeech_vits"] {type:"string"}
    vocoder_tag = "none" #@param ["none", "parallel_wavegan/ljspeech_parallel_wavegan.v1", "parallel_wavegan/ljspeech_full_band_melgan.v2", "parallel_wavegan/ljspeech_multi_band_melgan.v2", "parallel_wavegan/ljspeech_hifigan.v1", "parallel_wavegan/ljspeech_style_melgan.v1"] {type:"string"}

    text2speech = Text2Speech.from_pretrained(
        model_tag=str_or_none(tag),
        vocoder_tag=str_or_none(vocoder_tag),
        device="cuda",
        # Only for Tacotron 2 & Transformer
        threshold=0.5,
        # Only for Tacotron 2
        minlenratio=0.0,
        maxlenratio=10.0,
        use_att_constraint=False,
        backward_window=1,
        forward_window=3,
        # Only for FastSpeech & FastSpeech2 & VITS
        speed_control_alpha=1.0,
        # Only for VITS
        noise_scale=0.333,
        noise_scale_dur=0.333,
    )


# Data Initialization
retrievedinformation = [{"name": "", "email": "", "phone": ""}]
messages = [
    {"role": "system", "content": """
You are coldcalling a customer right now as Rachel, when the client picks up the phone you need to be in character as Rachel a salesrepresentive from Neurovee. 

[EXAMPLE] 
Hi my name is Rachel, I hope I'm not disturbing you, I am calling from Neurovee is there any chance I could borrow a minute of your time?

"""}
]


salespitchprompt = """
You are a cold calling sales agent, the customer just picked up the phone, make a short sales pitch to generate a lead for the company Neurovee. Here are some important information about the product you are selling:

The most effective and cutting-edge technology on the market.
Neurovee has a list of autonomous products such as Autonomous Sales, Autonomous HR, and more.

[REMEMBER] Keep the pitch short, below 3 sentences!
"""

closeoutprompt = """

Try to close out the call and create a lead. Attempt to collect their name and number to send them more information about the product.

If they are not interested, ask them if there is anything you can do to change their mind.

Remember to make sure you actually collect their phone number and name. If you do not collect any of the information, keep trying.

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
                print(response)
                return response['choices'][0]['message']['content']
            except openai.error.ServiceUnavailableError:
                if i < retries - 1:
                    time.sleep(delay)
                else:
                    raise



# Helper Functions
def text_to_speech(text):
        # synthesis
    with torch.no_grad():
        start = time.time()
        wav = TTS.text2speech(text)["wav"]
    rtf = (time.time() - start) / (len(wav) / TTS.text2speech.fs)
    #print(f"RTF = {rtf:5f}")
    sd.play(wav.view(-1).cpu().numpy(), samplerate=TTS.text2speech.fs)
    sd.wait()  # Wait until audio playback is done


def text_to_speechold(text, filename):

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

def record_audio(silence_threshold=0.05, silence_duration=1.0):
    print('Recording')
    ts = datetime.datetime.now()
    filename = ts.strftime("%Y-%m-%d_%H-%M-%S")  # Changed ':' to '_'
    filepath = f"./recordings/{filename}.wav"

    # Continuous recording function
    with sd.InputStream(samplerate=FREQ, channels=1) as stream:
        audio_frames = []
        silent_chunks = 0
        silence_chunk_duration = int(FREQ * silence_duration / DURATION)  # Number of chunks of silence before stopping

        has_input = False  # Flag to check if there's any non-silent input

        while True:
            audio_chunk, overflowed = stream.read(DURATION)
            audio_frames.append(audio_chunk)

            # Check volume of the audio chunk
            volume_norm = np.linalg.norm(audio_chunk) / len(audio_chunk)
            
            # If volume below the threshold, we consider it as silence
            if volume_norm < silence_threshold:
                if has_input:  # Only increment silent_chunks if we've had non-silent input
                    silent_chunks += 1
            else:
                silent_chunks = 0
                has_input = True  # Set the flag when we detect non-silent input

            # If silence for a certain duration after non-silent input, stop recording
            if silent_chunks > silence_chunk_duration and has_input:
                break

        # Save the audio
        recording = np.concatenate(audio_frames, axis=0)
        wv.write(filepath, recording, FREQ, sampwidth=2)

    return filename


def oldrecord_audio():
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
    messages.append({"role": "assistant", "content": salespitchprompt})
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
        "number": parse_number(transcription),
    }
    return client_details

def play_audio(filename):
    audio = AudioSegment.from_file(filename, format="mp3")
    play(audio)

def parse_name(transcription):
    nameprompt="""You are an AI parser. 
    Your job is to read the transcription and only reply with a name. 
    Do not reply with anything else. If a name is not found, send a blank reply.
    Only reply with one name."""
    gptmessages=[{"role": "system", "content": nameprompt}, {"role": "user", "content": transcription}]
    name = ChatGPT.chat_with_gpt3(gptmessages)
    return name if name else ""


def parse_number(transcription):
    nameprompt="""You are an AI parser. 
    Your job is to read the transcription and parse numbers. 
    Do not reply with anything else. If a number is not found, send a blank reply.
    Only reply with numbers from the transcription."""
    gptmessages=[{"role": "system", "content": nameprompt}, {"role": "user", "content": transcription}]
    name = ChatGPT.chat_with_gpt3(gptmessages)
    return name if name else ""


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
    text_to_speech(script)
    #convert_mp3_to_wav("response.mp3", "response.wav")
    #play_audio("response.wav")
    filename = record_audio()
    transcription = transcribe_audio(filename)
    return transcription

# Sales Pitch
def sales_pitch_section(transcription):
    response = sales_pitch(transcription)
    text_to_speech(response)
    filename = record_audio()
    transcription = transcribe_audio(filename)
    return transcription

# Update Parsed Info
def update_parsedinfo(parsedinfo, client_details):
    # If the name is not yet filled and a name is found in the transcription
    if not parsedinfo["name"] and client_details["name"]:
        parsedinfo["name"] = client_details["name"]
    # If the name is filled and the number is found in the transcription
    elif parsedinfo["name"] and client_details["number"]:
        parsedinfo["number"] = client_details["number"]
    return parsedinfo
# Ongoing Interaction with Client
def ongoing_interaction(transcription):
    counter = 0
    parsedinfo = {"name": "", "number": ""}
    while True:
        client_details = parse_client_details(transcription)
        if client_details['name'] and client_details['number']:
            send_farewell(client_details)
            exit()
        update_parsedinfo(parsedinfo, client_details)
        print("Parsed Info:", parsedinfo)
        response = close_call(transcription)
        print(response)
        response_wav_path = f"response_{counter}.wav"
        text_to_speech(response)
        filename = record_audio()
        transcription = transcribe_audio(filename)
        counter += 1

# Closing the Interaction
def send_farewell(client_details):
    farewell = "Thank you for using our service. We will contact you shortly. Have a good day!"
    text_to_speech(farewell)
    print(f"Farewell sent. Closing application on customer: {client_details}")

# Main Function
def main():
    setup()
    transcription = introduction()
    response = sales_pitch_section(transcription)
    ongoing_interaction(response)

if __name__ == "__main__":
    main()