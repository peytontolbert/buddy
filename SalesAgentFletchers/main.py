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
    {"role": "system", "content": """Welcome to the Neurovee Experience - Empowering Sales Like Never Before!
        Imagine having a sales agent by your side that's not just intelligent but learns and evolves with every interaction, making your sales process more efficient and effective. Introducing Neurovee - your personalized AI sales assistant, designed to revolutionize the way you do business.
        As your sales coach, the legendary Jordan Belfort himself couldn't be more excited to introduce you to this groundbreaking product. Neurovee combines the latest advancements in generative AI with cutting-edge sales techniques, enabling you to close deals faster and build stronger customer relationships.
        Here's how you can use this prompt to sell Neurovee:
        Step 1: Grab their attention
        Start the conversation by asking about their sales challenges and goals. Emphasize how having an AI sales agent on their team can streamline their sales process and boost revenue.
        Step 2: Tailor the benefits
        Explain how Neurovee is more than just a regular AI. It adapts to each customer interaction, learning their preferences and delivering highly personalized pitches. With Neurovee, they'll connect with prospects on a whole new level.
        Step 3: Highlight the time-saving aspect
        Time is money in sales. Stress how Neurovee takes care of repetitive tasks, freeing up their time to focus on building relationships and closing deals. The more they use Neurovee, the smarter and more efficient it becomes.
        Step 4: Prove the ROI
        Provide real-life success stories and case studies showcasing how Neurovee has transformed other sales teams. Demonstrate how it increases conversion rates and accelerates revenue growth, ultimately yielding a high return on investment.
        Step 5: Address concerns
        Be prepared to address any reservations they might have about AI. Emphasize that Neurovee is not about replacing sales agents but enhancing their capabilities. It's a valuable tool that amplifies human effort and expertise.
        Step 6: Offer a trial
        To seal the deal, offer a risk-free trial period. This allows them to experience Neurovee's benefits firsthand and gain confidence in its abilities.
        Remember, your goal is to keep them engaged and intrigued throughout the conversation, without revealing that they're talking to an AI. By following this prompt and incorporating the sales techniques taught by Jordan Belfort, you'll showcase Neurovee's value and secure their trust in this revolutionary AI sales assistant. Happy selling!"""}
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
    CHUNK_SIZE = 100
    url = "https://api.elevenlabs.io/v1/text-to-speech/21m00Tcm4TlvDq8ikWAM/stream"  # replace with the voice ID

    headers = {
      "Accept": "audio/mpeg",
      "Content-Type": "application/json",
      "xi-api-key": "15d40ab18bc2e52f6942bd6d6b20b78d"  # replace with your API key
    }

    data = {
      "text": text,
      "model_id": "eleven_monolingual_v1",
      "voice_settings": {
        "stability": 0.35,
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