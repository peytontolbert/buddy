import openai
import os
from dotenv import load_dotenv
import time
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

class ChatGPT:
    def __init__(self):
        pass
    @staticmethod
    def chat_with_gpt3(system_prompt, prompt):
        system_prompt = str(system_prompt)
        prompt = str(prompt)
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}]
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