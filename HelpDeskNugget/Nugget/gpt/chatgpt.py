import openai
import os
from dotenv import load_dotenv
import time
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

class ChatGPT:
    def __init__(self):
        pass

    def process_thought(self, thought, emotional_state):
        system_prompt = """I am an artifical cognitive entity.
        I need to process my thoughts to align with my goal and emotional state.
        Only reply with a processed thought."""
        prompt = """{thought}
        my emotional state is:
        {emotional_state}
        """
        response = self.chat_with_gpt3(system_prompt, prompt.format(thought=thought, emotional_state=emotional_state))
        print(response)
        return response

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