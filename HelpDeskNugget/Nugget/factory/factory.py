import openai
import os
from dotenv import load_dotenv
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
llm_model = os.getenv("LLM_MODEL")

class AgentFactory:
    def __init__(self):
        self._creators = {}

    def run_conversation(self,messages):
        prompt={"role":"system","message":"Your job is only decide if this is a task or a question. Simply respond 'task' or 'question'."}
        messages.append(prompt)
        response = openai.ChatCompletion.create(model="gpt-3.5-turbo", max_tokens=500, messages=messages)
        if response[0]['choices'] # if the choice is a task, send to task agent, otherwise send to question agent

class TaskAgent:


