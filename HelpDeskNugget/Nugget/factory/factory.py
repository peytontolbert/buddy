import openai
import os
from dotenv import load_dotenv
from gpt.chatgpt import ChatGPT
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
llm_model = os.getenv("LLM_MODEL")

class Factory:
    def __init__(self):
        self._creators = {}

    def run_conversation(self,messages=[],results=None):
        og_messages = messages.copy()
        prompt={"role":"system","content":"Your job is only decide if this is a task or a question. Simply respond 'task' or 'question'."}
        checkmessages = messages.append(prompt)
        print(messages)
        response = ChatGPT.chat_with_gpt3(messages)
        print(response)
        if "question" in response:
            results = self.run_question(og_messages)
        elif "task" in response:
            results = self.run_task(og_messages)
        else:
            results = self.run_clarify(og_messages)
        return results
    
    def run_question(self, question):
        print(question)
        response = ChatGPT.chat_with_gpt3(question)
        return response
    
    def run_task(self, task):
        response = ChatGPT.chat_with_gpt3(task)
        return response
    
    def run_clarify(self, clarify):
        question={"role":"assistant","content":"I am not sure what you mean. Please clarify."}
        clarify+=question
        return clarify
    


class AgentFactory:
    def create_agent(self, task):
        agent = Agent(task)
        return agent
    
class Agent:
    def __init__ (self, goal):
        self.goal = goal
        self.tasklist = []

    def create_task(self, task):
        prompt = """You are an AI agent who makes step-by-step plans to solve a problem under the help of external tools.
        For each step, make one plan followed by one tool-call, which will be executed later to retrieve evidence for that step.
        You should store each evidence into a distinct variable #E1, #E2, #E3 ... that can be referred to in later tool-call inputs.
        
        ###Available Tools###
        {tool_description}
        
        ###Output Format (Replace '<...>')###
        #Plan1: <describe your plan here>
        #E1: <toolname>[<input here>] (eg. Search[What is Python])
        #Plan2: <describe your plan here>
        #E2: <toolname>[<input here, you can use #E1 to represent its expected output>]
        And so on...

        ###Your Task###
        {task}

        ###Now Begin###
        """
