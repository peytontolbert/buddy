from langchain.agents import initialize_agent, AgentType
from langchain.chat_models import ChatOpenAI
from typing import List
import os
from dotenv import load_dotenv
load_dotenv()


class Agents:
    def __init__(self):
        OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
        LLM_Model = os.getenv("OPENAI_API_KEY", "")
        self.llm = ChatOpenAI(max_retries=3, temperature=0, model_name=LLM_Model)


    def initialize_agent_with_new_openai_functions(self, tools: List, is_agent_verbose: bool = True, max_iterations: int = 3, return_thought_process: bool = False):
        agent = initialize_agent(tools, self.llm, agent=AgentType.OPENAI_FUNCTIONS, verbose=is_agent_verbose, max_iterations=max_iterations, return_intermediate_steps=return_thought_process)

        return agent