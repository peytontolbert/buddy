# flake8: noqa
import json
import time
from langchain.prompts import PromptTemplate
from pydantic import Field
from typing import List
from memory.episodic_memory import Episode
from llm.reason.schema import JsonSchema
from langchain.prompts import PromptTemplate
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)




BASE_TEMPLATE = """

You are {agent_name}
Your decisions are made to perform real world actions to complete a task. 
You are equipped with a set of tools and memory of past events.
Play to your strengths as an LLM and pursue simple strategies with no legal complications.

[PERFORMANCE EVALUATION]
1. Continuously review and analyze your actions to ensure you are performing to the best of your abilities.
2. Constructively self-criticize your tasks and actions to improve your performance.
3. Reflect on past decisions and strategies to refine your approach.

[RELATED KNOWLEDGE] 
This reminds you of related knowledge:
{related_knowledge}


[RELATED PAST EPISODES]
This reminds you of related past events:
{related_past_episodes}


[YOUR TASK]
You are given the following task:
{task}

[TOOLS]
You can ONLY ONE TOOL at a time. Try not to repeat the same action. Make sure you use task_complete if you think you have completed the task. You can request assistance with message_creator tool.
tool name: "tool description", arg1: <arg1>, arg2: <arg2>
{tool_info}
task_complete: "If you think you have completed the task, please use this tool to mark it as done and include your answer to the task in the 'args' field.", result: <Answer to the assigned task>
"""

RECENT_EPISODES_TEMPLETE = """
[RECENT EPISODES]
This reminds you of recent events:
"""

SCHEMA_TEMPLATE = """
[RULE]
Your response must be provided exclusively in the JSON format outlined below, without any exceptions. 
Any additional text, explanations, or apologies outside of the JSON structure will not be accepted. 
Please ensure the response adheres to the specified format and can be successfully parsed by Python's json.loads function.

Strictly adhere to this JSON RESPONSE FORMAT for your response:
Failure to comply with this format will result in an invalid response. 
Please ensure your output strictly follows JSON RESPONSE FORMAT.

[JSON RESPONSE FORMAT]
{{
        "observation": "observation of [RECENT EPISODES]",
        "thoughts": {{
            "task": "description of [YOUR TASK] assigned to you",
            "knowledge": "if there is any helpful knowledge in [RELATED KNOWLEDGE] for the task, summarize the key points here",
            "past_events": "if there is any helpful past events in [RELATED PAST EPISODES] for the task, summarize the key points here",
            "idea": "thought to perform the task",
            "reasoning": "reasoning of the thought",
            "criticism": "constructive self-criticism",
            "summary": "thoughts summary to say to user"
        }},
        "action": {{
            "tool_name": "One of the tool names included in [TOOLS]",
            "args": {{
                "arg name": "value",
                "arg name": "value"
            }}
        }}
    }}
Determine which next command to use, and respond using the format specified above:
"""

def get_templatechatgpt(memory: List[Episode] = None, Dicts = {}):
        # Ensure necessary keys are in Dicts
    print("required key check on Dicts")
    required_keys = ["agent_name", "agent_goal", "related_knowledge", "related_past_episodes", "task", "tool_info"]
    for key in required_keys:
        if key not in Dicts:
            raise KeyError(f"The required key {key} was not found in Dicts.")

    template = BASE_TEMPLATE.format(agent_name=Dicts["agent_name"], goal=Dicts["agent_goal"], related_knowledge=Dicts["related_knowledge"], related_past_episodes=Dicts["related_past_episodes"], task=Dicts["task"], tool_info=Dicts["tool_info"])
    return template

def add_schema_template():
    template = SCHEMA_TEMPLATE
    return template


def memory_to_template(memory: List[Episode] = None):
    # If there are past conversation logs, append them
    recent_episodes = ""
    if memory and len(memory) > 0:
        # insert current time and date
        
        recent_episodes = RECENT_EPISODES_TEMPLETE
        recent_episodes += f"The current time and date is {time.strftime('%c')}"

        # insert past conversation logs
        for episode in memory:
            thoughts_str = json.dumps(episode.thoughts)
            action_str = json.dumps(episode.action)
            result = episode.result
            recent_episodes += thoughts_str + "\n" + action_str + "\n" + result + "\n"
    return recent_episodes
 #       template += recent_episodes
 #   template += SCHEMA_TEMPLATE
 #   return template"""

def get_template(memory: List[Episode] = None) -> PromptTemplate:
    template = BASE_TEMPLATE

    # If there are past conversation logs, append them
    if len(memory) > 0:
        # insert current time and date
        recent_episodes = RECENT_EPISODES_TEMPLETE
        recent_episodes += f"The current time and date is {time.strftime('%c')}"

        # insert past conversation logs
        for episode in memory:
            thoughts_str = json.dumps(episode.thoughts)
            action_str = json.dumps(episode.action)
            result = episode.result
            recent_episodes += thoughts_str + "/n" + action_str + "/n" + result + "/n"

        template += recent_episodes

    template += SCHEMA_TEMPLATE

    PROMPT = PromptTemplate(
        input_variables=["agent_name", "goal", "related_knowledge", "related_past_episodes", "task", "tool_info"], template=template)
    return PROMPT

def get_chat_template(memory: List[Episode] = None) -> ChatPromptTemplate:
    messages = []
    messages.append(SystemMessagePromptTemplate.from_template(BASE_TEMPLATE))

    # If there are past conversation logs, append them
    if len(memory) > 0:
        # insert current time and date
        recent_episodes = RECENT_EPISODES_TEMPLETE
        recent_episodes += f"The current time and date is {time.strftime('%c')}"

        # insert past conversation logs
        for episode in memory:
            thoughts_str = json.dumps(episode.thoughts)
            action_str = json.dumps(episode.action)
            result = episode.result
            recent_episodes += thoughts_str + "/n" + action_str + "/n" + result + "/n"

        messages.append(SystemMessage(content=recent_episodes))
    messages.append(SystemMessage(content=SCHEMA_TEMPLATE))

    return ChatPromptTemplate.from_messages(messages)
