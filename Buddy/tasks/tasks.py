import json
import openai
from pydantic import BaseModel, Field
from pydantic import BaseModel, Field
from langchain.llms.base import BaseLLM
from typing import List, Any
from langchain import LLMChain
from llm.generate_task_plan.prompt import get_template
from llm.list_output_parser import LLMListOutputParser


class Task(BaseModel):
    """Task model."""
    id: int = Field(..., description="Task ID")
    description: str = Field(..., description="Task description")
    is_done: bool = Field(False, description="Task done or not")
    result: str = Field("", description="The result of the task")


class TaskManager(BaseModel):
    """Task manager model."""
    tasks: List[Task] = Field([], description="The list of tasks")
    current_task_id: int = Field(1, description="The last task id")

    def generate_task_plan(self, agent_name: str, goal: str, thought: str, last_task=None):
        """Generate a task plan for the agent."""
        propmt = get_template()                
        BASE_TEMPLATE = """
        You are {agent_name}
        Your should create task that uses the result of an execution agent
        to create a new task with the following GOAL:

        [GOAL]
        {goal}

        [LAST TASK I DID]
        {last_task}

        [THOUGHTS]
        {thought}

        [YOUR MISSION]
        Based on the [GOAL], create new task to be completed by the AI system that do not overlap with incomplete tasks.
        - Tasks should be calculated backward from the GOAL, and effective arrangements should be made.
        - You can create any number of tasks.

        [RESPONSE FORMAT]
        Return the tasks as a list of string.
        - Enclose each task in double quotation marks.
        - Separate tasks with Tabs.
        - Reply in first-person.
        - Use [] only at the beginning and end

        ["Task 1 that I should perform"\t"Task 2 that I should perform",\t ...]

        [RESPONSE]
        """
        chat_input = BASE_TEMPLATE.format(thought=thought, agent_name=agent_name, goal=goal, last_task=last_task)
        try:
            
            results = openai.ChatCompletion.create(model="gpt-3.5-turbo-16k", messages=[{"role": "system", "content": chat_input}])
            result =  str(results['choices'][0]['message']['content'])
            print(result)
        except Exception as e:
            raise Exception(f"Error: {e}")

        # Parse and validate the result
        try:
            result_list = LLMListOutputParser.parse(result, separeted_string="\t")
        except Exception as e:
            raise Exception("Error: " + str(e))

        # Add tasks with a serial number
        for i, e in enumerate(result_list, start=1):
            id = int(i)
            description = e
            self.tasks.append(Task(id=id, description=description))

        self

    def get_task_by_id(self, id: int) -> Task:
        """Get a task by Task id."""
        for task in self.tasks:
            if task.id == id:
                return task
        return None

    def get_current_task(self) -> Task:
        """Get the current task agent is working on."""
        return self.get_task_by_id(self.current_task_id)

    def get_current_task_string(self) -> str:
        """Get the current task agent is working on as a string."""
        task = self.get_current_task()
        if task is None:
            return None
        else:
            return self._task_to_string(task)

    def complete_task(self, id: int, result: str) -> None:
        """Complete a task by Task id."""
        # Complete the task specified by ID
        if id > 0 and id <= len(self.tasks):
            self.tasks[id - 1].is_done = True
            self.tasks[id - 1].result = result
            self.current_task_id += 1
        else:
            print(f"Task with id {id} does not exist")

    def complete_current_task(self, result: str) -> None:
        """Complete the current task agent is working on."""
        self.complete_task(self.current_task_id, result=result)

    def _task_to_string(self, task: Task) -> str:
        """Convert a task to a string."""
        return f"{task.id}: {task.description}"

    def get_incomplete_tasks(self) -> List[Task]:
        """Get the list of incomplete tasks."""
        return [task for task in self.tasks if not task.is_done]

    def get_incomplete_tasks_string(self) -> str:
        """Get the list of incomplete tasks as a string."""
        result = ""
        for task in self.get_incomplete_tasks():
            result += self._task_to_string(task) + "\n"
        return result
