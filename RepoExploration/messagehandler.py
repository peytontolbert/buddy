from chatgpt import ChatGPT
from SerpAPIWrapper import SerpAPIWrapper
class MessageHandler:
    def __init__(self):
        pass
    def handle_message(self, message):
        response = ChatGPT.chat_with_gpt3("Answer the question if it is a simple question, otherwise  reply back with the answer 'task'", message)
        print(response)
        if response == "task":
            agent, reason = self.agent_selection(message)
            # Parse agent for `if` statements
            if agent == "Coding Agent":
                return self.codingAgent(message, reason)
            elif agent == "Google Agent":
                return self.searchAgent(message, reason)
            elif agent == "Brainstorm Agent":
                return self.brainstormAgent(message, reason)
            elif agent == "Multi-tool Agent":
                return self.multiToolAgent(message, reason)
            else:
                return "Unknown agent selected"
                
        else:
            return response
        

    def agent_selection(self, message, context=None):
        BASE_TEMPLATE = """
        You are an Agent-selection tool.
        Your mission is to analyze the user's request and decide which execution agent should handle it to best to help users with their tasks.

        [USER'S PROMPT]
        {user_prompt}

        [AGENT OPTIONS]
        Coding Agent - used for coding
        Google Agent - used to google 
        Brainstorm Agent - used to prompt with an llm 
        Multi-tool Agent - more than one tool, effective for complex tasks

        [INSTRUCTIONS]
        Analyze the user's prompt, objective, and context. Then, choose the most appropriate agent(s) to handle this request and explain why. If Multi-tool Agent is chosen, specify which agents it should combine.

        [RESPONSE FORMAT]
        Return the selected agent and reasoning as a dict, formatted as follows:

        Selected Agent: {Selected Agent}
        Reason: {Reason for selecting this agent}

        If Multi-tool Agent is selected, specify which agents it should combine.

        [EXAMPLE]
        User: 'Help me brainstorm ideas for a startup.'
        Assistant: 
        Selected Agent: Brainstorm Agent
        Reason: Brainstorm Agent is the best agent to handle this request because it can help the user brainstorm ideas for a startup.


        [RESPONSE]

        """

        response = ChatGPT.chat_with_gpt3("Select the best agent to handle the user's request and explain why. Follow the response format.", BASE_TEMPLATE)
        # Split the response by line and then by ': ' to get the key-value pairs
        response_dict = {line.split(": ")[0]: line.split(": ")[1] for line in response.strip().split("\n")}

        selected_agent = response_dict.get("Selected Agent", None)
        reason = response_dict.get("Reason", None)

        # Do something with the selected_agent and reason
        print(f"Selected Agent: {selected_agent}")
        print(f"Reason: {reason}")

        return selected_agent, reason
    
    def codingAgent(self, message, reason=None):
        return "Filler this will respond with code."
    
    def searchAgent(self, message, reason=None):
        serpapi = SerpAPIWrapper()
        response = serpapi.run(message)
        return response
    
    def brainstormAgent(self, message, reason=None):
        return "Filler this will respond with brainstorming."
    
    def multiToolAgent(self, message, reason=None):
        return "Filler this will respond with multi-tool."