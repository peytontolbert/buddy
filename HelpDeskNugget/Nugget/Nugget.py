import requests
from factory.factory import AgentFactory

class NuggetAGI:
    def __init__(self):
        self.messagequeue = []
        self.users = []
        self.standard_prompt = """This is a conversation between an AI assistant and a user."""
        self.conversationhistory = []
        self.factory = AgentFactory()
        pass

    def run(self):
        print("Checking messages...")
        self.check_messages()
        while len(self.messagequeue) > 0:
            message = self.messagequeue.pop(0) #Get the next message
            self.handle_message(message)


    def handle_message(self, message):
        user_id = message['user']
        if message['user'] not in self.users:
            print(f"New user: {message['user']}")
            self.users.append(message)
            self.start_conversation(message)
            print("New conversation started")
        print(f"Processing message: {message}")  

    def check_messages(self):
        messages = ""
        response = requests.get("http://localhost:5000/nuggetchat")
        if response.status_code == 200:
            response_data = response.json()
            for message in response_data['messages']:
                self.messagequeue.append(message)
                print(message)
        else:
            print("no new messages")
        return
    
    def start_conversation(self,message):
        self.conversationhistory.append({"user":message['user'], "messages":[{"role":"system","message":self.standard_prompt},{"role":"user","message":message['message']}]})
        return        
