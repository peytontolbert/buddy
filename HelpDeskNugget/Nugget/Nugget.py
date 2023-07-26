import requests
from factory.factory import Factory

class NuggetAGI:
    def __init__(self):
        self.messagequeue = []
        self.users = []
        self.standard_prompt = """This is a conversation between an AI assistant and a user."""
        self.conversationhistory = {}
        self.factory = Factory()

    def run(self):
        print("Checking messages...")
        self.check_messages()
        while len(self.messagequeue) > 0:
            message = self.messagequeue.pop(0) #Get the next message
            self.handle_message(message)


    def handle_message(self, message):
        user_id = message['user']
        if user_id not in self.users:
            print(f"New user: {message['user']}")
            self.users.append(message)
            self.start_conversation(message)
            print("New conversation started")
        print(f"Processing message: {message}")
        user_conversation = self.conversationhistory[user_id]
        response = self.factory.run_conversation(user_conversation)
        print(response)
        self.send_message(user_id,response)
        #I need to get the conversation of the user and send to the factory

    def send_message(self, user_id, message):
        print(user_id)
        messages = {"user":user_id,"message":message}
        requests.post("http://localhost:5000/privatechat",
                      json=messages)

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
        user_id = message['user']
        self.conversationhistory[user_id] = [
            {"role":"system","content":self.standard_prompt},
            {"role":"user","content":message['message']}
        ]
        return        
