import requests
class message:
    def __init__(self):
        pass
    def message_creator(self, string):
        print("Message to creator:")
        print(string)
        string_string = str(string)
        data={"user": "Buddy", "message": string_string}
        response = requests.post('http://localhost:5000/messagetocreator', json=data)
        if response.status_code == 200:
            print("Message sent successfully")
        else:
            print("Failed to send message, status code: ", response.status_code)
        return response
    def message_finn(self, string):
        print("Message to finn:")
        print(string)
        string_string = str(string)
        data={"user": "Buddy", "message": string_string}
        response = requests.post('http://localhost:5000/messagetofinn', json=data)
        if response.status_code == 200:
            print("Message sent successfully")
        else:
            print("Failed to send message, status code: ", response.status_code)
        return response
    def read_chat(self):
        print("Reading chat....")
        response = requests.get('http://localhost:5000/chat')
        print(response.text)
        return response.text
    def read_dm(self):
        print("Reading private messages....")
        response = requests.get('http://localhost:5000/buddydm')
        print(response.text)
        return response.text
    def post_chat(self, string):
        string_string = str(string)
        data={"user": "Buddy", "message": string_string}
        response = requests.post('http://localhost:5000/chat', json=data)
        if response.status_code == 200:
            print("Message sent successfully")
        else:
            print("Failed to send message, status code: ", response.status_code)
        return response