import requests

class message:
    def __init__(self):
        pass
    
    def message_creator(self, string):
        try:
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
        except requests.exceptions.RequestException as e:
            print(f"An error occurred: {e}")

    def message_finn(self, string):
        try:
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
        except requests.exceptions.RequestException as e:
            print(f"An error occurred: {e}")

    def read_chat(self):
        try:
            print("Reading chat....")
            response = requests.get('http://localhost:5000/chat')
            print(response.text)
            return response.text
        except requests.exceptions.RequestException as e:
            print(f"An error occurred: {e}")

    def read_dm(self):
        try:
            print("Reading private messages....")
            response = requests.get('http://localhost:5000/buddydm')
            print(response.text)
            return response.text
        except requests.exceptions.RequestException as e:
            print(f"An error occurred: {e}")

    def post_chat(self, string):
        try:
            string_string = str(string)
            data={"user": "Buddy", "message": string_string}
            response = requests.post('http://localhost:5000/chat', json=data)
            if response.status_code == 200:
                print("Message sent successfully")
            else:
                print("Failed to send message, status code: ", response.status_code)
            return response
        except requests.exceptions.RequestException as e:
            print(f"An error occurred: {e}")