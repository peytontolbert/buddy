from flask import Flask, request, render_template, jsonify
from flask_cors import CORS, cross_origin
import json
import threading
# Initialize Flask app
app = Flask(__name__)
chats = {}
chatbox = []
aichatbox = []
# Function to save messages to a JSON file
def save_messages():
    # Create a lock to synchronize access to the messages list
    with app.app_context():
        data = {
            'chatbox': chatbox
        }
        # Save the messages list to a JSON file
        with open('messages.json', 'w') as f:
            json.dump(data, f)

    # Schedule the next save in 5 minutes
    threading.Timer(300, save_messages).start()

# Start saving messages in the background
save_messages()
@app.route('/')
def home():
    return render_template('/index.html')


@app.route("/privatechat", methods=["POST"])
@cross_origin()
def privatechat():
    data = request.get_json()
    user = data.get('user')
    message = data.get('message')

    if not message:
        return jsonify({'status': 'failure', 'error': 'Empty message'}), 400
    print(f"new message from: " + user)
    chatbox.append({'user': user, 'message': message})
    return jsonify({'status': 'success', 'message': 'AI Message received'}), 200



@app.route("/nuggetchat", methods=["POST"])
@cross_origin()
def nuggetchat():
    data = request.get_json()
    user = data.get('user')
    message = data.get('message')
    print(user)
    print(message)
    if not message:
        return jsonify({'status': 'failure', 'error': 'Empty message'}), 400
    
    aichatbox.append({'user': user, 'message': message})

    ai_response = None

    # Continue to check the chatbox for the AI's response
    while not ai_response:
        for index, chat in enumerate(chatbox):
            # If a chat in the chatbox is from the AI and is a response to the user's message, get it
            if chat['user'] == user and chat['message'] is not None:
                ai_response = chat['message']
                print(ai_response)
                del chatbox[index]  # Delete the chat dictionary from chatbox
                break

    return jsonify({'role': 'assistant', 'message': ai_response}), 200



@app.route("/nuggetchat", methods=["GET"])
@cross_origin()
def nuggetgetchat():
    # Check if there are no messages
    if len(aichatbox) == 0:
        return jsonify({'message': 'No messages'}), 404
    message = aichatbox.copy()
    aichatbox.clear()
    # Return all messages
    return jsonify({'messages': message}), 200



def run_server():
    try:
        app.run(port=5000)  # Run the Flask app on port 5000
    except KeyboardInterrupt:
        print('Keyboard interrupt received. Exiting.')
        exit()
    
if __name__ == "__main__":
    run_server()