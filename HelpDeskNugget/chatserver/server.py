from flask import Flask, request, render_template, jsonify
import json
import threading
# Initialize Flask app
app = Flask(__name__)

messages = []
chatbox = []
unreaddm = []
finndm = []
buddydm = []
buddymessages = []
creatordm = []
nuggetchatbox = []
oldmessages = []
# Function to save messages to a JSON file
def save_messages():
    # Create a lock to synchronize access to the messages list
    with app.app_context():
        data = {
            'messages': messages,
            'chatbox': chatbox,
            'nuggetchatbox': nuggetchatbox,
            'unreaddm': unreaddm,
            'buddydm': buddydm,
            'creatordm': creatordm,
            'oldmessages': oldmessages
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

@app.route("/nuggetchat", methods=["POST"])
def nuggetchat():
    data = request.get_json()
    user = data.get('user')
    message = data.get('message')
    print(user)
    print(message)
    if not message:
        return jsonify({'status': 'failure', 'error': 'Empty message'}), 400
    
    nuggetchatbox.append({'user': user, 'message': message})
    return jsonify({'status': 'success', 'message': 'Message received'}), 200

@app.route("/nuggetchat", methods=["GET"])
def nuggetgetchat():
    # Check if there are no messages
    if len(nuggetchatbox) == 0:
        return jsonify({'message': 'No messages'}), 404
    message = nuggetchatbox.copy()
    nuggetchatbox.clear()
    # Return all messages
    return jsonify({'messages': message}), 200



def run_server():
    app.run(port=5000)  # Run the Flask app on port 5000

    
if __name__ == "__main__":
    run_server()