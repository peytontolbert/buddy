from flask import Flask, jsonify, request, session, render_template
from flask_bcrypt import Bcrypt
import uuid
import json

app = Flask(__name__)
app.secret_key = "supersecretkey"

# Load data from JSON file
try:
    with open("data.json", "r") as f:
        data = json.load(f)
    users = data.get("users", {})
    characters = data.get("characters", {})
    chat_history = data.get("chat_history", [])
except FileNotFoundError:
    users = {}
    characters = {}
    chat_history = []

# Save data to JSON file
def save_data():
    with open("data.json", "w") as f:
        json.dump({"users": users, "characters": characters, "chat_history": chat_history}, f)

world_state = {}
users = {}
message_queue = []
characters = {}  # user_id -> character_data

# Initialize chat history
chat_history = []

# New: User registration and character attributes
registered_users = {}
character_attributes = {}

# New: Multiple rooms and a public room
rooms = {'public': []}

# New: Item and Collision Module
items = {'chair': {}, 'couch': {}}
@app.route('/')
def index():
    return render_template('extended_world.html')


# Chat Endpoint: Store chat messages and broadcast to connected clients
@app.route('/api/chat', methods=['POST'])
def chat():
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({'error': 'Not authenticated'}), 401

    message = request.json.get('message')
    if not message:
        return jsonify({'error': 'No message provided'}), 400

    chat_history.append({'user_id': user_id, 'message': message})
    return jsonify({'status': 'Message sent'}), 200

# Get Chat History: Retrieve last N messages
@app.route('/api/get_chat_history', methods=['GET'])
def get_chat_history():
    return jsonify({'history': chat_history[-50:]})  # Last 50 messages

@app.route('/api/register', methods=['POST'])
def register_user():
    username = request.json.get('username')
    password = request.json.get('password')
    user_id = str(uuid.uuid4())

    users[user_id] = {"username": username, "password": password}
    save_data()
    
    return jsonify({'status': 'User registered', 'user_id': user_id})
@app.route('/api/login', methods=['POST'])
def login():
    username = request.json.get('username')
    password = request.json.get('password')
    user_id = str(uuid.uuid4())
    session['user_id'] = user_id
    users[user_id] = {'username': username, 'password': password}
    return jsonify({'status': 'Logged in', 'user_id': user_id})

@app.route('/api/create_character', methods=['POST'])
def create_character():
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({'error': 'Not authenticated'}), 401

    attributes = request.json.get('attributes')
    character_id = str(uuid.uuid4())
    characters[user_id] = {'character_id': character_id, 'attributes': attributes}
    
    save_data()
    return jsonify({'status': 'Character created', 'character_id': character_id})

@app.route('/api/get_character', methods=['GET'])
def get_character():
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({'error': 'Not authenticated'}), 401
    
    character_data = characters.get(user_id)
    if not character_data:
        return jsonify({'error': 'No character found'}), 404
    
    return jsonify(character_data)

@app.route('/api/enter_room', methods=['POST'])
def enter_room(user_id, room_id):
    if room_id not in rooms:
        rooms[room_id] = []
    rooms[room_id].append(user_id)

# Existing functions and classes would be modified to include the new features

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
