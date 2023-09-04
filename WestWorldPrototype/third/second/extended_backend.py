from flask import Flask, jsonify, request, session, render_template
from flask_bcrypt import Bcrypt
import uuid

app = Flask(__name__)
app.secret_key = "supersecretkey"

world_state = {}
users = {}
message_queue = []
characters = {}  # user_id -> character_data

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

@app.route('/api/register', methods=['POST'])
def register_user(username, password):
    user_id = uuid.uuid4().hex
    registered_users[user_id] = {'username': username, 'password': password}
    return True

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
