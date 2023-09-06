from flask import Blueprint, render_template, request, jsonify
from .models import User, Room, Character
from flask_socketio import join_room, leave_room, send


import json
bp = Blueprint("routes", __name__)

users = {}  # Username -> User object
rooms = {}  # Room name -> Room object

@bp.route("/")
def home():
    return render_template("index.html")

@bp.route("/register", methods=["POST"])
def register():
    data = request.json
    username = data["username"]
    password = data["password"]

    # Check if the username already exists
    with open("users.json", "r") as f:
        users_data = json.load(f)

    if username in users_data:
        return jsonify({"status": "error", "message": "Username already exists"}), 400

    users_data[username] = {"password": password, "character": None}

    with open("users.json", "w") as f:
        json.dump(users_data, f)

    return jsonify({"status": "success", "message": "User registered successfully"}), 201

@bp.route("/login", methods=["POST"])
def login():
    data = request.json
    username = data["username"]
    password = data["password"]

    with open("users.json", "r") as f:
        users_data = json.load(f)

    if username not in users_data:
        return jsonify({"status": "error", "message": "Username does not exist"}), 400

    if users_data[username]["password"] != password:
        return jsonify({"status": "error", "message": "Incorrect password"}), 400

    return jsonify({"status": "success", "message": "Logged in successfully"}), 200


@bp.route("/create_character", methods=["POST"])
def create_character():
    data = request.json
    username = data["username"]
    character_name = data["character_name"]
    strength = data["strength"]
    intelligence = data["intelligence"]

    with open("users.json", "r") as f:
        users_data = json.load(f)

    if username not in users_data:
        return jsonify({"status": "error", "message": "Username does not exist"}), 400

    users_data[username]["character"] = {
        "name": character_name,
        "strength": strength,
        "intelligence": intelligence
    }

    with open("users.json", "w") as f:
        json.dump(users_data, f)

    return jsonify({"status": "success", "message": "Character created successfully"}), 201


@bp.route("/create_room", methods=["POST"])
def create_room():
    data = request.json
    room_name = data["room_name"]
    is_public = data["is_public"]

    if room_name in rooms:
        return jsonify({"status": "error", "message": "Room already exists"}), 400

    rooms[room_name] = Room(room_name, is_public)
    return jsonify({"status": "success", "message": "Room created successfully"}), 201


@bp.route('/join_room', methods=['POST'])
def join_room_route():
    data = request.json
    room_name = data['room_name']
    username = data['username']

    if room_name not in rooms:
        return jsonify({'status': 'error', 'message': 'Room does not exist'}), 400

    join_room(room_name)
    send(username + ' has entered the room.', room=room_name)
    return jsonify({'status': 'success', 'message': 'Joined the room successfully'}), 200

@bp.route("/add_to_inventory", methods=["POST"])
def add_to_inventory():
    data = request.json
    username = data["username"]
    item = data["item"]
    # Add item to user's inventory (simplified)
    if username in users:
        users[username].inventory.append(item)
    return jsonify({"status": "success"})

@bp.route("/remove_from_inventory", methods=["POST"])
def remove_from_inventory():
    data = request.json
    username = data["username"]
    item = data["item"]
    # Remove item from user's inventory
    # ...

@bp.route("/drop_item", methods=["POST"])
def drop_item():
    data = request.json
    room_name = data["room_name"]
    item = data["item"]
    username = data["username"]
    # Drop item into room (simplified)
    if room_name in rooms and username in users:
        if item in users[username].inventory:
            rooms[room_name].items.append(item)
            users[username].inventory.remove(item)
    return jsonify({"status": "success"})

@bp.route("/pickup_item", methods=["POST"])
def pickup_item():
    data = request.json
    room_name = data["room_name"]
    item = data["item"]
    # Pick up item from room
    # ...



@bp.on("send_message")
def send_message_event(data):
    room_name = data["room_name"]
    message = data["message"]
    send(message, room=room_name)