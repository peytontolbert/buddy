class User:
    def __init__(self, username, password):
        self.username = username
        self.password = password
        self.character = None
        self.inventory = []  # List of item names

class Character:
    def __init__(self, name, strength, intelligence):
        self.name = name
        self.strength = strength
        self.intelligence = intelligence

class Room:
    def __init__(self, name, is_public):
        self.name = name
        self.is_public = is_public
        self.members = []
        self.items = []  # List of item names