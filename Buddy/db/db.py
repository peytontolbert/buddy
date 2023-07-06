
class DBManager:
    def __init__(self):
        self.database = {}  # This is a placeholder. Replace with your actual database.

    def upsert_user(self, user_id: str):
        if user_id not in self.database:
            self.database[user_id] = {"thoughts": []}

    def upsert_thought(self, user_id: str, thought: str):
        if user_id in self.database:
            self.database[user_id]["thoughts"].append(thought)

    def fetch_thought(self, user_id: str, thought_id: int) -> str:
        if user_id in self.database and thought_id < len(self.database[user_id]["thoughts"]):
            return self.database[user_id]["thoughts"][thought_id]
