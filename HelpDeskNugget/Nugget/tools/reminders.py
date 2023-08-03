class reminders:
    def __init__(self):
        self.reminders = {}

    def create_reminder(self, reminder: str, time: str):
        self.reminders[reminder] = time
        return f"Reminder set for {time}"

    def get_reminders(self):
        return self.reminders
    