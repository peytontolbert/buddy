from dbcontroller import DBController
from githubscraper import GithubScraper
from messagehandler import MessageHandler
import tkinter as tk
from tkinter import ttk
import psycopg2
import os
from dotenv import load_dotenv
load_dotenv()

def connect_to_database():
    try:
        # Connect to your PostgreSQL database. Replace these placeholders with your actual database credentials
        conn = psycopg2.connect(
            dbname=os.getenv("DB_NAME"),
            user=os.getenv("USER"),
            password=os.getenv("PASSWORD"),
            host=os.getenv("HOST"),
            port=os.getenv("PORT")
        )
        return conn
    except Exception as e:
        print(f"An error occurred while connecting to the database: {e}")
        return None
        

def send_message(event=None):
    user_message = user_input.get()
    chat_history.config(state=tk.NORMAL)
    chat_history.insert(tk.END, f"You: {user_message}\n")
    
    # Generate a response (replace this with actual chat logic)
    response = messagehandler.handle_message(user_message)
    
    chat_history.insert(tk.END, f"{response}\n")
    chat_history.config(state=tk.DISABLED)
    user_input.delete(0, tk.END)

def run_script():
    print("Running script...")
db_controller = DBController()
ghscraper = GithubScraper()
messagehandler = MessageHandler()
token = os.getenv("GITHUB_TOKEN")
# Initialize Tkinter window
root = tk.Tk()
root.title("Repository Manager")

chat_history = tk.Text(root, wrap=tk.WORD, state=tk.DISABLED)
chat_history.pack(expand=tk.YES, fill=tk.BOTH)
# Add an Entry widget for user input
user_input = tk.Entry(root)
user_input.pack(fill=tk.X)
# Bind the Enter key to send_message function
user_input.bind("<Return>", send_message)

# Add a Button to send the message
send_button = tk.Button(root, text="Send", command=send_message)
send_button.pack()

# Connect to the database on startup
conn = connect_to_database()

# Fetch and display complete repositories
complete_repos_label = tk.Label(root, text="Complete Repositories:")
complete_repos_label.pack()
complete_repos = db_controller.fetch_repos_by_completion_status(conn, True)  # Passing conn as the first argument
for repo in complete_repos:
    repo_id, repo_name, repo_url = repo
    repo_label = tk.Label(root, text=f"{repo_name} ({repo_url})")
    repo_label.pack()

# Fetch and display incomplete repositories
incomplete_repos_label = tk.Label(root, text="Incomplete Repositories:")
incomplete_repos_label.pack()
incomplete_repos = db_controller.fetch_repos_by_completion_status(conn, False)  # Passing conn as the first argument
for repo in incomplete_repos:
    repo_id, repo_name, repo_url = repo
    repo_label = tk.Label(root, text=f"{repo_name} ({repo_url})")
    repo_label.pack()

# Button to run a Python script
run_button = tk.Button(root, text="Run Script", command=run_script)
run_button.pack()
# Button to update the database from the JSON file
update_button = tk.Button(root, text="Update Database", command=lambda: db_controller.update_database_from_json)
update_button.pack()

# Button to update incomplete repositories
update_incomplete_button = tk.Button(root, text="Update Incomplete Repositories", command=lambda: db_controller.update_incomplete_repositories(conn, ghscraper, token))
update_incomplete_button.pack()



# Run the Tkinter event loop
root.mainloop()