# Sample Python code to read a repos.txt file containing a JSON list of repositories and insert them into a PostgreSQL database

import json
import psycopg2
from githubscraper import GithubScraper

# Path to the repos.txt file

class DBController:
    def __init__(self):
        self.file_path = "repos.json"
    def read_repo_file(self, file_path):
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            return data
        except Exception as e:
            print(f"An error occurred while reading the file: {e}")
    def update_database_from_json(self):
        # Your existing code that reads from the JSON file and updates the database
        repo_list = self.read_repo_file(self.file_path)
        if repo_list:
            for repo in repo_list:
                repo_name = repo.get("reponame")
                repo_url = repo.get("repourl")
                if repo_name and repo_url:
                    repo_id = self.insert_repo_name_and_url(repo_name, repo_url)
                    print(f"Inserted repo {repo_name} with ID: {repo_id}")
                else:
                    print(f"Skipped repo due to missing name or URL: {repo}")
    def insert_repo_name_and_url(self, repo_name, repo_url):
        try:
            # Connect to your PostgreSQL database. Replace these placeholders with your actual database credentials
            conn = psycopg2.connect(
                dbname="repos",
                user="postgres",
                password="1234",
                host="localhost",
                port="5432"
            )
            
            # Create a new database session and return a new instance of the Cursor class
            cur = conn.cursor()
            
            # Check for duplicate repository by name or URL
            cur.execute("""
                SELECT repo_id FROM Repositories WHERE repo_name = %s OR repo_url = %s;
            """, (repo_name, repo_url))
            
            existing_repo = cur.fetchone()
            
            if existing_repo:
                print(f"Skipped duplicate repo {repo_name} with URL {repo_url}")
                return None
            
            # Insert repository name and URL into the Repositories table
            cur.execute("""
                INSERT INTO Repositories (repo_name, repo_url)
                VALUES (%s, %s) RETURNING repo_id;
            """, (repo_name, repo_url))
            
            # Fetch the returned repo_id
            repo_id = cur.fetchone()[0]
            
            # Commit the changes and close the cursor and the connection
            conn.commit()
            cur.close()
            conn.close()
            
            return repo_id
        except Exception as e:
            print(f"An error occurred: {e}") 

    def fetch_repos_by_completion_status(self, conn, status):
        try:
            if conn is None:
                print("Database connection is None.")
                return None
            # Create a new database session and return a new instance of the Cursor class
            cur = conn.cursor()
            
            # Fetch all repositories based on completion status
            cur.execute("SELECT repo_id, repo_name, repo_url FROM Repositories WHERE complete = %s;", (status,))
            
            # Fetch all rows as a list of tuples and return it
            repos = cur.fetchall()

            if not repos:
                print(f"No repositories found with complete = {status}.")
                
            return repos
        except Exception as e:
            print(f"An error occurred while fetching repositories: {e}")
            return None
        
    def update_incomplete_repositories(self, conn, ghscraper, token):
        # Fetch all incomplete repositories
        incomplete_repos = self.fetch_repos_by_completion_status(conn, False)
        
        # Loop through each incomplete repo to update it
        for repo in incomplete_repos:
            repo_id, repo_name, repo_url = repo
            print(f"Updating {repo_name} ({repo_url})...")
            file_list, directory_list, description, keywords, readme = ghscraper.fetch_readme_with_pygithub(token, repo_url)
            for dir_path in directory_list:
                self.insert_into_table(conn, "Directories", ["repo_id", "dir_path"], [repo_id, dir_path], ["repo_id", "dir_path"])
            for file_path in file_list:
                self.insert_into_table(conn, "FileList", ["repo_id", "file_path", "description", "keywords"], [repo_id, file_path, description, keywords])
                if file_path.endswith('.py'):
                    self.insert_into_table(conn, "PythonFiles", ["repo_id", "file_path", "description", "keywords"], [repo_id, file_path, description, keywords], ["repo_id", "file_path"])
            if readme:
                self.insert_into_table(conn, "readmefiles", ["repo_id", "readme_content"], [repo_id, readme], ["repo_id"])
                # Mark the repository as complete
                self.mark_repository_complete(conn, repo_id)
            

    def mark_repository_complete(self,conn, repo_id):
        cur = conn.cursor()
        cur.execute(
            "UPDATE Repositories SET complete = TRUE WHERE repo_id = %s;",
            (repo_id,)
        )
        conn.commit()
        cur.close()
        
        

    def insert_into_table(self, conn, table_name, fields, values, unique_fields=None):
        cur = conn.cursor()

        # Create a string for SQL placeholders
        placeholders = ", ".join(["%s"] * len(fields))

        # Create SQL query string
        insert_query = f"INSERT INTO {table_name} ({', '.join(fields)}) VALUES ({placeholders});"

        # If unique fields are provided, check for duplicates
        if unique_fields:
            select_query = f"SELECT * FROM {table_name} WHERE "
            select_query += " AND ".join([f"{field} = %s" for field in unique_fields])
            
            cur.execute(select_query, [values[fields.index(field)] for field in unique_fields])
            
            existing_record = cur.fetchone()
            
            if existing_record:
                print(f"Skipped duplicate record in {table_name} for unique fields {unique_fields}")
                cur.close()
                return

        cur.execute(insert_query, values)
        conn.commit()
        cur.close()
def insert_directory(self, conn, repo_id, dir_path):
    cur = conn.cursor()
    
    # Check for duplicate directory by repo_id and dir_path
    cur.execute("""
        SELECT * FROM Directories WHERE repo_id = %s AND dir_path = %s;
    """, (repo_id, dir_path))
    
    existing_directory = cur.fetchone()
    
    if existing_directory:
        print(f"Skipped duplicate directory {dir_path} for repo_id {repo_id}")
    else:
        cur.execute("""
            INSERT INTO Directories (repo_id, dir_path)
            VALUES (%s, %s);
        """, (repo_id, dir_path))
        conn.commit()
    
    cur.close()


def insert_readme(self, conn, repo_id, readme):
    cur = conn.cursor()
    
    # Check for duplicate readme by repo_id
    cur.execute("""
        SELECT * FROM readmefiles WHERE repo_id = %s;
    """, (repo_id,))
    
    existing_readme = cur.fetchone()
    
    if existing_readme:
        print(f"Skipped duplicate readme for repo_id {repo_id}")
    else:
        cur.execute("""
            INSERT INTO readmefiles (repo_id, readme_content)
            VALUES (%s, %s);
        """, (repo_id, readme))
        conn.commit()
    
    cur.close()

    def insert_file_with_description_and_keywords(self, conn, repo_id, file_path, description, keywords):
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO FileList (repo_id, file_path, description, keywords)
            VALUES (%s, %s, %s, %s);
        """, (repo_id, file_path, description, keywords))
        conn.commit()
        cur.close()

        
    def insert_python_file(self, conn, repo_id, file_path, description, keywords):
        cur = conn.cursor()
        
        # Check for duplicate file_path for the same repo
        cur.execute("""
            SELECT file_id FROM PythonFiles WHERE repo_id = %s AND file_path = %s;
        """, (repo_id, file_path))
        
        existing_file = cur.fetchone()
        
        if existing_file:
            print(f"Skipped duplicate Python file {file_path} for repo ID {repo_id}")
            return None
        
        cur.execute("""
            INSERT INTO PythonFiles (repo_id, file_path, description, keywords)
            VALUES (%s, %s, %s, %s);
        """, (repo_id, file_path, description, keywords))
        conn.commit()
        cur.close()
