import os
import subprocess
class coding:
    def __init__(self):
        pass
    def create_python_script(self, args):
        print("Creating:", args)
        file_name = args.name
        file_contents = args.contents

        # Generate a unique filename
        unique_filename = str(file_name) + ".py"

        # Get the current directory
        current_dir = os.getcwd()

        # Create the workspace folder path
        workspace_folder = os.path.join(current_dir, "workspace")
        # Create the file path
        file_path = os.path.join(workspace_folder, unique_filename)

        # Write the contents to the file
        with open(file_path, "w") as file:
            file.write(file_contents)

        # Return the created file path
        return file_path
    def run_script(self, file_path):
        # Get the current directory
        current_dir = os.getcwd()

        # Create the workspace folder path
        workspace_folder = os.path.join(current_dir, "workspace")

        # Run the Python script using subprocess
        subprocess.run(["python", os.path.join(workspace_folder, file_path)]) 
    def edit_file(self, file_path, content):
        current_dir = os.getcwd()
        workspace_folder = os.path.join(current_dir, "workspace")
        file = os.path.join(workspace_folder, file_path)
        with open(file, "w") as f:
            f.write(content)

    def read_file(self, file_path):
        current_dir = os.getcwd()
        workspace_folder = os.path.join(current_dir, "workspace")
        file = os.path.join(workspace_folder, file_path)
        with open(file, "r") as f:
            return f.read()
        
    def view_workspace(self):
        current_dir = os.getcwd()
        workspace_folder = os.path.join(current_dir, "workspace")
        files = os.listdir(workspace_folder)
        return files
    
    def view_files(self, path):
        current_dir = os.getcwd()
        workspace_folder = os.path.join(current_dir, "workspace")
        files = os.listdir(os.path.join(workspace_folder, path))
        return files
