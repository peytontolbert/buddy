import os
import logging
class filemanager:
    def __init__(self):
        try:
            self.files = self.view_workspace()
            print(self.files)
        except Exception as e:
            print(f"An error occurred: {str(e)}")

    def view_file(self, file_path):
        current_dir = os.getcwd()
        workspace_folder = os.path.join(current_dir, "workspace","work")
        file = os.path.join(workspace_folder, file_path)
        try:
            with open(file, "r") as f:
                return f.read()
        except FileNotFoundError:
            print(f"The file {file_path} does not exist.")
            return (f"The file {file_path} does not exist.")

    def view_workspace(self):
        current_dir = os.getcwd()
        workspace_folder = os.path.join(current_dir, "workspace","work")
        try:
            files = os.listdir(workspace_folder)
            return files
        except FileNotFoundError:
            print(f"The workspace folder does not exist at location: {workspace_folder}")
            return f"The workspace folder does not exist at location: {workspace_folder}"

    def view_files(self, path):
        current_dir = os.getcwd()
        workspace_folder = os.path.join(current_dir, "workspace","work")
        full_path = os.path.join(workspace_folder, path)

        try:
            files = os.listdir(full_path)
            return files
        except FileNotFoundError as e:
            print(f"The specified path '{full_path}' does not exist.")
            return f"The specified path '{full_path}' does not exist."
            #raise Exception(f"The specified path '{full_path}' does not exist.")

    def edit_file(self, file_path, content):
        current_dir = os.getcwd()
        workspace_folder = os.path.join(current_dir, "workspace","work")
        file = os.path.join(workspace_folder, file_path)
        try:
            with open(file, "w") as f:
                f.write(content)
        except FileNotFoundError as e:
            logging.error(f"Error: {e}")
            return f"Error: {e}."