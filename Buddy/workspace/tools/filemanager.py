import os
class filemanager:
    def __init__(self):
        self.files = self.view_workspace()
        print(self.files)
        pass
    
    def view_file(self, file_path):
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


    def edit_file(self, file_path, content):
        current_dir = os.getcwd()
        workspace_folder = os.path.join(current_dir, "workspace")
        file = os.path.join(workspace_folder, file_path)
        with open(file, "w") as f:
            f.write(content)
