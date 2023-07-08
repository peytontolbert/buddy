import os
import json
import logging

class toolmanagement:
    def __init__(self):
        logging.basicConfig(filename='toolmanagement.log', level=logging.INFO)
        self.tools_path = os.path.join(os.getcwd(), 'workspace', 'tools')
        self.tools = []
        self.load_tools()

    def load_tools(self, args=None):
        try:
            with open(os.path.join('workspace','toolbag','tools.json'), 'r') as f:
                self.tools = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logging.error(f"Error: {e}")
            return f"Error: {e}"

    def create_tool(self, name, func, args, description, genre):
        if not all([name, func, args, description, genre]):
            return "All parameters must be provided."
            
        module_file = f'{genre}.py'
        module_path = os.path.join(self.tools_path, module_file)

        if os.path.exists(module_path):
            new_tool = {
                "name": name,
                "func": func,
                "args": args,
                "description": description,
                "class": genre,
                "user_permission_required": True
            }
            self.tools.append(new_tool)
            self.save_tools()
        else:
            print(f'Warning: Tool module file {module_file} not found.')
            return "Tool Class does not exist. Try creating a new class first."

    def save_tools(self, args=None):
        try:
            with open(os.path.join('workspace','toolbag','tools.json'), 'w') as f:
                json.dump(self.tools, f)
        except (FileNotFoundError, IOError) as e:
            logging.error(f"Error: {e}")
            return f"Error: {e}."
           #print("Error: File not found or unable to write file.")

    def list_by_class(self, class_name):
        if not class_name:
            print("Class name must be provided.")
            return "Class name must be provided."

        tools = [tool for tool in self.tools if tool['class'] == class_name]
        if not tools:
            print(f"No tools found for class '{class_name}'")
            return f"No tools found for class '{class_name}'"
        return tools

    def list_tool_classes(self, args=None):
        classes = [tool['class'] for tool in self.tools if tool['class'] not in classes]
        return classes

    def create_new_class(self, string):
        if not string or not string.isidentifier():
            error_message = "Invalid python class name."
            logging.error(error_message)
            return error_message
        
        file_path = os.path.join('workspace', 'tools', f'{string}.py')
        try:
            with open(file_path, 'w') as f:
                f.write(f"class {string}:\n")
                f.write(f"    def __init__(self):\n")
                f.write(f"        pass\n")
            return f"Created new class {string} in {file_path}"
        except (FileNotFoundError, IOError) as e:
            error_message = f"Unable to create class {string}."
            logging.error(error_message)
            return error_message