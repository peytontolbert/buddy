import os
import json
class toolmanagement:
    def __init__(self):
        self.tools_path = os.path.join(os.getcwd(), 'workspace', 'tools')
        self.tools = []
        self.load_tools()


    def load_tools(self):
        with open(os.path.join('workspace','toolbag','tools.json'), 'r') as f:
            self.tools = json.load(f)


    def create_tool(self, name, func, args, description, classe):
        module_file = f'{classe}.py'
        module_path = os.path.join(self.tools_path, module_file)
        print(module_path)
        if os.path.exists(module_path):
            new_tool = {
                "name": name,
                "func": func,
                "args": args,
                "description": description,
                "class": classe,
                "user_permission_required": True
            }
            self.tools.append(new_tool)
            self.save_tools()
        else:
            print(f'Warning: Tool module file {module_file} not found.')
            return "Tool Class does not exist. Try creating a new file first."

    def save_tools(self):
        with open(os.path.join('workspace','toolbag','tools.json'), 'w') as f:
            json.dump(self.tools, f)

    def list_by_class(self, class_name):
        tools = []
        for tool in self.tools:
            if tool['class'] == class_name:
                tools.append(tool)
        return tools

    def list_tool_classes(self):
        classes = []
        for tool in self.tools:
            if tool['class'] not in classes:
                classes.append(tool['class'])
        return classes
    
    def create_new_class(self, string):
        file_path = os.path.join('workspace', 'tools', f'{string}.py')
        with open(file_path, 'w') as f:
            f.write(f"class {string}:\n")
            f.write(f"    def __init__(self):\n")
            f.write(f"        pass\n")
        return f"Created new class {string} in {file_path}"