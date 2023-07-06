import time
import json
class notes:
    def __init__(self):
        pass
    def write(self, string, file_path="notebook.json"):
        print("Writing notes:")
        entry = {"entry": string}
        self.write_to_notebook(entry, file_path)
        return entry

    def write_to_notebook(self, entry, file_path="notebook.json"):
        entries = self.read_all(file_path)
        entries.append(entry)
        with open(file_path, "w") as f:
            json.dump(entries, f)

    def read(self, index, file_path="notebook.json"):
        print(f"Reading note at index {index}:")
        entries = self.read_all(file_path)
        try:
            return entries[index]
        except IndexError:
            print(f"No note at index {index}")
            return None

    def read_all(self, file_path="notebook.json"):
        try:
            with open(file_path, "r") as f:
                return json.load(f)
        except FileNotFoundError:
            return []
        except json.JSONDecodeError:
            return []