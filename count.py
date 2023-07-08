import os

def count_lines_in_file(file_path):
    with open(file_path, 'r') as f:
        return len(f.readlines())

def count_lines_in_dir(dir_path):
    total_lines = 0

    for root, dirs, files in os.walk(dir_path):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                total_lines += count_lines_in_file(file_path)
            except UnicodeDecodeError:
                # If the file is not a text file, skip it
                pass

    return total_lines

# replace "your_directory_path" with your actual directory path
dir_path = "Buddy"
print(f"Total lines: {count_lines_in_dir(dir_path)}")