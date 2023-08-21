import os

def rename_files_in_order(directory):
    # List all files in the directory
    files = sorted(os.listdir(directory))
    
    # Counter for the new file names
    counter = 1
    
    for file in files:
        # Construct the full path of the current file
        old_path = os.path.join(directory, file)
        
        # Construct the new file name
        new_name = f"{counter}.wav"
        new_path = os.path.join(directory, new_name)
        
        # Rename the file
        os.rename(old_path, new_path)
        
        # Increment the counter
        counter += 1

    print(f"Renamed {counter - 1} files in {directory}.")

# Call the function
rename_files_in_order('OUTPUT')