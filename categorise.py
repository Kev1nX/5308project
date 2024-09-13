import os
import shutil

def organize_files_by_prefix(source_folder, destination_base_folder):
    # Loop through all files in the source folder
    for filename in os.listdir(source_folder):
        if filename.endswith('.png'):  # Adjust the file type as necessary
            # Extract the first 3 characters of the filename (excluding extension)
            prefix = filename[:3]
            
            # Define the new folder path
            destination_folder = os.path.join(destination_base_folder, prefix)
            
            # Create the folder if it doesn't exist
            if not os.path.exists(destination_folder):
                os.makedirs(destination_folder)
            
            # Define the source file path and destination file path
            source_file = os.path.join(source_folder, filename)
            destination_file = os.path.join(destination_folder, filename)
            
            # Move the file to the new folder
            shutil.move(source_file, destination_file)
            print(f"Moved {filename} to {destination_folder}")

# Define the source folder (where the .txt files are) and the base destination folder
source_folder = '../TrafficSign-test-1'
destination_base_folder = '../testing_set'

# Call the function to organize the files
organize_files_by_prefix(source_folder, destination_base_folder)