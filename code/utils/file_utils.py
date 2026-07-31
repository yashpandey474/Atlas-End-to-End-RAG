from pathlib import Path

def check_file_exists(file_path: str) -> bool:
    # Create a path object
    file_path = Path(file_path)

    # Check if it exists and is a file
    if file_path.is_file():
        print("The file exists.")
        return True
    else:
        print("The file does not exist.")
        return False