from pathlib import Path
import json
import logging

logger = logging.getLogger(__name__)

def check_file_exists(file_path: str) -> bool:
    # Create a path object
    file_path = Path(file_path)

    # Check if it exists and is a file
    if file_path.is_file():
        logger.info("The file exists.")
        return True
    else:
        logger.info("The file does not exist.")
        return False

def read_from_json(file_path: str):
    if not check_file_exists(file_path=file_path):
        logger.info(f"File: {file_path} does not exist")
        return

    with open(file_path, "r") as f:
        data = json.load(f)

    return data

def write_to_json(file_path: str, to_write: dict):
    with open(file_path, "w") as f:
        json.dump(to_write, f, indent=4)