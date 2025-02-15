import os

DATA_DIR = "data"

def safe_path(filename):
    """Ensures files are only accessed inside data."""
    full_path = os.path.join(DATA_DIR, os.path.basename(filename))
    return full_path if full_path.startswith(DATA_DIR) else None

def read_file_content(file_path):
    """Reads and returns file contents if file exists."""
    try:
        with open(file_path, "r") as f:
            return f.read()
    except FileNotFoundError:
        return None
