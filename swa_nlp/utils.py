import os
from pathlib import Path


def get_dir(*paths) -> str:
    """
    Creates a dir from a list of directories (like os.path.join), runs os.makedirs and returns the name
    
    Args:
        paths (List[str]): list of string that constitutes the path
    Returns:
        str: the created or existing path
    """
    directory = os.path.join(*paths)
    os.makedirs(directory, exist_ok=True)
    return Path(directory)