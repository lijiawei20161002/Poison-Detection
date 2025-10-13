import os
import shutil

def remove_pycache_and_logs(root_dir='.'):
    """
    Recursively removes all __pycache__ directories and .log files
    starting from the given root directory.
    """
    for dirpath, dirnames, filenames in os.walk(root_dir):
        # Remove __pycache__ directories
        if "__pycache__" in dirnames:
            pycache_path = os.path.join(dirpath, "__pycache__")
            try:
                shutil.rmtree(pycache_path)
                print(f"Removed directory: {pycache_path}")
            except Exception as e:
                print(f"Failed to remove {pycache_path}: {e}")

        # Remove all .log files
        for filename in filenames:
            if filename.endswith(".log"):
                log_file_path = os.path.join(dirpath, filename)
                try:
                    os.remove(log_file_path)
                    print(f"Removed file: {log_file_path}")
                except Exception as e:
                    print(f"Failed to remove {log_file_path}: {e}")

if __name__ == "__main__":
    remove_pycache_and_logs()