# Custom logger
import logging
import os


def setup_logger(name, log_file, level=logging.INFO):
    create_file(name, log_file)
    log_file_path = os.path.join(log_file, f"{name.lower()}.log")
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    handler = logging.FileHandler(log_file_path)
    handler.setFormatter(formatter)
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)

    # Remove any existing StreamHandler
    for h in logger.handlers:
        if isinstance(h, logging.StreamHandler):
            logger.removeHandler(h)

    return logger


# Custom function to create directory if it doesn't exist
def create_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)


def create_file(name: str, path: str):
    # Construct the full path to the log file
    log_file_path = os.path.join(path, f"{name.lower()}.log")

    # Create the directory if it doesn't exist
    create_directory(os.path.dirname(log_file_path))

    # If the file doesn't exist, create it
    if not os.path.exists(log_file_path):
        with open(log_file_path, "w") as file:
            file.write(f"{name} log file\n")
