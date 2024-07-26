import re


def convert_to_traditional_path(path):
    return re.sub(r'^\\\\\?\\', '', path)