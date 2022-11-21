from distutils.util import strtobool

def str2bool(value: str) -> bool:
    return bool(strtobool(value))