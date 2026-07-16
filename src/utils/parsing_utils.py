import os


def repo_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def parse_csv_floats(value):
    return [float(x.strip()) for x in value.split(",") if x.strip()]


def parse_csv_ints(value):
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def parse_csv_strings(value):
    return [x.strip() for x in value.split(",") if x.strip()]