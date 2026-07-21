import os


def repo_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.dirname(__file__)))


def parse_csv_floats(value):
    return [float(x.strip()) for x in value.split(",") if x.strip()]


def parse_csv_ints(value):
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def parse_csv_strings(value):
    return [x.strip() for x in value.split(",") if x.strip()]


def parse_csv_bool_tuples(value):
    result = []
    for item in value.split(";"):
        item = item.strip("(").strip(")")
        if not item:
            continue
        parts = item.split(",")
        if len(parts) != 2:
            raise ValueError(f"Invalid format for boolean tuple: {item}")
        parts = [bool(int(x.strip())) for x in parts]
        result.append(tuple(parts))
    return result