
def get_lines_from_uploaded_file(file) -> list[str]:
    return file.getvalue().decode("utf-8").split("\n")