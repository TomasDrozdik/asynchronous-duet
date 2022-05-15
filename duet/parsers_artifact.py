def strip_contents(contents: str) -> str:
    return contents.strip()


def parse_lscpu(contents: str) -> str:
    for line in contents.splitlines():
        if "Model name:" in line:
            return line.split(":")[1].strip()
    return None


def parse_meminfo(contents: str) -> str:
    for line in contents.splitlines():
        if "MemTotal:" in line:
            return line.split(":")[1].strip()
    return None
