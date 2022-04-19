from typing import List
import enum
import yaml


class DuetBenchType(enum.Enum):
    AB = "A-B"
    AA = "A-A"


class DuetType(enum.Enum):
    A = "A"
    B = "B"


class DuetRepetititionType(enum.Enum):
    IN_ORDER = "in_order"
    RANDOM = "random"
    SWAP = "swap"


class ConfigException(Exception):
    pass


class BenchmarkConfig:
    def __init__(self, duetname: str, config: dict, duet_type: DuetType):
        self.duetname = duetname

        self.config = config

        self.duet_type = duet_type

        self.container = f"{duetname}.{duet_type.value}"

        self.image: str = config["image"]

        self.run_command: str = config["run"]

        self.result_files: List[str] = config["results"]

    def __str__(self):
        return str(vars(self))


class DuetConfig:
    def __init__(self, name: str, config: dict):
        self.name = name

        self.config = config

        self.remove_containers: bool = config["remove_containers"]

        self.repetitions: int = config["repetitions"]

        self.repetitions_type = DuetRepetititionType(
            config.get("repetitions_type", "swap")
        )

        self.type: DuetBenchType = (
            DuetBenchType.AB
            if DuetType.A.value in self.config and DuetType.B.value in self.config
            else DuetBenchType.AA
        )

        if DuetType.A.value not in config:
            raise ConfigException(f"Duet `{name}` is missing benchmark config for A")

        self.a = BenchmarkConfig(self.name, config[DuetType.A.value], DuetType.A)

        b_config = (
            config[DuetType.B.value]
            if self.type == DuetBenchType.AB
            else config[DuetType.A.value]
        )
        self.b = BenchmarkConfig(self.name, b_config, DuetType.B)

    def __str__(self):
        return str(vars(self))


# TODO: this setup may not be required, other option is to gather the data from lscpu and store them directly in results files, it will obtain similar information just automatically without need for config changes
class DuetBenchConfig:
    def __init__(self, config_filename):
        self.load_config(config_filename)

        self.duetbenchconfig: dict = self.config["duetbench"]

        self.name: str = self.duetbenchconfig["name"]

        self.verbose: bool = self.duetbenchconfig.get("verbose")

        self.seed: int = self.duetbenchconfig.get("seed")

        self.docker_command: str = self.duetbenchconfig.get("docker_command")

        self.duet_names: List[str] = self.duetbenchconfig["duets"]

        self.duets = [
            DuetConfig(duet_name, self.config[duet_name])
            for duet_name in self.duet_names
        ]

        self.check()

    def load_config(self, config_filename: str):
        try:
            with open(config_filename, "r") as config_file:
                self.config = yaml.safe_load(config_file)
        except Exception as e:
            raise ConfigException(
                f"Loading of duet config {config_filename} failed with exception: {e}"
            )

    def check(self):
        # Unique duets
        if len(set(self.duet_names)) != len(self.duet_names):
            raise ConfigException(
                f"Name clash in duets list: {self.duet_names}                      "
            )

        # Check presence of duet configs
        missing_configs = []
        for duet_name in self.duet_names:
            if duet_name not in self.config:
                missing_configs.append(duet_name)
        if missing_configs:
            raise ConfigException(f"Missing configs for duets: {missing_configs}")

    def __str__(self):
        return str(vars(self))
