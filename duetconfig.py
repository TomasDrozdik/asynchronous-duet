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
        # TODO: possibly check for existing images

    @property
    def image(self) -> str:
        return self.config["image"]

    @property
    def run_command(self) -> str:
        return self.config["run"]

    @property
    def result_files(self) -> List[str]:
        return self.config["results"]

    def __str__(self):
        return str(self.config)


class DuetConfig:
    def __init__(self, name: str, config: dict):
        self.name = name
        self.config = config

        if DuetType.A.value not in self.config: 
            raise ConfigException(f"Duet `{name}` is missing benchmark config for A")

        self.a = BenchmarkConfig(self.name, self.config[DuetType.A.value], DuetType.A)

        b_config = self.config[DuetType.B.value] if self.type == DuetBenchType.AB else self.config[DuetType.A.value]
        self.b = BenchmarkConfig(self.name, b_config, DuetType.B)

    @property
    def remove_containers(self) -> bool:
        return self.config["remove_containers"]

    @property
    def repetitions(self) -> int:
        return self.config["repetitions"]

    @property
    def repetitions_type(self) -> DuetRepetititionType:
        str_type = self.config.get("repetitions_type", "swap")
        return DuetRepetititionType(str_type)

    @property
    def type(self) -> DuetBenchType:
        return DuetBenchType.AB if DuetType.A.value in self.config and DuetType.B.value in self.config else DuetBenchType.AA

    def __str__(self):
        return str(self.config)


class SetupConfig:
    def __init__(self, name: str, config: dict):
        self.name = name
        self.config = config

        self.machine = config.get("machine")
        self.provider = config.get("provider")

    def __str__(self):
        return str(self.config)


class DuetBenchConfig:
    def __init__(self, config_filename):
        try:
            with open(config_filename, "r") as config_file:
                self.config = yaml.safe_load(config_file)
        except Exception as e:
            raise ConfigException(f"Loading of duet config {config_filename} failed with exception: {e}")

        # Unique duets
        if len(set(self.duet_names)) != len(self.duet_names):
            raise ConfigException(f"Name clash in duets list: {self.duet_names}")

        # Check presence of duet configs
        missing_configs = []
        for duet_name in self.duet_names:
            if duet_name not in self.config:
                missing_configs.append(duet_name)
        if missing_configs:
            raise ConfigException(f"Missing configs for duets: {missing_configs}")

        # Check existing setup
        setup_name = self.duetbenchconfig.get("setup")
        if setup_name and not setup_name in self.config:
            raise ConfigException(f"Specified setup `{setup_name}` is missing its description")

        self.duets = [DuetConfig(duet_name, self.config[duet_name]) for duet_name in self.duet_names]

    @property
    def duetbenchconfig(self) -> dict:
        return self.config["duetbench"]

    @property
    def name(self) -> str:
        return self.duetbenchconfig["name"]

    @property
    def verbose(self) -> bool:
        return self.duetbenchconfig.get("verbose", False)

    @property
    def seed(self) -> int:
        return self.duetbenchconfig.get("seed")

    @property
    def docker_command(self) -> str:
        return self.duetbenchconfig.get("docker_command")

    @property
    def duet_names(self) -> List[str]:
        return self.duetbenchconfig["duets"]

    @property
    def setup(self) -> SetupConfig:
        setup_name = self.duetbenchconfig.get("setup")
        return SetupConfig(self.name, self.config[setup_name]) if setup_name else SetupConfig(None, None)

    def __str__(self):
        return str(self.config)
