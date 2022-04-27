import os
from typing import List
import enum
import yaml
import re


class DuetBenchType(enum.Enum):
    AB = "A-B"
    AA = "A-A"


class Pair(enum.Enum):
    A = "A"
    B = "B"


class RepetititionType(enum.Enum):
    IN_ORDER = "in_order"
    RANDOM = "random"
    SWAP = "swap"


class Type(enum.Enum):
    SEQUENTIAL = "sequential"
    DUET = "duet"


class ConfigException(Exception):
    pass


ARTIFACTS_DIR = "artifacts"


def check_valid_keys(under_key: str, config: dict, valid_values):
    for key in config.keys():
        if key not in valid_values:
            raise ConfigException(
                f"Unknown key `{key}` in {under_key}, valid values {valid_values}"
            )


def unique(values):
    return len(set(values)) == len(values)


class BenchmarkConfig:
    VALUES = ["image", "run", "results"]

    def __init__(self, parent, config: dict, pair: Pair):
        check_valid_keys(parent, config, BenchmarkConfig.VALUES)

        self.parent = parent

        self.config = config

        self.duetname = self.parent.name

        self.pair = pair

        self.container = f"{self.duetname}.{pair.value}"

        self.image: str = self.get_or_inherit("image", None)

        self.run_command: str = config["run"]

        self.result_files: List[str] = self.get_or_inherit("results", None)

        self.check()

    def get_or_inherit(self, key, default):
        if key in self.config:
            return self.config[key]
        else:
            return self.parent.get_or_inherit(key, default)

    def check(self):
        if not self.image:
            raise ConfigException(
                f"DuetBenchmark {self.duetname}:{self.pair} does has no image"
            )

        if not self.result_files:
            raise ConfigException(
                f"DuetBenchmark {self.duetname}:{self.pair} does has no result files"
            )

    def __str__(self):
        return str(vars(self))


class DuetConfig:
    UNIQUE_VALUES = ["A", "B"]
    VALUES = [
        "remove_containers",
        "repetitions",
        "repetitions_type",
        "sequential_repetitions",
        "sequential_repetitions_type",
    ] + BenchmarkConfig.VALUES

    def __init__(self, name: str, config: dict, duetbenchconfig):
        check_valid_keys(name, config, DuetConfig.UNIQUE_VALUES + DuetConfig.VALUES)

        self.duetbenchconfig = duetbenchconfig

        self.name = name

        self.config = config

        self.remove_containers: bool = self.get_or_inherit(
            "remove_containers", default=True
        )

        self.repetitions: int = self.get_or_inherit("repetitions", default=1)

        self.repetitions_type = RepetititionType(
            self.get_or_inherit("repetitions_type", default=RepetititionType.SWAP.value)
        )

        self.sequential_repetitions: int = self.get_or_inherit(
            "sequential_repetitions", default=0
        )

        self.sequential_repetitions_type = RepetititionType(
            self.get_or_inherit(
                "sequential_repetitions_type", default=RepetititionType.SWAP.value
            )
        )

        self.type = (
            DuetBenchType.AB
            if Pair.A.value in self.config and Pair.B.value in self.config
            else DuetBenchType.AA
        )

        if Pair.A.value not in config:
            raise ConfigException(f"Duet `{name}` is missing benchmark config for A")

        self.a = BenchmarkConfig(self, config[Pair.A.value], Pair.A)

        b_config = (
            config[Pair.B.value]
            if self.type == DuetBenchType.AB
            else config[Pair.A.value]
        )
        self.b = BenchmarkConfig(self, b_config, Pair.B)

    def get_or_inherit(self, key, default):
        if key in self.config:
            return self.config[key]
        elif key in self.duetbenchconfig:
            return self.duetbenchconfig[key]
        elif default:
            return default

    def __str__(self):
        return str(vars(self))


class DuetBenchConfig:
    UNIQUE_VALUES = ["name", "verbose", "seed", "docker_command", "duets", "artifacts"]
    VALUES = DuetConfig.VALUES

    def __init__(self, config_filename):
        self.load_config(config_filename)

        self.duetbenchconfig: dict = self.config["duetbench"]

        check_valid_keys(
            "duetbench",
            self.duetbenchconfig,
            DuetBenchConfig.UNIQUE_VALUES + DuetBenchConfig.VALUES,
        )

        self.name: str = self.duetbenchconfig["name"]

        self.verbose: bool = self.duetbenchconfig.get("verbose")

        self.seed: int = self.duetbenchconfig.get("seed")

        self.docker_command: str = self.duetbenchconfig.get("docker_command")

        self.duet_names: List[str] = self.duetbenchconfig["duets"]

        self.artifacts: dict = (
            self.duetbenchconfig["artifacts"]
            if "artifacts" in self.duetbenchconfig
            else {}
        )

        self.duets = [
            DuetConfig(duet_name, self.config[duet_name], self.duetbenchconfig)
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
        # Valid values in duetbench
        for key in self.duetbenchconfig:
            if (
                key not in DuetBenchConfig.VALUES
                and key not in DuetBenchConfig.UNIQUE_VALUES
            ):
                raise ConfigException(f"Unknown key `{key}` in duetbench")

        # Unique result file names i.e. duets and artifacts directory
        if not unique(self.duet_names + [ARTIFACTS_DIR]):
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


class ResultFile:
    FILENAME_REGEX = re.compile(
        r"(?P<benchmark>[a-zA-Z0-9_-]+)\.(?P<run_id>\d+)\.(?P<type>duet|sequential)\.(?P<run_order>[AB]+)\.(?P<pair>[AB])\.(?P<result_file>.*)"
    )

    @staticmethod
    def parse(result_path: str):
        basename = os.path.basename(result_path)
        match = ResultFile.FILENAME_REGEX.match(basename)
        return (
            ResultFile(
                benchmark=match.group("benchmark"),
                run_id=match.group("run_id"),
                type=match.group("type"),
                run_order=match.group("run_order"),
                pair=match.group("pair"),
                result_file=match.group("result_file"),
                result_path=result_path,
            )
            if match
            else None
        )

    def __init__(
        self,
        benchmark: str,
        run_id: int,
        type: Type,
        run_order: str,
        pair: str,
        result_file: str,
        result_path: str = None,  # set by parse method only
    ):
        self.benchmark = benchmark
        self.run_id = run_id
        self.type = type
        self.run_order = run_order
        self.pair = pair
        self.result_file = result_file
        self.result_path = result_path

    def filename(self):
        return f"{self.benchmark}.{self.run_id}.{self.type.value}.{self.run_order}.{self.pair}.{self.result_file}"
