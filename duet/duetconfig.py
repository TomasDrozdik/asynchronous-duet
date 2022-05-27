import os
from typing import List
import enum
import yaml
import re


class DuetBenchType(enum.Enum):
    AB = "A-B"
    AA = "A-A"


class DuetOrder(enum.Enum):
    AB = "AB"
    BA = "BA"


class Pair(enum.Enum):
    A = "A"
    B = "B"


class Schedule(enum.Enum):
    SEQUENTIAL = "sequential"
    RANDOMIZED_INTERLEAVING_TRIALS = "randomized_interleaving_trials"


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
    VALUES = ["image", "run", "results", "run_base"]

    def __init__(self, parent, config: dict, pair: Pair):
        check_valid_keys(parent, config, BenchmarkConfig.VALUES)

        self.parent = parent

        self.config = config

        self.suite = self.get_or_inherit("suite")

        self.benchmark = self.parent.benchmark

        self.pair = pair

        self.container = f"{self.benchmark}.{pair.value}"

        self.image: str = self.get_or_inherit("image")

        self.base_run_command = self.get_or_inherit("run_base", "")

        self.run_command: str = f"{self.base_run_command} {self.config['run']}"

        self.result_files: List[str] = self.get_or_inherit("results")

        self.check()

    def get_or_inherit(self, key, default=None):
        if key in self.config:
            return self.config[key]
        else:
            return self.parent.get_or_inherit(key, default)

    def check(self):
        if not self.image:
            raise ConfigException(
                f"DuetBenchmark {self.suite}:{self.pair} does has no image"
            )

        if not self.result_files:
            raise ConfigException(
                f"DuetBenchmark {self.suite}:{self.pair} does has no result files"
            )

    def __str__(self):
        return str(vars(self))


class DuetConfig:
    UNIQUE_VALUES = ["A", "B"]
    VALUES = [
        "suite",
        "remove_containers",
        "duet_repetitions",
        "sequential_repetitions",
        "run_base",
    ] + BenchmarkConfig.VALUES

    def __init__(self, benchmark: str, config: dict, duetbenchconfig):
        check_valid_keys(
            benchmark, config, DuetConfig.UNIQUE_VALUES + DuetConfig.VALUES
        )

        self.benchmark = benchmark

        self.config = config

        self.duetbenchconfig = duetbenchconfig

        self.suite = self.get_or_inherit("suite")

        self.remove_containers: bool = self.get_or_inherit(
            "remove_containers", default=True
        )

        self.duet_repetitions: int = self.get_or_inherit("duet_repetitions", default=1)

        self.sequential_repetitions: int = self.get_or_inherit(
            "sequential_repetitions", default=0
        )

        self.type = (
            DuetBenchType.AB
            if Pair.A.value in self.config and Pair.B.value in self.config
            else DuetBenchType.AA
        )

        if Pair.A.value not in config:
            raise ConfigException(
                f"Duet `{self.benchmark}` is missing benchmark config for A"
            )

        self.a = BenchmarkConfig(self, config[Pair.A.value], Pair.A)

        b_config = (
            config[Pair.B.value]
            if self.type == DuetBenchType.AB
            else config[Pair.A.value]
        )
        self.b = BenchmarkConfig(self, b_config, Pair.B)

    def get_or_inherit(self, key, default=None):
        if key in self.config:
            return self.config[key]
        elif key in self.duetbenchconfig:
            return self.duetbenchconfig[key]
        elif default is not None:
            return default
        else:
            raise ConfigException(f"{self} can't inherit {key}")

    def __str__(self):
        return str(vars(self))


class DuetBenchConfig:
    UNIQUE_VALUES = [
        "suite",
        "verbose",
        "seed",
        "docker_command",
        "duets",
        "artifacts",
        "schedule",
        "run_base",
    ]
    VALUES = DuetConfig.VALUES

    def __init__(self, config_filename):
        self.load_config(config_filename)

        self.duetbenchconfig: dict = self.config["duetbench"]

        check_valid_keys(
            "duetbench",
            self.duetbenchconfig,
            DuetBenchConfig.UNIQUE_VALUES + DuetBenchConfig.VALUES,
        )

        self.suite: str = self.duetbenchconfig["suite"]

        self.verbose: bool = self.duetbenchconfig.get("verbose")

        self.seed: int = self.duetbenchconfig.get("seed")

        self.docker_command: str = self.duetbenchconfig.get("docker_command")

        self.duet_names: List[str] = self.duetbenchconfig["duets"]

        self.scheduling: Schedule = Schedule(
            self.duetbenchconfig.get("schedule", Schedule.SEQUENTIAL.value)
        )

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
        r"(?P<suite>[a-zA-Z0-9_-]+)\.(?P<benchmark>[a-zA-Z0-9_-]+)\.(?P<runid>\d+)\.(?P<type>duet|sequential)\.(?P<duet_order>[a-zA-Z0-9_-]+)\.(?P<pair>[AB])\.(?P<result_file>.*)"
    )

    @staticmethod
    def parse(result_path: str):
        basename = os.path.basename(result_path)
        match = ResultFile.FILENAME_REGEX.match(basename)
        return (
            ResultFile(
                suite=match.group("suite"),
                benchmark=match.group("benchmark"),
                runid=match.group("runid"),
                type=Type(match.group("type")),
                duet_order=match.group("duet_order"),
                pair=match.group("pair"),
                result_file=match.group("result_file"),
                result_path=result_path,
            )
            if match
            else None
        )

    def __init__(
        self,
        suite: str,
        benchmark: str,
        runid: int,
        type: Type,
        duet_order: str,
        pair: str,
        result_file: str,
        result_path: str = None,  # set by parse method only
    ):
        self.suite = suite
        self.benchmark = benchmark
        self.runid = runid
        self.type = type
        self.duet_order = duet_order
        self.pair = pair
        self.result_file = result_file
        self.result_path = result_path

    def __repr__(self):
        return f"ResultFile({self.filename()})"

    def filename(self):
        return f"{self.suite}.{self.benchmark}.{self.runid}.{self.type.value}.{self.duet_order}.{self.pair}.{self.result_file}"
