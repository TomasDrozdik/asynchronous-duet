import pandas as pd
from duet.process import convert_ns, determine_environment

# Scipy data types
from collections import namedtuple
from dataclasses import make_dataclass

fields = ["confidence_interval", "bootstrap_distribution", "standard_error"]
BootstrapResult = make_dataclass("BootstrapResult", fields)
ConfidenceInterval = namedtuple("ConfidenceInterval", ["low", "high"])


class StopExecution(Exception):
    def _render_traceback_(self):
        pass


def load_raw():
    df = pd.concat(
        [
            pd.read_csv("../results/results.bare-metal.csv"),
            pd.read_csv("../results/amazon-mid-july.csv"),
            pd.read_csv("../results/results.teaching.csv"),
        ]
    )
    df = convert_ns(df)
    df = determine_environment(df)
    return df
