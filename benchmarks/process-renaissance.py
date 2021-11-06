#!/usr/bin/env python3

import argparse
import pandas as pd
from datetime import datetime

results_format = [
    "benchmark",
    "wallclock_start_ms",
    "epoch_start_ms",
    "iteration_time_ns",
    "iteration",
    "machine",
    "provider",
    "jdk",
    "jdk_version",
    "time",
    "pair",
    "kind",
    "total_ms",
    "process_cpu_time_ns",
    "compilation_time_ms",
    "compilation_total_ms",
]


renaissance_results_format = [
    "benchmark",
    "duration_ns",
    "uptime_ns",
    "vm_start_unix_ms",
]


def convert_results(csv_filename: str, pair: str) -> pd.DataFrame:
    df = pd.read_csv(csv_filename)
    df["wallclock_start_ms"] = df["vm_start_unix_ms"].astype(int) + df["uptime_ns"].astype(int) / 1000
    df["epoch_start_ms"] = df["vm_start_unix_ms"]
    df["iteration_time_ns"] = df["duration_ns"]
    df["iteration"] = df.groupby("benchmark").cumcount(ascending=True)
    df["provider"] = None
    df["jdk"] = None
    df["jdk_version"] = None
    df["time"] = datetime.now()
    df["pair"] = pair
    df["kind"] = None
    df["process_cpu_time_ns"] = None
    df["compilation_time_ms"] = None
    df["compilation_total_ms"] = None
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-A", type=str, required=True, help="duet A results file")
    parser.add_argument("-B", type=str, required=True, help="duet B results file")
    parser.add_argument("--output", type=str, help="Concateneted output .csv file")

    args = parser.parse_args()

    resultsA = convert_results(args.A, pair="A")
    resultsB = convert_results(args.B, pair="B") 
    results = pd.concat([resultsA, resultsB])
    results.to_csv(args.output)


if __name__ == "__main__":
    main()
