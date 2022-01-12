#!/usr/bin/env python3

import argparse
import os
import pandas as pd
import json
import logging
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


def convert_results_csv(results_dir: str, pair: str) -> pd.DataFrame:
    df = pd.read_csv(f"{results_dir}/results.csv")
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


def convert_results_json(results_dir, pair: str, config: json) -> pd.DataFrame:
    with open(f"{results_dir}/results.json") as json_file:
        results_json = json.load(json_file)

    vm_start_ms = results_json["environment"]["vm"]["start_unix_ms"]
    results = []
    iteration_time_sum = 0
    for benchmark, benchmark_data in results_json["data"].items():
        for iteration, iteration_data in enumerate(benchmark_data["results"]):
            iteration_time_sum = iteration_time_sum + iteration_data["duration_ns"]
            results.append({
                "benchmark": benchmark,
                "wallclock_start_ms": vm_start_ms, # TODO: this does not seem correct
                "epoch_start_ms": vm_start_ms,
                "iteration_time_ns": iteration_data["duration_ns"],
                "iteration": iteration,
                "machine": config.get("machine", None) if config else None,
                "provider": config.get("provider", None) if config else None,
                "jdk": results_json["environment"]["jre"]["name"],
                "jdk_version": results_json["environment"]["jre"]["version"],
                "time": results_json["environment"]["vm"]["start_iso"],
                "pair": pair,
                "kind": None,
                "total_ms": None,
                "process_cpu_time_ns": None,
                "compilation_time_ms": None,
                "compilation_total_ms": results_json["environment"]["vm"]["compiler"]["compilation_time_ms"],
            })
    print(f"Pair{pair}: epoch_start: {vm_start_ms} iterations: {iteration_time_sum}")
    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("results_a", metavar="A", type=str, help="results of duet A")
    parser.add_argument("results_b", metavar="B", type=str, help="results of duet B")
    parser.add_argument("--output", type=str, help="merged result .csv file")
    parser.add_argument("--config", type=str, help="config describing given run")

    args = parser.parse_args()

    config = None
    if args.config:
        config = json.load(args.config)


    resultsA = convert_results_json(args.results_a, "A", config)
    resultsB = convert_results_json(args.results_b, "B", config) 

    results = pd.concat([resultsA, resultsB])

    out_csv = args.output if args.output else f"{os.path.commonprefix([args.results_a, args.results_b])}.csv"
    print(f"Write results to: {out_csv}")
    results.to_csv(out_csv, index=False)


if __name__ == "__main__":
    main()
