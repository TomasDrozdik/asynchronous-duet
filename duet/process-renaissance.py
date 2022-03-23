#!/usr/bin/env python3

import argparse
import os
import re
import sys
import traceback
import pandas as pd
import json
import logging

from duetconfig import DuetBenchConfig


RESULTS_FORMAT = [
    "benchmark",
    "runid",
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


class RenaissanceResult:
    FILENAME_REGEX = re.compile(
        r"(?P<duet_name>[a-zA-Z0-9_-]+)\.(?P<runid>\d+)\.(?P<duet_type>[AB])\.results.json"
    )

    def __init__(self, result_path: os.path, config: str):
        self.result_path = result_path
        self.result_filename = os.path.basename(result_path)

        match = self.FILENAME_REGEX.match(self.result_filename)
        if not match:
            logging.warning(
                f"Failed to match `{self.result_path}` with `{self.FILENAME_REGEX}`"
            )

        self.duet_name = match.group("duet_name")
        self.pair = match.group("duet_type")
        self.runid = match.group("runid")

        self.config = config

    def convert_results_json(self) -> pd.DataFrame:
        with open(self.result_path) as json_file:
            results_json = json.load(json_file)

        vm_start_ms = results_json["environment"]["vm"]["start_unix_ms"]
        results = []
        total_iterations = 0

        if len(results_json["data"]) != 1:
            logging.warning(
                f"Duet run expects only single running benchmark, results contain these: {results_json['data'].keys()}"
            )

        for benchmark, benchmark_data in results_json["data"].items():
            for iteration, iteration_data in enumerate(benchmark_data["results"]):
                total_iterations += 1
                results.append(
                    {
                        "benchmark": benchmark,
                        "runid": self.runid,
                        "pair": self.pair,
                        "iteration": iteration,
                        "epoch_start_ms": vm_start_ms,
                        "iteration_time_ns": iteration_data["duration_ns"],
                        "machine": self.config.setup.machine,
                        "provider": self.config.setup.provider,
                        "jdk": results_json["environment"]["jre"]["name"],
                        "jdk_version": results_json["environment"]["jre"]["version"],
                        # TODO: Dunno what to put in following fields
                        # "wallclock_start_ms": vm_start_ms,
                        # "time": results_json["environment"]["vm"]["start_iso"],
                        # "kind": None,
                        # "total_ms": None,
                        # "process_cpu_time_ns": None,
                        # "compilation_time_ms": None,
                        # "compilation_total_ms": results_json["environment"]["vm"]["compiler"]["compilation_time_ms"],
                    }
                )

        logging.info(
            f"Processed file: {self.result_path}, runid: {self.runid}, pair: {self.pair}, total_iterations: {total_iterations}"
        )
        return pd.DataFrame(results)

    def convert_results_csv(self) -> pd.DataFrame:
        df = pd.read_csv(self.result_path)
        df["epoch_start_ms"] = df["vm_start_unix_ms"]
        df["iteration_time_ns"] = df["duration_ns"]
        df["iteration"] = df.groupby("benchmark").cumcount(ascending=True)
        df["runid"] = self.runid
        df["pair"] = self.pair
        # df["provider"] = None
        # df["jdk"] = None
        # df["jdk_version"] = None
        # df["wallclock_start_ms"] = df["vm_start_unix_ms"].astype(int) + df["uptime_ns"].astype(int) / 1000
        # df["time"] = None
        # df["kind"] = None
        # df["process_cpu_time_ns"] = None
        # df["compilation_time_ms"] = None
        # df["compilation_total_ms"] = None
        return df

    def convert(self) -> pd.DataFrame:
        return self.convert_results_json()


def process_renaissance(results_dir: os.path, config: DuetBenchConfig) -> pd.DataFrame:
    renaissance_results = [
        RenaissanceResult(os.path.join(results_dir, file), config)
        for file in os.listdir(results_dir)
    ]

    result = pd.DataFrame()
    for renaissance_result in renaissance_results:
        result = pd.concat([result, renaissance_result.convert()])
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "results", type=str, help="Results directory of Renaissance Duet"
    )
    parser.add_argument(
        "-c", "--config", type=str, required=True, help="Duet config file"
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        help="Output results.csv file, default <results_dir_name>.csv",
    )
    args = parser.parse_args()

    try:
        config = DuetBenchConfig(args.config)
    except Exception as e:
        logging.critical(f"Critical config error: {e}")
        traceback.print_exception(e)
        sys.exit(1)

    logging.basicConfig(
        format="%(asctime)s  %(name)-20s %(levelname)-8s  %(message)s",
        level=logging.INFO,
        datefmt="%b %d %H:%M:%S",
    )

    try:
        results_df = process_renaissance(args.results, config)
    except Exception as e:
        logging.critical(f"Process renaissance failed with exception: {e}")
        traceback.print_exception(e)
        sys.exit(1)

    out_csv = args.output if args.output else f"{os.path.basename(args.results)}.csv"
    logging.info(f"Write results to: {out_csv}")
    results_df.to_csv(out_csv, index=False)


if __name__ == "__main__":
    main()
