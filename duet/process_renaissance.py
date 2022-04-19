#!/usr/bin/env python3

import argparse
import os
import re
import sys
import traceback
import pandas as pd
import json
import logging

# from duet.duetconfig import DuetBenchConfig

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

    def __init__(self, result_path: str, duet_name: str, pair: str, runid: str):
        self.result_path = result_path
        self.duet_name = duet_name
        self.pair = pair
        self.runid = runid

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
                        "jdk": results_json["environment"]["jre"]["name"],
                        "jdk_version": results_json["environment"]["jre"]["version"],
                        # "machine":
                        # "provider":
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


def parse_renaissance_result(result_path: str) -> RenaissanceResult:
    result_filename = os.path.basename(result_path)

    match = RenaissanceResult.FILENAME_REGEX.match(result_filename)
    if match:
        return RenaissanceResult(
            result_path,
            match.group("duet_name"),
            match.group("duet_type"),
            match.group("runid"),
        )
    else:
        logging.warning(f"Failed to match `{result_path}`")


def process_renaissance(results_dir: os.path) -> pd.DataFrame:
    renaissance_results = []
    for file in os.listdir(results_dir):
        result = parse_renaissance_result(f"{results_dir}/{file}")
        if result:
            renaissance_results.append(result)

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
        "-o",
        "--output",
        type=str,
        help="Output results.csv file, default <results_dir_name>.csv",
    )
    # For now deprecated parameter config
    # jparser.add_argument("-c", "--config", type=str, help="Duet config file")
    args = parser.parse_args()

    # config = None
    # if args.config:
    #    try:
    #        config = DuetBenchConfig(args.config)
    #    except Exception as e:
    #        logging.critical(f"Critical config error: {e}")
    #        traceback.print_exc()
    #        sys.exit(1)

    logging.basicConfig(
        format="%(asctime)s  %(name)-20s %(levelname)-8s  %(message)s",
        level=logging.INFO,
        datefmt="%b %d %H:%M:%S",
    )

    try:
        results_df = process_renaissance(args.results)
    except Exception as e:
        logging.critical(f"Process renaissance failed with exception: {e}")
        traceback.print_exc()
        sys.exit(1)

    out_csv = args.output if args.output else f"{os.path.basename(args.results)}.csv"
    logging.info(f"Write results to: {out_csv}")
    results_df.to_csv(out_csv, index=False)


if __name__ == "__main__":
    main()
