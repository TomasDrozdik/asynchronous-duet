#!/usr/bin/env python3

import argparse
import os
import sys
import traceback
import pandas as pd
import json
import logging

from duet.duetconfig import ResultFile


class RenaissanceResult:
    def __init__(self, result_file: ResultFile):
        self.result_file = result_file

    def convert_results_json(self) -> pd.DataFrame:
        df = None
        try:
            with open(self.result_file.result_path) as json_file:
                results_json = json.load(json_file)
            df = self._convert_to_df(results_json)
        except Exception:
            logging.error("ReneissanceResult failed with exception")
            traceback.print_exc()
            df = None
        return df

    def _convert_to_df(self, results_json) -> pd.DataFrame:
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
                        "runid": self.result_file.run_id,
                        "type": self.result_file.type,
                        "pair": self.result_file.pair,
                        "order": self.result_file.run_order,
                        "iteration": iteration,
                        "epoch_start_ms": vm_start_ms,
                        "iteration_time_ns": iteration_data["duration_ns"],
                        # TODO: Figure out what to put in the following fields
                        # "jdk": results_json["environment"]["jre"]["name"],
                        # "jdk_version": results_json["environment"]["jre"]["version"],
                        # "machine": # parse artifacts
                        # "provider": # parse artifacts
                        # "wallclock_start_ms": vm_start_ms,
                        # "time": results_json["environment"]["vm"]["start_iso"],
                        # "kind": None,
                        # "total_ms": None,
                        # "process_cpu_time_ns": None,
                        # "compilation_time_ms": None,
                        # "compilation_total_ms": results_json["environment"]["vm"]["compiler"]["compilation_time_ms"],
                    }
                )

        df = pd.DataFrame(results)
        logging.info(
            f"Processed file: {self.result_file.result_path}, runid: {self.result_file.run_id}, pair: {self.result_file.pair}, total_iterations: {total_iterations}"
        )
        return df


def process_renaissance(results_dir: os.path) -> pd.DataFrame:
    renaissance_results = []
    for file in os.listdir(results_dir):
        result_path = f"{results_dir}/{file}"
        result_file = ResultFile.parse(result_path)
        if not result_file:
            logging.warning(f"Failed to match result file `{result_path}")
        else:
            renaissance_results.append(RenaissanceResult(result_file))

    result = pd.DataFrame()
    for renaissance_result in renaissance_results:
        df = renaissance_result.convert_results_json()
        if df is not None:
            result = pd.concat([df, result])
    return result


def pre_process_iterations(results: pd.DataFrame) -> pd.DataFrame:
    results["iteration_start_ns"] = results.groupby(
        ["benchmark", "runid", "type", "pair"]
    )["iteration_time_ns"].transform(pd.Series.cumsum)

    results["iteration_start_ns"] = results.groupby(
        ["benchmark", "runid", "type", "pair"]
    )["iteration_start_ns"].shift(1, fill_value=0)

    results["iteration_start_ns"] = (
        results["iteration_start_ns"] + results["epoch_start_ms"] * 1_000_000
    )

    results["iteration_end_ns"] = (
        results["iteration_start_ns"] + results["iteration_time_ns"]
    )
    return results


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
    args = parser.parse_args()

    logging.basicConfig(
        format="%(asctime)s  %(name)-20s %(levelname)-8s  %(message)s",
        level=logging.INFO,
        datefmt="%b %d %H:%M:%S",
    )

    try:
        results_df = process_renaissance(args.results)
        results_df = pre_process_iterations(results_df)
    except Exception as e:
        logging.critical(f"Process renaissance failed with exception: {e}")
        traceback.print_exc()
        sys.exit(1)

    out_csv = args.output if args.output else f"{os.path.basename(args.results)}.csv"
    logging.info(f"Write results to: {out_csv}")
    results_df.to_csv(out_csv, index=False)


if __name__ == "__main__":
    main()
