#!/usr/bin/env python3

import argparse
from datetime import datetime
import os
import sys
import traceback
import pandas as pd
import logging
from enum import Enum

from duet.duetconfig import ARTIFACTS_DIR, ResultFile, DuetBenchConfig
from duet.parsers_benchmark import process_renaissance
from duet.parsers_artifact import (
    parse_lscpu,
    parse_meminfo,
    strip_contents,
)


BENCHMARK_PARSERS = {
    "renaissance": process_renaissance,
    "debug": lambda file, logger: pd.DataFrame(),
}


ARTIFACT_PARSERS = {
    "date": strip_contents,
    "hostname": strip_contents,
    "uname": strip_contents,
    "lscpu": parse_lscpu,
    "meminfo": parse_meminfo,
}


class SerializeEnum(Enum):
    JSON = "json"
    CSV = "csv"


def process_results(results, config: DuetBenchConfig) -> pd.DataFrame:
    full_df = pd.DataFrame()
    for result in results:
        try:
            result_df = process_result(result, config, logging.getLogger(result))
            result_df = compute_iteration_duration(result_df)
        except Exception:
            logging.error(f"Processing results {result} failed with exception.")
            traceback.print_exc()
        else:
            full_df = pd.concat([result_df, full_df])
    return full_df


def process_result(
    result_dir: str, duet_config: DuetBenchConfig, logger
) -> pd.DataFrame:
    result_df = pd.DataFrame()

    artifacts = {}
    for file in os.listdir(result_dir):
        result_path = f"{result_dir}/{file}"

        if file == ARTIFACTS_DIR:
            artifacts = parse_artifacts(result_path, duet_config, logger)
            continue

        # Otherwise a file has to be ResultFile with fixed format
        result_file = ResultFile.parse(result_path)
        if not result_file:
            logger.warning(f"Failed to match result file `{result_path}")
            continue

        if result_file.suite not in BENCHMARK_PARSERS:
            logger.error(f"No parser available for benchmark {result_file.benchmark}")
            continue

        parser = BENCHMARK_PARSERS[result_file.suite]
        logger.info(f"Parsing: {result_file}")
        partial_df = parser(result_file, logging.getLogger(file))
        logger.info(f"Parsed: {result_file} with {partial_df.size} iterations")
        result_df = pd.concat([partial_df, result_df])

    for key, value in artifacts.items():
        result_df[key] = value

    return result_df


def parse_artifacts(artifacts_dir: str, duet_config: DuetBenchConfig, logger):
    parsed_artifacts = {}
    # If filename matches some artifact from configuration, see if there is method to parse it
    for file in os.listdir(artifacts_dir):
        if file in duet_config.artifacts.keys():
            if file in ARTIFACT_PARSERS:
                with open(f"{artifacts_dir}/{file}", "r") as f:
                    contents = f.read()
                parsed_artifacts[file] = ARTIFACT_PARSERS[file](contents)
                logger.debug(f"Parsed artifact {file} as {parsed_artifacts[file]}")
            else:
                logger.info(
                    f"Found artifact {file} but there is no registered parser for it"
                )
    return parsed_artifacts


def compute_iteration_duration(results: pd.DataFrame) -> pd.DataFrame:
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


def store_results(result_df: pd.DataFrame, output: str, format: SerializeEnum):
    out_path = f"{output}.{format.value}"
    logging.info(f"Write results to: {out_path}")
    if format == SerializeEnum.JSON:
        result_df.to_json(out_path)
    elif format == SerializeEnum.CSV:
        result_df.to_csv(out_path, index=False)
    else:
        raise NotImplementedError()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=str, nargs="+", help="Results directory")
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        required=True,
        help="YAML config file for the duet benchmark",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=f"results.{datetime.now().strftime('%Y-%m-%d_%H:%M')}",
        help="Output file",
    )
    parser_group_output = parser.add_mutually_exclusive_group(required=False)
    parser_group_output.add_argument(
        "--json", action="store_true", help="Serialize DataFrame to JSON"
    )
    parser_group_output.add_argument(
        "--csv", action="store_true", help="Serialize DataFrame to CSV (default)"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    logging.basicConfig(
        format="%(asctime)s  %(name)-20s %(levelname)-8s  %(message)s",
        level=logging.INFO,
        datefmt="%b %d %H:%M:%S",
    )

    try:
        config = DuetBenchConfig(args.config)
    except Exception as e:
        logging.critical(f"Critical config error: {e}")
        traceback.print_exc()
        sys.exit(1)

    df = process_results(args.results, config)
    store_results(
        df, args.output, SerializeEnum.JSON if args.json else SerializeEnum.CSV
    )


if __name__ == "__main__":
    main()
