#!/usr/bin/env python3

import argparse
import os
import pandas as pd
import logging
from enum import Enum
from duet.constants import AF, ARTIFACT_COL, BASE_COL, RF, RUN_ID_COL

from duet.duetconfig import ARTIFACTS_DIR, ResultFile
from duet.parsers_benchmark import process_renaissance, process_dacapo, process_spec
from duet.parsers_artifact import (
    parse_lscpu,
    parse_meminfo,
    strip_contents,
)


BENCHMARK_PARSERS = {
    "renaissance": process_renaissance,
    "dacapo": process_dacapo,
    "scalabench": process_dacapo,
    "speccpu": process_spec,
    "debug": lambda file, logger: pd.DataFrame(),
}


ARTIFACT_PARSERS = {
    "date": strip_contents,
    "hostname": strip_contents,
    "uname": strip_contents,
    "lscpu": parse_lscpu,
    "meminfo": parse_meminfo,
}


REQUIRED_SCHEMA = set(
    [
        "suite",
        "benchmark",
        "type",
        "runid",
        "iteration",
        "iteration_start_ns",
        "iteration_end_ns",
    ]
)


class SerializeEnum(Enum):
    JSON = "json"
    CSV = "csv"


def parse_result_file(result_path):
    result_file = ResultFile.parse(result_path)
    if not result_file:
        logging.warning(f"Failed to match result file `{result_path}")
        return None

    if result_file.suite not in BENCHMARK_PARSERS:
        logging.error(f"No parser available for benchmark {result_file.benchmark}")
        return None

    parser = BENCHMARK_PARSERS[result_file.suite]
    result_df = None
    try:
        result_df = parser(result_file, logging.getLogger(result_file.filename()))
        logging.info(f"Parsed: {result_file} with {result_df.shape[0]} iterations")
    except Exception as e:
        logging.error(f"Parsing: {result_file} failed with exception {e}")
    return result_df


def process_result(result_dir: str) -> pd.DataFrame:
    result_df = pd.DataFrame()

    artifacts = {}
    for file in os.listdir(result_dir):
        result_path = f"{result_dir}/{file}"

        if file == ARTIFACTS_DIR:
            try:
                artifacts = parse_artifacts(result_path)
            except Exception as e:
                logging.error(
                    f"Parse artifacts: {result_path} failed with exception {e}"
                )
            continue

        # Otherwise a file has to be ResultFile with fixed format
        partial_result_df = parse_result_file(result_path)
        if partial_result_df is not None:
            result_df = pd.concat([partial_result_df, result_df])

    for key, value in artifacts.items():
        result_df[key] = value

    if not set(result_df.columns).issuperset(REQUIRED_SCHEMA):
        logging.error(f"Columns: {result_df.columns}")

    return result_df


def parse_artifacts(artifacts_dir: str):
    parsed_artifacts = {}
    for file in os.listdir(artifacts_dir):
        if file in ARTIFACT_PARSERS:
            with open(f"{artifacts_dir}/{file}", "r") as f:
                contents = f.read()
            parsed_artifacts[file] = ARTIFACT_PARSERS[file](contents)
            logging.debug(f"Parsed artifact {file} as {parsed_artifacts[file]}")
        else:
            logging.info(
                f"Found artifact {file} but there is no registered parser for it"
            )
    return parsed_artifacts


def store_results(result_df: pd.DataFrame, output: str, format: SerializeEnum):
    logging.info(f"Write results to: {output}")
    if format == SerializeEnum.JSON:
        result_df.to_json(output)
    elif format == SerializeEnum.CSV:
        result_df.to_csv(output, index=False)
    else:
        raise NotImplementedError()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=str, nargs="+", help="Results directories")
    parser.add_argument(
        "-o",
        "--output",
        type=str,
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
        format="%(asctime)s  %(name)-12s %(levelname)-.1s  %(message)s",
        level=logging.INFO,
        datefmt="%b %d %H:%M:%S",
    )

    df = pd.DataFrame()
    for result_file in args.results:
        try:
            df = pd.concat([df, process_result(result_file)])
        except Exception as e:
            logging.error(f"Failed to parse {result_file} with: {e}")

    serialization = SerializeEnum.JSON if args.json else SerializeEnum.CSV
    if not args.output:
        if len(args.results) == 1:
            args.output = (
                f"{os.path.basename(args.results[0].rstrip('/'))}.{serialization.value}"
            )
        else:
            args.output = f"results.{serialization.value}"

    store_results(df, args.output, serialization)


if __name__ == "__main__":
    main()
