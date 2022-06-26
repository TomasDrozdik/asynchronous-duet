#!/usr/bin/env python3

import argparse
from typing import List
import os
import traceback
import pandas as pd
import logging
from enum import Enum

from duet.constants import AF, ARTIFACT_COL, BASE_COL, RF, RUN_ID_COL
from duet.config import ARTIFACTS_DIR, ResultFile
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
    AF.date: strip_contents,
    AF.hostname: strip_contents,
    AF.uname: strip_contents,
    AF.lscpu: parse_lscpu,
    AF.meminfo: parse_meminfo,
}


REQUIRED_SCHEMA = set(BASE_COL)


class SerializeEnum(Enum):
    JSON = "json"
    CSV = "csv"


def parse_result_files(results: List[str]):
    df = pd.DataFrame()
    for result_file in results:
        try:
            df = pd.concat([df, process_result(result_file)])
        except Exception as e:
            logging.error(f"Failed to parse {result_file} with: {e}")
    return df


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


def compute_overlaps(input_df):
    def overlap(interval1, interval2):
        start = max(interval1[0], interval2[0])
        end = min(interval1[1], interval2[1])
        return True if end - start > 0 else False

    input_df = input_df[input_df[RF.type] == "duet"]
    input_df = input_df[BASE_COL + ARTIFACT_COL]

    runs = input_df.groupby(by=RUN_ID_COL + ARTIFACT_COL)
    overlap_indices = []
    for name, run in runs:
        logging.debug(f"process run group {name}")
        dfA = run[run[RF.pair] == "A"]
        dfB = run[run[RF.pair] == "B"]
        dfA = dfA.sort_values(by=[RF.start_ns])
        dfB = dfB.sort_values(by=[RF.start_ns])

        iA = 0
        iB = 0
        while iA < dfA.shape[0] and iB < dfB.shape[0]:
            rowA = dfA.iloc[iA]
            intervalA = rowA[RF.start_ns], rowA[RF.end_ns]

            rowB = dfB.iloc[iB]
            intervalB = rowB[RF.start_ns], rowB[RF.end_ns]

            if overlap(intervalA, intervalB):
                overlap_indices.append((dfA.index[iA], dfB.index[iB]))

            if intervalA[0] == intervalB[0]:
                if intervalA[1] <= intervalB[1]:
                    iA += 1
                else:
                    iB += 1
            elif intervalA[0] < intervalB[0]:
                iA += 1
            else:
                iB += 1

    df = pd.DataFrame(
        {
            "indexA": [indexA for indexA, _ in overlap_indices],
            "indexB": [indexB for _, indexB in overlap_indices],
        }
    )
    df = df.join(input_df, on="indexA")
    df = df.join(input_df, on="indexB", rsuffix="_B")
    df = df.drop([RF.pair] + [f"{col}_B" for col in RUN_ID_COL], axis=1)
    df = df.rename(
        {
            RF.iteration: RF.iteration + "_A",
            RF.start_ns: RF.start_ns + "_A",
            RF.end_ns: RF.end_ns + "_A",
        },
        axis=1,
    )
    df[RF.overlap_start_ns] = df[[RF.start_ns + "_A", RF.start_ns + "_B"]].max(axis=1)
    df[RF.overlap_end_ns] = df[[RF.end_ns + "_A", RF.end_ns + "_B"]].min(axis=1)
    df[RF.overlap_time_ns] = df[RF.overlap_end_ns] - df[RF.overlap_start_ns]
    df = df.drop(["indexA", "indexB"], axis=1)
    return df


def parse_args():
    parser = argparse.ArgumentParser()
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

    subparsers = parser.add_subparsers(title="command", dest="command")

    sub_parse = subparsers.add_parser("parse")
    sub_parse.add_argument("results", type=str, nargs="+", help="Results directories")

    sub_overlaps = subparsers.add_parser("overlaps")
    sub_overlaps.add_argument("parsed_csv", type=str, help="Previously parsed csv")

    return parser.parse_args()


def main():
    args = parse_args()

    logging.basicConfig(
        format="%(asctime)s  %(name)-12s %(levelname)-.1s  %(message)s",
        level=logging.INFO,
        datefmt="%b %d %H:%M:%S",
    )

    try:
        if args.command == "parse":
            df = parse_result_files(args.results)
        elif args.command == "overlaps":
            input_df = pd.read_csv(args.parsed_csv)
            df = compute_overlaps(input_df)
    except Exception:
        logging.critical(f"Command {args} failed  with exception")
        traceback.print_exc()
        exit(1)

    serialization = SerializeEnum.JSON if args.json else SerializeEnum.CSV
    if not args.output:
        if args.command == "parse" and len(args.results) == 1:
            basename = os.path.basename(args.results[0]).rstrip("/")
            args.output = f"{basename}.{serialization.value}"
        elif args.command == "overlaps":
            basename = ".".join(os.path.basename(args.parsed_csv).split(".")[:-1])
            args.output = f"{basename}.overlaps.{serialization.value}"
        else:
            args.output = f"results.{serialization.value}"

    store_results(df, args.output, serialization)


if __name__ == "__main__":
    main()
