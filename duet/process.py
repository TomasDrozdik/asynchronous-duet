#!/usr/bin/env python3

import argparse
from enum import Enum
import logging
import numpy as np
import os
import pandas as pd
from scipy.stats import (
    bootstrap,
    gmean,
)
import traceback
from typing import List

from duet.constants import (
    AF,
    ARTIFACT_COL,
    BASE_COL,
    BENCHMARK_ID_COL,
    DF,
    PAIR_ID_COL,
    RF,
    RUN_ID_COL,
    TIME_D_NS_SUFFIX_COL,
)
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


def compute_overlaps(input_df: pd.DataFrame) -> pd.DataFrame:
    def overlap(interval1, interval2):
        start = max(interval1[0], interval2[0])
        end = min(interval1[1], interval2[1])
        return True if end - start > 0 else False

    input_df = input_df[input_df[RF.type] == "duet"]

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
    df = df.join(input_df, on="indexB", rsuffix=RF.b_suffix)
    b_suffix_columns_to_remove = set(
        [x for x in df.columns if x.endswith(RF.b_suffix)]
    ) - set(TIME_D_NS_SUFFIX_COL)
    df = df.drop(list(b_suffix_columns_to_remove), axis=1)
    df = df.rename(
        {
            RF.iteration: RF.iteration_A,
            RF.start_ns: RF.start_ns_A,
            RF.end_ns: RF.end_ns_A,
            RF.time_ns: RF.time_ns_A,
        },
        axis=1,
    )
    df[RF.overlap_start_ns] = df[[RF.start_ns_A, RF.start_ns_B]].max(axis=1)
    df[RF.overlap_end_ns] = df[[RF.end_ns_A, RF.end_ns_B]].min(axis=1)
    df[RF.overlap_time_ns] = df[RF.overlap_end_ns] - df[RF.overlap_start_ns]
    df = df.drop(["indexA", "indexB"], axis=1)
    return df


def compute_ci_seqn(df: pd.DataFrame, sample_type: str) -> pd.DataFrame:
    df = df[df[RF.type] == "seqn"]
    df = preprocess_data(df)

    if sample_type == "run_means":
        df = df.groupby(by=PAIR_ID_COL).agg(sample=(RF.time_ns, np.mean)).reset_index()
    elif sample_type == "one_sample":
        df = df.groupby(by=PAIR_ID_COL).agg(sample=(RF.time_ns, "sample")).reset_index()
        raise NotImplementedError(f"Sample type: {sample_type} does not work yet")
    elif sample_type == "run_merge":
        df = df.rename({RF.time_ns: "sample"}, axis=1)
    else:
        raise NotImplementedError(f"Unknown sample_type: {sample_type}")

    return (
        df.groupby(by=BENCHMARK_ID_COL + [RF.pair])
        .apply(
            lambda x: pd.Series(
                {"ci": bootstrap(data=(x["sample"],), statistic=np.mean)}
            )
        )
        .reset_index()
    )


def compute_ci_syncduet(df: pd.DataFrame, sample_type: str) -> pd.DataFrame:
    df = df[df[RF.type] == "syncduet"]
    df = preprocess_data(df)

    # Pair A/B runs to single row
    df = df.pivot_table(
        index=ARTIFACT_COL + RUN_ID_COL + [RF.iteration],
        columns=RF.pair,
        values=[RF.time_ns],
    ).reset_index()
    df.columns = [f"{i}_{j}" if j else i for i, j in df.columns]

    df[DF.pair_speedup] = df[RF.time_ns_A] / df[RF.time_ns_B]

    if sample_type == "run_means":
        df = df.groupby(ARTIFACT_COL + RUN_ID_COL).agg(sample=(DF.pair_speedup, gmean))
    elif sample_type == "one_sample":
        df = (
            df.groupby(by=PAIR_ID_COL)
            .agg(sample=(DF.pair_speedup, "sample"))
            .reset_index()
        )
        raise NotImplementedError(f"Sample type: {sample_type} does not work yet")
    elif sample_type == "run_merge":
        df = df.rename({DF.pair_speedup: "sample"}, axis=1)
    else:
        raise NotImplementedError(f"Unknown sample_type: {sample_type}")

    # Expected result (internally computed by bootstrap as the sample statistics):
    # df_ggmsr = df_gmsr.groupby(ARTIFACT_COL + BENCHMARK_ID_COL).agg(ggmsr=("gmsr", gmean))

    return (
        df.groupby(BENCHMARK_ID_COL)
        .apply(
            lambda x: pd.Series({"ci": bootstrap(data=(x["sample"],), statistic=gmean)})
        )
        .reset_index()
    )


def compute_ci_duet_no_overlaps(df: pd.DataFrame, sample_type: str) -> pd.DataFrame:
    df = df[df[RF.type] == "duet"]
    df = preprocess_data(df)

    if sample_type == "run_means":
        df = df.groupby(by=PAIR_ID_COL).agg(sample=(RF.time_ns, np.mean)).reset_index()
    elif sample_type == "one_sample":
        df = df.groupby(by=PAIR_ID_COL).agg(sample=(RF.time_ns, "sample")).reset_index()
        raise NotImplementedError(f"Sample type: {sample_type} does not work yet")
    elif sample_type == "run_merge":
        df = df.rename({RF.time_ns: "sample"}, axis=1)
    else:
        raise NotImplementedError(f"Unknown sample_type: {sample_type}")

    return (
        df.groupby(by=BENCHMARK_ID_COL + [RF.pair])
        .apply(
            lambda x: pd.Series(
                {"ci": bootstrap(data=(x["sample"],), statistic=np.mean)}
            )
        )
        .reset_index()
    )


def expand_confidence_interval(df: pd.DataFrame, ci_col="ci") -> pd.DataFrame:
    df["lo"] = df.apply(lambda x: x[ci_col].confidence_interval[0], axis=1)
    df["hi"] = df.apply(lambda x: x[ci_col].confidence_interval[1], axis=1)
    df["err"] = (df["hi"] - df["lo"]) / 2
    df["mid"] = df["lo"] + df["err"]
    df["se"] = df.apply(lambda x: x[ci_col].standard_error, axis=1)
    df = df.drop(ci_col, axis=1)
    return df


def preprocess_java(df: pd.DataFrame) -> pd.DataFrame:
    # Delete first half of iterations
    max_iteration = "iter_max"
    df[max_iteration] = df.groupby(ARTIFACT_COL + RUN_ID_COL)[RF.iteration].transform(
        max
    )
    df = df[df[RF.iteration] >= df[max_iteration] / 2]
    df = df.drop(max_iteration, axis=1)
    return df


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    is_java = df[RF.suite].isin(["renaissance", "dacapo", "scalabench"])
    df_java = df[is_java]
    df_rest = df[~is_java]
    df_java = preprocess_java(df_java)
    return pd.concat([df_java, df_rest])


def preprocess_overlaps(df: pd.DataFrame) -> pd.DataFrame:
    """Filter out small overlaps i.e. <10% of mean run time"""
    pass


def alter_score(df: pd.DataFrame, rate: float, pair="B") -> pd.DataFrame:
    """Alter speedup/slowdown score of results by given rate.

    For seqn/syncduet types simply speedup the time_ns, for duet it needs to
    re-compute start end timestamps to somewhat keep overlaps representative
    of actual slower pair.
    """
    pass


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
