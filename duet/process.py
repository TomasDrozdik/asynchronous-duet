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
    mannwhitneyu,
)
import traceback
from typing import List

from duet.constants import (
    AF,
    ARTIFACT_COL,
    BASE_COL,
    BENCHMARK_ENV_COL,
    DF,
    RF,
    RUN_ID_COL,
    TIME_D_NS_COL,
)
from duet.config import ARTIFACTS_DIR, ResultFile
from duet.parsers_benchmark import (
    post_process_dacapo,
    process_renaissance,
    process_dacapo,
    process_spec,
)
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
            # Check if there are old Dacapo/Scalabench results
            df = post_process_dacapo(df, logger=logging)
            logging.info(f"Finished parsing {result_file} with {df.shape[0]} records")
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
        logging.debug(f"Parsed: {result_file} with {result_df.shape[0]} iterations")
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
            logging.warning(
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
    overlap_df = _compute_overlaps(input_df)
    return _cleanup_overlap_columns(overlap_df)


def _compute_overlaps(input_df: pd.DataFrame) -> pd.DataFrame:
    def overlap(interval1, interval2):
        start = max(interval1[0], interval2[0])
        end = min(interval1[1], interval2[1])
        return end - start > 0

    input_df = input_df[input_df[RF.type] == "duet"].reset_index()

    # Find indices of overlapping A/B iterations
    runs = input_df.groupby(by=RUN_ID_COL + ARTIFACT_COL)
    overlap_indices = []
    for _, run in runs:
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

    # Join A/B indices together
    df = pd.DataFrame(
        {
            "indexA": [indexA for indexA, _ in overlap_indices],
            "indexB": [indexB for _, indexB in overlap_indices],
        }
    )
    df = df.join(input_df, on="indexA")
    df = df.join(input_df, on="indexB", rsuffix=RF.b_suffix)

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

    # for col in BENCHMARK_ID_COL:
    #    assert((df[col] == df[col + RF.b_suffix]).all())
    # assert((df[RF.overlap_time_ns] >= 0).all())
    return df


def _cleanup_overlap_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop(["indexA", "indexB"], axis=1)
    b_suffix_columns_to_remove = set(
        [x for x in df.columns if x.endswith(RF.b_suffix)]
    ) - set([x + RF.b_suffix for x in TIME_D_NS_COL + [RF.iteration]])
    a_columns_to_remove = [RF.pair]
    df = df.drop(list(b_suffix_columns_to_remove) + a_columns_to_remove, axis=1)
    return df


def _apply_sampling(df: pd.DataFrame, sample_col: str, statistics, sample_type: str):
    if sample_type == "run_means":
        df = (
            df.groupby(by=ARTIFACT_COL + RUN_ID_COL)
            .agg(sample=(sample_col, statistics))
            .reset_index()
        )
    elif sample_type == "run_merge":
        df = df.rename({sample_col: DF.sample}, axis=1)
    else:
        raise NotImplementedError(f"Unknown sample_type: {sample_type}")
    return df


def pair_iterations_sequentially(df: pd.DataFrame) -> pd.DataFrame:
    """Pair runs on matching indexes together.
    Seqn runs are matched to random matching run because of RMIT. Syncduet runs are matched properly
    as their iterations are synced. Duet runs are matched per iteration but they might not overlap.
    """
    df = df.pivot_table(
        index=ARTIFACT_COL + RUN_ID_COL + [RF.iteration],
        columns=RF.pair,
        values=[RF.time_ns],
    ).reset_index()
    df.columns = [f"{i}_{j}" if j else i for i, j in df.columns]
    # Drop rows with Nan as these don't have matching iteration result because of a timeout
    df = df.dropna(subset=[RF.time_ns_A, RF.time_ns_B])
    return df


def pair_overlapping_duet_iterations(df: pd.DataFrame, overlap: float) -> pd.DataFrame:
    """Pair duet iterations only if iteration_time_A * overlap < overlap_time_AB and iteration_time_B * overlap < overlap_time_AB."""
    assert overlap > 0 and overlap <= 1
    df_overlaps = compute_overlaps(df)
    return preprocess_overlaps(df_overlaps, p=overlap)


def compute_ci_pair_difference(
    df: pd.DataFrame, sample_type: str, pair_function, **kwargs
) -> pd.DataFrame:
    # Compute grand mean so that we can later compute relative CI width
    df_grand_mean = df.groupby(BENCHMARK_ENV_COL).agg(grand_mean=(RF.time_ns, np.mean))

    df = pair_function(df)
    df[DF.pair_diff] = df[RF.time_ns_A] - df[RF.time_ns_B]

    # Compute CI with given sampling
    df = _apply_sampling(df, DF.pair_diff, np.mean, sample_type)
    df = (
        df.groupby(by=BENCHMARK_ENV_COL)
        .apply(
            lambda x: pd.Series(
                {DF.ci: bootstrap(data=(x[DF.sample],), statistic=np.mean, **kwargs)}
            )
            if len(x) >= 2
            else None
        )
        .reset_index()
    )
    df = df.dropna()
    df = expand_confidence_interval(df)

    df = df.merge(df_grand_mean, on=BENCHMARK_ENV_COL)
    df[DF.ci_width] = (df["hi"] - df["lo"]) / df["grand_mean"]
    return df


def compute_ci_pair_speedup(
    df: pd.DataFrame, sample_type: str, **kwargs
) -> pd.DataFrame:
    df = pair_iterations_sequentially(df)
    df[DF.pair_speedup] = df[RF.time_ns_A] / df[RF.time_ns_B]

    # Compute CI with given sampling
    # Expected result (internally computed by bootstrap as the sample statistics):
    # df_ggmsr = df_gmsr.groupby(ARTIFACT_COL + BENCHMARK_ID_COL).agg(ggmsr=("gmsr", gmean))
    df = _apply_sampling(df, DF.pair_speedup, np.mean, sample_type)
    df = (
        df.groupby(BENCHMARK_ENV_COL)
        .apply(
            lambda x: pd.Series(
                {DF.ci: bootstrap(data=(x["sample"],), statistic=gmean, **kwargs)}
            )
            if len(x) >= 2
            else None
        )
        .reset_index()
    )
    df = df.dropna()

    df = expand_confidence_interval(df)
    df[DF.ci_width] = df["hi"] - df["lo"]
    return df


def compute_ci(df_input: pd.DataFrame, overlap_rates=[0.4]) -> pd.DataFrame:
    dfs = []
    df = compute_ci_pair_difference(
        df_input[df_input[RF.type] != "duet"],
        sample_type="run_means",
        pair_function=pair_iterations_sequentially,
    )
    df[DF.overlap_rate] = None
    dfs.append(df)

    for overlap_rate in overlap_rates:
        df = compute_ci_pair_difference(
            df_input[df_input[RF.type] == "duet"],
            sample_type="run_means",
            pair_function=lambda x: pair_overlapping_duet_iterations(x, overlap_rate),
        )
        df[DF.overlap_rate] = overlap_rate
        dfs.append(df)

    return pd.concat(dfs).reset_index()


def arbiter_ci_contains_zero(df_ci: pd.DataFrame) -> pd.DataFrame:
    """Decide if A/B pairs differ based on difference CI encompassing 0"""
    df_ci[DF.err_ci] = 0
    mask = np.absolute(df_ci["mid"]) <= df_ci["err"]
    df_ci.loc[~mask, DF.err_ci] = df_ci["mid"] - df_ci["err"]
    df_ci[DF.match_ci] = df_ci[DF.err_ci] == 0
    return df_ci


def arbiter_utest(df: pd.DataFrame, pvalue=0.05, **kwargs) -> pd.DataFrame:
    """Decide if A/B pairs differ based on Mann-Whitney u-test"""
    df_data = []
    groups = df.groupby(BENCHMARK_ENV_COL)
    for group, values in groups:
        values_A = values[values[RF.pair] == "A"]
        values_B = values[values[RF.pair] == "B"]
        if values_A[RF.time_ns].count() > 0 and values_B[RF.time_ns].count() > 0:
            utest = mannwhitneyu(x=values_A[RF.time_ns], y=values_B[RF.time_ns])
            df_data.append(group + (utest.statistic, utest.pvalue))
        else:
            print(f"Utest failed for: {group}")
    df_pred = pd.DataFrame(
        df_data, columns=BENCHMARK_ENV_COL + [DF.u_test_statistics, DF.u_test_pvalue]
    )
    df_pred[DF.match_utest] = df_pred[DF.u_test_pvalue] > pvalue
    return df_pred


def arbiter(df_prep: pd.DataFrame, df_ci: pd.DataFrame) -> pd.DataFrame:
    df_pred_ci = arbiter_ci_contains_zero(df_ci)
    df_pred_utest = arbiter_utest(df_prep)
    return df_pred_ci.merge(df_pred_utest, validate="1:1")


def group_predictions(df: pd.DataFrame, ci=True, utest=True) -> pd.DataFrame:
    ci_kwargs = {
        "total_count_ci": (DF.match_ci, "count"),
        "match_ci": (DF.match_ci, "sum"),
        "err_ci": (DF.err_ci, "sum"),
    }
    utest_kwargs = {
        "total_count_utest": (DF.match_utest, "count"),
        "match_utest": (DF.match_utest, "sum"),
    }
    kwargs = {}
    if ci:
        kwargs = {**kwargs, **ci_kwargs}
    if utest:
        kwargs = {**kwargs, **utest_kwargs}

    df = df.groupby(by=[DF.env, RF.suite, RF.type]).agg(**kwargs).reset_index()
    if ci:
        df[DF.match_ratio_ci] = df["match_ci"] / df["total_count_ci"]
    if utest:
        df[DF.match_ratio_utest] = df["match_utest"] / df["total_count_utest"]
    return df


def expand_confidence_interval(df: pd.DataFrame, ci_col=DF.ci) -> pd.DataFrame:
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


def preprocess_overlaps(df_overlaps: pd.DataFrame, p=0.1) -> pd.DataFrame:
    """Filter out small overlaps i.e. <10% of mean run time"""
    assert 0 < p and p < 1
    return df_overlaps[
        (df_overlaps[RF.time_ns_A] * p < df_overlaps[RF.overlap_time_ns])
        & (df_overlaps[RF.time_ns_B] * p < df_overlaps[RF.overlap_time_ns])
    ]


def alter_score(df: pd.DataFrame, slowdown: float, pair="B") -> pd.DataFrame:
    """Alter - slowdown score of results by given rate.

    For seqn/syncduet types simply speeding up the time_ns is sufficient,
    for duet it needs to re-compute start/end timestamps to somewhat keep
    overlaps representative of actual slower pair.
    Thus we can unify it and recompute the timestamps.

    Requires DF with ARTIFACT_COL + BASE_COL
    """

    def recompute_timestamps(df: pd.DataFrame):
        df[RF.time_ns] = (1 + slowdown) * df[RF.time_ns]
        df[RF.end_ns] = df[RF.start_ns] + df[RF.time_ns]
        return df

    def recompute_timestamps_duet(df: pd.DataFrame):
        delta_time = "delta_time"
        df[delta_time] = slowdown * df[RF.time_ns]
        df[delta_time] = df[delta_time].cumsum()
        df[RF.start_ns] += df[delta_time].shift(1, fill_value=0)
        df[RF.end_ns] += df[delta_time]
        # assert all((1 + rate) * df[RF.time_ns] == df[RF.end] - df[RF.start]) - does not hold supposedly because of float operations
        df[RF.time_ns] = df[RF.end_ns] - df[RF.start_ns]
        df = df.drop(delta_time, axis=1)
        return df

    duet_mask = df[RF.type] == "duet"
    alter_pair_mask = df[RF.pair] == pair
    return pd.concat(
        [
            df[alter_pair_mask & duet_mask]
            .groupby(by=ARTIFACT_COL + RUN_ID_COL, group_keys=False)
            .apply(recompute_timestamps_duet),
            df[alter_pair_mask & ~duet_mask]
            .groupby(by=ARTIFACT_COL + RUN_ID_COL, group_keys=False)
            .apply(recompute_timestamps),
            df[~alter_pair_mask],
        ]
    )


def run_time_seqn(df: pd.DataFrame):
    df = (
        df[df[RF.type] == "seqn"]
        .groupby(by=ARTIFACT_COL + RUN_ID_COL + [RF.pair])
        .agg(
            pair_start=(RF.start_ns, min),
            pair_end=(RF.end_ns, max),
            iteration_count=(RF.iteration, len),
        )
        .reset_index()
    )
    df["pair_duration"] = df["pair_end"] - df["pair_start"]
    return (
        df.groupby(by=ARTIFACT_COL + RUN_ID_COL)
        .agg(
            run_time_ns=("pair_duration", sum), iteration_count=("iteration_count", sum)
        )
        .reset_index()
    )


def run_time_duet(df: pd.DataFrame):
    df = (
        df[df[RF.type].isin(["duet", "syncduet"])]
        .groupby(by=ARTIFACT_COL + RUN_ID_COL)
        .agg(
            run_start=(RF.start_ns, min),
            run_end=(RF.end_ns, max),
            iteration_count=(RF.iteration, len),
        )
        .reset_index()
    )
    df[DF.run_time_ns] = df["run_end"] - df["run_start"]
    df = df.drop(["run_start", "run_end"], axis=1)
    return df


def run_time(df: pd.DataFrame):
    df = pd.concat(
        [
            run_time_seqn(df),
            run_time_duet(df),
        ]
    )
    df[DF.run_time] = df[DF.run_time_ns].apply(
        lambda x: pd.Timedelta(value=x, unit="ns").seconds
    )
    return df


def determine_environment(df: pd.DataFrame) -> pd.DataFrame:
    conditions = [
        df[AF.hostname].str.startswith("cirrus"),
        df[AF.hostname].str.startswith("teaching"),
        df[AF.hostname].str.startswith("ip-"),
    ]
    choices = ["bare-metal", "shared-vm", "AWS t3.medium"]
    df[DF.env] = np.select(conditions, choices, default=None)
    return df


def convert_ns(df: pd.DataFrame) -> pd.DataFrame:
    df[RF.start] = pd.to_datetime(df[RF.start_ns], unit="ns")
    df[RF.end] = pd.to_datetime(df[RF.end_ns], unit="ns")
    df[RF.time] = (df[RF.end] - df[RF.start]).dt.seconds
    return df


def convert_ns_overlaps(df: pd.DataFrame) -> pd.DataFrame:
    df[RF.overlap_start] = pd.to_datetime(df[RF.overlap_start_ns], unit="ns")
    df[RF.overlap_end] = pd.to_datetime(df[RF.overlap_end_ns], unit="ns")
    df[RF.overlap_time] = (df[RF.overlap_end] - df[RF.overlap_start]).dt.seconds

    df[RF.start_A] = pd.to_datetime(df[RF.start_ns_A], unit="ns")
    df[RF.end_A] = pd.to_datetime(df[RF.end_ns_A], unit="ns")
    df[RF.time_A] = (df[RF.end_A] - df[RF.start_A]).dt.seconds

    df[RF.start_B] = pd.to_datetime(df[RF.start_ns_B], unit="ns")
    df[RF.end_B] = pd.to_datetime(df[RF.end_ns_B], unit="ns")
    df[RF.time_B] = (df[RF.end_B] - df[RF.start_B]).dt.seconds
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

    subparsers = parser.add_subparsers(title="command", dest="command", required=True)

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
