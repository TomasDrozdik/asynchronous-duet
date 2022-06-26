import re
import pandas as pd
import json
from pathlib import Path

from duet.constants import RF, NS_PER_S, MS_PER_NS
from duet.config import ResultFile


def add_result_file_columns(df: pd.DataFrame, result_file: ResultFile) -> pd.DataFrame:
    df[RF.suite] = result_file.suite
    df[RF.benchmark] = result_file.benchmark
    df[RF.type] = result_file.type.value
    df["order"] = result_file.duet_order
    df[RF.pair] = result_file.pair
    df[RF.runid] = result_file.runid
    return df


def process_renaissance(result_file: ResultFile, logger) -> pd.DataFrame:
    with open(result_file.result_path) as json_file:
        results_json = json.load(json_file)

    vm_start_ms = results_json["environment"]["vm"]["start_unix_ms"]
    results = []

    if len(results_json["data"]) != 1:
        logger.warning(
            f"Duet run expects only single running benchmark, results contain these: {results_json['data'].keys()}"
        )

    for benchmark, benchmark_data in results_json["data"].items():
        for iteration, iteration_data in enumerate(benchmark_data["results"]):
            results.append(
                {
                    RF.benchmark: benchmark,
                    RF.iteration: iteration,
                    "epoch_start_ms": vm_start_ms,
                    RF.time_ns: iteration_data["duration_ns"],
                    "uptime_ns": iteration_data["uptime_ns"],
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
    df[RF.start_ns] = (df["epoch_start_ms"] * MS_PER_NS) + df["uptime_ns"]
    df[RF.end_ns] = df[RF.start_ns] + df[RF.time_ns]

    df = add_result_file_columns(df, result_file)
    return df


def process_dacapo(result_file: ResultFile, logger):
    df = pd.read_csv(result_file.result_path)
    df.rename({RF.time_ns: RF.time_ns}, axis=1, inplace=True)
    df = add_result_file_columns(df, result_file)
    df["c"] = 1
    df[RF.iteration] = df.groupby(
        by=[RF.suite, RF.benchmark, RF.type, RF.pair, RF.runid]
    )["c"].cumsum()
    df.drop("c", axis=1, inplace=True)

    if "start_unix" in df.columns:
        df[RF.start_ns] = df["start_unix"] * MS_PER_NS
    else:
        df[RF.start_ns] = df["total_ms"] * MS_PER_NS

    df[RF.end_ns] = df[RF.start_ns] + df[RF.time_ns]
    return df


def process_spec(result_file: ResultFile, logger):
    """
    SPEC CPU logs report timings of iterations in logs in a following way:

    Benchmark Times:
     Run Start:    2022-06-01 14:04:52 (1654092292)
     Rate Start:   2022-06-01 14:04:52 (1654092292.28303)
     Rate End:     2022-06-01 14:12:37 (1654092757.17651)
     Run Stop:     2022-06-01 14:12:37 (1654092757)
     Run Elapsed:  00:07:45 (465)
    """
    result_dir = Path(result_file.result_path)
    assert result_dir.is_dir()
    runlog = list(result_dir.glob("*002.log"))
    assert len(runlog) == 1
    runlog_lines = runlog[0].read_text().splitlines()

    rate_start_re = re.compile(r".*Rate Start:.*\((?P<timestamp>.*)\)")
    rate_end_re = re.compile(r".*Rate End:.*\((?P<timestamp>.*)\)")

    rate_starts = []
    rate_ends = []

    for line in runlog_lines:
        for regex, container in [
            (rate_start_re, rate_starts),
            (rate_end_re, rate_ends),
        ]:
            match = regex.match(line)
            if match:
                container.append(float(match.group(1)) * NS_PER_S)

    df = pd.DataFrame(
        {
            RF.start_ns: rate_starts,
            RF.end_ns: rate_ends,
        }
    )
    df[RF.iteration] = df.index
    df[RF.time_ns] = df[RF.end_ns] - df[RF.start_ns]

    df = add_result_file_columns(df, result_file)
    return df
