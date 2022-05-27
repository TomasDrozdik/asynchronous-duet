import pandas as pd
import json

from duet.duetconfig import ResultFile


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
                    "suite": result_file.suite,
                    "benchmark": benchmark,
                    "runid": result_file.runid,
                    "iteration": iteration,
                    "type": result_file.type.value,
                    "pair": result_file.pair,
                    "order": result_file.duet_order,
                    "epoch_start_ms": vm_start_ms,
                    "iteration_duration_ns": iteration_data["duration_ns"],
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
    df["iteration_start_ns"] = (df["epoch_start_ms"] * 1_000_000) + df["uptime_ns"]
    df["iteration_end_ns"] = df["iteration_start_ns"] + df["iteration_duration_ns"]
    return df


def process_dacapo(result_file: ResultFile, logger):
    df = pd.read_csv(result_file.result_path)
    df.rename({"iteration_time_ns": "iteration_duration_ns"}, axis=1, inplace=True)
    df["suite"] = result_file.suite
    df["type"] = result_file.type.value
    df["order"] = result_file.duet_order
    df["pair"] = result_file.pair
    df["runid"] = result_file.runid

    df["c"] = 1
    df["iteration"] = df.groupby(by=["suite", "type", "order", "pair", "runid"])[
        "c"
    ].cumsum()
    df.drop("c", axis=1, inplace=True)

    df["iteration_start_ns"] = df["total_ms"] + df["iteration_duration_ns"]
    df["iteration_end_ns"] = df["iteration_start_ns"] + df["iteration_duration_ns"]
    return df
