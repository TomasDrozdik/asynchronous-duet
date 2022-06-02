#!/bin/env python3

import argparse
import datetime
import logging
import os
import random
import signal
import subprocess
import sys
import traceback
from typing import List

from duet.duetconfig import (
    ARTIFACTS_DIR,
    BenchmarkConfig,
    DuetBenchConfig,
    DuetConfig,
    DuetOrder,
    Pair,
    ResultFile,
    Schedule,
    Type,
)


DOCKER = "docker"


class Runner:
    def __init__(self, logger):
        self.logger = logger
        self.async_processes_next_id = 0
        self.async_processes = {}  # handle to process mapping

    @staticmethod
    def split_cmd(cmd) -> List[str]:
        if isinstance(cmd, list):
            return cmd
        elif isinstance(cmd, str):
            return cmd.split()

    def run(self, cmd):
        self.logger.debug(f"run> {cmd}")
        p = subprocess.run(
            self.split_cmd(cmd), stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        self.log(p.stdout, p.stderr)
        if p.returncode:
            raise RuntimeError(
                f"Running '{cmd}' finished with non-zero exit status {p.returncode}"
            )

    def run_async(self, cmd) -> int:
        self.logger.debug(f"run_async({self.async_processes_next_id})> {cmd}")
        try:
            async_process = subprocess.Popen(
                self.split_cmd(cmd), stdout=subprocess.PIPE, stderr=subprocess.STDOUT
            )
        except Exception as e:
            raise RuntimeError(f"Launch of async process '{cmd}' threw exception: {e}")
        async_handle = self.async_processes_next_id
        self.async_processes[async_handle] = async_process
        self.async_processes_next_id = self.async_processes_next_id + 1
        return async_handle

    def wait_async(self, handle: int):
        assert handle in self.async_processes.keys()
        self.logger.debug(f"wait_async({handle})")

        p = self.async_processes.pop(handle)
        (stdout, stderr) = p.communicate()
        returncode = p.wait()
        self.log(stdout, stderr)
        if returncode:
            raise RuntimeError(
                f"Running of async task finished with non-zero exit status: {returncode}"
            )

    def log(self, stdout, stderr):
        if stdout:
            for line in stdout.decode("utf-8").splitlines():
                line = line.strip()
                if line:
                    self.logger.info(">    " + line)
        if stderr:
            for line in stderr.decode("utf-8").strip().splitlines():
                line = line.strip()
                if line:
                    self.logger.warning(">     " + line)


class Benchmark:
    def __init__(self, config: BenchmarkConfig, logger):
        self.config = config
        self.logger = logger

        self.runner = Runner(self.logger)
        self.async_handle = None

    def start_instance(self):
        self.runner.run(
            f"{DOCKER} run --name {self.config.container} -it -d {self.config.image} bash"
        )

    def run_async(self):
        split_cmd = f"{DOCKER} exec {self.config.container} bash -c".split()
        split_cmd.append(f"{self.config.run_command}")
        self.async_handle = self.runner.run_async(split_cmd)

    def wait(self):
        assert self.async_handle is not None
        self.runner.wait_async(self.async_handle)

    def get_results(self, results_dir: str, runid: int, type: Type, duet_order: str):
        for remote_result_path in self.config.result_files:
            # Since docker cp cannot rename the file (remote_result_path), we first copy the file
            # (local_tmp_path) and then rename it (local_result_path).
            filename = os.path.basename(remote_result_path)

            local_tmp_path = f"{results_dir}/{filename}"

            if os.path.exists(local_tmp_path):
                raise RuntimeError(
                    f"Result name conflict, move of `{remote_result_path}` to `{local_tmp_path}` failed"
                )

            self.runner.run(
                f"{DOCKER} cp {self.config.container}:{remote_result_path} {local_tmp_path}"
            )

            result_file = ResultFile(
                self.config.suite,
                self.config.benchmark,
                runid,
                type,
                duet_order,
                pair=self.config.pair.value,
                result_file=filename,
            )

            local_result_path = f"{results_dir}/{result_file.filename()}"
            os.rename(local_tmp_path, local_result_path)

    def cleanup(self, rm: bool):  # nothrow
        try:
            self.runner.run(f"{DOCKER} stop {self.config.container}")
        except RuntimeError as e:
            self.logger.warn(f"CLEANUP {DOCKER} stop hit exception: {e}")
        if rm:
            try:
                self.runner.run(f"{DOCKER} rm {self.config.container}")
            except RuntimeError as e:
                self.logger.warn(f"CLEANUP {DOCKER} rm hit exception: {e}")


class SequentialBenchmarkRunner:
    def __init__(self, config: DuetConfig, results_dir: str, runid: int, pair: Pair):
        self.config = config
        self.results_dir = results_dir
        self.runid = runid

        benchmark_config = self.config.a if pair == Pair.A else self.config.b
        self.benchmark = Benchmark(
            benchmark_config, logging.getLogger(self.config.a.container)
        )

        self.logger = logging.getLogger(str(self))

    def run(self):
        self.logger.info(f"{self} RUN")
        self.benchmark.start_instance()
        self.benchmark.run_async()
        self.benchmark.wait()

        self.logger.info(f"{self} RESULTS")
        self.benchmark.get_results(
            self.results_dir, self.runid, Type.SEQUENTIAL, duet_order=None
        )

    def cleanup(self):
        self.logger.info(f"{self} CLEANUP")
        self.benchmark.cleanup(self.config.remove_containers)

    def __str__(self):
        return f"seqn:{self.config.benchmark}:{self.runid}"


class DuetBenchmarkRunner:
    def __init__(
        self, config: DuetConfig, results_dir: str, runid: int, duet_order: DuetOrder
    ):
        self.config = config
        self.results_dir = results_dir
        self.runid = runid
        self.duet_order = duet_order

        self.a = Benchmark(config.a, logging.getLogger(config.a.container))
        self.b = Benchmark(config.b, logging.getLogger(config.b.container))

        self.logger = logging.getLogger(str(self))

    def run(self):
        self.logger.info(f"{self} RUN")
        self.a.start_instance()
        self.b.start_instance()

        if self.duet_order == DuetOrder.AB:
            self.a.run_async()
            self.b.run_async()
        elif self.duet_order == DuetOrder.BA:
            self.b.run_async()
            self.a.run_async()
        else:
            raise NotImplementedError()

        self.a.wait()
        self.b.wait()

        self.logger.info(f"{self} GET")
        for benchmark in [self.a, self.b]:
            benchmark.get_results(
                self.results_dir, self.runid, Type.DUET, self.duet_order.value
            )

    def cleanup(self):  # nothrow
        self.logger.info(f"{self} CLEANUP")
        for benchmark in [self.a, self.b]:
            benchmark.cleanup(self.config.remove_containers)

    def __str__(self):
        return f"duet:{self.config.benchmark}:{self.runid}"


class Harness:
    def __init__(self, config: DuetBenchConfig, results_dir: str, logger):
        self.config = config
        self.results_dir = results_dir
        self.logger = logger

    def run(self):
        plan = self.plan()

        if self.config.scheduling == Schedule.SEQUENTIAL:
            pass
        elif self.config.scheduling == Schedule.RANDOMIZED_INTERLEAVING_TRIALS:
            random.shuffle(plan)
        else:
            raise NotImplementedError()

        self.logger.info(f"PLAN: {[str(x) for x in plan]}")

        self.execute(plan)

    def plan(self):
        plan = []
        for duet_config in self.config.duets:
            for duet_runid in range(duet_config.duet_repetitions):
                plan.append(
                    DuetBenchmarkRunner(
                        duet_config, self.results_dir, duet_runid, DuetOrder.AB
                    )
                )
            for seq_runid in range(duet_config.sequential_repetitions):
                plan.append(
                    SequentialBenchmarkRunner(
                        duet_config, self.results_dir, seq_runid, Pair.A
                    )
                )
                plan.append(
                    SequentialBenchmarkRunner(
                        duet_config, self.results_dir, seq_runid, Pair.B
                    )
                )
        return plan

    def execute(self, plan):
        signal.signal(signal.SIGINT, self.handle_interrupt_and_exit)
        for benchmark in plan:
            try:
                benchmark.run()
            except RuntimeError as e:
                self.logger.error(f"{benchmark} failed with exception: {e}")
                traceback.print_exc()
            except Exception as e:
                self.logger.critical(f"{benchmark} unexpected exception: {e}")
                traceback.print_exc()
            finally:
                benchmark.cleanup()

    def handle_interrupt_and_exit(self, *args):
        self.logger.info("CAUGHT INTERRUPT, CLEANUP AND EXIT")
        # Throw SystemExit exception that gets caught in execute and does cleaup if viable
        sys.exit(1)


def create_results_dir(config: DuetBenchConfig, outdir: str, force: bool, logger):
    if outdir:
        results_dir = outdir
    else:
        time_tag = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        results_dir = f"results.{config.suite}.{time_tag}"

    try:
        os.makedirs(f"{results_dir}/{ARTIFACTS_DIR}")
    except (OSError, FileExistsError) as e:
        if isinstance(e, FileExistsError) and force:
            pass
        else:
            raise RuntimeError(
                f"Failed to create results directory `{results_dir}`"
            ) from e

    logger.info(f"Results will be placed in `{results_dir}`")
    return results_dir


def gather_artifacts(artifacts_config: dict, results_dir: str, logger):
    logger.info("Gather artifacts")
    for artifact, command in artifacts_config.items():
        result_path = f"{results_dir}/{ARTIFACTS_DIR}/{artifact}"
        logger.info(f"Artifact `{artifact}` from `{command}` to `{result_path}`")
        try:
            with open(result_path, "w") as output:
                p = subprocess.run(
                    command.split(), stdout=output, stderr=subprocess.PIPE
                )
            if p.returncode != 0:
                logger.error(
                    f"Artifact `{artifact}` command failed with error {p.returncode}\n"
                    f"sterr: {p.stderr.decode('utf-8')}"
                )
        except Exception:
            logging.error(f"Artifact `{artifact}` command failed with exception")
            traceback.print_exc()


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "config", type=str, help="YAML config file for the duet benchmark"
    )
    parser.add_argument("-o", "--outdir", type=str, help="Output directory")
    parser.add_argument(
        "-f",
        "--force",
        action="store_true",
        help="Overwrite output directory if present",
    )
    parser.add_argument(
        "-a",
        "--artifacts-only",
        action="store_true",
        help="Only gather artifacts, do not run duet",
    )
    parser.add_argument(
        "-d",
        "--docker",
        type=str,
        help="Docker command to use - docker/podman",
    )

    return parser.parse_args()


def override_config_from_args(config: DuetBenchConfig, args):
    if args.docker:
        config.docker_command = args.docker

    return config


def main():
    args = parse_arguments()
    try:
        config = DuetBenchConfig(args.config)
    except Exception as e:
        logging.critical(f"Critical config error: {e}")
        traceback.print_exc()
        sys.exit(1)

    config = override_config_from_args(config, args)

    logging.basicConfig(
        format="%(asctime)s  %(name)-20s %(levelname)-8s  %(message)s",
        level=logging.DEBUG if config.verbose else logging.INFO,
        datefmt="%b %d %H:%M:%S",
    )
    logger = logging.getLogger("duet")

    if config.seed:
        logger.info(f"Seed: {config.seed}")
        random.seed(config.seed)

    if config.docker_command:
        global DOCKER
        logger.info(f"Docker command: {config.docker_command}")
        DOCKER = config.docker_command

    try:
        results_dir = create_results_dir(config, args.outdir, args.force, logger)
        gather_artifacts(config.artifacts, results_dir, logger)
        if not args.artifacts_only:
            harness = Harness(config, results_dir, logger)
            harness.run()
    except Exception as e:
        logging.critical(f"Critical run duets error: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
