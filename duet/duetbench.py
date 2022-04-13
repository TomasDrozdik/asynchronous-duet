#!/bin/env python3

import argparse
import datetime
import logging
import os
import random
import subprocess
import sys
import traceback
from typing import List

from duet.duetconfig import (
    BenchmarkConfig,
    DuetBenchConfig,
    DuetConfig,
    DuetRepetititionType,
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
                    self.logger.info(line)
        if stderr:
            for line in stderr.decode("utf-8").strip().splitlines():
                line = line.strip()
                if line:
                    self.logger.debug(line)
                self.logger.warning(line)


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

    def get_results(self, results_dir, tag):
        for remote_result_path in self.config.result_files:
            # Since docker cp cannot rename the file (remove_result_path), we first copy the file
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

            local_result_path = (
                f"{results_dir}/{tag}.{self.config.duet_type.value}.{filename}"
            )
            os.rename(local_tmp_path, local_result_path)

    def cleanup(self, rm: bool):  # nothrow
        try:
            self.runner.run(f"{DOCKER} stop {self.config.container}")
        except RuntimeError as e:
            self.logger.warn(f"Cleanup stop hit exception: {e}")
        if rm:
            try:
                self.runner.run(f"{DOCKER} rm {self.config.container}")
            except RuntimeError as e:
                self.logger.warn(f"Cleanup rm hit exception: {e}")


class DuetBenchmark:
    def __init__(self, config: DuetConfig, results_dir: str, logger):
        self.config = config
        self.results_dir = results_dir
        self.logger = logger

        self.a = Benchmark(self.config.a, logging.getLogger(self.config.a.container))
        self.b = Benchmark(self.config.b, logging.getLogger(self.config.b.container))

    def get_run_order(self, run_id):
        in_order = [self.a, self.b]
        reverse_order = list(reversed(in_order))
        if self.config.repetitions_type == DuetRepetititionType.IN_ORDER:
            return in_order
        elif self.config.repetitions_type == DuetRepetititionType.RANDOM:
            return in_order if random.choice([True, False]) else reverse_order
        elif self.config.repetitions_type == DuetRepetititionType.SWAP:
            return reverse_order if run_id % 2 else in_order
        else:
            raise NotImplementedError(
                f"Unknown repetition type: {self.repetitions_type}"
            )

    def run(self):
        self.logger.info("Start duet instances")
        self.a.start_instance()
        self.b.start_instance()

        self.logger.info(
            f"Run duet: {self.config.name}, type: {self.config.type.value}, repetitions: {self.config.repetitions}, repetitions_type: {self.config.repetitions_type}"
        )

        for run_id in range(self.config.repetitions):
            run_order = self.get_run_order(run_id)
            self.logger.info(
                f"Run - runid: {run_id}, order: {[x.config.container for x in run_order]}"
            )
            for benchmark in run_order:
                benchmark.run_async()

            self.logger.info(f"Wait - runid: {run_id}")
            self.a.wait()
            self.b.wait()

            self.logger.info(f"Get results - runid: {run_id}")
            tag = f"{self.config.name}.{run_id}"
            self.a.get_results(self.results_dir, tag)
            self.b.get_results(self.results_dir, tag)

    def cleanup(self):  # nothrow
        self.logger.info("Cleanup")
        self.a.cleanup(self.config.remove_containers)
        self.b.cleanup(self.config.remove_containers)


def create_results_dir(config: DuetBenchConfig, logger):
    time_tag = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    results_dir = f"results.{config.name}.{time_tag}"
    try:
        os.mkdir(results_dir)
    except (OSError, FileExistsError) as e:
        raise RuntimeError(
            f"Failed to create results directory `{results_dir}` with exception: {e}"
        )

    logger.info(f"Results will be placed in `{results_dir}`")
    return results_dir


def run_duets(config: DuetBenchConfig, logger):
    results_dir = create_results_dir(config, logger)
    for duet_config in config.duets:
        logger.info(f"Duet `{duet_config.name}`")
        duet = None
        try:
            duet = DuetBenchmark(
                duet_config, results_dir, logging.getLogger(duet_config.name)
            )
            duet.run()
        except RuntimeError as e:
            logger.error(f"Duet `{duet_config.name}` failed with exception: {e}")
            traceback.print_exception(e)
        finally:
            if duet:
                duet.cleanup()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "config", type=str, help="YAML config file for the duet benchmark"
    )
    args = parser.parse_args()

    try:
        config = DuetBenchConfig(args.config)
    except Exception as e:
        logging.critical(f"Critical config error: {e}")
        traceback.print_exception(e)
        sys.exit(1)

    logging.basicConfig(
        format="%(asctime)s  %(name)-20s %(levelname)-8s  %(message)s",
        level=logging.DEBUG if config.verbose else logging.INFO,
        datefmt="%b %d %H:%M:%S",
    )
    logger = logging.getLogger("duet")
    logger.info(f"Start duet benchmark with config: {config}")

    if config.seed:
        logger.info(f"Seed: {config.seed}")
        random.seed(config.seed)

    if config.docker_command:
        global DOCKER
        logger.info(f"Docker command: {config.docker_command}")
        DOCKER = config.docker_command

    try:
        run_duets(config, logger)
    except RuntimeError as e:
        logging.critical(f"Critical run error: {e}")
        traceback.print_exception(e)
        sys.exit(1)


if __name__ == "__main__":
    main()