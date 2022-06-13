#!/bin/env python3

import argparse
from collections import defaultdict
import datetime
import logging
import os
import random
import shutil
import signal
import subprocess
import sys
from time import sleep
import traceback
from typing import List

from duet.duetconfig import (
    ARTIFACTS_DIR,
    BenchmarkConfig,
    ConfigException,
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
        stdout, stderr = p.stdout.decode("utf-8"), p.stderr.decode("utf-8")
        self.log(stdout, stderr)
        if p.returncode:
            raise RuntimeError(
                f"Running '{cmd}' finished with non-zero exit status {p.returncode}"
            )
        return stdout, stderr

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

    def wait_async(
        self, handle: int, timeout=None
    ):  # throws TimeoutExpired, RuntimeError
        assert handle in self.async_processes.keys()
        self.logger.debug(f"wait_async({handle}, timeout={timeout})")

        p = self.async_processes.pop(handle)
        stdout, stderr = None, None
        try:
            stdout, stderr = p.communicate(timeout=timeout)
            if p.returncode:
                raise RuntimeError(
                    f"Running of async task finished with non-zero exit status: {p.returncode}"
                )
        except subprocess.TimeoutExpired as exc:
            self.logger.debug(f"wait_async({handle}, timeout={timeout}) TIMEOUT")
            raise exc
        finally:
            self.log(
                stdout.decode("utf-8") if stdout else None,
                stderr.decode("utf-8") if stderr else None,
            )

    def log(self, stdout, stderr):
        if stdout:
            for line in stdout.splitlines():
                line = line.strip()
                if line:
                    self.logger.info(">    " + line)
        if stderr:
            for line in stderr.splitlines():
                line = line.strip()
                if line:
                    self.logger.warning(">     " + line)


class Benchmark:
    def __init__(self, config: BenchmarkConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger

        self.runner = Runner(self.logger)
        self.async_handle = None

    def start_instance(self, mount_shm=False):
        self.runner.run(
            f"{DOCKER} run --name {self.config.container}{' -v /dev/shm:/dev/shm --ipc host' if mount_shm else ''} -it -d {self.config.image} bash"
        )

    def run_async(self, barrier_name=None):
        split_cmd = f"{DOCKER} exec {self.config.container} bash -c".split()
        if barrier_name:
            assert self.config.syncduet_run_command
            run_cmd_with_barrier = self.config.syncduet_run_command.replace(
                "$$", barrier_name
            )
            split_cmd.append(run_cmd_with_barrier)
        else:
            split_cmd.append(self.config.run_command)
        self.async_handle = self.runner.run_async(split_cmd)

    def wait(self, timeout=None):  # throws TimeoutExpired, RuntimeError
        assert self.async_handle is not None
        self.runner.wait_async(self.async_handle, timeout)
        self.async_handle = None

    def stop(self):
        benchmark_program = self.config.run_command.split()[0].split("/")[-1]
        try:
            pid, _ = self.runner.run(
                f"{DOCKER} exec {self.config.container} pgrep {benchmark_program}"
            )
            self.runner.run(f"{DOCKER} exec {self.config.container} kill -15 {pid}")
            sleep(5)  # wait a bit for benchmark to process signal and write results
        except RuntimeError:
            self.logger.warning("STOP not successful")

    def get_results(self, results_dir: str, runid: int, type: Type, duet_order: str):
        for remote_result_path in self.config.result_files:
            # Since docker cp cannot rename the file (remote_result_path), we first copy the file
            # (local_tmp_path) and then rename it (local_result_path).
            filename = os.path.basename(remote_result_path)

            local_tmp_path = f"{results_dir}/{filename}"

            self._delete_if_exists(local_tmp_path)

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
            self._delete_if_exists(local_result_path)
            os.rename(local_tmp_path, local_result_path)

    def make_barrier(self, name):
        self.runner.run(
            f"{DOCKER} exec {self.config.container} /barrier/make-barrier {name} 2"
        )

    def remove_barrier(self, name):
        self.runner.run(
            f"{DOCKER} exec {self.config.container} bash /barrier/rm-barrier {name}"
        )

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

    def _delete_if_exists(self, path: str):
        if os.path.exists(path):
            self.logger.warning(f"Delete {path} due to conflict")
            if os.path.isdir(path):
                shutil.rmtree(path)
            else:
                os.remove(path)


class SequentialBenchmarkRunner:
    def __init__(
        self,
        config: DuetConfig,
        results_dir: str,
        runid: int,
        pair: Pair,
        base_logger: logging.Logger,
    ):
        self.config = config
        self.results_dir = results_dir
        self.runid = runid
        self.logger = logging.getLogger(f"{base_logger.name}.{self}")

        benchmark_config = self.config.a if pair == Pair.A else self.config.b
        self.benchmark = Benchmark(
            benchmark_config, logging.getLogger(f"{self.logger.name}.{pair.value}")
        )

    def run(self):
        self.logger.info(f"{self} RUN")
        self.benchmark.start_instance()
        self.benchmark.run_async()
        self.benchmark.wait(timeout=self.config.timeout)

    def stop(self):
        self.logger.info(f"{self} STOP")
        self.benchmark.stop()

    def cleanup(self):
        self.logger.info(f"{self} CLEANUP")
        self.benchmark.cleanup(self.config.remove_containers)

    def get_results(self):
        self.logger.info(f"{self} RESULTS")
        self.benchmark.get_results(
            self.results_dir, self.runid, Type.SEQUENTIAL, duet_order=None
        )

    def __str__(self):
        return f"seqn.{self.config.benchmark}.{self.runid}"


class DuetBenchmarkRunner:
    def __init__(
        self,
        config: DuetConfig,
        results_dir: str,
        runid: int,
        duet_order: DuetOrder,
        synchronized: bool,
        base_logger: logging.Logger,
    ):
        self.config = config
        self.results_dir = results_dir
        self.runid = runid
        self.duet_order = duet_order
        self.synchronized = synchronized
        self.logger = logging.getLogger(f"{base_logger.name}.{self}")

        self.barrier_name = None

        self.a = Benchmark(
            config.a, logging.getLogger(f"{self.logger.name}.{Pair.A.value}")
        )
        self.b = Benchmark(
            config.b, logging.getLogger(f"{self.logger.name}.{Pair.B.value}")
        )

    def run(self):
        self.logger.info(f"{self} RUN")
        self.a.start_instance(mount_shm=self.synchronized)
        self.b.start_instance(mount_shm=self.synchronized)

        if self.synchronized:
            self.barrier_name = f"{str(self)}:barrier"
            self.a.make_barrier(name=self.barrier_name)

        if self.duet_order == DuetOrder.AB:
            self.a.run_async(self.barrier_name)
            self.b.run_async(self.barrier_name)
        elif self.duet_order == DuetOrder.BA:
            self.b.run_async(self.barrier_name)
            self.a.run_async(self.barrier_name)
        else:
            raise NotImplementedError()

        if self.config.timeout:
            # On timeout raise TimeoutExpired and terminate both benchmarks in cleanup
            # If a completes before timeout B might still hit timeout, thus compute time left
            a_start = datetime.datetime.now()
            self.a.wait(timeout=self.config.timeout)
            a_run_time = datetime.datetime.now() - a_start
            self.b.wait(timeout=max(1, self.config.timeout - a_run_time.seconds))
        else:
            self.a.wait()
            self.b.wait()

    def get_results(self):
        self.logger.info(f"{self} GET")
        for benchmark in [self.a, self.b]:
            type = Type.SYNCDUET if self.barrier_name else Type.DUET
            benchmark.get_results(
                self.results_dir, self.runid, type, self.duet_order.value
            )

    def stop(self):
        self.logger.info(f"{self} STOP")
        for benchmark in [self.a, self.b]:
            benchmark.stop()

    def cleanup(self):  # nothrow
        self.logger.info(f"{self} CLEANUP")
        if self.barrier_name:
            try:
                self.a.remove_barrier(self.barrier_name)
            except RuntimeError:
                self.logger.error(f"Failed to remove barrier {self.barrier_name}")
        for benchmark in [self.a, self.b]:
            benchmark.cleanup(self.config.remove_containers)

    def __str__(self):
        return (
            f"{Type.SYNCDUET.value}.{self.config.benchmark}.{self.runid}"
            if self.synchronized
            else f"{Type.DUET.value}.{self.config.benchmark}.{self.runid}"
        )


class Harness:
    def __init__(
        self,
        config: DuetBenchConfig,
        results_dir: str,
        benchmark_filter: List[str],
        logger,
    ):
        self.config = config
        self.results_dir = results_dir
        self.benchmark_filter = benchmark_filter
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
        return self.execute(plan)

    def plan(self):
        plan = []
        benchmarks_filtered_out = 0
        for duet_config in self.config.duets:
            if (
                self.benchmark_filter
                and duet_config.benchmark not in self.benchmark_filter
            ):
                benchmarks_filtered_out += 1
                continue

            for duet_runid in range(duet_config.duet_repetitions):
                plan.append(
                    DuetBenchmarkRunner(
                        duet_config,
                        self.results_dir,
                        duet_runid,
                        DuetOrder.AB,
                        synchronized=False,
                        base_logger=self.logger,
                    )
                )
            for syncduet_runid in range(duet_config.syncduet_repetitions):
                plan.append(
                    DuetBenchmarkRunner(
                        duet_config,
                        self.results_dir,
                        syncduet_runid,
                        DuetOrder.AB,
                        synchronized=True,
                        base_logger=self.logger,
                    )
                )
            for seq_runid in range(duet_config.sequential_repetitions):
                plan.append(
                    SequentialBenchmarkRunner(
                        duet_config,
                        self.results_dir,
                        seq_runid,
                        Pair.A,
                        base_logger=self.logger,
                    )
                )
                plan.append(
                    SequentialBenchmarkRunner(
                        duet_config,
                        self.results_dir,
                        seq_runid,
                        Pair.B,
                        base_logger=self.logger,
                    )
                )

        self.logger.info(
            f"Excluded {benchmarks_filtered_out} benchmarks not matching filter {self.benchmark_filter}"
        )
        return plan

    def execute(self, plan):
        errors = []
        signal.signal(signal.SIGINT, self.handle_interrupt_and_exit)
        self.benchmark = None
        for benchmark in plan:
            self.benchmark = benchmark
            try:
                benchmark.run()
            except subprocess.TimeoutExpired:
                errors.append(f"TIMEOUT:{benchmark}")
                self.logger.warning(f"{benchmark} hit timeout")
                benchmark.stop()
            except RuntimeError as e:
                errors.append(f"RUN:{benchmark}")
                self.logger.error(f"{benchmark} failed with exception: {e}")
            except Exception as e:
                errors.append(f"CRITICAL:{benchmark}")
                self.logger.critical(f"{benchmark} unexpected exception: {e}")
                traceback.print_exc()
            finally:
                # Run may have failed on Exception but some results might be
                # present try to obtain them.
                try:
                    benchmark.get_results()
                except RuntimeError:
                    self.logger.error(f"{benchmark} failed to get results")
                    errors.append(f"RESULTS:{benchmark}")

                benchmark.cleanup()
        return errors

    def handle_interrupt_and_exit(self, *args):
        self.logger.info("CAUGHT INTERRUPT, CLEANUP AND EXIT")
        # Throw SystemExit exception that gets caught in execute and does cleaup if viable
        sys.exit(1)


def create_results_dir(outdir: str, force: bool):
    if outdir:
        results_dir = outdir
    else:
        time_tag = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        results_dir = f"results.{time_tag}"

    try:
        os.makedirs(f"{results_dir}/{ARTIFACTS_DIR}")
    except (OSError, FileExistsError) as e:
        if isinstance(e, FileExistsError) and force:
            pass
        else:
            raise RuntimeError(
                f"Failed to create results directory `{results_dir}`"
            ) from e
    return results_dir


def gather_artifacts(config: DuetBenchConfig, results_dir: str, logger):
    logger.info("Gather artifacts")
    for artifact, command in config.artifacts.items():
        result_path = f"{results_dir}/{ARTIFACTS_DIR}/{artifact}"

        if os.path.exists(result_path):
            logger.info(f"Artifact `{artifact}` already exists")
            continue
        else:
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
        except Exception as e:
            logger.error(f"Artifact `{artifact}` command failed with exception {e}")


def run_config(
    config: DuetBenchConfig, results_dir, benchmark_filter: List[str], logger
):
    gather_artifacts(config, results_dir, logger)

    if config.seed:
        logger.info(f"Seed: {config.seed}")
        random.seed(config.seed)

    if config.docker_command:
        global DOCKER
        logger.info(f"Docker command: {config.docker_command}")
        DOCKER = config.docker_command

    errors = []
    try:
        harness = Harness(config, results_dir, benchmark_filter, logger)
        errors += harness.run()
    except Exception as e:
        logger.critical(f"Critical run duets error: {e}")
        traceback.print_exc()

    return errors


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "configs", type=str, nargs="+", help="YAML config files for the duet benchmark"
    )
    parser.add_argument("-o", "--outdir", type=str, help="Output directory")
    parser.add_argument(
        "-f",
        "--force",
        action="store_true",
        help="Overwrite output directory if present",
    )
    parser.add_argument(
        "-d",
        "--docker",
        type=str,
        help="Docker command to use - docker/podman",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Use DEBUG log level",
    )
    parser.add_argument(
        "--filter",
        type=str,
        nargs="+",
        help="Only run benchmarks matching these filters",
    )

    return parser.parse_args()


def override_config_from_args(config: DuetBenchConfig, args):
    if args.docker:
        config.docker_command = args.docker
    if args.verbose:
        config.verbose = True

    return config


def configure_logging():
    logging.basicConfig(
        format="%(asctime)s %(name)-35s %(levelname).1s  %(message)s",
        level=logging.INFO,
        datefmt="%b %d %H:%M:%S",
    )


def main():
    configure_logging()
    args = parse_arguments()

    results_dir = create_results_dir(args.outdir, args.force)
    config_errors = defaultdict(list)

    for config_path in args.configs:
        logging.info(f"START DUET {config_path}")
        try:
            config = DuetBenchConfig(config_path)
        except ConfigException as e:
            logging.error(f"Config error: {e}")
            continue
        except Exception as e:
            logging.critical(f"Critical config error: {e}")
            continue

        config = override_config_from_args(config, args)

        logger = logging.getLogger(f"{config.name}")
        logger.setLevel(logging.DEBUG if config.verbose else logging.INFO)

        config_errors[config.name] += run_config(
            config, results_dir, args.filter, logger
        )

    logging.info("SUMMARY:")
    for config_name, errors in config_errors.items():
        if errors:
            logger.error(f"{config_name}: {len(errors)} errors: {errors}")
        else:
            logger.info(f"{config_name}: 0 errors")

    logging.info(f"RESULTS: {results_dir}")


if __name__ == "__main__":
    main()
