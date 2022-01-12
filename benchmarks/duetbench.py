#!/bin/env python3

import argparse
import datetime
import subprocess
import logging


class Runner:
    def __init__(self, logger):
        self.logger = logger
        self.async_processes_next_id = 0
        self.async_processes = {} # handle to process mapping

    def run(self, cmd: str):
        self.logger.debug(f"run> {cmd}")
        p = subprocess.run(cmd.split(), capture_output=True)
        self.log(p.stdout, p.stderr)
        if p.returncode:
            raise RuntimeError(f"Running '{cmd}' finished with non-zero exit status {p.returncode}")

    def run_async(self, cmd: str) -> int:
        self.logger.debug(f"run_async({self.async_processes_next_id})> {cmd}")
        try:
            async_process = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        except Exception as e:
            raise RuntimeError(f"Launch of async process '{cmd}' threw exception: {e}")
        async_handle = self.async_processes_next_id
        self.async_processes[async_handle] = async_process
        self.async_processes_next_id = self.async_processes_next_id + 1
        return async_handle

    def wait_async(self, handle: int):
        assert(handle in self.async_processes.keys())
        self.logger.debug(f"wait_async({handle})")

        p = self.async_processes.pop(handle)
        (stdout, stderr) = p.communicate()
        returncode = p.wait()
        self.log(stdout, stderr)
        if returncode:
            raise RuntimeError(f"Running of async task finished with non-zero exit status: {returncode}")

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
    def __init__(self, image, container, docker_cmd, logger):
        self.image = image
        self.container = container
        self.docker_cmd = docker_cmd
        self.logger = logger
        self.runner = Runner(logger)
        self.async_handle = None

    def start_instance(self):
        self.runner.run(f"{self.docker_cmd} run --name {self.container} -it -d {self.image} bash")

    def setup(self):
        self.runner.run(f"{self.docker_cmd} exec {self.container} ./setup.sh")

    def run_async(self):
        self.async_handle = self.runner.run_async(f"{self.docker_cmd} exec {self.container} ./run.sh")

    def wait(self):
        assert(self.async_handle is not None)
        self.runner.wait_async(self.async_handle)

    def get_results(self, tag):
        results_dir=f"results.{tag}.{self.container}.{self.image}"
        self.runner.run(f"{self.docker_cmd} cp {self.container}:/duet/results/ {results_dir}")

    def cleanup(self, rm: bool):
        try:
            self.runner.run(f"{self.docker_cmd} stop {self.container}")
        except RuntimeError as e:
            self.logger.warn("Exception hit: {e}")
        if rm:
            try:
                self.runner.run(f"{self.docker_cmd} rm {self.container}")
            except RuntimeError as e:
                self.logger.warn("Exception hit: {e}")


class DuetBenchmark:
    def __init__(self, image_a: str, image_b:str, docker_cmd: str, rm: bool, logger):
        self.a = Benchmark(image_a, "duetA", docker_cmd, logging.getLogger("duetA"))
        self.b = Benchmark(image_b, "duetB", docker_cmd, logging.getLogger("duetB"))
        self.rm = rm
        self.logger = logger

    def run(self):
        try:
            self.logger.info("Start duet instances")
            self.a.start_instance()
            self.b.start_instance()

            self.logger.info("Setup benchmarks")
            self.a.setup()
            self.b.setup()

            self.logger.info("Run benchmarks")
            self.a.run_async()
            self.b.run_async()

            self.logger.info("Wait for benchmarks to finish")
            self.a.wait()
            self.b.wait()

            self.logger.info("Get results")
            time_tag = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            self.a.get_results(time_tag)
            self.b.get_results(time_tag)
        except RuntimeError as e:
            self.logger.error(f"Duet benchmark hit exception: {e}")

        self.cleanup()

    def cleanup(self):
        self.logger.info("Cleanup")
        self.a.cleanup(self.rm)
        self.b.cleanup(self.rm)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image_a", metavar="A", type=str, help="docker image to run as duet A")
    parser.add_argument("image_b", metavar="B", type=str, help="docker image to run as duet B")
    parser.add_argument("--docker", type=str, default="docker", help="docker command to run, default is 'docker' other option e.g. podaman")
    parser.add_argument("--rm", action="store_true", help="remove created containers on stop")
    parser.add_argument("--verbose", "-v", action="store_true", help="debug prints")

    args = parser.parse_args()

    # logging.basicConfig(filename='example.log', filemode='w', level=logging.DEBUG)
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level)

    logger = logging.getLogger("duet")

    logger.info(f"Start duet benchmark with: {args}")
    duet = DuetBenchmark(args.image_a, args.image_b, args.docker, args.rm, logger)
    duet.run()


if __name__ == "__main__":
    main()
