# Asynchronous Duet Benchmarking

Tool for running Asynchronous Duet Benchmarks.
This is "practical part" of a [Master Thesis](https://github.com/TomasDrozdik/asynchronous-duet-thesis).

- [Asynchronous Duet Benchmarking](#asynchronous-duet-benchmarking)
  - [Build and Benchmark](#build-and-benchmark)
    - [Build docker image(s)](#build-docker-images)
    - [Run config](#run-config)
    - [How `duetbench.py` does it?](#how-duetbenchpy-does-it)
    - [Install](#install)
    - [Run asynchronous-duet benchmark!](#run-asynchronous-duet-benchmark)
  - [Process results](#process-results)
  - [Interpret results](#interpret-results)
  - [Development setup](#development-setup)
    - [Tests](#tests)

## Build and Benchmark

Simple guide on how to run asynchronous duet benchmark.


### Build docker image(s)

Example: [renaissance docker image](./benchmarks/renaissance/Dockerfile) for [Renaissance suite](https://renaissance.dev/)

It is built from `openjdk11` image and it downloads renaissance suite jar file.

``` bash
cd ./benchmarks/renaissance/Dockerfile
docker build --tag renaissance .
```

### Run config

For each asynchronous-duet run you need a config file.
This config file is uniquely tied to docker images it is build for e.g. `renaissance` image above uses `/duet` workdir and config file has to reflect that.

Example: [renaissance duet YAML config](./benchmarks/renaissance/renaissance.duet.yml)

``` yml
duetbench:
  name: renaissance                 # name of the duet benchmark
  verbose: true                     # debug logging
  seed: 42                          # seed for repetitions
  docker_command: podman            # command to invoke docker in shell, default is docker

  remove_containers: true           # remove containers after finish

  repetitions: 2                    # number of harness repetitions
  repetitions_type: swap            # how to interleave harness repetitions, options: random, swap, in_order

  sequential_repetitions: 2         # sequential repetitions, default 0
  sequential_repetitions_type: swap # sequential repetitions type, same as `repetitions_type`

  artifacts:                        # artifacts to gather before the run
    uname: uname -a                 # `uname` artifact, run `uname -a > {results_dir}/artifacts/uname`
    lscpu: lscpu                    # similar, results placed to `{results_dir}/artifacts/lscpu`

  image: renaissance                # docker image to run for a duet run, must exist on the system
  results:                          # list of result files to copy from the container after each run
    - /duet/results                 # has to be an absolute path in the context of a Docker image it runs in,
                                    # for renaissance image it is workdir `/duet/`

  duets:                            # list of duets to run, these must be top level YAML elements,
    - a-benchmark                   # invalid names: `artifacts`, `duetbench`

a-benchmark:
  repetitions: 4                       # overwrite value from duetbench config
  A:                                   # A run
    run: echo RunningA > /duet/results # command to run the benchmark harness in the container

  B:                                   # B run, optional, if not present does A/A run
    image: renaissance-graal           # override inherited image
    run: echo RunningB > /duet/results
    results:                           # override where are the result files in context of renaissance-graal docker image
      - /results
```

### How `duetbench.py` does it?

In order to write correct run configs it is worth understanding roughly what `duetbench.py` does:

```bash
# Start containers, in interactive and detaiched mode running bash - that keeps them running
docker run --name containerA -it -d <image_a> bash
docker run --name containerB -it -d <image_b> bash

for repetition in <repetitions> ; do
    # Launch benchmarks in parallel in specified order (random, swap, in_order) and wait
    docker exec containerA bash -c "<run-command-for-A>" &
    docker exec containerB bash -c "<run-command-for-B>" &
    wait

    # Copy each result file
    docker cp containerA:<result_path> local_host_output_path
    docker cp containerB:<result_path> local_host_output_path
done

docker stop containerA
docker stop containerB

if <remove_containers> ; then
    docker rm containerA
    docker rm containerB
fi
```


### Install

Ideally use python virtual environment
```
(venv)$ pip install .[all]
```

> If using `zsh` shell you need to escape first '[' i.e. `pip install .\[all]`

This should install scripts `duetbench.py` and `process_renaissance.py` as well as dependencies for [`analyze.ipynb`](./duet/analyze.ipynb).

Running:
```
(venv)$ duetbench --help # if installed via `pip install .`
(venv)$ python -m duet.duetbench --help # or this if requirements.txt are installed
usage: duetbench [-h] config

positional arguments:
  config      YAML config file for the duet benchmark

optional arguments:
  -h, --help  show this help message and exit
```

### Run asynchronous-duet benchmark!

Once you have the tools installed, config read, and docker image built it is as simple as `duetbench <config>` e.g.

```
(venv)$ duetbench ./benchmarks/renaissance/duet.yml
```

Logging should give you an idea of what is going on, it also logs all the outputs from the benchmarks themselves so you can examine that as well.

Results will be in output directory of with format like `results.renaissance.20220413-115622` where `renaissance` is the name of the duet from config and it is followed by timestamp.

Results themselves have fixed naming schema `<benchmark_name>.<run_id>.<run_type>.<results_file_from_config>`.
Example results directory in the repo [`results.renaissance.test\`](./results.renaissance.test/)
```
(venv)$ ls -lh results.renaissance.test
total 48K
-rwxrwxrwx 1 root root 8.4K Mar 20 12:17 chi-square.0.A.results.json
-rwxrwxrwx 1 root root 8.4K Mar 20 12:17 chi-square.0.B.results.json
-rwxrwxrwx 1 root root 8.4K Mar 20 12:17 chi-square.1.A.results.json
-rwxrwxrwx 1 root root 8.4K Mar 20 12:17 chi-square.1.B.results.json
```

This fixed naming schema is utilized later in processing.


## Process results

To compare the A and B benchmarks, first process the raw copied results from the containers.
Each benchmark suite returns results in its own format.
To compare them results need to be unified.

**Requirements**: benchmark suite has to give us this information:
* _wallclock start_ of an inner harness iteration
* _wallclock end_ of an inner harness iteration

Renaissance suite does not give this to us directly, but it is fairly easy to calculate that.
For renaissance specifically, there is [`process_renaissance.py`](./duet/process_renaissance.py) that does exactly that given you've installed via pip you already have it available:
```
(venv)$ process_renaissance --help
usage: process_renaissance [-h] -c CONFIG [-o OUTPUT] results

positional arguments:
  results               Results directory of Renaissance Duet

optional arguments:
  -h, --help            show this help message and exit
  -c CONFIG, --config CONFIG
                        Duet config file
  -o OUTPUT, --output OUTPUT
                        Output results.csv file, default <results_dir_name>.csv
```

As you can see it requires config file of that duet and its results directory, it creates single results CSV file.
For example:
```
(venv)$ process_renaissance --config ./benchmarks/renaissance/duet.yml --output results.renaissance.test.csv results.renaissance.test
Apr 13 13:58:43  root                 INFO      Processed file: results.renaissance.test/chi-square.0.A.results.json, runid: 0, pair: A, total_iterations: 10
Apr 13 13:58:43  root                 INFO      Processed file: results.renaissance.test/chi-square.0.B.results.json, runid: 0, pair: B, total_iterations: 10
Apr 13 13:58:43  root                 INFO      Processed file: results.renaissance.test/chi-square.1.A.results.json, runid: 1, pair: A, total_iterations: 10
Apr 13 13:58:43  root                 INFO      Processed file: results.renaissance.test/chi-square.1.B.results.json, runid: 1, pair: B, total_iterations: 10
Apr 13 13:58:43  root                 INFO      Write results to: results.renaissance.test.csv
```

**Unified Results CSV format**

| Field             | Description                                           |
| ----------------- | ----------------------------------------------------- |
| benchmark         | benchmark name                                        |
| runid             | run number of one A/B duet                            |
| pair              | type - A or B                                         |
| epoch_start_ms    | start of the run in wallclock time                    |
| iteration_time_ns | time per single internall iteration in harness        |
| machine           | machine this ran on e.g. localhost, EC2 instance name |
| provider          | provider name e.g. AWS, localhost                     |
| jdk               | JDK name                                              |
| jdk_version       | JDK version                                           |

> Some benchmarks might have another "score" metric number e.g. how many things it managed to do instead of just time of iteration, in those cases we might add another field.


## Interpret results

Then to interpret the unified results format there is [analyze notebook](./duet/analyze.ipynb).
There is some sample results data present in [results.renaissance.test](./results.renaissance.test.csv) and the notebook points to it directly.


## Development setup

For development you may install all the dependencies directly to your venv.
This installs python code formatters `black` and `flake8` to which CI enforces compliance. To enable automatic checks localy it also installs `pre-commit` and that enforces, checks, and auto repairs multiple formatting issues described in [`.pre-commit-config.yaml`](./.pre-commit-config.yaml)


```
(venv)$ pip install -r requirements.txt
(venv)$ pre-commit install # install git hooks
(venv)$ pre-commit # run all autoformatters
(venv)$ python -m duet.duetbench
```

Or you can install the package via pip in editable mode, i.e. binaries would reflect changes made to files. And run from your `$PATH`.
```
(venv)$ pip install --editable .[all]
(venv)$ duetbench --help
```

### Tests

There are no tests so far :-)

```
(venv)$ pytest
```
