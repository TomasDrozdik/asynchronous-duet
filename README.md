# Asynchronous Duet Benchmarking

Tool for running Asynchronous Duet Benchmarks.
This is "practical part" of a [Master Thesis](https://github.com/TomasDrozdik/asynchronous-duet-thesis).

- [Asynchronous Duet Benchmarking](#asynchronous-duet-benchmarking)
  - [Install/Buid/Run TL;DR](#installbuidrun-tldr)
  - [Build and Benchmark](#build-and-benchmark)
    - [Build docker image(s)](#build-docker-images)
    - [Install](#install)
    - [Run asynchronous-duet benchmark!](#run-asynchronous-duet-benchmark)
    - [Run config](#run-config)
    - [How `duetbench.py` does it?](#how-duetbenchpy-does-it)
  - [Process results](#process-results)
  - [Interpret results](#interpret-results)
  - [Development setup](#development-setup)
    - [Tests](#tests)


## Install/Buid/Run TL;DR

```
# Install duet tools
python3 -m venv venv
source venv/bin/activate
pip install .

# Build docker images, WARNING: need git LFS
git lfs checkout
./build.sh -d <docker|podman>

# Run test configs
./run.sh -d <docker|podman> -o <existing_result_dir> -t

# Run
./run.sh -d <docker|podman> -o <existing_result_dir>
```

## Build and Benchmark

Simple guide on how to run asynchronous duet benchmark.

### Build docker image(s)

Docker images data are stored in git LFS storage ([more info and install instructions](https://git-lfs.github.com/))
After fresh checkout you need to run:
```
git lfs checkout
```

**Build using script:**
```
$ ./build.sh -d <docker|podman>
```

**Build manually:**
Example: [renaissance docker image](./benchmarks/renaissance/Dockerfile) for [Renaissance suite](https://renaissance.dev/)

It is built from `openjdk11` image and it downloads renaissance suite jar file.

``` bash
cd ./benchmarks/renaissance/Dockerfile
docker build --tag renaissance .
```

### Install

Ideally use python virtual environment
```
(venv)$ pip install .[all]
```

> If using `zsh` shell you need to escape first '[' i.e. `pip install .\[all]`

This should install scripts `duetbench.py` and `duetprocess.py` as well as dependencies for [`analyze.ipynb`](./duet/analyze.ipynb).

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

Results themselves have fixed naming schema `<suite_name>.<benchmark_name>.<runid>.<duet_order>.<run_type>.<results_file_from_config>`.
```
(venv) *[feature/duet2][~/d/school/thesis-master/asynchronous-duet]$ ls -lRh results.renaissance.test
results.renaissance.test:
total 36K
drwxrwxrwx 1 root root 4.0K May 15 18:49 artifacts
-rwxrwxrwx 1 root root 7.7K May 15 18:49 renaissance.chi-square.0.duet.AB.A.results.json
-rwxrwxrwx 1 root root 7.8K May 15 18:49 renaissance.chi-square.0.duet.AB.B.results.json
-rwxrwxrwx 1 root root 7.7K May 15 18:50 renaissance.chi-square.1.duet.BA.A.results.json
-rwxrwxrwx 1 root root 7.7K May 15 18:50 renaissance.chi-square.1.duet.BA.B.results.json

results.renaissance.test/artifacts:
total 9.5K
-rwxrwxrwx 1 root root   33 May 15 18:49 date
-rwxrwxrwx 1 root root    7 May 15 18:49 hostname
-rwxrwxrwx 1 root root 2.6K May 15 18:49 lscpu
-rwxrwxrwx 1 root root 1.5K May 15 18:49 meminfo
-rwxrwxrwx 1 root root   93 May 15 18:49 uname
```

This fixed naming schema is utilized later in processing.


### Run config

For each asynchronous-duet run you need a config file.
This config file is uniquely tied to docker images it is build for e.g. `renaissance` image above uses `/duet` workdir and config file has to reflect that.

Example: [renaissance duet YAML config](./benchmarks/renaissance/duet.yml)

``` yml
duetbench:
  suite: renaissance                # name of the benchmark suite - used for parsing results
  verbose: true                     # debug logging
  seed: 42                          # random seed
  docker_command: podman            # command to invoke docker in shell, default is docker

  remove_containers: true           # remove containers after finish

  duet_repetitions: 2               # number of harness repetitions
  sequential_repetitions: 2         # sequential repetitions, default 0

  schedule: randomized_interleaving_trials # what schedule to use on benchmarks

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
  duet_repetitions: 4                  # overwrite value from duetbench config
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


## Process results

To compare the A and B benchmarks, first process the raw copied results from the containers.
Each benchmark suite returns results in its own format.
To compare them results need to be unified.

**Requirements**: benchmark suite has to give us this information:
* _wallclock start_ of an inner harness iteration
* _wallclock end_ of an inner harness iteration

Renaissance suite does not give this to us directly, but it is fairly easy to calculate that.
For renaissance specifically, there is [`duetprocess.py`](./duet/duetprocess.py) that does exactly that given you've installed via pip you already have it available:
```
(venv)$ duetprocess --help
usage: duetprocess [-h] -c CONFIG [-o OUTPUT] [--json | --csv] results [results ...]

positional arguments:
  results               Results directory

optional arguments:
  -h, --help            show this help message and exit
  -c CONFIG, --config CONFIG
                        YAML config file for the duet benchmark
  -o OUTPUT, --output OUTPUT
                        Output file
  --json                Serialize DataFrame to JSON
  --csv                 Serialize DataFrame to CSV (default)
```

As you can see it requires config file of that duet and its results directory, it creates single results CSV file.
For example:
```
(venv)$ duetprocess --config ./benchmarks/renaissance/duet.yml --output results.renaissance.test results.renaissance.test
Apr 13 13:58:43  root                 INFO      Processed file: results.renaissance.test/chi-square.0.A.results.json, runid: 0, pair: A, total_iterations: 10
Apr 13 13:58:43  root                 INFO      Processed file: results.renaissance.test/chi-square.0.B.results.json, runid: 0, pair: B, total_iterations: 10
Apr 13 13:58:43  root                 INFO      Processed file: results.renaissance.test/chi-square.1.A.results.json, runid: 1, pair: A, total_iterations: 10
Apr 13 13:58:43  root                 INFO      Processed file: results.renaissance.test/chi-square.1.B.results.json, runid: 1, pair: B, total_iterations: 10
Apr 13 13:58:43  root                 INFO      Write results to: results.renaissance.test.csv
```

**Results schema**

| Field                | Description                                    |
| -------------------- | ---------------------------------------------- |
| suite                | suite name                                     |
| benchmark            | benchmark name                                 |
| runid                | run number of one A/B duet                     |
| iteration            | internal harness iteartion number              |
| pair                 | type - A or B                                  |
| order                | order - AB/BA                                  |
| epoch_start_ms       | start of the run in wallclock time             |
| iteration_time_ns    | time per single internall iteration in harness |
| iteration_start_ns   | time per single internall iteration in harness |
| iteration_end_ns     | time per single internall iteration in harness |
| *artifact_resuls*... | any artifact value can be parsed and included  |


## Interpret results

Then to interpret the unified results format there is [analyze notebook](./duet/analyze.ipynb).

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
