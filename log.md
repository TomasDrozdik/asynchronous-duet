# Asynchronous Duet Benchmarking - Work Log

## Core research review:

* Initial Experiments with Duet Benchmarking: Performance Testing Interference in the Cloud

    - *benchmarks:* ScalaBench with an adapted harness to support synchronized inner loops and accurate timings
        - run on Graal and HotSpot JVM
        - JVM params:
            - fixed heap size
            - disabled GC during an iteration, forced in between iterations
            - 30 harness runs, 20 iterations per run, first 15 discarded

    - *hardware:* TODO:
        -bind each benchmark to single CPU core in 2 core VM

    - *processing:*
        - compute grand geometric mean of speedups utilizing synchronized iterations

* Duet Benchmarking: Improving measurement accuracy in the cloud

    - *benchmarks:* TODO:

    - *hardware:* TODO:

    - *processing:* TODO:

* Study of performance variation and predictability in public IAAS clouds - TODO:

* Conducting repeatable measurements in the cloud environment - TODO:

---

## Nov 06 2021

* Planning
* Papers

### Renaissance benchmarks

Single harness run:

```
benchmark,duration_ns,uptime_ns,vm_start_unix_ms
scrabble,911379929,2515700351,1636206612905
scrabble,529459711,3750527761,1636206612905
```

> Q: How does uptime_ns relate to duration_ns?

Need to extract start and end of each iteration:
```
iteration_start = vm_start_unix_ms + uptime_ns // ?
iteration_end = vm_start_unix_ms + uptime_ns + duration_ns // ?
```

Run as:
```
$ java -jar {renaissance_dir}/{jar_file} {harness_loops_arg} --csv {outcsv_path} {workload_name}
```

Processing scrip thus needs to parse some path with output file.
Creates run artifacts:
```
harness-144715-8722645056205070637  launcher-144719-1454091004567985825
```

## Nov 07 2021

### Create docker image with Renaissance benchmark

Use some java docker image e.g. `openjdk` (it uses *oraclelinux* but has `curl`, `java` that is sufficient for Renaissance).
Built image contains installed benchmark (`curl -L -O <renaissance-jar-link>`) in the `/duet` folder.

Steps:
1. `install.sh` run during the containers build
2. `setup.sh` run at the start of the execution
3. `run.sh` run the benchmark - must be executed simultaneously over multiple runs

> Q: Should runs (start or run.sh) be synchronized?
> It might impact the interference (measure the overhead of run start).

> Q: Should `run.sh` execute only a single run and the harness takes care of repeated runs?
> Probably yes...

4. stop phase - optional termination?
5. `process.sh` takes benchmark file results and converts them to a fixed format, stored as `/duet/results.csv`

```
$ docker run --name duetbench -d renaissance bash  # detached run (need bash to prevent immediate exit)
$ docker exec -it duetbench "./setup.sh"
$ docker exec -it duetbench "./run.sh"
$ docker exec -it duetbench "./process.sh"
$ docker cp duetbench:/duet/results.csv .
```

**restults.csv** format:
```
benchmark, wallclock_start_ms, epoch_start_ms, iteration_time_ns, iteration, machine, provider, jdk, jdk_version, time, pair, kind, total_ms, process_cpu_time_ns, compilation_time_ms, compilation_total_ms
```

## Nov 27 2021

Script that runs duet [duetbench.sh](./benchmarks/duetbench.sh) that just executes docker commands (runs are in parallel).

Used `docker commit <container> <new_image_name>` to update container image.

Now it contains python3 pandas and script `process-renaissance.py` that converts renaissance results to results.csv format.

Duet pair i.e. identification of given parallel run can be done via an environment variable passed to docker run e.g. `DUET_PAIR`.

Create an alpine image, easier to modify with `docker commit`.


## Nov 28 2021

Create Graal image.


