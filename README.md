# Asynchronous Duet Benchmarking - Master Thesis

Goal of this thesis is to create tools for running duet benchmarks with unsynchronized inner harness iterations utilizing docker containers for isolation.

Thesis work can be split into three parts:

1. **Tool for running the benchmarks** defines what is the input and desired output of an asynchronous duet benchmark run.
    It describes how to incorporate a benchmark of some software into a duet benchmarking procedure to identify performance regression on commit basis.
    Output should be in standardized format for the next tool to functions.
    Tool should also provide a procedure of converting any benchmark output in to this standardized output format.

2. **Tool for results processing** should process the standardized output and decide whether there is a difference in performance between the two versions.

3. **Result analysis and different approaches study**

## Overall plan and progress

1. **Part 1**
    1. [ ] Create POC for running concurrent Renaissance benchmark in A/A duet utilizing 2 Docker containers *- Expected: Nov 07 2021*
    2. [ ] Identify parameters for actual run *- Expected: mid Nov 2021*
        - hardware, cloud providers capabilities
        - benchmarks and versions for A/B runs, their setup, input parameters, output parameters, teardown
        - resource contention of given benchmark (CPU, Mem, I/O)
    3. [ ] Create alpha taking required capabilities into considerations *- Expected: end of Nov 2021*
        - make a test run on identified benchmarks
        - make A/A and A/B runs with known regressions
    4. [ ] Run benchmarks on different cloud instances and reference server *- Expected: mid Dec 2021*

2. **Part 2**
    1. [ ] Study data processing of synchronous duet *- Expected: min Nov 2021*
    2. [ ] Analyse A/A duets on 1.1 *- Expected: mid Nov 2021*
    3. [ ] Analyse A/B duets benchmarks from 1.3 *- Expected: Dec-Jan 2021*
    4. [ ] Explore processing options for data processing *- Expected: Jan 2022*
    5. [ ] Run the data processing from 2.4 on data from 1.4 *- Expected: Jan-Feb 2022*

3. **Part 3**
    1. [x] Review core papers *- Done: Nov 06 2021*
    2. [ ] Review related work *- Expected: Nov 14 2021*
    3. [ ] Analyze different interferences and observe their impact on designed scenarios *- Expected: Jan 2022*
    4. [ ] Assess viability of 2.3 *- Expected Feb-Mar 2022*
