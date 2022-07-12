class ArtifactFiels:
    date = "date"
    hostname = "hostname"
    lscpu = "lscpu"
    meminfo = "meminfo"
    uname = "uname"


class ResultsFields:
    suite = "suite"
    benchmark = "benchmark"
    type = "type"
    runid = "runid"
    pair = "pair"
    iteration = "iteration"
    duetorder = "duet_order"

    start_ns = "iteration_start_ns"
    end_ns = "iteration_end_ns"
    time_ns = "iteration_time_ns"

    start = "iteration_start"
    end = "iteration_end"
    time = "iteration_time"

    overlap_start_ns = "overlap_start_ns"
    overlap_end_ns = "overlap_end_ns"
    overlap_time_ns = "overlap_time_ns"

    overlap_start = "overlap_start"
    overlap_end = "overlap_end"
    overlap_time = "overlap_time"

    a_suffix = "_A"
    b_suffix = "_B"

    iteration_A = "iteration" + a_suffix
    iteration_B = "iteration" + b_suffix

    start_ns_A = "iteration_start_ns" + a_suffix
    end_ns_A = "iteration_end_ns" + a_suffix
    time_ns_A = "iteration_time_ns" + a_suffix
    start_ns_B = "iteration_start_ns" + b_suffix
    end_ns_B = "iteration_end_ns" + b_suffix
    time_ns_B = "iteration_time_ns" + b_suffix

    start_A = "iteration_start" + a_suffix
    end_A = "iteration_end" + a_suffix
    time_A = "iteration_time" + a_suffix
    start_B = "iteration_start" + b_suffix
    end_B = "iteration_end" + b_suffix
    time_B = "iteration_time" + b_suffix


class DerivedFields:
    env = "environment"

    sample = "sample"
    ci = "ci"
    ci_width = "ci_width"

    pair_diff = "pair_diff"

    pair_speedup = "pair_speedup"
    gmsr = "run_gmean_speedup"
    ggmsr = "bench_gmean_speedup"

    run_time_ns = "run_time_ns"
    run_time = "run_time"
    iteration_count = "iteration_count"
    run_time_per_iteration = "run_time_per_iteration"
    run_time_speedup = "run_time_speedup"


AF = ArtifactFiels()
RF = ResultsFields()
DF = DerivedFields()

MS_PER_S = 1000
US_PER_S = 1000 * MS_PER_S
NS_PER_S = 1000 * US_PER_S

MS_PER_NS = NS_PER_S / MS_PER_S

ARTIFACT_COL = [AF.date, AF.hostname, AF.lscpu, AF.meminfo, AF.uname, DF.env]

BENCHMARK_ID_COL = [RF.suite, RF.benchmark, RF.type]
RUN_ID_COL = BENCHMARK_ID_COL + [RF.runid]
PAIR_ID_COL = RUN_ID_COL + [RF.pair]
ITER_ID_COL = PAIR_ID_COL + [RF.iteration]

BENCHMARK_ENV_COL = BENCHMARK_ID_COL + [DF.env]

TIME_NS_COL = [RF.start_ns, RF.end_ns]
TIME_NS_SUFFIX_COL = [RF.start_ns_A, RF.end_ns_A, RF.start_ns_B, RF.end_ns_B]
TIME_D_NS_COL = TIME_NS_COL + [RF.time_ns]
TIME_D_NS_SUFFIX_COL = TIME_NS_SUFFIX_COL + [RF.time_ns_A, RF.time_ns_B]

BASE_COL = ITER_ID_COL + TIME_D_NS_COL

TIME_COL = [RF.start, RF.end]
TIME_D_COL = TIME_COL + [RF.time]

OVERLAP_NS_COL = [RF.overlap_start_ns, RF.overlap_end_ns]
OVERLAP_D_NS_COL = OVERLAP_NS_COL + [RF.overlap_time_ns]

OVERLAP_COL = [RF.overlap_start, RF.overlap_end]
OVERLAP_D_COL = OVERLAP_COL + [RF.overlap_time]
