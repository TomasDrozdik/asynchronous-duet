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


AF = ArtifactFiels()
RF = ResultsFields()

MS_PER_S = 1000
US_PER_S = 1000 * MS_PER_S
NS_PER_S = 1000 * US_PER_S

MS_PER_NS = NS_PER_S / MS_PER_S

ARTIFACT_COL = [AF.date, AF.hostname, AF.lscpu, AF.meminfo, AF.uname]

RUN_ID_COL = [RF.suite, RF.benchmark, RF.type, RF.runid]
PAIR_ID_COL = RUN_ID_COL + [RF.pair]
ITER_ID_COL = PAIR_ID_COL + [RF.iteration]

TIME_NS_COL = [RF.start_ns, RF.end_ns]
TIME_D_NS_COL = TIME_NS_COL + [RF.time_ns]

BASE_COL = ITER_ID_COL + TIME_D_NS_COL

TIME_COL = [RF.start_ns, RF.end_ns]
TIME_D_COL = TIME_COL + [RF.time_ns]

OVERLAP_NS_COL = [RF.overlap_start_ns, RF.overlap_end_ns]
OVERLAP_D_NS_COL = OVERLAP_NS_COL + [RF.overlap_time_ns]

OVERLAP_COL = [RF.overlap_start, RF.overlap_end]
OVERLAP_D_COL = OVERLAP_COL + [RF.overlap_time]
