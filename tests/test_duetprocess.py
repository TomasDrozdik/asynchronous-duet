import pytest
import pandas as pd
from duet.constants import RF, AF
from duet.process import compute_overlaps

af = {AF.date: "", AF.hostname: "", AF.lscpu: "", AF.meminfo: "", AF.uname: ""}
runid_1 = {RF.suite: "xxx", RF.benchmark: "yyy", RF.type: "duet", RF.runid: 0, **af}
runid_2 = {RF.suite: "aaa", RF.benchmark: "bbb", RF.type: "duet", RF.runid: 0, **af}


@pytest.mark.parametrize(
    "data,overlap_count",
    # fmt: off
    [
        (
            [
                {**runid_1, RF.pair: "A", RF.iteration: 0, RF.start_ns: 1, RF.end_ns: 2},
                {**runid_1, RF.pair: "B", RF.iteration: 0, RF.start_ns: 1, RF.end_ns: 2},
            ],
            1
        ),
        (
            [
                {**runid_1, RF.pair: "A", RF.iteration: 0, RF.start_ns: 0, RF.end_ns: 4},
                {**runid_1, RF.pair: "A", RF.iteration: 1, RF.start_ns: 6, RF.end_ns: 10},
                {**runid_1, RF.pair: "B", RF.iteration: 0, RF.start_ns: 2, RF.end_ns: 6},
                {**runid_1, RF.pair: "B", RF.iteration: 1, RF.start_ns: 8, RF.end_ns: 12},
            ],
            2
        ),
        (
            [
                {**runid_1, RF.pair: "A", RF.iteration: 0, RF.start_ns: 0, RF.end_ns: 4},
                {**runid_1, RF.pair: "A", RF.iteration: 1, RF.start_ns: 6, RF.end_ns: 10},
                {**runid_1, RF.pair: "B", RF.iteration: 0, RF.start_ns: 2, RF.end_ns: 6},
                {**runid_1, RF.pair: "B", RF.iteration: 1, RF.start_ns: 8, RF.end_ns: 12},
            ],
            2
        ),
        (
            [
                {**runid_1, RF.pair: "A", RF.iteration: 0, RF.start_ns: 0, RF.end_ns: 4},
                {**runid_1, RF.pair: "A", RF.iteration: 1, RF.start_ns: 4, RF.end_ns: 8},
                {**runid_1, RF.pair: "B", RF.iteration: 0, RF.start_ns: 0, RF.end_ns: 5},
                {**runid_1, RF.pair: "B", RF.iteration: 1, RF.start_ns: 5, RF.end_ns: 10},
            ],
            3
        ),
        (
            [
                {**runid_1, RF.pair: "B", RF.iteration: 0, RF.start_ns: 0, RF.end_ns: 4},
                {**runid_1, RF.pair: "B", RF.iteration: 1, RF.start_ns: 4, RF.end_ns: 8},
                {**runid_1, RF.pair: "A", RF.iteration: 0, RF.start_ns: 0, RF.end_ns: 5},
                {**runid_1, RF.pair: "A", RF.iteration: 1, RF.start_ns: 5, RF.end_ns: 10},
            ],
            3
        ),
        (
            [
                {**runid_1, RF.pair: "A", RF.iteration: 0, RF.start_ns: 0, RF.end_ns: 4},
                {**runid_1, RF.pair: "A", RF.iteration: 1, RF.start_ns: 4, RF.end_ns: 8},
                {**runid_1, RF.pair: "B", RF.iteration: 0, RF.start_ns: 0, RF.end_ns: 5},
                {**runid_1, RF.pair: "B", RF.iteration: 1, RF.start_ns: 5, RF.end_ns: 10},
                {**runid_2, RF.pair: "A", RF.iteration: 0, RF.start_ns: 0, RF.end_ns: 4},
                {**runid_2, RF.pair: "A", RF.iteration: 1, RF.start_ns: 4, RF.end_ns: 8},
                {**runid_2, RF.pair: "B", RF.iteration: 0, RF.start_ns: 0, RF.end_ns: 5},
                {**runid_2, RF.pair: "B", RF.iteration: 1, RF.start_ns: 5, RF.end_ns: 10},
            ],
            6
        ),
    ]
    # fmt: on
)
def test_benchmark_start_instances(data, overlap_count):
    df = pd.DataFrame(data)
    odf = compute_overlaps(df)
    assert odf.shape[0] == overlap_count
