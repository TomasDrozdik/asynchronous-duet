import pandas as pd
import pytest

from duet.constants import AF, DF, RF
from duet.process import alter_score, compute_overlaps

af = {
    DF.env: "test",
    AF.date: "",
    AF.hostname: "",
    AF.lscpu: "",
    AF.meminfo: "",
    AF.uname: "",
}
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
def test_compute_overlaps(data, overlap_count):
    df = pd.DataFrame(data)
    odf = compute_overlaps(df)
    assert odf.shape[0] == overlap_count


@pytest.mark.parametrize(
    "data,rate,expected",
    # fmt: off
    [
        # 200% Slowdown, check that it does not affect other pair
        (
            [
                {**runid_1, RF.pair: "A", RF.iteration: 0, RF.start_ns: 1, RF.end_ns: 2, RF.time_ns: 1},
                {**runid_1, RF.pair: "A", RF.iteration: 1, RF.start_ns: 3, RF.end_ns: 4, RF.time_ns: 1},
                {**runid_1, RF.pair: "B", RF.iteration: 0, RF.start_ns: 1, RF.end_ns: 2, RF.time_ns: 1},
                {**runid_1, RF.pair: "B", RF.iteration: 1, RF.start_ns: 3, RF.end_ns: 4, RF.time_ns: 1},
            ],
            2,
            [
                {**runid_1, RF.pair: "A", RF.iteration: 0, RF.start_ns: 1, RF.end_ns: 2, RF.time_ns: 1},
                {**runid_1, RF.pair: "A", RF.iteration: 1, RF.start_ns: 3, RF.end_ns: 4, RF.time_ns: 1},
                {**runid_1, RF.pair: "B", RF.iteration: 0, RF.start_ns: 1, RF.end_ns: 4, RF.time_ns: 3},
                {**runid_1, RF.pair: "B", RF.iteration: 1, RF.start_ns: 5, RF.end_ns: 8, RF.time_ns: 3},
            ],
        ),
        # 50% Speedup
        (
            [
                {**runid_1, RF.pair: "B", RF.iteration: 0, RF.start_ns: 0, RF.end_ns: 2, RF.time_ns: 2},
                {**runid_1, RF.pair: "B", RF.iteration: 1, RF.start_ns: 2, RF.end_ns: 4, RF.time_ns: 2},
            ],
            -0.5,
            [
                {**runid_1, RF.pair: "B", RF.iteration: 0, RF.start_ns: 0, RF.end_ns: 1, RF.time_ns: 1},
                {**runid_1, RF.pair: "B", RF.iteration: 1, RF.start_ns: 1, RF.end_ns: 2, RF.time_ns: 1},
            ],
        ),
        # Copy of the 200 slowdown but this time with 2 runid, to verify that it does the groupby correctly
        (
            [
                {**runid_1, RF.pair: "A", RF.iteration: 0, RF.start_ns: 1, RF.end_ns: 2, RF.time_ns: 1},
                {**runid_1, RF.pair: "A", RF.iteration: 1, RF.start_ns: 3, RF.end_ns: 4, RF.time_ns: 1},
                {**runid_1, RF.pair: "B", RF.iteration: 0, RF.start_ns: 1, RF.end_ns: 2, RF.time_ns: 1},
                {**runid_1, RF.pair: "B", RF.iteration: 1, RF.start_ns: 3, RF.end_ns: 4, RF.time_ns: 1},
                {**runid_2, RF.pair: "A", RF.iteration: 0, RF.start_ns: 1, RF.end_ns: 2, RF.time_ns: 1},
                {**runid_2, RF.pair: "A", RF.iteration: 1, RF.start_ns: 3, RF.end_ns: 4, RF.time_ns: 1},
                {**runid_2, RF.pair: "B", RF.iteration: 0, RF.start_ns: 1, RF.end_ns: 2, RF.time_ns: 1},
                {**runid_2, RF.pair: "B", RF.iteration: 1, RF.start_ns: 3, RF.end_ns: 4, RF.time_ns: 1},
            ],
            2,
            [
                {**runid_1, RF.pair: "A", RF.iteration: 0, RF.start_ns: 1, RF.end_ns: 2, RF.time_ns: 1},
                {**runid_1, RF.pair: "A", RF.iteration: 1, RF.start_ns: 3, RF.end_ns: 4, RF.time_ns: 1},
                {**runid_1, RF.pair: "B", RF.iteration: 0, RF.start_ns: 1, RF.end_ns: 4, RF.time_ns: 3},
                {**runid_1, RF.pair: "B", RF.iteration: 1, RF.start_ns: 5, RF.end_ns: 8, RF.time_ns: 3},
                {**runid_2, RF.pair: "A", RF.iteration: 0, RF.start_ns: 1, RF.end_ns: 2, RF.time_ns: 1},
                {**runid_2, RF.pair: "A", RF.iteration: 1, RF.start_ns: 3, RF.end_ns: 4, RF.time_ns: 1},
                {**runid_2, RF.pair: "B", RF.iteration: 0, RF.start_ns: 1, RF.end_ns: 4, RF.time_ns: 3},
                {**runid_2, RF.pair: "B", RF.iteration: 1, RF.start_ns: 5, RF.end_ns: 8, RF.time_ns: 3},
            ],
        ),
    ]
    # fmt: on
)
def test_alter_score(data, rate, expected):
    df = pd.DataFrame(data)
    df_altered = alter_score(df, rate)
    df_expected = pd.DataFrame(expected)

    pd.testing.assert_frame_equal(
        left=df_altered,
        right=df_expected,
        check_like=True,
        check_dtype=False,
        check_column_type=False,
    )
