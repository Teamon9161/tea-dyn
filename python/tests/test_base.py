from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.testing import assert_array_equal
from tea_dyn import s


def test_base():
    @dataclass
    class test:
        name: str

    ctx = [[1, -2, 3], np.array(()), "hello", test("aab")]
    # result backend will not be changed because the output of expression is not a iter
    assert s(0).eval(ctx, backend="np") == [1, -2, 3]
    assert_array_equal(s(1).eval(ctx), np.array(()))
    assert s(2).eval(ctx) == "hello"
    assert s(3).eval(ctx) == test("aab")
    assert s(0).abs().shift(-1, 0).eval(ctx, backend="vec") == [2, 3, 0]


def test_time():
    time_arr = np.array(
        [
            np.datetime64("2020-01-01"),
            np.datetime64("2020-01-02"),
            np.datetime64("2020-01-03"),
        ]
    )
    ctx = [time_arr, np.datetime64("2023-01-01")]
    assert_array_equal(s(0).eval(ctx), time_arr)
    assert s(1).eval(ctx) == np.datetime64("2023-01-01")


def test_expr_for_ndarray():
    arr = np.array([[1, -2, 3], [4, -5, 6], [7, -8, 9]])
    ctx = [arr]
    expr = s(0).abs().shift(1, 0, axis=1)
    assert_array_equal(expr.eval(ctx), np.array([[0, 1, 2], [0, 4, 5], [0, 7, 8]]))


def test_pd_backend():
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, -5, 6]})
    assert_array_equal(s("a").eval(df), np.array([1, 2, 3]))
    assert_array_equal(s(1).abs().eval(df), np.array([4, 5, 6]))
    assert s("b").shift(1, 0).eval(df, backend="list") == [0, 4, -5]
