from dataclasses import dataclass
from tea_dyn import s
import numpy as np
from numpy.testing import assert_array_equal

def test_base():
    @dataclass
    class test:
        name: str

    ctx = [[1, -2, 3], np.array(()), "hello", test("aab")]
    # result backend will not be changed because the output of expression is not a iter
    assert s(0).eval(ctx, backend='np') == [1, -2, 3]
    assert_array_equal(s(1).eval(ctx), np.array(()))
    assert s(2).eval(ctx) == "hello"
    assert s(3).eval(ctx) == test("aab")
    assert s(0).abs().shift(-1, 0).eval(ctx, backend="vec") == [2, 3, 0]

def test_expr_for_ndarray():
    arr = np.array([[1, -2, 3], [4, -5, 6], [7, -8, 9]])
    ctx = [arr]
    expr = s(0).abs().shift(1, 0, axis=1)
    assert_array_equal(expr.eval(ctx), np.array([[0, 1, 2], [0, 4, 5], [0, 7, 8]]))
