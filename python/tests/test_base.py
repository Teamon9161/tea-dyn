from tea_dyn import s
# import os
# os.environ['RUST_BACKTRACE'] = "full"

def test_base():
    ctx = [[1, -2, 3]]
    expr = s(0).abs().shift(-1, 0)
    assert expr.eval(ctx) == [2, 3, 0]

def test_nd():
    import numpy as np
    from numpy.testing import assert_array_equal
    arr = np.array([[1, -2, 3], [4, -5, 6], [7, -8, 9]])
    ctx = [arr]
    expr = s(0).abs().shift(1, 0, axis=1)
    assert_array_equal(expr.eval(ctx), np.array([[2, 3, 0], [5, 6, 0], [8, 9, 0]]))

test_base()
