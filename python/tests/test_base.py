from tea_dyn import s


def test_base():
    ctx = [[1, -2, 3]]
    expr = s(0).abs().shift(-1, 0)
    assert expr.eval(ctx) == [2, 3, 0]
