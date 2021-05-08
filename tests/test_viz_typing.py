from sinnfull.viz import StrTuple

def test_strtuple():
    assert StrTuple((5,4)) == StrTuple(5,4) == StrTuple("(5,4)") == StrTuple("(5, 4)")
    assert StrTuple((5,4)) < StrTuple((5,10))
    assert str(StrTuple((5,4))) == '(5, 4)'
    assert StrTuple((5,4)) == '(5,4)'
    assert str(StrTuple((5,4))[1]) == '(4,)'
