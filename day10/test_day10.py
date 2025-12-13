# Regression tests

import unittest
from day10 import pm, reduce_mat


class Day10Tests(unittest.TestCase):

    def test_missing(self):
        mat = """
0 1 0 0 1 0 1 21
1 0 0 1 0 1 1 41
0 0 1 0 1 1 0 34
1 0 0 1 0 1 0 27
0 1 0 1 1 1 0 27
0 0 0 1 0 0 0 10
"""
        mat = [list(map(int, row.split())) for row in mat.strip().split("\n")]
        reduce_mat(mat)
        pm(mat)
        assert False


    def test_ordering(self):
        mat = """
1 0 0 1 1 0 1 0 53
1 0 1 0 0 0 0 1 30
1 1 0 0 0 0 0 1 43
0 1 0 0 0 1 1 1 41
0 1 0 0 0 1 1 1 41
1 0 1 0 0 0 0 0 18
0 0 1 0 1 1 0 0 24
0 0 1 1 1 1 0 0 35
"""
        mat = [list(map(int, row.split())) for row in mat.strip().split("\n")]
        reduce_mat(mat)
        pm(mat)
