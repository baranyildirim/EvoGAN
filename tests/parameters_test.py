#!/usr/bin/env python3
import sys
sys.path.insert(0, '..')

from parameters import Parameters
import unittest

class TestParameters(unittest.TestCase):
    def test_serialize(self):
        p = Parameters(0, 0, 1, 1)
        actual = p.serialize()
        expected = int("0011")
        self.assertEqual(actual, expected)


if __name__ == '__main__':
    unittest.main()
