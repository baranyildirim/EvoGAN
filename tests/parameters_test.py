#!/usr/bin/env python3
from model.parameters import Parameters
import unittest

class TestParameters(unittest.TestCase):
    def test_serialize(self):
        p = Parameters(0, 0, 1, 1)
        actual = p.serialize()
        expected = list(0, 0, 1, 1)
        self.assertEqual(actual, expected)

    def test_from_serial(self):
        p = Parameters(0, 0, 1, 1)


