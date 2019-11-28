import unittest
import numpy as np

from GrayScottPDE import GrayScott

class TestSum(unittest.TestCase):

    def test_time_dimensions(self):
        N = 200
        model = GrayScott(N)
        parameters = [0.14, 0.06, 0.060, 0.062]
        times = np.arange(0, 10 * 10 + 10, 10)
        values = model.simulate(parameters, times)
        self.assertEqual(len(times), values.shape[0])

    def test_spatial_dimensions(self):
        N = 200
        model = GrayScott(N)
        parameters = [0.14, 0.06, 0.060, 0.062]
        times = np.arange(0, 10 * 10 + 10, 10)
        values = model.simulate(parameters, times)
        self.assertEqual(2*N*N, values.shape[1])

    def test_coral(self):
        N = 200
        model = GrayScott(N)
        parameters = [0.14, 0.06, 0.060, 0.062]
        times = np.arange(0, 100 * 10 + 10, 10)
        values = model.simulate(parameters, times)
        self.assertEqual(values[len(times)-1][0], 0.2689803513215552)

if __name__ == '__main__':
    unittest.main()
