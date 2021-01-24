import unittest
import Autoencodeur as AE


class DimensionTest(unittest.TestCase):

    def setUp(self):
        self.ae = AE.AutoEncodeur(2, 512, 128, 16, 0.01)

    def test_symmetry(self):
        self.assertEqual(self.ae.inputs.shape, self.ae.outputs.shape)


if __name__ == '__main__':
    unittest.main()
