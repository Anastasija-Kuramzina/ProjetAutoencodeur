import unittest
import Autoencodeur as AE


class InputTest(unittest.TestCase):

    def set_up(self):
        self.ae = AE.autoencodeur(2, 512, 128, 16, 0.01)

    def test_something(self):
        self.assertEqual(self.ae.input_dim, 784)


if __name__ == '__main__':
    unittest.main()
