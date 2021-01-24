import unittest
from source import Autoencodeur as AE


class InputTest(unittest.TestCase):
    """ Test responsable de la confirmation que la dimension
    d'entrée d'une instance d'autoencodeur correspond aux
    dimensions d'une image MNIST aplatie. """

    def test_input(self):
        self.ae = AE.AutoEncodeur(2, 512, 128, 16, 0.01)

        self.assertEqual(self.ae.input_dim, 784)


if __name__ == '__main__':
    unittest.main()
