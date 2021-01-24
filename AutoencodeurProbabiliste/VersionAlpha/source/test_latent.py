import unittest
from source import Autoencodeur as AE


class LatentTest(unittest.TestCase):
    """ Test chargé d'affirmer si la dimension latente est effectivement plus petite que la dimension d'entrée.
    Si cette condition n'est pas remplie, le codeur automatique n'apprendra jamais à compresser les données."""

    def setUp(self):
        self.ae = AE.AutoEncodeur(2, 512, 128, 16, 0.01)

    def test_latent_dim(self):
        self.assertLess(self.ae.latent_dim ,self.ae.input_dim)


if __name__ == '__main__':
    unittest.main()