import unittest
import Autoencodeur as AE


class LatentTest(unittest.TestCase):

    def setUp(self):
        self.ae = AE.AutoEncodeur(2, 512, 128, 16, 0.01)

    def latent_test(self):
        self.assertLess(self.ae.dim_latent,self.ae.input_dim)


if __name__ == '__main__':
    unittest.main()