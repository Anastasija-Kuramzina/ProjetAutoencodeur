import unittest
import AutoencodeurProbabiliste.sourceae as projetae

class TestSymmetrie(unittest.TestCase):
    """ Tests responsables de la confirmation que la k-eme et (n-k) couche ont la même dimension
    """

    def test_symmetrie(self):
        """Test de symmétrie: couche d'entrée et couche de sortie"""
        self.ae = projetae.AutoEncodeur(2, 512, 128, 16, 0.01, 0.0012)
        inputdim = self.ae.input_dim
        outputdim = self.ae.decoder.output_dim
        self.assertEqual(inputdim, outputdim)

    def test_symmetrie_c1(self):
        """Test de symmétrie: premiere couche apres entrée, derniere couche avant sortie"""
        self.ae = projetae.AutoEncodeur(2, 512, 128, 16, 0.01, 0.0012)
        premiere_couche = self.ae.encoder.c1.units
        derniere_couche = self.ae.decoder.c3.units
        self.assertEqual(premiere_couche, derniere_couche)

    def test_symmetrie_c2(self):
        """Test de symmétrie: deuxieme couche apres entrée, deixueme couche avant sortie"""
        self.ae = projetae.AutoEncodeur(2, 512, 128, 16, 0.01, 0.0012)
        premiere_couche = self.ae.encoder.c2.units
        derniere_couche = self.ae.decoder.c2.units
        self.assertEqual(premiere_couche, derniere_couche)

    def test_symmetrie_c3(self):
        """Test de symmétrie: derniere couche avant couche latente, premiere couche apres couche latente"""
        self.ae = projetae.AutoEncodeur(2, 512, 128, 16, 0.01, 0.0012)
        premiere_couche = self.ae.encoder.c3.units
        derniere_couche = self.ae.decoder.c1.units
        self.assertEqual(premiere_couche, derniere_couche)


if __name__ == '__main__':
    unittest.main()
