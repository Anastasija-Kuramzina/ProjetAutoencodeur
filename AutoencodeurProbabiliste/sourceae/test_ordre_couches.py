import unittest
import AutoencodeurProbabiliste.sourceae as projetae

class TestOrdreCouches(unittest.TestCase):
    """Test vérifiant la réaction de l'interface aux si l'utilisateur saisira les couches qui ne sont
    pas dans l'ordre strictement décroissant
    """

    def test_ordre_couches(self):
        interface = projetae.UI_fenetre()
        params = [2, 128, 64, 256, 10]
        valeur = interface.entrainement.verifier_params(params, lr=0.001)
        self.assertEqual(valeur, -1)

if __name__ == '__main__':
    unittest.main()
