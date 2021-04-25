import unittest
import AutoencodeurProbabiliste.sourceae as projetae

class TestCoucheNulle(unittest.TestCase):
    """Test vérifiant la réaction de l'interface si l'utilisateur saisit une couche de taille nulle
    """

    def test_couche_nulle(self):
        interface = projetae.UI_fenetre()
        # Taille d'une couche n'est pas un entier strictement positif
        params = [2, 256, 128, 0, 10]
        valeur = interface.entrainement.verifier_params(params, lr=0.001)
        self.assertEqual(valeur, -1)



if __name__ == '__main__':
    unittest.main()
