import unittest
import AutoencodeurProbabiliste.sourceae as projetae

class TestLearningRate(unittest.TestCase):
    """Test vérifiant la réaction de l'interface si l'utilisateur saisira un taux d'aprentissage qui n'est pas
    compris strictement entre 0 et 1.
    """

    def test_learning_rate(self):
        interface = projetae.UI_fenetre()
        # Taux d'aprentissage n'est pas strictement entre 0 et 1
        params = [2, 256, 128, 64, 10]
        valeur = interface.entrainement.verifier_params(params, lr=2.5)
        self.assertEqual(valeur, -1)

if __name__ == '__main__':
    unittest.main()
