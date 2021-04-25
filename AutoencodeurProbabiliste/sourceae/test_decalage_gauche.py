import unittest
import AutoencodeurProbabiliste.sourceae as projetae

class TestDecalageGauche(unittest.TestCase):
    """Test vérifiant le fonctionnement du décalage de la galerie
    """

    def test_decalage_gauche(self):
        interface = projetae.UI_fenetre()
        indices = [2,3,4,5]
        indices_apres = [3,4,5,6]
        interface.gallerie.images_courantes = indices
        interface.gallerie.image_suivante()
        result = interface.gallerie.images_courantes

        self.assertEqual(result, indices_apres)

if __name__ == '__main__':
    unittest.main()