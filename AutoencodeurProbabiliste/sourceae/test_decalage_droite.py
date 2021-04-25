import unittest
import AutoencodeurProbabiliste.sourceae as projetae

class TestDecalageDroite(unittest.TestCase):
    """Test vérifiant le fonctionnement du décalage de la galerie
    """

    def test_decalage_droite(self):
        interface = projetae.UI_fenetre()
        indices =  [3,4,5,6]
        indices_apres = [2,3,4,5]
        interface.gallerie.images_courantes = indices
        interface.gallerie.image_precedente()
        result = interface.gallerie.images_courantes

        self.assertEqual(result, indices_apres)

if __name__ == '__main__':
    unittest.main()