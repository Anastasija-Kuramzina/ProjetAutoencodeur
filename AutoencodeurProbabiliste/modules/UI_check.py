import AutoencodeurProbabiliste.modules as modules
import matplotlib.pyplot as plt
import random
import tensorflow as tf

class UI_check():
    """ Classe responsable de l'affichage de 10 images et leurs reconstructions par l'autoencodeur afin de vérifier la
     qualité de ses performances"""

    def __init__(self, fenetre, entrainement):
        # Fenetre principale
        self.fenetre = fenetre
        self.train = entrainement
        self.donnees, labels = modules.Donnees.test_donnees_mnist()

    def preparer_autoencoder(self):
        if self.train.training_status == 0:
            print('Autoencodeur pas entraîné')
            return -1
        elif self.train.training_status == 1:
            self.autoencodeur = self.train.autoencoder
            reconstructed = self.autoencodeur(self.donnees)
            return reconstructed


    def comparer_resultats(self):
        """ Visualisation de la performance d'une instance de classe AutoEncodeur. Cette méthode affiche 10 images originales
                    et leurs recostructions """

        images_originales = self.donnees
        images_reconstruites = self.preparer_autoencoder()

        digit_size = 28
        plt.figure(figsize=(20, 4))
        j = random.randint(0, 64)
        for i in range(10):
            plt.gray()
            inputaxis = plt.subplot(2, 10, i + 1)
            plt.imshow(images_originales[j].reshape(digit_size, digit_size))
            inputaxis.get_xaxis().set_visible(False)
            inputaxis.get_yaxis().set_visible(False)

            outputaxis = plt.subplot(2, 10, i + 11)
            to_plot = tf.reshape(images_reconstruites[j], [digit_size, digit_size])
            plt.imshow(to_plot)
            outputaxis.get_xaxis().set_visible(False)
            outputaxis.get_yaxis().set_visible(False)
            j = j + 1
        plt.show()
