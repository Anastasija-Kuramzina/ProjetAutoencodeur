import AutoencodeurProbabiliste.sourceae as projetae
import matplotlib.pyplot as plt
import random
import tensorflow as tf

class UI_check():
    """ Classe responsable de l'affichage de 10 images et leurs reconstructions par l'autoencodeur afin de vérifier la
     qualité de ses performances.

     :param fenetre: fenêtre principale
     :type fenetre: 'tkinter.Tk'
     :param train: écran de l'entraînement ou l'instance de UI_check va être ajoutée
     :type train: class 'AutoencodeurProbabiliste.modules.UI_entrainement.UI_entrainement'
     :param donnees: les images a utiliser
     :type donnees: class 'numpy.ndarray'
     """

    def __init__(self, fenetre, entrainement):
        # Fenetre principale
        self.fenetre = fenetre
        self.train = entrainement
        self.donnees, labels = projetae.Donnees.test_donnees_mnist()


    def preparer_autoencoder(self):
        """ Méthode vérifiant si un autoencodeur entraîné est disponible.

        :return:  liste si aucun autoencodeur entraîné est disponible, et les images reconstuites par l'autoencodeur sinon
        :rtype: list/ class 'numpy.ndarray'
        """

        if self.train.training_status == 0:
            print('Autoencodeur pas entraîné')
            return []
        elif self.train.training_status == 1:
            self.autoencodeur = self.train.autoencoder
            reconstructed = self.autoencodeur(self.donnees)
            return reconstructed


    def comparer_resultats(self):
        """ Visualisation de la performance d'une instance de classe AutoEncodeur aprés l'entraînement.

        Cette méthode affiche 10 images originales et leurs recostructions """

        images_originales = self.donnees
        images_reconstruites = self.preparer_autoencoder()

        if images_reconstruites == []:
            return

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
