import AutoencodeurProbabiliste.modules as modules
import matplotlib.pyplot as plt
from scipy.stats import norm
import numpy as np
import tensorflow as tf

class UI_morceau():
    """ Classe responsable de l'affichage d'un graphique 2D des images données encodées dans l'espace latent """

    def __init__(self, fenetre, entrainement):
        # Fenetre principale
        self.fenetre = fenetre
        self.train = entrainement
        self.donnees, self.labels = modules.Donnees.test_donnees_mnist()

    def preparer_autoencoder(self):
        if self.train.training_status == 0:
            print('Autoencodeur pas encore entraîné')
            return -1
        elif self.train.training_status == 1:
            autoencodeur = self.train.autoencoder
            return autoencodeur


    def decoder_grille(self):
        """Méthode pour décodage et affichage d'un morceau aléatoire de l'espace latent"""
        num_digits = 20
        digit_size = 28
        autoencodeur = self.preparer_autoencoder()
        figure = np.zeros((digit_size * num_digits, digit_size * num_digits))

        # Construction d'une grille aléatoire de l'espace latent continu
        grid_x = norm.ppf(np.linspace(0.05, 0.95, num_digits))
        grid_y = norm.ppf(np.linspace(0.05, 0.95, num_digits))

        # Pour chacun des 400 points aléatoires de la grille,
        for i, yi in enumerate(grid_x):
            for j, xi in enumerate(grid_y):
                # Prendre un point aléatoire et le convertir en un tenseur
                z_echantillon = tf.convert_to_tensor([xi, yi], dtype=tf.float32)
                z_echantillon = tf.reshape(z_echantillon, [1, 2])
                # Decoder ce tenseur en un tenseur de sortie de dimension 784
                x_decode = autoencodeur.decoder(z_echantillon)

                # Remodelation en une image de 28 par 28 pixels
                chiffre = tf.reshape(x_decode, [digit_size, digit_size])

                figure[i * digit_size: (i + 1) * digit_size,
                j * digit_size: (j + 1) * digit_size] = chiffre

        plt.figure(figsize=(15, 15))
        plt.imshow(figure)
        plt.show()
