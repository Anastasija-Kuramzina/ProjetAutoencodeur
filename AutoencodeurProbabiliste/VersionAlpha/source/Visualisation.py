import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import tensorflow as tf
import numpy as np
import random
from scipy.stats import norm
import tensorflow.keras.datasets.mnist as mnist
import AutoencodeurProbabiliste.VersionAlpha.source.Autoencodeur as AE
import AutoencodeurProbabiliste.VersionAlpha.source.Donnees as Donnees



def preparer_autoencodeur(test_images):
    """ Méthode chargée de l'instantiation d'un autoencodeur, de son entrainement et de
    preparation des images reconstruites """

    # Chargement et entrainement d'un autoencodeur
    autoencodeur = AE.AutoEncodeur(input_dim, latent_dim, dim_couche_1, dim_couche_2, dim_couche_3, kl_poids)
    AE.train(autoencodeur, learning_rate, num_epochs)

    # Preparation des images pour une visualisation:
    reconstructed = autoencodeur(test_images)
    return reconstructed, autoencodeur


def comparer_resultats():
    """ Visualisation de la performance d'une instance de classe AutoEncodeur. Cette méthode affiche 10 images originales
    et leurs recostructions """
    digit_size = 32
    plt.figure(figsize=(20, 4))
    j = random.randint(0,64)
    for i in range(10):
        inputaxis = plt.subplot(2, 10, i + 1)
        plt.imshow(test_images[j].reshape(digit_size, digit_size,3))
        plt.gray()
        inputaxis.get_xaxis().set_visible(False)
        inputaxis.get_yaxis().set_visible(False)

        outputaxis = plt.subplot(2, 10, i + 11)
        to_plot = tf.reshape(reconstructed[j],[digit_size, digit_size,3] )
        plt.imshow(to_plot)
        plt.gray()
        outputaxis.get_xaxis().set_visible(False)
        outputaxis.get_yaxis().set_visible(False)
        j = j + 1
    plt.show()




def colors(labels):
    """" Assignation d'un couleur a chaque chiffre (chaque classe de MNIST) pour pouvoir visualiser l'espace latent"""
    cols = {0: 'lightcoral', 1: 'olivedrab', 2: 'goldenrod', 3: 'darkseagreen', 4: 'saddlebrown',
                5: 'steelblue', 6: 'cornflowerblue', 7: 'limegreen', 8: 'darkviolet', 9: 'slateblue'}
    colors = list(map(cols.get, labels))
    return colors


def latent_plot(z_mu, z_log_var, test_labels):
    """" Preparation des encodages pour une visualisation, conversion de chaque vecteur latent en
    un couple des coordonnées et cisualisation des points obtenues dans une grille"""
    X_latent = []
    Y_latent = []

    # Conversion les vecteurs latentes en couples de coordonnées (x,y)
    for i in range(len(encoded)):
        X_latent.append(encoded[i][0])
        Y_latent.append(encoded[i][1])

    plt.figure(figsize=(14, 14))
    plt.scatter(X_latent, Y_latent, c=colors(test_labels))

    #Légende
    patchlist = []
    for i in range(10):
        patchlist.append(mpatches.Patch(color=colors([i])[0], label=i))
    plt.legend(handles=patchlist)
    plt.show()



def decoder_grille(autoencodeur):
    """Décodage et affichage d'un morceau de l'espace latent"""
    num_digits = 20
    digit_size = 32
    figure = np.zeros((digit_size * num_digits, digit_size * num_digits))

    # Construction d'une grille aléatoire de l'espace continu
    grid_x = norm.ppf(np.linspace(0.05, 0.95, num_digits))
    grid_y = norm.ppf(np.linspace(0.05, 0.95, num_digits))

    # Pour chacun des 20 points aléatoires de la grille,
    for i, yi in enumerate(grid_x):
        for j, xi in enumerate(grid_y):
            # Prendre un point aléatoire et le convertir en un tenseur
            z_echantillon = tf.convert_to_tensor([xi, yi], dtype=tf.float32)
            z_echantillon = tf.reshape(z_echantillon, [1,2])
            # Decoder ce tenseur en un tenseur de sortie de dimension 1024
            x_decode = autoencodeur.decoder(z_echantillon)

            # Remodelation en une image de 32 par 32 pixels
            chiffre = tf.reshape(x_decode,[digit_size, digit_size])

            figure[i * digit_size: (i + 1) * digit_size,
                           j * digit_size: (j + 1) * digit_size] = chiffre

    plt.figure(figsize=(15, 15))
    plt.imshow(figure)#, cmap='Greys')
    plt.show()


if __name__=='__main__':

    # Choix des hyperparametres
    learning_rate = 0.0002
    num_epochs = 10
    dim_couche_1 = 1024
    dim_couche_2 = 784
    dim_couche_3 = 512
    latent_dim = 16
    input_dim = 1024*3
    kl_poids = 0.000002

    # Preparation dún auto-encodeur
    test_images, test_labels = Donnees.Donnees.test_donnees_cifar()

    # Conversion des etiquettes
    test_labels = np.array(test_labels)
    labels = []
    for i in test_labels:
        labels.append(i[0])

    test_labels = labels


    reconstructed, autoencodeur = preparer_autoencodeur(test_images)

    # Visualisation des 10 images reconstruites par l'autoencodeur
    comparer_resultats()

    # Visualisation de l'espace latent
    z_mu, z_log_var, encoded = autoencodeur.encoder(test_images)
    latent_plot(z_mu, z_log_var, test_labels)

    # Decodage d'un morceau de l'espace latent
    decoder_grille(autoencodeur)


