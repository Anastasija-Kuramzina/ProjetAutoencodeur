# Fichier contenant méthodes nécessaires pour les visualisations de la performance de l'autoencodeur
# Code ancien, pas utilisé depuis version Alpha
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import tensorflow as tf
import numpy as np
import random
from scipy.stats import norm
import AutoencodeurProbabiliste.sourceae as modules


def preparer_autoencodeur(test_images):
    """ Méthode chargée de l'instantiation d'un autoencodeur, de son entrainement et de
    preparation des images reconstruites

    :param test_images: tenseur 2D de taille (batch_size, 784) contenant les images d'entrée
    :param type: class 'tensorflow.python.framework.ops.EagerTensor'
    :return: tenseur 2D de taille (batch_size, 784) contenant les images reconstruites, instance de classe AutoEncodeur
    :rtype: list
    """

    # Chargement et entrainement d'un autoencodeur
    autoencodeur = projetae.AutoEncodeur(input_dim, latent_dim, dim_couche_1, dim_couche_2, dim_couche_3, kl_poids)
    projetae.train(autoencodeur, learning_rate, num_epochs)

    # Preparation des images pour une visualisation:
    reconstructed = autoencodeur(test_images)
    return reconstructed, autoencodeur


def comparer_resultats():
    """ Visualisation de la performance d'une instance de classe AutoEncodeur. Cette méthode affiche 10 images originales
    et leurs recostructions.
    """
    digit_size = 28
    plt.figure(figsize=(20, 4))
    j = random.randint(0,64)
    for i in range(10):
        inputaxis = plt.subplot(2, 10, i + 1)
        plt.imshow(test_images[j].reshape(digit_size, digit_size))
        plt.gray()
        inputaxis.get_xaxis().set_visible(False)
        inputaxis.get_yaxis().set_visible(False)

        outputaxis = plt.subplot(2, 10, i + 11)
        to_plot = tf.reshape(reconstructed[j],[digit_size, digit_size] )
        plt.imshow(to_plot)
        plt.gray()
        outputaxis.get_xaxis().set_visible(False)
        outputaxis.get_yaxis().set_visible(False)
        j = j + 1
    plt.show()



def colors(labels):
    """" Assignation d'un couleur a chaque chiffre (chaque classe de MNIST) pour pouvoir visualiser l'espace latent.

    :param labels: liste d'entiers - étiquettes des images MNIST
    :type labels: class 'numpy.ndarray'
    :return: liste des strings - noms de couleurs associés aux étiquettes
    :rtype: class 'numpy.ndarray'

    """
    cols = {0: 'lightcoral', 1: 'olivedrab', 2: 'goldenrod', 3: 'darkseagreen', 4: 'saddlebrown',
                5: 'steelblue', 6: 'cornflowerblue', 7: 'limegreen', 8: 'darkviolet', 9: 'slateblue'}
    colors = list(map(cols.get, labels))
    return colors


def latent_plot(encoded, test_labels):
    """" Préparation des encodages pour une visualisation, conversion de chaque vecteur latent en
    un couple des coordonnées et visualisation des points obtenues sous la forme d'un scatterplot.

    :param encoded: un tenseur 2D de taille (batch_size, latent_dim) contenant les représentations latented des images test de MNIST
    :type encoded: class 'tensorflow.python.framework.ops.EagerTensor'
    :param test_labels: array contenant les étiquettes, une pour chaque image
    :type test_labels: class 'numpy.ndarray'

    """
    X_latent = []
    Y_latent = []

    # Conversion les vecteurs latentes en couples de coordonnées (x,y)
    for i in range(len(encoded)):
        X_latent.append(encoded[i][0])
        Y_latent.append(encoded[i][1])

    plt.figure(figsize=(14, 14))
    plt.scatter(X_latent, Y_latent, c=colors(test_labels))

    #Etiquettes
    etiquettes = {0: 'T-shirt', 1: 'Pantalons', 2: 'Pull', 3: 'Robe', 4: 'Manteau', 5: 'Sandale', 6: 'Chemise', 7: 'Sneaker',
                  8: 'Sac', 9: 'Botte'}

    #Légende
    patchlist = []
    for i in range(10):
        patchlist.append(mpatches.Patch(color=colors([i])[0], label=etiquettes[i]))
    plt.legend(handles=patchlist)
    plt.show()


def decoder_grille(autoencodeur):
    """Méthode décodage un morceau aléatoirement chosi de l'espace latent en une grille de 20x20 images.

    Pour chaque couple [x,y] des coordonnées générés alétoirement a partir d'un morceau 20x20 de l'espace latent,
    on lui passe par le décodeur pour générer une image. Ensuite on affiche les 400 images obtenues sous la
    forme d'une grille montrant la continuité de l'espace latent.

    :param autoencodeur: autoencodeur dont le décodeur sera utilisé pour le décodage des coordonnées aléatoires
    :type autoencodeur: classe: 'modules.AutoEncodeur'
    """
    num_digits = 20
    digit_size = 28
    figure = np.zeros((digit_size * num_digits, digit_size * num_digits))

    # Construction d'une grille aléatoire de l'espace continu
    grid_x = norm.ppf(np.linspace(0.05, 0.95, num_digits))
    grid_y = norm.ppf(np.linspace(0.05, 0.95, num_digits))

    # Pour chacun des 400 points aléatoires de la grille,
    for i, yi in enumerate(grid_x):
        for j, xi in enumerate(grid_y):
            # Prendre un point aléatoire et le convertir en un tenseur
            z_echantillon = tf.convert_to_tensor([xi, yi], dtype=tf.float32)
            z_echantillon = tf.reshape(z_echantillon, [1,2])
            # Decoder ce tenseur en un tenseur de sortie de dimension 784
            x_decode = autoencodeur.decoder(z_echantillon)

            # Remodelation en une image de 28 par 28 pixels
            chiffre = tf.reshape(x_decode,[digit_size, digit_size])

            figure[i * digit_size: (i + 1) * digit_size,
                           j * digit_size: (j + 1) * digit_size] = chiffre

    plt.figure(figsize=(15, 15))
    plt.imshow(figure)
    plt.show()


if __name__=='__main__':

    # Choix des hyperparametres
    learning_rate = 0.005
    num_epochs = 2
    dim_couche_1 = 512
    dim_couche_2 = 256
    dim_couche_3 = 16
    latent_dim = 2
    input_dim = 784
    kl_poids = 0.012

    # Preparation dún auto-encodeur
    test_images, test_labels = projetae.Donnees.test_donnees_mnist()

    # Conversion des etiquettes
    test_labels = np.array(test_labels)
    labels = []
    for i in test_labels:
        labels.append(i)
    test_labels = labels

    reconstructed, autoencodeur = preparer_autoencodeur(test_images)

    # Visualisation des 10 images reconstruites par l'autoencodeur
    comparer_resultats()

    # Visualisation de l'espace latent
    z_mu, z_log_var, encoded = autoencodeur.encoder(test_images)
    latent_plot(encoded, test_labels)

    # Decodage d'un morceau de l'espace latent
    decoder_grille(autoencodeur)




