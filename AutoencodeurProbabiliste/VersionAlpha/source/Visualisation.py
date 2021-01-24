import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import tensorflow as tf
import numpy as np
from scipy.stats import norm
import tensorflow.keras.datasets.mnist as mnist
from source import Autoencodeur as AE


def preparer_donnees_test():
    """"Methode qui charge et prepare des images de test MNIST"""

    #Chargement de l'ensemble de données MNIST composé de 60000 paires
    # image/étiquette d'entraînement et 10000 paires image/étiquette de test:
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

    # Remodelation des donnees: 10000 vecteurs de dimension 784 au lieu des 10000 matrices de dimensions 28x28
    test_images = test_images.reshape(10000, 784)

    # Conversion de chaque pixel en un nombre flottant 32 bits
    test_images = test_images.astype('float32')

    #Normalisation des valeurs de chaque pixel pour les rendre entre 0 et 1
    test_images = test_images/255

    #Création d'un jeu de données dont les éléments sont des tranches des tenseurs donnés -
    # les tenseurs donnés sont découpés le long de leur première dimension
    # (division des lots en tenseurs individuels pour itérer sur l'ensemble de données):
    test_dataset = tf.data.Dataset.from_tensor_slices(test_images)

    # Mélanger les images (tenseurs)
    test_dataset = test_dataset.shuffle(buffer_size=1024).batch(1)

    return test_images, test_labels



def preparer_autoencodeur(test_images):
    """ Méthode chargée de l'instantiation d'un autoencodeur, de son entrainement et de
    preparation des images reconstruites """

    # Chargement et entrainement d'un autoencodeur
    autoencodeur = AE.AutoEncodeur(latent_dim, dim_couche_1, dim_couche_2, dim_couche_3, kl_poids)
    AE.train(autoencodeur, learning_rate, num_epochs)

    # Preparation des images pour une visualisation:
    reconstructed = autoencodeur(test_images)

    return reconstructed, autoencodeur



def comparer_resultats():
    """ Visualisation de la performance d'une instance de classe AutoEncodeur. Cette méthode affiche 10 images originales
    et leurs recostructions """

    plt.figure(figsize=(20, 4))
    j = 15
    for i in range(10):
        inputaxis = plt.subplot(2, 10, i + 1)
        plt.imshow(test_images[j].reshape(28, 28))
        plt.gray()
        inputaxis.get_xaxis().set_visible(False)
        inputaxis.get_yaxis().set_visible(False)

        outputaxis = plt.subplot(2, 10, i + 11)
        to_plot = tf.reshape(reconstructed[j],[28, 28] )
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
    digit_size = 28
    figure = np.zeros((28 * num_digits, 28 * num_digits))

    # Construction d'une grille aléatoire de l'espace continu
    grid_x = norm.ppf(np.linspace(0.05, 0.95, num_digits))
    grid_y = norm.ppf(np.linspace(0.05, 0.95, num_digits))

    # Pour chacun des 20 points aléatoires de la grille,
    for i, yi in enumerate(grid_x):
        for j, xi in enumerate(grid_y):
            # Prendre un point aléatoire et le convertir en un tenseur
            z_echantillon = tf.convert_to_tensor([xi, yi], dtype=tf.float32)
            z_echantillon = tf.reshape(z_echantillon, [1,2])
            # Decoder ce tenseur en un tenseur de sortie de dimension 784
            x_decode = autoencodeur.decoder(z_echantillon)

            # Remodelation en une image de 28 par 28 pixels
            chiffre = tf.reshape(x_decode,[28, 28])

            figure[i * 28: (i + 1) * 28,
                           j * 28: (j + 1) * 28] = chiffre

    plt.figure(figsize=(15, 15))
    plt.imshow(figure, cmap='Greys')
    plt.show()


if __name__=='__main__':

    # Choix des hyperparametres
    learning_rate = 0.0035
    num_epochs = 3
    dim_couche_1 = 256
    dim_couche_2 = 64
    dim_couche_3 = 16
    latent_dim = 2
    kl_poids = 0.0008

    # Preparation dún auto-encodeur
    test_images, test_labels = preparer_donnees_test()
    reconstructed, autoencodeur = preparer_autoencodeur(test_images)

    # Visualisation des 10 images reconstruites par l'autoencodeur
    comparer_resultats()

    # Visualisation de l'espace latent
    z_mu, z_log_var, encoded = autoencodeur.encoder(test_images)
    latent_plot(z_mu, z_log_var, test_labels)

    # Decodage d'un morceau de l'espace latent
    decoder_grille(autoencodeur)


