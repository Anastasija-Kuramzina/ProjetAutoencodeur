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

    #Etiquettes
    #etiquettes = {0:'avion', 1:'voiture', 2:'oiseau', 3:'chat', 4:'cerf',5:'chien',6:'grenouille',7:'cheval', 8:'bateau',9:'camion'}
    etiquettes = {0: 'T-shirt', 1: 'Pantalons', 2: 'Pull', 3: 'Robe', 4: 'Manteau', 5: 'Sandale', 6: 'Chemise', 7: 'Sneaker',
                  8: 'Sac', 9: 'Botte'}

    #Légende
    patchlist = []
    for i in range(10):
        patchlist.append(mpatches.Patch(color=colors([i])[0], label=etiquettes[i]))
    plt.legend(handles=patchlist)
    plt.show()



def decoder_grille(autoencodeur):
    """Décodage et affichage d'un morceau de l'espace latent"""
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
    num_epochs = 60
    dim_couche_1 = 512
    dim_couche_2 = 256
    dim_couche_3 = 16
    latent_dim = 2
    input_dim = 784
    kl_poids = 0.012

    # Preparation dún auto-encodeur
    test_images, test_labels = Donnees.Donnees.test_donnees_mnist()

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
    latent_plot(z_mu, z_log_var, test_labels)

    # Decodage d'un morceau de l'espace latent
    decoder_grille(autoencodeur)




