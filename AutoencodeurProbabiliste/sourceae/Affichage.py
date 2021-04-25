import AutoencodeurProbabiliste.sourceae as modules
import matplotlib.pyplot as plt
import tensorflow as tf


class Affichage():
    """Classe contenant des methodes utilisées par l'interface graphique pour le gestion des images"""

    @classmethod
    def afficher_image_originale(cls,images,n, axis):
        """Methode prenant en entree un tenseur d'images et une indice n, qui renvoye l'image d'indice n et lui
        affiche sur l'axe donné en entrée.

        :param images: tenseur d'images
        :type images: class 'tensorflow.python.framework.ops.EagerTensor'
        :param n: indice de l'image a afficher
        :type n: int
        :param axis: l'axe ou il faut afficher l'image
        :type axis: class 'matplotlib.axes._subplots.AxesSubplot'
        :return: l'image affichée sous la forme d'un tenseur
        :rtype: class 'tensorflow.python.framework.ops.EagerTensor'
        """
        plt.gray()
        image = images[n]
        axis.imshow(image.reshape(28, 28))
        axis.get_xaxis().set_visible(False)
        axis.get_yaxis().set_visible(False)
        return image

    @classmethod
    def obtenir_vecteur(cls,images,n, autoencodeur):
        """Methode prenant en entree un tenseur d'images et une indice n, qui encode l'image d'indice n a l'aide de l'encodeur.

        :param images: tenseur d'images
        :type images: class 'tensorflow.python.framework.ops.EagerTensor'
        :param n: indice de l'image a afficher
        :type n: int
        :param autoencodeur: autoencodeur dont l'encodeur sera utilisé pour l'encodage de l'image
        :type autoencodeur: class 'AutoencodeurProbabiliste.modules.Autoencodeur.AutoEncodeur'
        :return: la représentation latente vectorielle de l'image d'entrée
        :rtype: class 'tensorflow.python.framework.ops.EagerTensor'"""
        image = images[n]
        image = tf.convert_to_tensor(image, dtype=tf.float32)
        image = tf.reshape(image, [1, 784])
        mu, log_var, vecteur = autoencodeur.encoder(image)
        return vecteur

    @classmethod
    def vecteur_prep(cls,vect, latent_dim):
        """Methode prenant en entree un vecteur et lui rendant dans le format accepte par l'autoencodeur.

        :param vect: vecteur latent sous la forme d'un numpy array
        :type vect: class 'numpy.ndarray'
        :param latent_dim: la dimension latente de l'autoencodeur
        :type latent_dim: int
        :return: vecteur latent sous la forme d'un tenseur
        :rtype: class 'tensorflow.python.framework.ops.EagerTensor'
        """
        vect = tf.convert_to_tensor(vect, dtype=tf.float32)
        vect = tf.reshape(vect, [1, latent_dim])
        return vect

    @classmethod
    def afficher_image_reconstruite(cls,vecteur, autoencodeur, axis):
        """Methode prenant en entree un vecteur et lui decodant a l'aide du decodeur, puis affichant et renvoyant le resultat.

        :param vecteur: vecteur a décoder
        :type vecteur: class 'numpy.ndarray'
        :param autoencoder: autoencodeur dont le décodeur sera utilisé pour le décodage
        :type autoencodeur: class 'AutoencodeurProbabiliste.modules.Autoencodeur.AutoEncodeur'
        :param axis: l'axe ou il faut afficher l'image
        :type axis: class 'matplotlib.axes._subplots.AxesSubplot'
        :return image: image a afficher sous la forme d'un tenseur
        :rtype: class 'tensorflow.python.framework.ops.EagerTensor'
        """
        plt.gray()
        vecteur = projetae.Affichage.vecteur_prep(vecteur, autoencodeur.latent_dim)
        image = autoencodeur.decoder(vecteur)
        image = tf.reshape(image,[28, 28])
        axis.imshow(image)
        return image


    @classmethod
    def melange_images(cls,vect1, vect2):
        """Méthode calculant la moyenne des deux vecteurs (point qui se situe exactement entre les deux)

        :param vect1: premier vecteur
        :type vect1: class 'numpy.ndarray'
        :param vect2: deuxieme vecteur
        :type vect2: class 'numpy.ndarray'
        :return: la moyenne des deux vecteurs
        :rtype: class 'numpy.ndarray'

        """
        moyenne = (vect1+vect2)/2
        return moyenne


