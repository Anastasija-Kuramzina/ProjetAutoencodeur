import AutoencodeurProbabiliste.modules as modules
import matplotlib.pyplot as plt
import tensorflow as tf


class Affichage():
    """Classe contenant des methodes utilisees par l'interface graphique"""

    @classmethod
    def afficher_image_originale(cls,images,n, axis):
        """Methode prenant en entree un tenseur d'images et une indice n, qui affiche et renvoye l'image d'indice n"""
        plt.gray()
        image = images[n]
        axis.imshow(image.reshape(28, 28))
        axis.get_xaxis().set_visible(False)
        axis.get_yaxis().set_visible(False)
        return image

    @classmethod
    def obtenir_vecteur(cls,images,n, autoencodeur):
        """Methode prenant en entree un tenseur d'images et une indice n, qui encode l'image d'indice n a l'aide de l'encodeur"""
        image = images[n]
        image = tf.convert_to_tensor(image, dtype=tf.float32)
        image = tf.reshape(image, [1, 784])
        mu, log_var, vecteur = autoencodeur.encoder(image)
        return vecteur

    @classmethod
    def vecteur_prep(cls,vect):
        """Methode prenant en entree un vecteur et lui rendant dans le format accepte par l'autoencodeur."""
        vect = tf.convert_to_tensor(vect, dtype=tf.float32)
        vect = tf.reshape(vect, [1, 2])
        return vect

    @classmethod
    def afficher_image_reconstruite(cls,vecteur, autoencodeur, axis):
        """Methode prenant en entree un vecteur et lui decodant a l'aide du decodeur, puis affichant et renvoyant le resultat."""
        plt.gray()
        vecteur = modules.Affichage.vecteur_prep(vecteur)
        image = autoencodeur.decoder(vecteur)
        image = tf.reshape(image,[28, 28])
        axis.imshow(image)
        return image

    @classmethod
    def melange_images(cls,vect1, vect2, autoencodeur):
        moyenne = (vect1+vect2)/2
        return moyenne


if __name__=='__main__':

    print("hello")
