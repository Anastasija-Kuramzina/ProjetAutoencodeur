import AutoencodeurProbabiliste.modules as modules
import matplotlib.pyplot as plt
import tensorflow as tf

class Affichage():
    """Classe contenant des methodes utilisees par l'interface graphique"""

    @classmethod
    def preparer_autoencodeur(cls,input_dim, latent_dim, dim_couche_1, dim_couche_2, dim_couche_3, kl_poids, learning_rate, num_epochs):
        """ Méthode chargée de l'instantiation d'un autoencodeur et de son entrainement """

        autoencodeur = modules.AutoEncodeur(input_dim, latent_dim, dim_couche_1, dim_couche_2, dim_couche_3, kl_poids)
        modules.train(autoencodeur, learning_rate, num_epochs)
        return autoencodeur


    @classmethod
    def afficher_informations(cls,autoencodeur):
        return

    @classmethod
    def afficher_image_originale(cls,images,n):
        """Methode prenant en entree un tenseur d'images et une indice n, qui affiche et renvoye l'image d'indice n"""
        plt.gray()
        axis = plt.subplot(1, 1, 1)
        image = images[n]
        plt.imshow(image.reshape(28, 28))
        axis.get_xaxis().set_visible(False)
        axis.get_yaxis().set_visible(False)
        text = "Image "+str(n)
        print(text)
        plt.text(0, 0, text)
        plt.show()
        return image

    @classmethod
    def obtenir_vecteur(cls,images,n):
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
    def afficher_image_reconstruite(cls,vecteur):
        """Methode prenant en entree un vecteur et lui decodant a l'aide du decodeur, puis affichant et renvoyant le resultat."""
        plt.gray()
        vecteur = modules.Affichage.vecteur_prep(vecteur)
        image = autoencodeur.decoder(vecteur)
        image = tf.reshape(image,[28, 28])
        axis = plt.subplot(1, 1, 1)
        titre = str(vecteur)
        plt.text(0, 0, titre)
        axis.get_xaxis().set_visible(False)
        axis.get_yaxis().set_visible(False)
        plt.imshow(image)
        plt.show()
        return image

    @classmethod
    def melange_images(cls,vect1, vect2):
        moyenne = (vect1+vect2)/2
        image = modules.Affichage.afficher_image_reconstruite(moyenne)
        return image


if __name__=='__main__':

    # Choix des hyperparametres
    learning_rate = 0.005
    num_epochs = 20
    dim_couche_1 = 512
    dim_couche_2 = 128
    dim_couche_3 = 16
    latent_dim = 2
    input_dim = 784
    kl_poids = 0.0012

    # Preparation d'un auto-encodeur et des donnees
    test_images, test_labels = modules.Donnees.test_donnees_mnist()
    autoencodeur = modules.Affichage.preparer_autoencodeur(input_dim, latent_dim, dim_couche_1, dim_couche_2, dim_couche_3, kl_poids,  learning_rate, num_epochs)

    for i in range(10):
        x = i/8 * 0.5 * (-1)**i
        y =  i/7 * 1.5 * (-1)**i
        vect = [x,y]
        modules.Affichage.afficher_image_reconstruite(vect)


