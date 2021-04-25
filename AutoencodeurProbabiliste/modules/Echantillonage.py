from tensorflow.keras import layers
import tensorflow as tf

class Echantillonage(layers.Layer):
  """ Classe personnalisée Echantillonage, étendant la classe Layer. Une instance de cette classe est responsable
  de l'échantillonnage a partir d'une distribution normale cetrée réduite. Elle prend en entrée deux tenseurs z_mu et
  z_log_var et renvoie un échantillon d'un vecteur de variables aléatoires suivant une loi gausienne de moyenne z_mu et
  de variance exp(z_log_var)"""

  def call(self, inputs):
    """La méthode de la superclasse Layer, définissant ce qui se passe à chaque étape de l'entraînement au tenseur entrant

    :param inputs: liste contenant deux tenseurs 2D de taille (batch_size, latent_dim) - le vecteur des moyennes et le vecteur des log variances
    :type inputs: list
    :return: tenseurs 2D de taille (batch_size, latent_dim) représentant la variable latente z
    :rtype: class 'tensorflow.python.framework.ops.EagerTensor'
    """
    z_mean, z_log_var = inputs
    batch = tf.shape(z_mean)[0]
    dim = tf.shape(z_mean)[1]
    # Noeaud aleatoire: echantillonage a partir de la distribution normale centrée réduite
    epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
    # Réparametrization: z = moyenne + epsilon*exp(0.5*log_variance)
    return z_mean + tf.exp(z_log_var/2) * epsilon
