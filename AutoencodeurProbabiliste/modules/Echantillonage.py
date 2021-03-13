from tensorflow.keras import layers
import tensorflow as tf

class Echantillonage(layers.Layer):
  """ Classe personnalisée Echantillonage, étendant la classe Layer. Une instance de cette classe est responsable
  de l'échantillonnage a partir d'une distribution normale cetrée réduite. Elle prend en entrée deux tenseurs z_mu et
  z_log_var et renvoie un échantillon d'un vecteur de variables aléatoires suivant une loi gausienne de moyenne z_mu et
  de variance exp(z_log_var)"""

  def call(self, inputs):
    z_mean, z_log_var = inputs
    batch = tf.shape(z_mean)[0]
    dim = tf.shape(z_mean)[1]
    # Noeaud aleatoire: echantillonage a partir de la distribution normale centrée réduite
    epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
    # Réparametrization: z = moyenne + epsilon*exp(0.5*log_variance)
    return z_mean + tf.exp(z_log_var/2) * epsilon
