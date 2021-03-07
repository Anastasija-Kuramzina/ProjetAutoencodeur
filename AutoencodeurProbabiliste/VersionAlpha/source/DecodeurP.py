import tensorflow as tf
from tensorflow.keras import layers

class DecodeurP(layers.Layer):
  """Classe du décodeur P, étendant la classe Layer. Une instance de cette classe prend en entrée un tenseur latent z
  et renvoie sa reconstruction x """

  def __init__(self, couche1, couche2, couche3, output_dim):

    super(DecodeurP, self).__init__()
    # 3 couches cachées de dimensions couche1, couche2 et couche3 respectivement
    self.c1 = layers.Dense(couche1, activation='relu')
    self.c2 = layers.Dense(couche2, activation='relu')
    self.c3 = layers.Dense(couche3, activation='relu')

    # Couche de sortie de 784 dimensions avec une activation sigmoide pour que ces
    # valeurs sont entre 0 et 1 comme les pixels de l'image d'entrée
    self.sortie = layers.Dense(output_dim, activation='sigmoid')

  def call(self, inputs):
    x = self.c1(inputs)
    x = self.c2(x)
    x = self.c3(x)
    sortie = self.sortie(x)
    return sortie