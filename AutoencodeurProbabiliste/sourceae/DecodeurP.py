from tensorflow.keras import layers

class DecodeurP(layers.Layer):
  """Classe du décodeur P, étendant la classe Layer.

  :param couche1: la taille de la derniere couche avant la couche de sortie
  :type couche1: int
  :param couche2: la taille de la deuxieme couche
  :type couche2: int
  :param couche3: la taille de la premiere couche apres la couche latente
  :type couche3: int
  :param output_dim: la taille de la couche de sortie
  :type output_dim: int
  :param c1: couche cachée de taille couche1
  :type c1: class 'tensorflow.python.keras.layers.core.Dense'
  :param c2: couche cachée de taille couche2
  :type c2: class 'tensorflow.python.keras.layers.core.Dense'
  :param c3: couche cachée de taille couche3
  :type c3: class 'tensorflow.python.keras.layers.core.Dense'
  :param sortie: couche de sortie
  :type sortie: class 'tensorflow.python.keras.layers.core.Dense'
  """

  def __init__(self, couche1, couche2, couche3, output_dim):

    super(DecodeurP, self).__init__()
    # 3 couches cachées de dimensions couche1, couche2 et couche3 respectivement
    self.c1 = layers.Dense(couche1, activation='relu')
    self.c2 = layers.Dense(couche2, activation='relu')
    self.c3 = layers.Dense(couche3, activation='relu')
    self.output_dim = output_dim

    # Couche de sortie de 784 dimensions avec une activation sigmoide pour que ces
    # valeurs sont entre 0 et 1 comme les pixels de l'image d'entrée
    self.sortie = layers.Dense(self.output_dim, activation='sigmoid')

  def call(self, inputs):
    """La méthode de la superclasse Layer, définissant ce qui se passe à chaque étape de l'entraînement au tenseur entrant.

    Elle prend en entrée un tenseur latent z, passe lui par les 3 couches cachées et renvoie sa
    reconstruction en une image x encore sous la forme d'un tenseur.

    :param inputs:  tenseur 2D de taille (batch_size, dim_latente) représentant la variable latente échantilonnée z
    :type inputs: class 'tensorflow.python.framework.ops.EagerTensor'
    :return: tenseur 2D de taille (batch_size,784) représentant les images décodées
    :rtype: class 'tensorflow.python.framework.ops.EagerTensor'
    """
    x = self.c1(inputs)
    x = self.c2(x)
    x = self.c3(x)
    sortie = self.sortie(x)
    return sortie