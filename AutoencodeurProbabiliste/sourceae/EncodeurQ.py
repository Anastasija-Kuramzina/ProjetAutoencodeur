from tensorflow.keras import layers
import AutoencodeurProbabiliste.sourceae as projetae

class EncodeurQ(layers.Layer):
  """ Classe de l'encodeur Q, étendant la classe Layer.

  :param couche1: la taille de la premiere couche apres la couche d'entrée
  :type couche1: int
  :param couche2: la taille de la deuxieme couche
  :type couche2: int
  :param couche3: la taille de la derniere couche avant la couche latente
  :type couche3: int
  :param latent_dim: la dimension latente
  :type latent_dim: int
  :param c1: couche cachée de taille couche1
  :type c1: class 'tensorflow.python.keras.layers.core.Dense'
  :param c2: couche cachée de taille couche2
  :type c2: class 'tensorflow.python.keras.layers.core.Dense'
  :param c3: couche cachée de taille couche3
  :type c3: class 'tensorflow.python.keras.layers.core.Dense'
  :param z_mu: couche de sortie de l'encodeur  contenant les moyennes
  :type z_mu: class 'tensorflow.python.keras.layers.core.Dense'
  :param z_log_var: couche de sortie de l'encodeur contenant les log variances
  :type z_log_var:  class 'tensorflow.python.keras.layers.core.Dense'
  :param echantillonage: couche latente contenant la variable latente z
  :type echantillonage: class'AutoencodeurProbabiliste.modules.Echantillonage.Echantillonage'
  """

  def __init__(self, latent_dim, couche1, couche2, couche3):
    super(EncodeurQ, self).__init__()
    # 3 couches cachées de dimensions couche1, couche2 et couche3 respectivement
    self.c1 = layers.Dense(couche1, activation='relu')
    self.c2 = layers.Dense(couche2, activation='relu')
    self.c3 = layers.Dense(couche3, activation=None)

    # Deux tenseurs de dimension latent_dim provenant de la couche 3: le tenseur de moyennes et le tenseur de log variances.
    self.z_mu = layers.Dense(latent_dim, activation=None)
    self.z_log_var = layers.Dense(latent_dim, activation=None)

    # Tenseur échantilloné z
    self.echantillonage = projetae.Echantillonage()


  def call(self, inputs):
    """La méthode de la superclasse Layer, définissant ce qui se passe à chaque étape de l'entraînement au tenseur entrant.

    Elle prends en entrée un tenseur, passe lui par les trois couches Dense cachées et le
    transforme en deux tenseurs z_mu et z_log_var. Ces deux tenseurs passent ensuite par une couche d'echantillonage,
    et finalement un tenseur latent z est renvoyé.

    :param inputs:  tenseur 2D de taille (batch_size, 784) représentant l'image d'entrée sous la forme vctorielle
    :type inputs: class 'tensorflow.python.framework.ops.EagerTensor'
    :return: liste de 3 tenseurs 2D de taille (batch_size,dim_latente) représentant le tenseur des moyennes, tenseur des log variances rt la variable latente z
    :rtype: list
    """
    x = self.c1(inputs)
    x = self.c2(x)
    x = self.c3(x)

    z_mu = self.z_mu(x)
    z_log_var = self.z_log_var(x)
    z = self.echantillonage((z_mu, z_log_var))

    return z_mu, z_log_var, z