from tensorflow.keras import layers
import AutoencodeurProbabiliste.modules as modules

class EncodeurQ(layers.Layer):
  """ Classe de l'encodeur Q, étendant la classe Layer. Une instance de cette classe prend en entrée un tenseur x (image MNIST), passe la par
   3 couches Dense cachées et le transforme en deux tenseurs z_mu et z_log_var. Ces deux tenseurs passent ensuite par une couche
   d'echantillonage, et finalement un tenseur latent z est renvoyé"""
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
    self.echantillonage = modules.Echantillonage()

  def call(self, inputs):

    x = self.c1(inputs)
    x = self.c2(x)
    x = self.c3(x)

    z_mu = self.z_mu(x)
    z_log_var = self.z_log_var(x)
    z = self.echantillonage((z_mu, z_log_var))

    return z_mu, z_log_var, z