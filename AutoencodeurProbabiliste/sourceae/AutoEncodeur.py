import tensorflow as tf
import AutoencodeurProbabiliste.sourceae as projetae
import math

class AutoEncodeur(tf.keras.Model):
  """ Classe d'autoencoder, étendant la classe Model, qui relie l'encodeur et le décodeur ensemble en un modèle.

  :param input_dim: la taille de la couche d'entrée
  :type input_dim: int
  :param latent_dim: la dimension de l'espace latent
  :type latent_dim: int
  :param kl_poids:
  :type kl_poids: float
  :param encoder: l'encodeur
  :type encoder: class 'AutoencodeurProbabiliste.modules.EncodeurQ.EncodeurQ'
  :param decoder: le décodeur
  :type decoder: class 'AutoencodeurProbabiliste.modules.DecodeurP.DecodeurP'

  """
  def __init__(self, input_dim, latent_dim, dim_couche_1,dim_couche_2, dim_couche_3, kl_poids):
    """Constructeur de la classe AutoEncodeur"""
    super(AutoEncodeur, self).__init__()
    self.input_dim = input_dim
    self.latent_dim = latent_dim
    self.kl_poids = kl_poids
    self.encoder = projetae.EncodeurQ(latent_dim, dim_couche_1, dim_couche_2, dim_couche_3)
    self.decoder = projetae.DecodeurP(dim_couche_3, dim_couche_2, dim_couche_1, self.input_dim)

  def call(self, inputs):
    """La méthode de la superclasse Model, définissant ce qui se passe à chaque étape de l'entraînement au tenseur entrant

    :param inputs: un lot des images d'entrée - tenseur 2D de taille (batch_size, 784) entrant dans le modele
    :type inputs: class 'tensorflow.python.framework.ops.EagerTensor'
    :return: un lot des images réconstruites - tenseur 2D de taille (batch_size, 784)
    :rtype: class 'tensorflow.python.framework.ops.EagerTensor'
    """

    z_mu, z_log_var, z = self.encoder(inputs)
    reconstructed = self.decoder(z)
    # Calcule la perte de divergence KL
    kl_loss = - 0.5 * tf.reduce_mean(tf.reduce_sum(z_log_var - tf.square(z_mu) - tf.exp(z_log_var) + 1))
    kl_loss = kl_loss * self.kl_poids
    self.add_loss(kl_loss)

    return reconstructed


def train(model,  learning_rate, num_epochs):
  """ Méthode d'entraînement personnalisée.

  Elle prend en entrée un modele un (autoencodeur), le taux d'aprentissage et le nobre d'epoques.
  Apres la selection d'un optimiseur at d'une perte de reconstruction, la méthode train() parcourt l’ensemble des
  données d’entraînement a chaque époque d’apprentissage, lot par lot, en calculant les gradients et en optimisant
  les paramètres de l’auto-encodeur ("trainable_weights"). À chaque époque, le processus suivant est répété
  (pour chaque lot) : train() passe chaque lot par l’autoencoder, calcule la perte de reconstruction et lui ajoute
  la perte KL, calculée dans la méthode call() de l’auto-encodeur. Tout au long de ce processus, train() utilise
  tf.GradientTape()pour enregistrer les gradients calculés au cours de ce processus. Cela permet de les utiliser
  plus tard dans l’époque pour l’optimisation du poids.

  :param model: l'autoencodeur a entrainer
  :type model: class 'AutoencodeurProbabiliste.modules.Autoencodeur.AutoEncodeur'
  :param learning_rate: le taux d'aprentissage
  :type learning_rate: float
  :param num_epochs: le nombre d'epoques (itérations) a effectuer pendant l'entrainement
  :return: la perte moyenne finale
  :rtype: class 'numpy.float32'

  """

  # Acquisition des donnees:
  train_dataset = projetae.Donnees.train_donnees_mnist()

  # Choix d'un optimiseur
  optimizer = tf.keras.optimizers.Adam(learning_rate)

  # La perte moyenne pour afficher plus tard, pendant l'entainement
  loss_metric = tf.keras.metrics.Mean()

  # Itération sur l'ensemble de données pour chaque époque:
  for epoch in range(num_epochs):

    # Iterations pour chaque lot (batch) des donnees:
    # Quoi faire a chaque pas et pour chaque batch (lot)
    for step, training_batch in enumerate(train_dataset):

      #GradientTape enregistre les gradients calculés par l'optimiseur
      with tf.GradientTape() as tape:

       # Images reconstruites par l'autoencodeur
        reconstructed = model(training_batch)

        # Calcule la perte de reconstruction en utilisant l'entropie croisée
        perte = tf.reduce_mean(tf.reduce_sum(tf.keras.losses.binary_crossentropy(training_batch, reconstructed))) #perte_reconstruction(training_batch, reconstructed)

       # Ajoute la perte de divergence KL calculée dans la méthode call() de l'autoencodeur
       # La perte KL est calculée dans la méthode call() car c'est là qu'elle a un accès direct à z_mu et z_log_var
        perte += sum(model.losses)

      # Gradients stockés appliqués a l'optimiseur
      grads = tape.gradient(perte, model.trainable_weights)
      optimizer.apply_gradients(zip(grads, model.trainable_weights))

      # Calcule la perte actuelle:
      loss_metric(perte)


      # Affiche la perte moyenne actuelle pour chaque cent pas
      if step % 100 == 0:
        texte = ' Époque %d / %d. Pas %s / 1000. Perte %s pourcent' % (epoch+1, num_epochs, step, loss_metric(perte))
        #print(texte)

    texte = ' ÉPOQUE %d / %d. PERTE MOYENNE: %s ' % (
    epoch + 1, num_epochs, loss_metric(perte).numpy())
    print(texte)

    if math.isnan(loss_metric(perte).numpy()):
      print("ÉCHEC D'ENTRAÎNEMENT")
      return

  return loss_metric(perte).numpy()


