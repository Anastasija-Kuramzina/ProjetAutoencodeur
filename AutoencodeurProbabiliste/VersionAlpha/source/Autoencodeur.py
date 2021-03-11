from tensorflow.keras import layers
import tensorflow as tf
import tensorflow.keras.datasets.mnist as mnist

import AutoencodeurProbabiliste.VersionAlpha.source.EncodeurQ as Enc
import AutoencodeurProbabiliste.VersionAlpha.source.DecodeurP as Dec
import AutoencodeurProbabiliste.VersionAlpha.source.Donnees as Donnees



class AutoEncodeur(tf.keras.Model):
  """ Classe d'autoencoder, étendant la classe Model, qui relie l'encodeur et le décodeur ensemble en un modèle. """
  def __init__(self, input_dim, latent_dim, dim_couche_1,dim_couche_2, dim_couche_3, kl_poids):
    super(AutoEncodeur, self).__init__()
    # Poids de la perte KL dans

    self.input_dim = input_dim
    self.latent_dim = latent_dim
    self.kl_poids = kl_poids
    self.encoder = Enc.EncodeurQ(latent_dim, dim_couche_1, dim_couche_2, dim_couche_3)

    self.decoder = Dec.DecodeurP(dim_couche_3, dim_couche_2, dim_couche_1, self.input_dim)

  def call(self, inputs):
    # self._set_inputs(inputs)
    z_mu, z_log_var, z = self.encoder(inputs)
    reconstructed = self.decoder(z)

    # Calcule la perte de divergence KL
    kl_loss = - 0.5 * tf.reduce_mean(z_log_var - tf.square(z_mu) - tf.exp(z_log_var) + 1)
    kl_loss = kl_loss * self.kl_poids
    self.add_loss(kl_loss)
    return reconstructed

  def display(self):
    """Methode affichant les informations sur la structure de l'autoencodeur"""
    print("Structure de l'autoencodeur : ")




def train(model,  learning_rate, num_epochs):
  """ Méthode d'entraînement personnalisée permettant des ajustements futurs faciles. Elle prend en entrée un modele
  un (autoencodeur), le taux d'aprentissage et e mobre d'epoques. Apres la selection d'un optimiseur at d'une perte
  de reconstruction, la méthode train() A chaque époque d’apprentissage, elle parcourts l’ensemble desdonnées d’entraînement,
  lot par lot, en calculant les gradients et enoptimisant les paramètres de l’auto-encodeur ("trainable_weights").
  À chaque époque, le processus suivant est répété pour chaque étape d’apprentissage (et chaque lot) : train() passe le lot par
  l’au-toencoder, calcule la perte de reconstruction et ajoute la perte KL, calculée dans la méthode d’appel de l’auto-encoder.
  Tout au long de ce processus, train() utilise tf.GradientTape()pour enregistrer les gradients calculés au cours de ce processus.
  Cela permet de les utiliser plus tard dans l’époque pour l’optimisation du poids. """


  # Acquerir les donnees:
  train_dataset = Donnees.Donnees.train_donnees_mnist()

  # Choix d'un optimiseur
  optimizer = tf.keras.optimizers.Adam(learning_rate)

  # La perte Binary Cross Entropy (entropie croisée)
 # perte_reconstruction= tf.keras.losses.binary_crossentropy()

  # La perte moyenne pour afficher plus tard, pendant l'entainement
  loss_metric = tf.keras.metrics.Mean()

  # Itération sur l'ensemble de données pour chaque époque:
  for epoch in range(num_epochs):
    print('Époque %d / %d commence.' % (epoch+1, num_epochs))

    # Iterations pour chaque lot (batch) des donnees:
    # Quoi faire a chaque pas et por chaque batch
    for step, training_batch in enumerate(train_dataset):

      #GradientTape enregistre les gradients calculés par l'optimiseur
      with tf.GradientTape() as tape:

       # Images reconstruites par l'autoencodeur
        reconstructed = model(training_batch)

        # Calcule la perte de reconstruction en utilisant l'entropie croisée
        perte = tf.keras.losses.binary_crossentropy(training_batch, reconstructed) #perte_reconstruction(training_batch, reconstructed)

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
        print('Pas %s: perte moyenne = %s' % (step, loss_metric.result()))



