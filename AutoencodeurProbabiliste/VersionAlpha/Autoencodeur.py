from tensorflow.keras import layers
import tensorflow as tf
import tensorflow.keras.datasets.mnist as mnist


#Chargement de l'ensemble de données MNIST composé de 60000 paires
# image/étiquette d'entraînement et 10000 paires image/étiquette de test:
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Remodelation des donnees: 60000 vecteurs de dimension 784 au lieu des 60000 matrices de dimensions 28x28
train_images = train_images.reshape(60000, 784)

# Conversion de chaque pixel en un nombre flottant 32 bits
train_images = train_images.astype('float32')

#Normalisation des valeurs de chaque pixel pour les rendre entre 0 et 1
train_images = train_images/255

#Création d'un jeu de données dont les éléments sont des tranches des tenseurs donnés -
# les tenseurs donnés sont découpés le long de leur première dimension
# (division des lots en tenseurs individuels pour itérer sur l'ensemble de données):
train_dataset = tf.data.Dataset.from_tensor_slices(train_images)

# Mélanger les images (tenseurs)
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(64)



# Classe personnalisée, étendant la classe Layer, responsable de l'échantillonnage
class Echantillonage(layers.Layer):

  def call(self, inputs):
    z_mean, z_log_var = inputs
    batch = tf.shape(z_mean)[0]
    dim = tf.shape(z_mean)[1]
    # Noeaud aleatoire: echantillonage a partir de la distribution normale centrée réduite
    epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
    # Réparametrization: z = moyenne + epsilon*exp(0.5*log_variance)
    return z_mean + tf.exp(z_log_var/2) * epsilon


# Classe de l'encodeur Q, étendant la classe Layer
class EncodeurQ(layers.Layer):
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
    self.echantillonage = Echantillonage()

  def call(self, inputs):
    x = self.c1(inputs)
    x = self.c2(x)
    x = self.c3(x)
    z_mu = self.z_mu(x)
    z_log_var = self.z_log_var(x)
    z = self.echantillonage((z_mu, z_log_var))
    return z_mu, z_log_var, z


# Classe du décodeur P, étendant la classe Layer
class DecodeurP(layers.Layer):

  def __init__(self, couche1, couche2, couche3):

    super(DecodeurP, self).__init__()
    # 3 couches cachées de dimensions couche1, couche2 et couche3 respectivement
    self.c1 = layers.Dense(couche1, activation='relu')
    self.c2 = layers.Dense(couche2, activation='relu')
    self.c3 = layers.Dense(couche3, activation='relu')

    # Couche de sortie de 784 dimensions avec une activation sigmoide pour que ces
    # valeurs sont entre 0 et 1 comme les pixels de l'image d'entrée
    self.sortie = layers.Dense(784, activation='sigmoid')

  def call(self, inputs):
    x = self.c1(inputs)
    x = self.c2(x)
    x = self.c3(x)
    sortie = self.sortie(x)
    return sortie



# Classe d'autoencoder, étendant la classe Model, qui relie l'encodeur et le décodeur ensemble en un modèle:
class AutoEncodeur(tf.keras.Model):
  def __init__(self, latent_dim, dim_couche_1,dim_couche_2, dim_couche_3, kl_poids ):
    super(AutoEncodeur, self).__init__()
    # Poids de la perte KL dans
    self.input_dim = 784
    self.kl_poids = kl_poids
    self.encoder = EncodeurQ(latent_dim, dim_couche_1, dim_couche_2, dim_couche_3)

    self.decoder = DecodeurP(dim_couche_3, dim_couche_2, dim_couche_1)

  def call(self, inputs):
    # self._set_inputs(inputs)
    z_mu, z_log_var, z = self.encoder(inputs)
    reconstructed = self.decoder(z)

    # Calcule la perte de divergence KL
    kl_loss = - 0.5 * tf.reduce_mean(
        z_log_var - tf.square(z_mu) - tf.exp(z_log_var) + 1)
    kl_loss = kl_loss * self.kl_poids
    self.add_loss(kl_loss)
    return reconstructed

# Méthode d'entraînement personnalisée permettant des ajustements futurs faciles
def train(model,  learning_rate, num_epochs):
  # Choix d'un optimiseur
  optimizer = tf.keras.optimizers.Adam(learning_rate)

  # La perte Binary Cross Entropy (entropie croisée)
  perte_reconstruction= tf.keras.losses.BinaryCrossentropy()

  # La perte moyenne pour afficher plus tard, pendant l'entainement
  loss_metric = tf.keras.metrics.Mean()

  # Itération sur l'ensemble de données pour chaque époque:
  for epoch in range(num_epochs):
    print('Époque %d / %d commence.' % (epoch+1, num_epochs))

    # Iterations pour chaque lot (batch) des donnees:
    # Quoi faire a chaque pas et por chaque batch
    for step, training_batch in enumerate(train_dataset):
      #GradientTape enregistre les dgradients calculés par l'optimiseur
      with tf.GradientTape() as tape:

       # Images reconstruites par l'autoencodeur
        reconstructed = model(training_batch)

        # Calcule la perte de reconstruction en utilisant l'entropie croisée
        perte = perte_reconstruction(training_batch, reconstructed)
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

