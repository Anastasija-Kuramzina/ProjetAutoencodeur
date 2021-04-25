import tensorflow.keras.datasets.fashion_mnist as mnist
import tensorflow.keras.datasets.cifar10 as cifar
import tensorflow as tf
import numpy


class Donnees():

    @classmethod
    def train_donnees_mnist(cls):
        """"Methode qui charge et prepare des images de l'entrainement FashionMNIST dans le format nécessaire pour l'entraînement.

        :return: dataset a utiliser pour l'entraînement
        :rtype: class 'tensorflow.python.data.ops.dataset_ops.BatchDataset'
        """

        # Chargement de l'ensemble de données MNIST composé de 60000 paires
        # image/étiquette d'entraînement et 10000 paires image/étiquette de test
        (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

        # Remodelation des donnees: 60000 vecteurs de dimension 784 au lieu des 60000 matrices de dimensions 28x28
        train_images = train_images.reshape(60000, 784)

        # Conversion de chaque pixel en un nombre flottant 32 bits
        train_images = train_images.astype('float32')

        # Normalisation des valeurs de chaque pixel pour les rendre entre 0 et 1
        train_images = train_images / 255
        train_images = 1 - train_images

        # Création d'un jeu de données dont les éléments sont des tranches des tenseurs donnés -
        # les tenseurs donnés sont découpés le long de leur première dimension
        # (division des lots en tenseurs individuels pour itérer sur l'ensemble de données):
        train_dataset = tf.data.Dataset.from_tensor_slices(train_images)

        # Mélanger les images (tenseurs)
        train_dataset = train_dataset.shuffle(buffer_size=1024).batch(64)

        return train_dataset



    @classmethod
    def test_donnees_mnist(cls):
        """"Methode qui charge et prepare des images de test MNIST

        :return: deux objets tu type 'numpy.ndarray', un contenant les images et un contenant leurs étiquettes
        :rtype: tuple
        """


        #Chargement de l'ensemble de données MNIST composé de 60000 paires
        # image/étiquette d'entraînement et 10000 paires image/étiquette de test:
        (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

        # Remodelation des donnees: 10000 vecteurs de dimension 784 au lieu des 10000 matrices de dimensions 28x28
        test_images = test_images.reshape(10000, 784)

        # Conversion de chaque pixel en un nombre flottant 32 bits
        test_images = test_images.astype('float32')

        #Normalisation des valeurs de chaque pixel pour les rendre entre 0 et 1
        test_images = test_images/255
        test_images = 1 - test_images

        return test_images, test_labels


   # @classmethod
    #def test_donnees_mnist_origin(cls):
    #    (test_images, test_labels) = mnist.load_data()
     #   return test_images, test_labels


    @classmethod
    def train_donnees_cifar(cls):
        """"Methode qui charge et prepare des images de l'entrainement Cifar10

        :return: dataset a utiliser pour l'entraînement
        :rtype: class 'tensorflow.python.data.ops.dataset_ops.BatchDataset'
        """

        # Chargement de l'ensemble de données Cifar10 composé de 50000 paires
        # image/étiquette d'entraînement et 10000 paires image/étiquette de test

        (train_images, train_labels), (test_images, test_labels) = cifar.load_data()
        # Remodelation des donnees: 50000 vecteurs de dimension 1024 au lieu des 50000 matrices de dimensions 32x32
        train_images = train_images.reshape(50000, 1024, 3)
        print(tf.shape(train_images))
        train_images = numpy.array(train_images)
        print(train_images.shape)
        train_images = train_images[:,:,0]
        print(train_images.shape)

        # Conversion de chaque pixel en un nombre flottant 32 bits
        train_images = train_images.astype('float32')

        # Normalisation des valeurs de chaque pixel pour les rendre entre 0 et 1
        train_images = train_images / 255

        # Création d'un jeu de données dont les éléments sont des tranches des tenseurs donnés -
        # les tenseurs donnés sont découpés le long de leur première dimension
        # (division des lots en tenseurs individuels pour itérer sur l'ensemble de données):
        train_dataset = tf.data.Dataset.from_tensor_slices(train_images)

        # Mélanger les images (tenseurs)
        train_dataset = train_dataset.shuffle(buffer_size=1024).batch(64)
        return train_dataset

    @classmethod
    def test_donnees_cifar(cls):
        """"Methode qui charge et prepare des images de test Cifar10

        :return: deux objets tu type 'numpy.ndarray', un contenant les images et un contenant leurs étiquettes
        :rtype: tuple
        """

        #Chargement de l'ensemble de données Cifar10 composé de 50000 paires
        # image/étiquette d'entraînement et 10000 paires image/étiquette de test:
        (train_images, train_labels), (test_images, test_labels) = cifar.load_data()

        # Remodelation des donnees: 10000 vecteurs de dimension 1024 au lieu des 10000 matrices de dimensions 32x32
        test_images = test_images.reshape(10000, 1024, 3)
        print(tf.shape(test_images))
        test_images = numpy.array(test_images)
        print(test_images.shape)
        test_images = test_images[:,:,0]
        print(test_images.shape)

        # Conversion de chaque pixel en un nombre flottant 32 bits
        test_images = test_images.astype('float32')

        #Normalisation des valeurs de chaque pixel pour les rendre entre 0 et 1
        test_images = test_images/255

        return test_images, test_labels

