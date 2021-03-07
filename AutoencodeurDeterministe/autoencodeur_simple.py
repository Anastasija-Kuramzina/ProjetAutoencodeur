from tensorflow.keras.layers import Dense, Input
from tensorflow.keras import Model
from tensorflow.keras.datasets import mnist
import tensorflow.keras.datasets.cifar10 as cifar
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import tensorflow

# Preparation des donnees MNIST
(train_images, train_labels), (test_images, test_labels) = cifar.load_data()
train_images = train_images.reshape(50000, 3*1024)
test_images = test_images.reshape(10000, 3*1024)
train_images = train_images.astype('float32')
test_images = test_images.astype('float32')
train_images /= 255
test_images /= 255

# Classe Autoencodeur
class Autoencoder(Model):
    def __init__(self):
        super(Autoencoder, self).__init__()

        self.encoder = tensorflow.keras.Sequential([
            Dense(1024, activation='relu')(Input(shape=(3*1024,))),
            Dense(512, activation='relu'),
            Dense(64, activation='relu')
        ])

        self.decoder = tensorflow.keras.Sequential([
            Dense(512, activation='relu'),
            Dense(1024, activation='relu'),
            Dense(3*1024, activation='sigmoid'),
        ])


    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


def train(self, numEpochs, batchSize):
        self.compile(optimizer='adam', loss='binary_crossentropy')
        self.fit(train_images, train_images, epochs=numEpochs, batch_size=batchSize, shuffle=True,
                        validation_data=(test_images, test_images))


if __name__=='__main__':
    # Instantiation d'un auto-encodeur
    latent_values = []
    autoencoder = Autoencoder()

    #L'entrainement
    train(autoencoder,6, 256)

    # Preparation des images pour une visualisation:
    encoded_images = autoencoder.encoder(test_images).numpy()
    decoded_images = autoencoder.decoder(encoded_images).numpy()

    # Visualisation des resultats

    plt.figure(figsize=(20, 4))
    j = 15
    for i in range(10):
        inputaxis = plt.subplot(2, 10, i + 1)
        plt.imshow(test_images[j].reshape(32, 32,3))
        plt.gray()
        inputaxis.get_xaxis().set_visible(False)
        inputaxis.get_yaxis().set_visible(False)

        outputaxis = plt.subplot(2, 10, i + 11)
        plt.imshow(decoded_images[j].reshape(32, 32,3))
        plt.gray()
        outputaxis.get_xaxis().set_visible(False)
        outputaxis.get_yaxis().set_visible(False)
        j = j + 1
    plt.show()


    # Assignation d'un couleur a chaque chiffre
    def colors(labels):
        cols = {0: 'lightcoral', 1: 'olivedrab', 2: 'goldenrod', 3: 'darkseagreen', 4: 'saddlebrown',
                5: 'steelblue', 6: 'cornflowerblue', 7: 'limegreen', 8: 'darkviolet', 9: 'slateblue'}
        colors = list(map(cols.get, labels))
        return colors

    # Visualisation de l'espace latent
    X = []
    Y = []
    for i in range(len(encoded_images)):
        X.append(encoded_images[i][0])
        Y.append(encoded_images[i][1])
    X /= 255
    Y /= 255

    plt.figure(figsize=(14, 14))
    plt.scatter(X, Y, c=colors(test_labels))

    # Legende
    patchlist = []
    for i in range(10):
        patchlist.append(mpatches.Patch(color=colors([i])[0], label=i))
    plt.legend(handles=patchlist)
    plt.show()





