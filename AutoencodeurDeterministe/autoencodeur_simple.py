from tensorflow.keras.layers import Dense, Input
from tensorflow.keras import Model
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import tensorflow

# Preparation des donnees MNIST
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images = train_images.reshape(60000, 784)
test_images = test_images.reshape(10000, 784)
train_images = train_images.astype('float32')
test_images = test_images.astype('float32')
train_images /= 255
test_images /= 255

# Classe Autoencodeur
class Autoencoder(Model):
    def __init__(self):
        super(Autoencoder, self).__init__()

        self.encoder = tensorflow.keras.Sequential([
            Dense(512, activation='relu')(Input(shape=(784,))),
            Dense(64, activation='relu'),
            Dense(2, activation='relu')
        ])

        self.couche1 = Dense(512, activation='relu')
        self.couche2 = Dense(64, activation='relu')
        self.couche3 = Dense(2, activation='relu')


        self.decoder = tensorflow.keras.Sequential([
            Dense(64, activation='relu'),
            Dense(512, activation='relu'),
            Dense(784, activation='sigmoid'),
        ])

        couche1 = tensorflow.keras.Dense(32, activation='relu')

        couche2 = Dense(64, activation='relu')(couche1)



    def call(self, x):
        x = self.couche1(x)
        x = self.couche2(x)
        encoded = self.couche3(x)

        #encoded = self.encoder(x)
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
        plt.imshow(test_images[j].reshape(28, 28))
        plt.gray()
        inputaxis.get_xaxis().set_visible(False)
        inputaxis.get_yaxis().set_visible(False)

        outputaxis = plt.subplot(2, 10, i + 11)
        plt.imshow(decoded_images[j].reshape(28, 28))
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





