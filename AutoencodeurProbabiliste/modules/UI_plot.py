import AutoencodeurProbabiliste.modules as modules
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


class UI_plot():
    """ Classe responsable de l'affichage d'un graphique 2D des images données encodées dans l'espace latent """

    def __init__(self, fenetre, entrainement):
        # Fenetre principale
        self.fenetre = fenetre
        self.train = entrainement
        self.donnees, self.labels = modules.Donnees.test_donnees_mnist()

    def preparer_autoencoder(self):
        if self.train.training_status == 0:
            print('Autoencodeur pas entraîné')
            return -1
        elif self.train.training_status == 1:
            self.autoencodeur = self.train.autoencoder
            encodages = self.autoencodeur.encoder(self.donnees)
            return encodages


    def colors(self, labels):
        """" Assignation d'un couleur a chaque chiffre (chaque classe de MNIST) pour pouvoir visualiser l'espace latent"""
        cols = {0: 'lightcoral', 1: 'olivedrab', 2: 'goldenrod', 3: 'darkseagreen', 4: 'saddlebrown',
                5: 'steelblue', 6: 'cornflowerblue', 7: 'limegreen', 8: 'darkviolet', 9: 'slateblue'}
        colors = list(map(cols.get, labels))
        return colors


    def latent_plot(self):
        """" Preparation des encodages pour une visualisation, conversion de chaque vecteur latent en
        un couple des coordonnées et cisualisation des points obtenues dans une grille"""

        z_mu, z_log_var, encoded = self.preparer_autoencoder()
        labels = self.labels

        X_latent = []
        Y_latent = []

        # Conversion les vecteurs latentes en couples de coordonnées (x,y)
        for i in range(len(encoded)):
            X_latent.append(encoded[i][0])
            Y_latent.append(encoded[i][1])

        # Scatterplot
        plt.figure(figsize=(14, 14))
        plt.scatter(X_latent, Y_latent, c=self.colors(labels))

        # Création des étiquettes
        etiquettes = {0: 'T-shirt', 1: 'Pantalons', 2: 'Pull', 3: 'Robe', 4: 'Manteau', 5: 'Sandale', 6: 'Chemise',
                      7: 'Sneaker',
                      8: 'Sac', 9: 'Botte'}

        # Création de la légende
        patchlist = []
        for i in range(10):
            patchlist.append(mpatches.Patch(color=self.colors([i])[0], label=etiquettes[i]))
        plt.legend(handles=patchlist)

        plt.show()


