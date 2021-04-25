import matplotlib.pyplot as plt
import AutoencodeurProbabiliste.sourceae as projetae

class Hypotheses():
    """Classe utilisée pour l'étude de l'influence des hyperparametres sur les aspects différents de la
    performance de l'autoencodeur.

    :param params: les hyperparametres initials
    :type params: list
    :param test_params: les valeurs de l'hyperparametre a varier
    :type test_params: list
    :param param_non_fixe: l'indice de hyperparametre a varier
    :type param_non_fixe: int
    :param pertes: la liste qui va contenir les pertes finales calculées pour chaque valeur de hyperparametre choisi
    :type pertes: list
    :param autoencodeur: l'autoencodeur a utiliser pour les expériences
    :type autoencodeur: class 'AutoencodeurProbabiliste.modules.Autoencodeur.AutoEncodeur'
    """

    def __init__(self, params, test_params, param_non_fixe):
        self.params = params
        self.test_params = test_params
        # indice de hyperparametre a etudier; les autres restent fixes
        self.param_non_fixe = param_non_fixe #indice de hyperparametre a varier

        self.pertes = []

        self.title_dict = {0:'Taille de couche cachée 1', 1:'Taille de couche cachée 2',2:'Taille de couche cachée 3',
                           3:'Dimension latente', 4:'Nombre d époques', 5:'Taux d aprentissage', 6: 'Poids de la divergence KL'}

    def train_record(self):
        """Méthode qui pour tout element p de test_params substitue p comme le hyperparametre a essayer, entraîne
         un autoencodeur avec cet hyperparametre et sauvegarde la perte finale, tout en gardant les
         autres hyperparametres constants.
         """

        self.x = []
        for i in range(len(self.test_params)):
            self.params[self.param_non_fixe] = self.test_params[i]
            couche1 = self.params[0]
            couche2 = self.params[1]
            couche3 = self.params[2]
            dim_lat = self.params[3]
            epochs = self.params[4]
            learning_rate = self.params[5]
            kl_poids = self.params[6]

            self.autoencodeur = projetae.AutoEncodeur(784, dim_lat, couche1, couche2, couche3, kl_poids)
            perte = projetae.train(self.autoencodeur, learning_rate, epochs)
            if perte is None:
                return
            self.pertes.append((perte))
            self.x.append(self.test_params[i])
            print('Expérience ', i, ' Param: ', self.test_params[i], " Perte: ", perte )


    def visualiser(self):
        """Méthode visualisant l'influence de hyperparametre choisi sur la perte moyenne finale"""
        plt.figure(figsize=(14, 14))
        plt.plot(self.x, self.pertes, '-o')
        plt.xlabel(self.title_dict[self.param_non_fixe])
        plt.ylabel("Perte moyenne")
        plt.show()


if __name__ == '__main__':

    params_initials = [256, 128, 64, 2, 10, 0.001, 0.001]

    # Taux d'aprentissage
    learningrates_overview = [0.000001, 0.000005, 0.00001, 0.00005, 0.0001, 0.0005,
                     0.001, 0.005, 0.01, 0.05 ,0.1,  0.5]

    learningrates_detail = [ 0.000001, 0.000005, 0.0001, 0.0002, 0.0005, 0.0007,
                     0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.01, 0.02, 0.03, 0.04, 0.05,
                     0.06, 0.07, 0.08, 0.09, 0.1, 0.2, 0.5, 0.9]
    test_learningrate = Hypotheses(params_initials, learningrates_detail, 5)
    test_learningrate.train_record()
    test_learningrate.visualiser()

    # Poids de la divergence KL
    poids_kl = [ 0.000001, 0.000005, 0.0001, 0.0002, 0.0005, 0.0007,
                     0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.01, 0.02, 0.03, 0.04, 0.05,
                     0.06, 0.07, 0.08, 0.09, 0.1, 0.2, 0.5, 0.9]
    test_kl = Hypotheses(params_initials, poids_kl, 6)
    test_kl.train_record()
    test_kl.visualiser()



