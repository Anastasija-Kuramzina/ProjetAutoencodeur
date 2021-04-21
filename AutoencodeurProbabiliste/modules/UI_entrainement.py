import tkinter as tk
import AutoencodeurProbabiliste.modules as modules
from PIL import ImageTk, Image

class UI_entrainement():
    """Classe responsable de l'écran de l'entrainement, """

    def __init__(self, fenetre, mainwindow):

        self.fenetre = fenetre
        self.mainwindow = mainwindow
        self.donnees, labels = modules.Donnees.test_donnees_mnist()

        # Page de l'entrainement
        self.training = tk.Frame(master=self.fenetre, width=900, height=700, bg='black', relief='sunken')
        self.training.place(x=120, y=120)

        # Initialisations des hyperparametres, autoencodeur vide et status de l'entrainement
        self.params = [2, 512, 128, 32, 1]
        self.training_status = 0
        self.autoencoder = ""

        # Titre
        im = Image.open("../images/titre1v2.PNG")
        im = im.resize((700, 60), Image.ANTIALIAS)
        im = ImageTk.PhotoImage(im)
        titre = tk.Label(self.training, image = im, highlightthickness = 0, bg = 'black')
        titre.place(x=125, y=0)
        titre.image = im

        texte = tk.Label(self.training, text="LA PREMIÈRE ÉTAPE EST LA SÉLECTION \nDES HYPERPARAMÈTRES. \nVEUILLEZ SÉLECTIONNER LES \nTAILLES DE COUCHES DE L'AUTOENCODEUR, \nLE TAUX D'APPRENTISSAGE (~ 0,015) \n ET LE NOMBRE D'ÉPOQUES.\n  \nVOUS POUVEZ ENSUITE ENTRAÎNER \nL'AUTOENCODEUR ET SURVEILLER LA \nPROGRESSION À L'ÉCRAN."
                         , bg='black', fg='cyan', justify = 'left',
                          font=("Courier New", 11)).place(x=520, y=100)


        # Choix des hyperparametres
        couche1 = tk.Label(self.training, text="  DIMENSION DE COUCHE 1", bg='black', fg='blue',
                           font=("Courier New", 16, "bold")).place(x=30, y=100)
        self.couche1 = tk.Entry(self.training, bg='darkslategrey', fg='cyan', highlightbackground='cyan')
        self.couche1.place(x=380, y=100)

        couche2 = tk.Label(self.training, text="  DIMENSION DE COUCHE 2", bg='black', fg='blue',
                           font=("Courier New", 16, "bold")).place(x=30, y=140)
        self.couche2 = tk.Entry(self.training, bg='darkslategrey', fg='cyan', highlightbackground='cyan')
        self.couche2.place(x=380, y=140)

        couche3 = tk.Label(self.training, text="  DIMENSION DE COUCHE 3", bg='black', fg='blue',
                           font=("Courier New", 16, "bold")).place(x=30, y=180)
        self.couche3 = tk.Entry(self.training, bg='darkslategrey', fg='cyan', highlightbackground='cyan')
        self.couche3.place(x=380, y=180)

        couchelatente = tk.Label(self.training, text="  DIMENSION LATENTE", bg='black', fg='blue',
                                 font=("Courier New", 16, "bold")).place(x=30, y=220)
        self.couchelatente = tk.Entry(self.training, bg='darkslategrey', fg='cyan', highlightbackground='cyan')
        self.couchelatente.place(x=380, y=220)

        learningrate = tk.Label(self.training, text="  LEARNING RATE", bg='black', fg='blue',
                                font=("Courier New", 16, "bold")).place(x=30, y=260)

        self.lr = tk.Entry(self.training, bg='darkslategrey', fg='cyan', highlightbackground='cyan')
        self.lr.place(x=380, y=260)

        poidskl = tk.Label(self.training, text="  EPOCHS", bg='black', fg='blue',
                           font=("Courier New", 16, "bold")).place(x=30, y=300)

        self.epochs = tk.Entry(self.training, bg='darkslategrey', fg='cyan', highlightbackground='cyan')
        self.epochs.place(x=380, y=300)


        # Ecran montrant le processus de l'entrainement
        self.ecran = tk.Frame(self.training, width=800, height=170, bg='darkslategrey', highlightbackground="blue", highlightthickness = 3, relief='sunken').place(x=50, y=350)
        self.progress = tk.Label(self.training, text = " ", bg = 'darkslategrey', fg = 'white', font=("Courier New", 14, "bold"))
        self.progress.place(x=100,y=440)


        # Bouton d'entrainement
        train = Image.open("../images/entrainement.PNG")
        train = train.resize((370, 42), Image.ANTIALIAS)
        train = ImageTk.PhotoImage(train)

        self.entrainer = tk.Button(self.training, bg="black",activebackground='black', relief='flat', state="active",
                                    font=("Courier New", 16, "bold"), image = train, command=self.train)
        self.entrainer.place(x=232,y=525)
        self.entrainer.image = train


        # Visualisation de qualité de l'autoencodeur
        self.check = modules.UI_check(self.fenetre, self)

        # Bouton de vérification de qualité resultats
        self.checkbouton = tk.Button(self.training, text="VÉRIFIER LES\nRÉSULTATS", justify='center',
                               bg="black", activebackground='black', fg='blueviolet',
                               activeforeground='blueviolet', relief='raised', state="disabled",
                               font=("Courier New", 14, "bold"), command=self.check.comparer_resultats)
        self.checkbouton.place(x=380, y=615)


        # Visualisation des images encodées dans l'espace latent 2D
        self.plot = modules.UI_plot(self.fenetre, self)
        # Bouton d'affichage de la graphique
        self.plotbouton = tk.Button(self.training, text="PLOT: STRUCTURE\nLATENTE", justify='center',
                                   bg="black", activebackground='black', fg='blueviolet',
                                   activeforeground='blueviolet', relief='raised', state="disabled",
                                   font=("Courier New", 14, "bold"), command=self.plot.latent_plot)
        self.plotbouton.place(x=120, y=615)



        # Décodage d'un morceau aléatoire de l'espace latent régularisé
        self.morceau= modules.UI_morceau(self.fenetre, self)

        # Bouton d'affichage de l'espace latent (morceau decode)
        self.morceaubouton = tk.Button(self.training, text="ESPACE LATENTE:\nDÉCODAGE", justify='center',
                               bg="black", activebackground='black', fg='blueviolet',
                               activeforeground='blueviolet', relief='raised', state="disabled",
                             font=("Courier New", 14, "bold"), command=self.morceau.decoder_grille)

        self.morceaubouton.place(x=600, y=615)


    def train(self):
        '''Méthode principale de l'entrainement. Elle vérifie que les hyperparametres sont correctement choisis a l'aide
         de la méthode verifier_params() et, si les hyperparametres sont choisis correctement elle entraine l'autoencodeur a
         l'aide de la méthode train() de classe Autoencodeur. Finalement cette méthode active les boutons dependantses aux résultats
         de l'entrainement et change le statut de l'entrainement.'''
        print('VÉRIFICATION DES HYPERPARAMETRES')
        self.progress.configure(text="              VÉRIFICATION DES HYPERPARAMETRES                  ")
        self.progress.place(x=100, y=410)

        # Preparation des hyperparametres entieres et de learning rate
        self.params = self.toint([self.couchelatente.get(),self.couche1.get(), self.couche2.get(), self.couche3.get(),self.epochs.get()])
        lr = float(self.lr.get())
        # Parametres incorrectement saisis: signaler ceci a l'utilisateur et sortir
        if (self.verifier_params(self.params, lr) == -1):
            print('ECHEC DES HYPERPARAMETRES')
            return
        else:
            self.progress.configure(text="           ENTRAÎNEMENT COMMENCE                ")
            print('ENTRAINEMENT COMMENCE')

            self.autoencoder = modules.AutoEncodeur(input_dim=784, latent_dim=self.params[0],
                                                    dim_couche_1=self.params[1], dim_couche_2=self.params[2],
                                                    dim_couche_3=self.params[3], kl_poids=0.0012)
            # Entraînement
            perte = modules.train(self.autoencoder, lr, self.params[4], self.progress)
            # Message de fin de l'entraînement
            self.progress = tk.Label(self.training,
                                     text="ENTRAÎNEMENT FINI AVEC PERTE FINALE " + str(perte) + " %"      ,
                                     bg='darkslategrey', fg='white', font=("Courier New", 16, "bold"))
            self.progress.place(x=200, y=410)

            # Activation des boutons:
            if self.params[0] == 2:
                self.plotbouton.configure(state='active') # Graphique disponible seulement pour dim = 2
            self.checkbouton.configure(state='active')
            self.morceaubouton.configure(state='active')

            self.training_status = 1


    def toint(self, params):
        """Méthode convertissant les entrées (hyperparametres) en entiers"""
        int_params = []
        for i in params:
            int_params.append(int(i))
        return int_params


    def verifier_params(self, params, lr):
        """Méthode vérifiant que les hyperparametres saisis satisfont les conditions nécessaires minimales pour que
        l'autoencodeur fonctionne correctement et signalant les problemes et potentiels solutions a l'utilisateur """

        for i in params:
            if i < 1 or i >= 784:
                self.progress.configure(text = "SAISISSEZ UN ENTIER STRICTEMENT POSITIF INFÉRIEUR A 784         ")
                return -1

        if lr <= 0 or lr >= 1:
            self.progress.configure(text="CHOISISSEZ UN LEARNING RATE ENTRE 0 ET 1                    ")
            return -1

        if (params[0] >= params[3]) or (params[3] >= params[2]) or (params[2] >= params[1]) or (params[1] >= 784):
            self.progress.configure(text="CHAQUE COUCHE DOIT ÊTRE PLUS PETITE QUE LA COUCHE PRÉCÉDENTE  ")
            return -1

        return 0


    def on_enter_check(self):
        self.check_hover.configure(text = "Does this work?")

    def on_leave_check(self):
        self.check_hover.configure(text = "")

