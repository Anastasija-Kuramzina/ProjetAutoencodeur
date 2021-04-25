import tkinter as tk
import AutoencodeurProbabiliste.projetae as modules
import matplotlib.pyplot as plt
from PIL import ImageTk, Image
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class UI_resultats():
    """ Classe responsable de l'affichage des images choisiers dans la gallérie, de leur encodage et calcul de la moyenne
    et d'affichage d'image mélagée.

    :param fenetre: fenêtre principale
    :type fenetre: class 'AutoencodeurProbabiliste.modules.UI_fenetre.UI_fenetre'
    :param frame: endroit sur l'écran de gallerie ou il faut afficher les résultats
    :type frame: class 'tkinter.Frame'>
    :param train: écran de l'entraînement d'ou provient l'autoencodeur entraîné
    :type train: class 'AutoencodeurProbabiliste.modules.UI_entrainement.UI_entrainement'
    :param gal: gallerie
    :type gal: class 'AutoencodeurProbabiliste.modules.UI_gallerie.UI_gallerie'
    :param donnees: images test a utiliser pour les visualisations
    :type donnees: class 'numpy.ndarray'
    :param figure1: figure matplotlib ou il faut ajouter l'image 1
    :type figure1: class 'matplotlib.figure.Figure'
    :param im1: canevas pour afficher l'image 1
    :type im1: class 'matplotlib.backends.backend_tkagg.FigureCanvasTkAgg'
    :param axis1: l'axe ou il faut ajouter l'image 1
    :type axis1: class 'matplotlib.axes._subplots.AxesSubplot'
    :param vect1: texte affichant la représentation vectorielle de l'image 1
    :type vect1: class 'tkinter.Label'
    :param figure2: figure matplotlib ou il faut ajouter l'image 2
    :type figure2: class 'matplotlib.figure.Figure'
    :param im2: canevas pour afficher l'image 2
    :type im2: class 'matplotlib.backends.backend_tkagg.FigureCanvasTkAgg'
    :param axis2: l'axe ou il faut ajouter l'image 2
    :type axis2: class 'matplotlib.axes._subplots.AxesSubplot'
    :param vect2: texte affichant la représentation vectorielle de l'image 2
    :type vect2: class 'tkinter.Label'
    :param figure3: figure matplotlib ou il faut ajouter l'image résultante
    :type figure3: class 'matplotlib.figure.Figure'
    :param im3: canevas pour afficher l'image résultante
    :type im3: class 'matplotlib.backends.backend_tkagg.FigureCanvasTkAgg'
    :param axis3: l'axe ou il faut ajouter l'image résultante
    :type axis3: class 'matplotlib.axes._subplots.AxesSubplot'
    :param vect3: texte affichant la représentation vectorielle de l'image résultante
    :type vect3: class 'tkinter.Label'
    :param button1: bouton pour sélectionner l'image 1
    :type button1: class 'tkinter.Button'
    :param button2: bouton pour sélectionner l'image 2
    :type button2: class 'tkinter.Button'
    :param melanger: bouton pour mélanger les images
    :type melanger: class 'tkinter.Button'
    """
    def __init__(self, fenetre, gallerie, mainwindow):

        # Fenetre principale
        self.fenetre = fenetre
        self.frame = gallerie
        self.train = mainwindow.entrainement
        self.gal = mainwindow.gallerie
        self.donnees, labels = projetae.Donnees.test_donnees_mnist()

        # Endroit pour image1:
        self.figure1 = plt.Figure(figsize=(1.6, 1.6))
        self.figure1.patch.set_facecolor('black')
        self.im1 = FigureCanvasTkAgg(self.figure1, master=self.frame)
        self.im1.get_tk_widget().place(x=60, y=370)
        self.axis1 = self.figure1.add_subplot(1, 1, 1)
        self.axis1.set_facecolor('black')
        self.axis1.get_xaxis().set_visible(False)
        self.axis1.get_yaxis().set_visible(False)
        titre1 = tk.Label(self.frame, text="PREMIERE IMAGE CHOISIE", bg='black', fg='cyan',
                          font=("Courier New", 14, "bold")).place(x=35, y=345)

        label1 = tk.Label(self.frame, text="VECTEUR 1: ", bg='black', fg='blueviolet',
                          font=("Courier New", 14, 'bold')).place(x=90, y=525)

        self.vect1 = tk.Label(self.frame, text=" ", bg='black', fg='blueviolet',
                       font=("Courier New", 14, 'bold'))
        self.vect1.place(x=30, y=555)

        # Endroit pour image 2:
        titre2 = tk.Label(self.frame, text="DEUXIEME IMAGE CHOISIE", bg='black', fg='cyan',
                          font=("Courier New", 14, "bold")).place(x=625, y=345)
        self.figure2 = plt.Figure(figsize=(1.6, 1.6))
        self.figure2.patch.set_facecolor('black')
        self.im2 = FigureCanvasTkAgg(self.figure2, master=self.frame)
        self.im2.get_tk_widget().place(x=680, y=370)
        self.im2.draw()
        self.axis2 = self.figure2.add_subplot(1, 1, 1)
        self.axis2.set_facecolor('black')
        self.axis2.get_xaxis().set_visible(False)
        self.axis2.get_yaxis().set_visible(False)
        label2 = tk.Label(self.frame, text="VECTEUR 2: ",bg='black', fg='blueviolet',
                          font=("Courier New", 14, 'bold')).place(x=710, y=525)
        self.vect2 = tk.Label(self.frame, text=" ", bg='black', fg='blueviolet',
                          font=("Courier New", 14, 'bold'))
        self.vect2.place(x=630, y=555)

        # Endroit pour afficher les resultats (image mélangée)
        self.figure3 = plt.Figure(figsize=(2.75, 2.75))
        self.figure3.patch.set_facecolor('black')
        self.im3 = FigureCanvasTkAgg(self.figure3, master=self.frame)
        self.im3.get_tk_widget().place(x=315, y=305)
        self.im3.draw()
        self.axis3 = self.figure3.add_subplot(1, 1, 1)
        self.axis3.set_facecolor('darkslategrey')
        self.axis3.get_xaxis().set_visible(False)
        self.axis3.get_yaxis().set_visible(False)
        label3 = tk.Label(self.frame, text="VECTEUR RÉSULTANT ", bg='black', fg='blueviolet',
                          font=("Courier New", 14, "bold")).place(x=350, y=572)
        self.vect3 = tk.Label(self.frame, text=" ", bg='black', fg='magenta',
                              font=("Courier New", 14, "bold"))
        self.vect3.place(x=320, y=600)


        # Boutons pour choisir des images
        self.button1 = tk.Button(self.frame, text = 'AJOUTER COMME\nIMAGE 1', justify='center', bg="black",activebackground='black', fg='limegreen', activeforeground='limegreen', relief='raised', state="active", cursor="star", font=("Courier New", 12, "bold"), highlightbackground="magenta",
                              highlightthickness=2, command=self.affichEntree1)
        self.button1.place(x=295, y=270)

        self.button2 = tk.Button(self.frame, text = 'AJOUTER COMME\nIMAGE 2', justify='center',bg="black",activebackground='black', fg='limegreen', activeforeground='limegreen', relief='raised', state="active", cursor="star", font=("Courier New", 12, "bold"), highlightbackground="magenta",
                              highlightthickness=2, command=self.affichEntree2)
        self.button2.place(x=460, y=270)

        # Bouton pour mélanger les images
        melanger = Image.open("../images/melanger1.PNG")
        melanger = melanger.resize((180, 40), Image.ANTIALIAS)
        melanger = ImageTk.PhotoImage(melanger)
        self.melanger = tk.Button(self.frame, image=melanger, bg="black", relief = 'flat', activebackground='black', highlightbackground = 'black', highlightthickness=0, state="active",
                                  command=self.melanger)
        self.melanger.place(x=352, y=645)
        self.melanger.image = melanger

        # Images décoratives
        arrowleft = Image.open("../images/arrowleft.PNG")
        arrowleft = arrowleft.resize((90, 60), Image.ANTIALIAS)
        arrowleft = ImageTk.PhotoImage(arrowleft)
        al= tk.Label(self.frame, image=arrowleft, highlightthickness=0, bg='black')
        al.place(x=564, y=420)
        al.image = arrowleft

        arrowright = Image.open("../images/arrowright.PNG")
        arrowright = arrowright.resize((90, 60), Image.ANTIALIAS)
        arrowright = ImageTk.PhotoImage(arrowright)
        ar = tk.Label(self.frame, image=arrowright, highlightthickness=0, bg='black')
        ar.place(x=240, y=420)
        ar.image = arrowright


    def affichEntree1(self):
        """Méthode affichant la premiere image chosie et, si l'entraînement est fini, sa représentation vectorielle"""
        self.gal.frame_selection(self.gal.axis2, self.gal.im2,'cyan')
        im1 = self.gal.select_image_1()

        if self.train.training_status == 1:
            self.autoencoder = self.train.autoencoder
            self.vecteur1 = projetae.Affichage.obtenir_vecteur(self.donnees, im1, self.autoencoder)
            self.vect1.configure(text=str(self.vecteur1.numpy()))
        else:
            print("Entrainement pas fini.")
        projetae.Affichage.afficher_image_originale(self.donnees, im1, self.axis1)
        self.im1.draw()


    def affichEntree2(self):
        """Méthode affichant la deuxieme image chosie et, si l'entraînement est fini, sa représentation vectorielle"""
        self.gal.frame_selection(self.gal.axis3, self.gal.im3, 'cyan')
        plt.gray()
        im2 = self.gal.select_image_2()

        if self.train.training_status == 1:
            self.autoencoder = self.train.autoencoder
            self.vecteur2 = projetae.Affichage.obtenir_vecteur(self.donnees, im2, self.autoencoder)
            self.vect2.configure(text=str(self.vecteur2.numpy()))
        else:
            print("Entrainement pas fini.")
        projetae.Affichage.afficher_image_originale(self.donnees, im2, self.axis2)
        self.im2.draw()


    def melanger(self):
        """Méthode mélangeant les deux images choisies - si l'entraînement est fini - et affichant cette image et sa
        représentation vectorielle."""
        if self.train.training_status == 1:
            self.autoencoder = self.train.autoencoder
            vecteur1 = self.vecteur1
            vecteur2 = self.vecteur2
            moyenne = projetae.Affichage.melange_images(vecteur1, vecteur2)
            self.vect3.configure(text=str(moyenne.numpy()))
            projetae.Affichage.afficher_image_reconstruite(moyenne, self.autoencoder, self.axis3)
            self.im3.draw()
        else:
            print("Entrainement pas fini.")
