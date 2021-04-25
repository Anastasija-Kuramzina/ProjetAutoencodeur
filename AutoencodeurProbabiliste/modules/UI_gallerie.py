import tkinter as tk
import AutoencodeurProbabiliste.projetae as modules
import matplotlib.pyplot as plt
from PIL import ImageTk, Image
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class UI_gallerie():
    """Classe responsable de l'écran de gestion des images.

    Elle comporte une galerie défilante de 10000 images de test FashionMNIST, dans lesquelles l'utilisateur
    peut sélectionner des images pour que l'autoencodeur les fusionne.

    :param fenetre: fenêtre principale de l'interface graphique ou l'écran sera ajouté
    :type fenetre: class: 'tkinter.Tk'
    :param donnees: images test a utiliser pour les visualisations
    :type donnees: class 'numpy.ndarray'
    :param gallerie: page de gestion des images contenant tous les autres composants
    :type gallerie: class 'tkinter.Frame'
    :param ecran: écran constituant la gallerie, qui affiche 4 images a la fois
    :type ecran: class 'tkinter.Frame'
    :param images_courantes: indices des 4 images affichées sur l'écran
    :type images_courantes: list
    :param next: bouton pour décaler la gallerie vers la droite
    :type next: class 'tkinter.Button'>
    :param previous: bouton pour décaler la gallerie vers la gauche
    :type previous: class 'tkinter.Button'>
    :param resultats: endroit ou on instancie UI_resultats
    :type resultats: class 'tkinter.Frame'>
    :param figure1: figure matplotlib affichant la premiere image de la galerie
    :type figure1: class 'matplotlib.figure.Figure'
    :param im1: canevas pour afficher l'image 1
    :type im1: class 'matplotlib.backends.backend_tkagg.FigureCanvasTkAgg'
    :param axis1: l'axe ou il faut ajouter l'image 1
    :type axis1: class 'matplotlib.axes._subplots.AxesSubplot'
    :param figure2: figure matplotlib affichant la deuxieme image de la galerie
    :type figure2: class 'matplotlib.figure.Figure'
    :param im2: canevas pour afficher l'image 2
    :type im2: class 'matplotlib.backends.backend_tkagg.FigureCanvasTkAgg'
    :param axis2: l'axe ou il faut ajouter l'image 2
    :type axis2: class 'matplotlib.axes._subplots.AxesSubplot'
    :param figure3: figure matplotlib affichant la troisieme image de la galerie
    :type figure3: class 'matplotlib.figure.Figure'
    :param im3: canevas pour afficher l'image résultante
    :type im3: class 'matplotlib.backends.backend_tkagg.FigureCanvasTkAgg'
    :param axis3: l'axe ou il faut ajouter l'image résultante
    :type axis3: class 'matplotlib.axes._subplots.AxesSubplot'
    :param figure4: figure matplotlib affichant la quatrieme image de la galerie
    :type figure4: class 'matplotlib.figure.Figure'
    :param im4: canevas pour afficher l'image résultante
    :type im4: class 'matplotlib.backends.backend_tkagg.FigureCanvasTkAgg'
    :param axis4: l'axe ou il faut ajouter l'image résultante
    :type axis4: class 'matplotlib.axes._subplots.AxesSubplot'

    """
    def __init__(self, fenetre):
        self.fenetre = fenetre
        self.donnees, labels = projetae.Donnees.test_donnees_mnist()

        # Page de l'affichage et selection des images
        self.gallerie = tk.Frame(master=self.fenetre, width=900, height=700, bg='black', relief='sunken')
        self.gallerie.place(x=120, y=120)

        im = Image.open("../images/titre2v3.PNG")
        im = im.resize((600, 60), Image.ANTIALIAS)
        im = ImageTk.PhotoImage(im)
        titre = tk.Label(self.gallerie, image=im, highlightthickness = 0, bg = 'black')
        titre.place(x=140, y=0)
        titre.image = im

        # Ecran affichant les images
        self.ecran = tk.Frame(self.gallerie, width=800, height=184, bg='darkslategrey', highlightbackground="blue",
                              highlightthickness=5, relief='raised').place(x=50, y=77)

        # Liste contenant a tout moment les indices des images affichées
        self.images_courantes = [0,1,2,3]

        # Gallerie
        self.afficher_images(self.images_courantes)

        # Acceder a l'image suivante
        self.next = tk.Button(self.gallerie, text=">", bg="darkslategrey",
                              activebackground='darkslategrey', fg='blue', activeforeground='blue', relief='flat',
                              font=("Courier New", 28, "bold"), state = "active", command=self.image_suivante)
        self.next.place(x=790, y=135)


        # Acceder a l'image precedente
        self.previous = tk.Button(self.gallerie, text="<", bg="darkslategrey",
                                  activebackground='darkslategrey', fg='blue', activeforeground='blue', relief='flat',
                                  font=("Courier New", 28, "bold"), state="active", command = self.image_precedente)
        self.previous.place(x=55, y=132)


        # L'endroit pour les resultats
        self.resultats = tk.Frame(master=self.gallerie, width=800, height=260, bg='black')
        self.resultats.place(x=50, y=335)


    def afficher_images(self, ind):
        """Méthode de mise a jour de la galerie.

        Elle affiche 4 nouvelles images - dont les indices sont passées en argument - dans les 4 cases de l'écran.
        Les images proviennent de la base de données FashionMNIST

        :param ind: liste des 4 nouvelles indices
        :type ind: list
        """
        n1, n2, n3, n4 = ind[0], ind[1], ind[2], ind[3]
        #Image1
        self.figure1 = plt.Figure(figsize=(1.72, 1.72))
        self.figure1.patch.set_facecolor('darkslategrey')
        self.im1 = FigureCanvasTkAgg(self.figure1, master=self.gallerie)
        self.im1.get_tk_widget().place(x=120, y=83)
        self.axis1 = self.figure1.add_subplot(1, 1, 1)
        self.axis1.get_xaxis().set_visible(False)
        self.axis1.get_yaxis().set_visible(False)
        projetae.Affichage.afficher_image_originale(self.donnees, n1, self.axis1)
        self.im1.draw()

        #Image2
        self.figure2 = plt.Figure(figsize=(1.72, 1.72))
        self.figure2.patch.set_facecolor('darkslategrey')
        self.im2 = FigureCanvasTkAgg(self.figure2, master=self.gallerie)
        self.im2.get_tk_widget().place(x=285, y=83)
        self.axis2 = self.figure2.add_subplot(1, 1, 1)
        self.axis2.get_xaxis().set_visible(False)
        self.axis2.get_yaxis().set_visible(False)
        projetae.Affichage.afficher_image_originale(self.donnees, n2, self.axis2)
        self.im2.draw()

        #Image3
        self.figure3 = plt.Figure(figsize=(1.72, 1.72))
        self.figure3.patch.set_facecolor('darkslategrey')
        self.im3 = FigureCanvasTkAgg(self.figure3, master=self.gallerie)
        self.im3.get_tk_widget().place(x=450, y=83)
        self.axis3 = self.figure3.add_subplot(1, 1, 1)
        self.axis3.get_xaxis().set_visible(False)
        self.axis3.get_yaxis().set_visible(False)
        projetae.Affichage.afficher_image_originale(self.donnees, n3, self.axis3)
        self.im3.draw()

        #Image4
        self.figure4 = plt.Figure(figsize=(1.72, 1.72))
        self.figure4.patch.set_facecolor('darkslategrey')
        self.im4 = FigureCanvasTkAgg(self.figure4, master=self.gallerie)
        self.im4.get_tk_widget().place(x=615, y=83)
        self.axis4 = self.figure4.add_subplot(1, 1, 1)
        self.axis4.get_xaxis().set_visible(False)
        self.axis4.get_yaxis().set_visible(False)
        projetae.Affichage.afficher_image_originale(self.donnees, n4, self.axis4)
        self.im4.draw()


    def image_suivante(self):
        """Méthode de décalage de la galerie.

         Elle décale la galerie vers la gauche en affichant l'image de l'indice n+1 au lieu de l'image
         de l'indice n sur l'écran de la galerie.
         """
        self.frame_selection(self.axis2, self.im2, 'darkslategrey')
        self.frame_selection(self.axis3, self.im3,  'darkslategrey')
        ims = self.images_courantes
        for i in range(4):
            self.images_courantes[i] = ims[i] + 1
        self.afficher_images(self.images_courantes)


    def image_precedente(self):
        """Méthode de décalage de la galerie.

         Elle décale la galerie vers la gauche en affichant l'image de l'indice n-1 au lieu de l'image
         de l'indice n sur l'écran de la galerie.
         """
        self.frame_selection(self.axis2, self.im2,'darkslategrey')
        self.frame_selection(self.axis3, self.im3, 'darkslategrey')
        ims = self.images_courantes
        for i in range(4):
            self.images_courantes[i] = ims[i] - 1
        self.afficher_images(self.images_courantes)

    def select_image_1(self):
        """Méthode retournant l'indice dans la base de données FashionMNIST de la premiere image sur l'écran

        :return: indice de l'image a utiliser
        :rtype: int
        """
        image = self.images_courantes[1]
        return image

    def select_image_2(self):
        """Méthode retournant l'indice dans la base de données FashionMNIST de la deuxieme image sur l'écran

        :return: indice de l'image a utiliser
        :rtype: int"""
        image = self.images_courantes[2]
        return image

    def frame_selection(self, axis, im, color):
        """ Méthode ajoutant un cadre autout d'une image (pour montrer que elle a été sélectionnée) ou l'enlévant (pour
        montrer que l'image a été déselectionnée.

        :param axis: l'axe ou il faut afficher l'image
        :type axis: class 'matplotlib.axes._subplots.AxesSubplot'
        :param im: canevas ou se trouve image a encadrer
        :type im: class 'matplotlib.backends.backend_tkagg.FigureCanvasTkAgg'
        """
        axis.spines['left'].set_color(color)
        axis.spines['left'].set_linewidth(3)
        axis.spines['right'].set_color(color)
        axis.spines['right'].set_linewidth(3)
        axis.spines['top'].set_color(color)
        axis.spines['top'].set_linewidth(3)
        axis.spines['bottom'].set_color(color)
        axis.spines['bottom'].set_linewidth(3)
        im.draw()




