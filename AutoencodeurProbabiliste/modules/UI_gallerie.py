import tkinter as tk
import AutoencodeurProbabiliste.modules as modules
import matplotlib.pyplot as plt
from PIL import ImageTk, Image
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


class UI_gallerie():

    def __init__(self, fenetre):
        self.fenetre = fenetre
        self.donnees, labels = modules.Donnees.test_donnees_mnist()

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
        self.initialiser_images(self.images_courantes)

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



    def initialiser_images(self, ind):
        n1, n2, n3, n4 = ind[0], ind[1], ind[2], ind[3]
        #Image1
        self.figure1 = plt.Figure(figsize=(1.72, 1.72))
        self.figure1.patch.set_facecolor('darkslategrey')
        self.im1 = FigureCanvasTkAgg(self.figure1, master=self.gallerie)
        self.im1.get_tk_widget().place(x=120, y=83)
        self.axis1 = self.figure1.add_subplot(1, 1, 1)
        self.axis1.get_xaxis().set_visible(False)
        self.axis1.get_yaxis().set_visible(False)
        modules.Affichage.afficher_image_originale(self.donnees, n1, self.axis1)
        self.im1.draw()

        #Image2
        self.figure2 = plt.Figure(figsize=(1.72, 1.72))
        self.figure2.patch.set_facecolor('darkslategrey')
        self.im2 = FigureCanvasTkAgg(self.figure2, master=self.gallerie)
        self.im2.get_tk_widget().place(x=285, y=83)
        self.axis2 = self.figure2.add_subplot(1, 1, 1)
        self.axis2.get_xaxis().set_visible(False)
        self.axis2.get_yaxis().set_visible(False)
        modules.Affichage.afficher_image_originale(self.donnees, n2, self.axis2)
        self.im2.draw()

        #Image3
        self.figure3 = plt.Figure(figsize=(1.72, 1.72))
        self.figure3.patch.set_facecolor('darkslategrey')
        self.im3 = FigureCanvasTkAgg(self.figure3, master=self.gallerie)
        self.im3.get_tk_widget().place(x=450, y=83)
        self.axis3 = self.figure3.add_subplot(1, 1, 1)
        self.axis3.get_xaxis().set_visible(False)
        self.axis3.get_yaxis().set_visible(False)
        modules.Affichage.afficher_image_originale(self.donnees, n3, self.axis3)
        self.im3.draw()

        #Image4
        self.figure4 = plt.Figure(figsize=(1.72, 1.72))
        self.figure4.patch.set_facecolor('darkslategrey')
        self.im4 = FigureCanvasTkAgg(self.figure4, master=self.gallerie)
        self.im4.get_tk_widget().place(x=615, y=83)
        self.axis4 = self.figure4.add_subplot(1, 1, 1)
        self.axis4.get_xaxis().set_visible(False)
        self.axis4.get_yaxis().set_visible(False)
        modules.Affichage.afficher_image_originale(self.donnees, n4, self.axis4)
        self.im4.draw()


    def image_suivante(self):
        self.frame_selection(self.axis2, self.im2, 'darkslategrey')
        self.frame_selection(self.axis3, self.im3,  'darkslategrey')
        ims = self.images_courantes
        for i in range(4):
            self.images_courantes[i] = ims[i] + 1
        self.initialiser_images(self.images_courantes)


    def image_precedente(self):
        self.frame_selection(self.axis2, self.im2,'darkslategrey')
        self.frame_selection(self.axis3, self.im3, 'darkslategrey')
        ims = self.images_courantes
        for i in range(4):
            self.images_courantes[i] = ims[i] - 1
        self.initialiser_images(self.images_courantes)

    def select_image_1(self):
        image = self.images_courantes[1]
        return image

    def select_image_2(self):
        image = self.images_courantes[2]
        return image

    def frame_selection(self, axis, im, color):
        axis.spines['left'].set_color(color)
        axis.spines['left'].set_linewidth(3)
        axis.spines['right'].set_color(color)
        axis.spines['right'].set_linewidth(3)
        axis.spines['top'].set_color(color)
        axis.spines['top'].set_linewidth(3)
        axis.spines['bottom'].set_color(color)
        axis.spines['bottom'].set_linewidth(3)
        im.draw()




