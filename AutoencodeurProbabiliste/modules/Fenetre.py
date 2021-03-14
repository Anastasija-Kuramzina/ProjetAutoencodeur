import tkinter as tk
import AutoencodeurProbabiliste.modules as modules
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import threading


class InterfaceGraphique():
    """ Classe responsable pour l'affichage de l'interface graphique """

    def __init__(self):

        # Fenetre principale
        self.fenetre = tk.Tk()
        self.fenetre.geometry("800x670")

        # Donnees et autoencodeur
        self.donnees, labels = modules.Donnees.test_donnees_mnist()
        self.autoencoder = modules.AutoEncodeur(input_dim = 784, latent_dim = 2, dim_couche_1 = 512, dim_couche_2 = 256, dim_couche_3 = 32, kl_poids = 0.0012)

        # Endroit pour l'entrainement
        self.training = tk.Frame(master=self.fenetre, width=700, height=95, bg='cornflowerblue')
        self.training.place(x = 50, y = 35)
        titre1 = tk.Label(self.fenetre, text = "1. Entraînez un autoencodeur:").place(x = 50,y = 10)
        self.entrainer = tk.Button(self.fenetre, text="Commencer l'entraînement", bg="steelblue",
                                   activebackground="limegreen", fg='white', activeforeground='white', command=self.train).place(x=75,
                                                                                                             y=75)
        self.ecran = tk.Frame(master=self.fenetre, width=490, height=82, bg='navy').place(x=250, y=42)
        self.progress = tk.Label(self.fenetre, text = ". . .", bg = 'navy', fg = 'white').place(x = 500,y = 70)

        # Endroit pour le choix des images
        self.training = tk.Frame(master=self.fenetre, width=700, height=95, bg='cornflowerblue')
        self.training.place(x=50, y=152)
        titre2 = tk.Label(self.fenetre, text="2. Choisissez deux images:").place(x=50, y=131)
        instruction= tk.Label(self.fenetre, text="Entrez deux entiers entre 0 et 9999 pour choisir deux images", bg='cornflowerblue', fg = 'white').place(x=56, y=160)
        self.entree1 = tk.Entry(self.fenetre)
        self.entree1.place(x=80, y=200)
        self.entree2 = tk.Entry(self.fenetre)
        self.entree2.place(x=320, y=200)

        # Endroit pour afficher les images choisies
        self.images = tk.Frame(master=self.fenetre, width=335, height=350, bg='cornflowerblue')
        self.images.place(x=50, y=275)
        titre3 = tk.Label(self.fenetre, text="3. Images choisies et leurs représentations vectorielles:").place(x=50, y=250)
        self.melanger = tk.Button(self.fenetre, text="Mélanger les images", bg="steelblue",
                                   activebackground="limegreen", fg='white', activeforeground='white', command=self.affichEntree).place(x=150,
                                                                                                             y=630)
        # Endroit pour image1:
        self.figure1 = plt.Figure(figsize=(1.5, 1.5))
        self.im1 = FigureCanvasTkAgg(self.figure1, master=self.fenetre)
        self.im1.get_tk_widget().place(x = 60, y = 290)
        self.axis1 = self.figure1.add_subplot(1, 1, 1)
        self.axis1.get_xaxis().set_visible(False)
        self.axis1.get_yaxis().set_visible(False)
        label1 = tk.Label(self.fenetre, text="Vecteur 1: ", bg='cornflowerblue', fg='white', font=("Courier", 12)).place(x=240, y=300)
        self.vect1 = tk.Label(self.fenetre, text = "...",bg = 'cornflowerblue', fg = 'white').place(x=230, y=340)

        # Endroit pour image 2:
        self.figure2 = plt.Figure(figsize=(1.5, 1.5))
        self.im2 = FigureCanvasTkAgg(self.figure2, master=self.fenetre)
        self.im2.get_tk_widget().place(x=60, y=460)
        self.im2.draw()
        self.axis2 = self.figure2.add_subplot(1, 1, 1)
        self.axis2.get_xaxis().set_visible(False)
        self.axis2.get_yaxis().set_visible(False)
        label2 = tk.Label(self.fenetre, text = "Vecteur 2: ", bg = 'cornflowerblue', fg = 'white', font=("Courier", 12)).place(x = 240, y = 470)
        self.vect2 = tk.Label(self.fenetre, text="...", bg = 'cornflowerblue', fg = 'white').place(x=230, y=510)

        # Endroit pour afficher les resultats
        self.images = tk.Frame(master=self.fenetre, width=335, height=350, bg='cornflowerblue')
        self.images.place(x=411, y=275)
        titre4 = tk.Label(self.fenetre, text="4. Image résultante:").place(x=411,y=250)

        label3 = tk.Label(self.fenetre, text = "Vecteur résultant: ", bg = 'cornflowerblue', fg = 'white', font=("Courier", 12)).place(x = 420, y = 280)
        self.vect3 = tk.Label(self.fenetre, text="...", bg = 'cornflowerblue', fg = 'white').place(x=425, y=315)

        self.figure3 = plt.Figure(figsize=(2.75, 2.75))
        self.im3 = FigureCanvasTkAgg(self.figure3, master=self.fenetre)
        self.im3.get_tk_widget().place(x=440, y=340)
        self.im3.draw()
        self.axis3 = self.figure3.add_subplot(1, 1, 1)
        self.axis3.get_xaxis().set_visible(False)
        self.axis3.get_yaxis().set_visible(False)

        recommencer = tk.Label(self.fenetre, text="Pour récommencer, saisissez deux nouveaux entiers",  fg='black').place(x=400,
                                                                                                            y=630)

    def affichEntree(self):
        plt.gray()
        image1 = self.entree1.get()
        image2 = self.entree2.get()
        if image1 == "" or image2 == "":
            print("Selectionnez deux images!")

        # Afficher image 1 et vecteur 1
        im1 = int(image1)
        vecteur1 = modules.Affichage.obtenir_vecteur(self.donnees, im1, self.autoencoder)
        self.vect1 = tk.Label(self.fenetre, text=str(vecteur1.numpy()), bg='cornflowerblue', fg='white').place(x=230,
                                                                                                               y=340)
        modules.Affichage.afficher_image_originale(self.donnees, im1, self.axis1)
        self.im1.draw()


        # Afficher image 2 et vecteur 2
        im2 = int(image2)
        vecteur2 = modules.Affichage.obtenir_vecteur(self.donnees, im2, self.autoencoder)
        self.vect2 = tk.Label(self.fenetre, text=str(vecteur2.numpy()), bg='cornflowerblue', fg='white').place(x=230,
                                                                                                               y=510)
        modules.Affichage.afficher_image_originale(self.donnees, im2, self.axis2)
        self.im2.draw()

        # Melanger les images
        moyenne = modules.Affichage.melange_images(vecteur1, vecteur2, self.autoencoder)
        self.vect3 = tk.Label(self.fenetre, text=str(moyenne.numpy()), bg='cornflowerblue', fg='white').place(x=430, y=315)
        modules.Affichage.afficher_image_reconstruite(moyenne, self.autoencoder, self.axis3)
        self.im3.draw()


    def train(self):
        modules.train(self.autoencoder,0.001,20)
        self.progress = tk.Label(self.fenetre, text="Entraînement fini!", bg = 'navy', fg = 'white').place(x = 460,y = 70)

if __name__=='__main__':

    interface = InterfaceGraphique()
    interface.fenetre.mainloop()