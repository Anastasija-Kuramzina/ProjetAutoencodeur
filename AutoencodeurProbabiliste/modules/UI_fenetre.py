import tkinter as tk
from PIL import ImageTk, Image
import AutoencodeurProbabiliste.projetae as modules


class UI_fenetre():
    """ Classe responsable de l'affichage de l'interface graphique.

    Une instance de cette classe posséde une fenêtre principale contenant trois écrans: écran de titre, écran de l'entraînement
    et écran de gestion des images. C'est la classe principale de ce projet, connectant tous les autres classes.

    :param fenetre: la fenêtre principale de l'interface graphique
    :type fenetre: class 'tkinter.Tk'
    :param background: le fond de la fenêtre principale
    :type background: class 'tkinter.Label'
    :param gallerie: gallerie affichant les images a choisir, partie de l'écran de gestion des images
    :type gallerie: class 'AutoencodeurProbabiliste.modules.UI_gallerie.UI_gallerie'
    :param entrainement: écran de gestion de l'entrainement
    :type entrainement: class 'AutoencodeurProbabiliste.modules.UI_entrainement.UI_entrainement
    :param resultats: partie de l'écran de gestion des images qui affiche les résultats
    :type resultats: class 'AutoencodeurProbabiliste.modules.UI_resultats.UI_resultats'
    :param next: bouton pour passer a l'écran suivant
    :type next:class 'tkinter.Button'
    :param previous: bouton pour passer a l'écran précédent
    :type previous: class 'tkinter.Button'
    :param startbouton: bouton pour fermer l'écran de titre et ouvrir l'écran de l'entraînement
    :type startbouton: class 'tkinter.Button'
    """
    def __init__(self):
        """"Constructeur de la classe UI_fenetre.

        Il construit d'abord la fenetre principale, puis il initialise les objets des classes UI_gallerie et UI_resultats
        ,qui font partie de l'eécran de gestion des images, et un objet de classe UI_entrainement - un écran de
        l'entrainement. À la fin, il construit l'écran de titre qui est créé au-dessus des autres écrans."""

        # Fenêtre principale
        self.fenetre = tk.Tk()
        self.fenetre.geometry("1300x1000")
        self.fenetre['bg'] = 'white'

        # Fond de la fenêtre principale
        im_background = Image.open("../images/monitor.png")
        im_background = im_background.resize((1300, 1000), Image.ANTIALIAS)
        im_background = ImageTk.PhotoImage(im_background)
        self.background = tk.Label(master=self.fenetre, bg='white', relief='raised', image = im_background)
        self.background.place(x=0,y=0)
        self.background.image = im_background
        backgroundtop = tk.Frame(self.fenetre,bg='black', relief='sunken',  width=950, height=750)
        backgroundtop.place(x=90, y=90)

        # Page de gestion des images
        self.gallerie = projetae.UI_gallerie(self.fenetre)

        # Page de gestion de l'entrainement
        self.entrainement = projetae.UI_entrainement(self.fenetre, self)

        # Page de gestion des resultats
        self.resultats = projetae.UI_resultats(self.fenetre, self.gallerie.gallerie, self)


        # Bouton pour acceder a l'ecran suivant
        self.next = tk.Button(self.fenetre, text=">", bg="black",
                              activebackground="black", fg='limegreen', activeforeground='limegreen',
                              font=("Courier New", 24, "bold"), command=self.suivant, relief='raised')
        self.next.place(x=930,y=700)

        # Bouton pour acceder a l'ecran precedent
        self.previous = tk.Button(self.fenetre, text="<", bg="black",
                              activebackground="black", fg='limegreen', activeforeground='limegreen',
                              font=("Courier New", 24, "bold"), command=self.precedent, state="disabled", relief='raised')
        self.previous.place(x=130,y=700)

        # Titre
        img_title = Image.open("../images/wallpaper2.jpg")
        img_title = img_title.resize((890, 540), Image.ANTIALIAS)
        img_title = ImageTk.PhotoImage(img_title)
        self.titlepage = tk.Frame(master=self.fenetre, width=900, height=700, bg='black', relief='raised',
                                  highlightbackground="black",highlightthickness=5)
        self.titlepage.place(x=120, y=120)
        title = tk.Label(self.titlepage, image=img_title, borderwidth = 0, highlightthickness = 0)
        title.image = img_title
        title.place(x=0, y=80)

        img_commencer = Image.open("../images/commencer.PNG")
        img_commencer = img_commencer.resize((300, 55), Image.ANTIALIAS)
        img_commencer = ImageTk.PhotoImage(img_commencer)

        self.startbouton = tk.Button(master=self.titlepage, command=self.start, state="active",
                                     relief='raised', image=img_commencer,  borderwidth = 0, highlightthickness = 0)
        self.startbouton.place(x=280, y=636)
        self.startbouton.image = img_commencer


    def suivant(self):
        """Commande pour le bouton self.next: elle ferme l'écran de gestion des images et affiche l'écran de l'entraînement."""

        self.entrainement.training.lower(self.gallerie.gallerie)
        self.next.configure( state="disabled")
        self.previous.configure( state="active")

    def precedent(self):
        """Commande pour le bouton self.previous: elle ferme l'écran de l'entraînement et affiche l'écran de gestion des images."""

        self.gallerie.gallerie.lower(self.entrainement.training)
        self.previous.configure(state="disabled")
        self.next.configure( state="active")

    def start(self):
        """Commande pour le bouton self.startbouton: elle ferme l'écran de titre et affiche l'écran de l'entraînement."""

        self.titlepage.lower(self.background)


if __name__=='__main__':

    interface = UI_fenetre()
    interface.fenetre.mainloop()

