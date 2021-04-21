import tkinter as tk
from PIL import ImageTk, Image
import AutoencodeurProbabiliste.modules as modules


class UI_fenetre():
    """ Classe responsable de l'affichage de l'interface graphique """
    def __init__(self):

        # Fenetre principale
        self.fenetre = tk.Tk()

        im_background = Image.open("../images/monitor.png")
        im_background = im_background.resize((1300, 1000), Image.ANTIALIAS)
        im_background = ImageTk.PhotoImage(im_background)
        self.background = tk.Label(master=self.fenetre, bg='white', relief='raised', image = im_background)
        self.background.place(x=0,y=0)
        self.background.image = im_background

        self.background2 = tk.Frame(self.fenetre,bg='black', relief='sunken',  width=950, height=750)
        self.background2.place(x=90, y=90)
        # Page de gestion des images
        self.gallerie = modules.UI_gallerie(self.fenetre)

        # Page de l'entrainement
        self.entrainement = modules.UI_entrainement(self.fenetre, self)

        # Resultats
        self.resultats = modules.UI_resultats(self.fenetre, self.gallerie.gallerie, self)
        self.fenetre.geometry("1300x1000")
        self.fenetre['bg']='white'


        # Acceder a l'ecran suivant
        self.next = tk.Button(self.fenetre, text=">", bg="black",
                              activebackground="black", fg='limegreen', activeforeground='limegreen',
                              font=("Courier New", 24, "bold"), command=self.suivant, relief='raised')
        self.next.place(x=930,y=700)

        # Acceder a l'ecran precedent
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
        self.entrainement.training.lower(self.gallerie.gallerie)
        self.next.configure( state="disabled")
        self.previous.configure( state="active")

    def precedent(self):
        self.gallerie.gallerie.lower(self.entrainement.training)
        self.previous.configure(state="disabled")
        self.next.configure( state="active")

    def start(self):
        self.titlepage.lower(self.background)



if __name__=='__main__':

    interface = UI_fenetre()
    interface.fenetre.mainloop()

