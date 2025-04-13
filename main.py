from core.run_model import run_model


#  Wiederholschleife
def hauptmenue():
    while True:
        print("Wähle das Modell:")
        print("1 = LBPH")
        print("2 = Eigenfaces")
        print("3 = Fisherfaces")
        print("q = Beenden")
        choice = input("Deine Wahl (1/2/3/q): ").strip()

        if choice == "1":
            from models.lbph import LBPHModel as Model
            run_model(Model, "LBPH")
        elif choice == "2":
            from models.eigenfaces import EigenfacesModel as Model
            run_model(Model, "Eigenfaces")
        elif choice == "3":
            from models.fisherfaces import FisherfacesModel as Model
            run_model(Model, "Fisherfaces")
        elif choice.lower() == "q":
            print("Beendet.")
            break
        else:
            print("Ungültige Eingabe. Bitte 1, 2 oder q eingeben.\n")

if __name__ == "__main__":
    hauptmenue()