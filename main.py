from core.run_model import run_model


#  Wiederholschleife
def hauptmenue():
    while True:
        print("Wähle das Modell:")
        print("1 = LBPH")
        print("2 = Eigenfaces")
        print("3 = Fisherfaces")
        print("4 = Dlib-ResNet")
        print("5 = FaceNet")
        print("6 = ArcFace")
        print("q = Beenden")
        choice = input("Deine Wahl (1/2/3/4/5/6/q): ").strip()

        if choice == "1":
            from models.lbph import LBPHModel as Model
            run_model(Model, "LBPH")
        elif choice == "2":
            from models.eigenfaces import EigenfacesModel as Model
            run_model(Model, "Eigenfaces")
        elif choice == "3":
            from models.fisherfaces import FisherfacesModel as Model
            run_model(Model, "Fisherfaces")
        elif choice == "4":
            from models.dlib_resnet import DlibResNetModel as Model
            run_model(Model, "Dlib-ResNet")
        elif choice == "5":
            from models.FaceNet import FaceNetModel as Model
            run_model(Model, "FaceNet")

        elif choice == "6":
            from models.ArcFace import ArcFaceModel as Model
            run_model(Model, "ArcFace")
        elif choice.lower() == "q":
            print("Beendet.")
            break
        else:
            print("Ungültige Eingabe. Bitte 1, 2, 3, 4, 5, 6 oder q eingeben.\n")

if __name__ == "__main__":
    hauptmenue()