import pandas as pd
from regresja_liniowa import linear_regression
from liniowy_model_prawdopodobienstwa import linear_regression_logit_possibility
from bayes_durny import NaiveBayesClassifier
from bayes_durny_z_laplace import NaiveBayesClassifier_Laplace
from knn_model import KNeighborsClassifier
def wczytaj_dane():
    plik=input("Podaj nazwę pliku z danymi: ")
    print("Wczytywanie danych z pliku", plik)
    try:
        dane=pd.read_csv(plik, sep=";")
        print("Dane wczytane poprawnie")
        return dane
    except:
        print("Błąd wczytywania danych")
        return None
    
def wybierz_model():
    print("Wybierz model:")
    print("1. Regresja liniowa")
    print("2. Liniowy model prawdopodobieństwa/Model logitowy")
    print("3. Model Naive Bayes -LAB3")
    print("4. Model Naive Bayes -LAB3- z wygładzaniem Lapleace'a")
    print("5. KNN -ktore nie działa")
    print("6. zakoncz")

def obsluz_model(wybor, dane):
    if wybor == "1":
        print("Regresja liniowa.")
        lr = linear_regression(dane)
        lr.regresja_liniowa_1()
    elif wybor == "2":
        print("Liniowy model prawdopodobieństwa/Model logitowy.")
        lr = linear_regression_logit_possibility(dane)
        lr.regresja_liniowa_1()
    elif wybor == "3":
        print("Model Naive Bayes.")
        nbc = NaiveBayesClassifier()
        nbc.bayes()
    elif wybor == "4":
        print("Model Naive Bayes z wygładzaniem Laplace'a.")
        nbc = NaiveBayesClassifier_Laplace()
        nbc.bayes()
    elif wybor == "5":
        kn=KNeighborsClassifier()
        kn.main_knn()
    elif wybor == "6":
        print("Koniec programu.")
        return False
    else:
        print("Niepoprawny wybór. Spróbuj ponownie.")
    return True

def main():
    dane=wczytaj_dane()
    if dane is None:
        return
    
    while True:
        wybierz_model()
        wybor=input("Twój wybór: ")
        if not obsluz_model(wybor, dane):
            break

if __name__ == "__main__":
    main()