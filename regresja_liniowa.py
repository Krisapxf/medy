import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import statsmodels.api as sm

class linear_regression:
    def __init__(self, dane):
        self.dane = dane

        
    def wybierz_kolumny(self,dane):
        print("\nDostępne kolumny w danych:")
        print(dane.columns)
        
        x_column = input("Podaj nazwę kolumny dla zmiennej niezależnej (X): ")
        y_column = input("Podaj nazwę kolumny dla zmiennej zależnej (Y): ")

        if x_column not in dane.columns or y_column not in dane.columns:
            print("Niepoprawne nazwy kolumn. Spróbuj ponownie.")
            return None, None
        
        return x_column, y_column

    def regresja_liniowa(self, dane, x_column, y_column):
        X=dane[x_column].values.reshape(-1,1)
        Y=dane[y_column].values.reshape(-1,1)
        model=LinearRegression()
        model.fit(X,Y)
        Y_pred=model.predict(X)
        mse=mean_squared_error(Y,Y_pred)
        r2=r2_score(Y,Y_pred)
        a = model.coef_[0][0] 
        b = model.intercept_[0]  

        # Obliczanie statystyk
        odchylenie_std_zaleznej = np.std(Y)  
        srednia_zaleznej = np.mean(Y)  
        reszty = Y - Y_pred  # Reszty
        blad_standardowy_reszt = np.std(reszty)  
        suma_kwadratow_reszt = np.sum(reszty ** 2) 
        
        print(f"\nWzór prostej regresji: y = {a:.4f} * X + {b:.4f}")
        print(f"Odchylenie standardowe zmiennej zależnej: {odchylenie_std_zaleznej:.4f}")
        print(f"Średnia arytmetyczna zmiennej zależnej: {srednia_zaleznej:.4f}")
        print(f"Błąd standardowy reszt: {blad_standardowy_reszt:.4f}")
        print(f"Suma kwadratów reszt: {suma_kwadratow_reszt:.4f}")
        print(f"\nWzór prostej regresji: y = {a:.4f} * X + {b:.4f}")
        
        X_with_const = sm.add_constant(X)
        model_stats = sm.OLS(Y, X_with_const).fit()
        print("\nPodsumowanie modelu regresji (statsmodels):")
        print(model_stats.summary())


        plt.scatter(X, Y, color='blue', label='Dane rzeczywiste')
        plt.plot(X, Y_pred, color='red', label='Linia regresji')
        plt.xlabel(x_column)
        plt.ylabel(y_column)
        plt.title('Regresja Liniowa')
        plt.legend()
        plt.show()

    def regresja_liniowa_1(self):
        print(self.dane)
        
        x_column, y_column = self.wybierz_kolumny(self.dane)
        
        if x_column and y_column:
            self.regresja_liniowa(self.dane, x_column, y_column)