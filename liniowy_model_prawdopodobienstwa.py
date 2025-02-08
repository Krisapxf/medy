import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import statsmodels.api as sm

class linear_regression_logit_possibility:
    def __init__(self, dane):
        self.dane = dane

    def wybierz_kolumny(self):
        print("\nDostępne kolumny w danych:")
        print(self.dane.columns)
        
        print("UWAGA, jeżeli model tego wymaga, to stwórz w danych kolumnę, która będzie zawierała prawdopodobieństwo, które model ma badać")
        wymagane = input("Czy model wymaga stworzenia kolumny z prawdopodobieństwem? (t/n): ")
        if wymagane == "t":
            kolumna_a = input("Podaj nazwę kolumny dla zmiennej a: ")
            kolumna_omega = input("Podaj nazwę kolumny dla zmiennej omega: ")
            self.dane["p_i"] = self.dane[kolumna_a] / self.dane[kolumna_omega]
            self.dane["L_i"] = np.log(self.dane["p_i"] / (1 - self.dane["p_i"]))


        self.dane["wiek_srodek"] = self.dane["Wiek"].apply(lambda x: np.mean([int(y) for y in x.split("-")]))
        self.dane["staz_pracy"] = self.dane["avg"]

        print("Dostępne kolumny w danych:")
        print(self.dane.columns)

        x1_column = input("Podaj nazwę kolumny dla 1 zmiennej niezależnej (X1): ")
        x2_column = input("Podaj nazwę kolumny dla 2 zmiennej niezależnej (X2): ")
        y_column = input("Podaj nazwę kolumny dla zmiennej zależnej (Y): ")

        if (x1_column or x2_column) not in self.dane.columns or y_column not in self.dane.columns:
            print("Niepoprawne nazwy kolumn. Spróbuj ponownie.")
            return None, None
        
        return x1_column,x2_column, y_column

    def linear_model(self, X, b0, b1, b2):
        const, x1_zmienna, x2_zmienna = X
        return b0 + b1 * x1_zmienna + b2 * x2_zmienna

    def regresja_liniowa(self, x1_column, x2_column,  y_column):
        x1_zmienna = self.dane[x1_column].values
        x2_zmienna = self.dane[x2_column].values
        prawdziwe_p = self.dane[y_column].values

        X = np.array([np.ones_like(x1_zmienna), x1_zmienna, x2_zmienna])
        popt_linear, pcov_linear = curve_fit(self.linear_model, X, prawdziwe_p, p0=[0, 0, 0], maxfev=10000)
        b0_linear, b1_linear, b2_linear = popt_linear
        predicted_linear = self.linear_model(X, b0_linear, b1_linear, b2_linear)

        # Obliczenie statystyk dopasowania dla modelu liniowego
        mse_linear = np.mean((prawdziwe_p - predicted_linear) ** 2)
        ss_total_linear = np.sum((prawdziwe_p - np.mean(prawdziwe_p)) ** 2)
        ss_residual_linear = np.sum((prawdziwe_p - predicted_linear) ** 2)
        r_squared_linear = 1 - (ss_residual_linear / ss_total_linear)
        
        print("Współczynniki modelu liniowego:")
        print(f"Intercept (b0): {b0_linear:.4f}")
        print(f"x1_zmienna (b1): {b1_linear:.4f}")
        print(f"x2_zmienna (b2): {b2_linear:.4f}")
        print("\nStatystyki dopasowania (model liniowy):")
        print(f"Błąd średniokwadratowy (MSE): {mse_linear:.4f}")
        print(f"R-squared: {r_squared_linear:.4f}")
        print("\nFunkcja liniowa:")
        print(f"p(X) = {b0_linear:.4f} + {b1_linear:.4f} * x1_zmienna + {b2_linear:.4f} * x2_zmienna")


        x_zmienna_range = np.linspace(self.dane[x1_column].min() - 3, self.dane[x1_column].max() + 3, 100)
        x2_zmienna_range = np.linspace(self.dane[x2_column].min() - 3, self.dane[x2_column].max() + 3, 100)

        x1_zmienna_sredni = self.dane[x2_column].mean()
        predicted_linear_x_zmienna = [self.linear_model([1, x1_zmienna_sredni, x2_zmienna], b0_linear, b1_linear, b2_linear) for x1_zmienna in x_zmienna_range]

        plt.figure(figsize=(10, 6))
        plt.plot(x_zmienna_range, predicted_linear_x_zmienna, label=f"x1_zmienna = {x1_zmienna_sredni:.1f} (model liniowy)", color="orange")
        plt.scatter(self.dane[x1_column], self.dane["p_i"], color="red", label="Empiryczne p_i")
        plt.title("Prawdopodobieństwo znalezienia pracy w funkcji x_zmiennau (model liniowy)")
        plt.xlabel("x1_zmienna")
        plt.ylabel("Prawdopodobieństwo")
        plt.legend()
        plt.grid()
        plt.show()

        x2_zmienna_sredni = self.dane[x2_zmienna].mean()
        predicted_linear_x2_zmienna = [self.linear_model([1, x2_zmienna_sredni, x1_zmienna], b0_linear, b1_linear, b2_linear) for x2_zmienna in x2_zmienna_range]

        plt.figure(figsize=(10, 6))
        plt.plot(x2_zmienna_range, predicted_linear_x2_zmienna, label=f"x2_zmienna = {x2_zmienna_sredni:.1f} lat (model liniowy)", color="purple")
        plt.scatter(self.dane["x2_zmienna_pracy"], self.dane["p_i"], color="blue", label="Empiryczne p_i")
        plt.title("Prawdopodobieństwo znalezienia pracy w funkcji stażu pracy (model liniowy)")
        plt.xlabel("zmienna x2")
        plt.ylabel("Prawdopodobieństwo")
        plt.legend()
        plt.grid()
        plt.show()

        probabilities_linear = np.zeros((len(x_zmienna_range), len(x2_zmienna_range)))
        for i, x1_zmienna in enumerate(x_zmienna_range):
            for j, x2_zmienna in enumerate(x2_zmienna_range):
                probabilities_linear[i, j] = self.linear_model([1, x1_zmienna, x2_zmienna], b0_linear, b1_linear, b2_linear)

        plt.figure(figsize=(12, 8))
        plt.imshow(probabilities_linear, aspect="auto", origin="lower",
                   extent=[x2_zmienna_range.min(), x2_zmienna_range.max(), x_zmienna_range.min(), x_zmienna_range.max()],
                   cmap="viridis")
        plt.colorbar(label="Prawdopodobieństwo (model liniowy)")
        plt.scatter(self.dane[x2_zmienna], self.dane[x1_zmienna], color="white", edgecolor="black", label="Empiryczne dane")
        plt.title("Heatmapa prawdopodobieństwa znalezienia pracy (model liniowy)")
        plt.xlabel("Staż pracy (lata)")
        plt.ylabel("x1_zmienna")
        plt.legend()
        plt.grid(False)
        plt.show()

    def regresja_liniowa_1(self):
        print(self.dane)
        x1_column, x2_column, y_column = self.wybierz_kolumny()
        if x1_column and x2_column and y_column:
            self.regresja_liniowa(x1_column,x2_column, y_column)