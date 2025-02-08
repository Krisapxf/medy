import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.calibration import CalibratedClassifierCV

class KNeighborsClassifier:
    def load_data(self, file_path):
        return pd.read_csv(file_path, sep=';')

    def preprocess_data(self, data, features, target_column):
        data[features] = data[features].apply(pd.to_numeric, errors='coerce')
        data = data.dropna()
        X = data[features]
        y = (data[target_column] == 1).astype(int)
        return X, y

    def select_model(self, model_name, k=None):
        if model_name == 'KNN':
            knn = KNN(n_neighbors=k)
            knn = CalibratedClassifierCV(knn, method='sigmoid', cv='prefit')  # Use probability estimation
        elif model_name == 'SVM':
            return SVC(probability=True)
        elif model_name == 'DecisionTree':
            return DecisionTreeClassifier()
        else:
            raise ValueError(f"Model {model_name} jest nieznany!")

    def evaluate_model(self, model, X_train, X_test, y_train, y_test):
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        auc_roc = roc_auc_score(y_test, y_pred_proba)
        
        print("Informacje o modelu:")
        print(f"Dokładność modelu: {model.score(X_test, y_test):.4f}")
        print("\nRaport klasyfikacji:")
        print(classification_report(y_test, y_pred))
        print(f"\nAUC-ROC: {auc_roc:.4f}")

        conf_matrix = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=['Nie', 'Tak'], yticklabels=['Nie', 'Tak'])
        plt.xlabel("Przewidywane klasy")
        plt.ylabel("Prawdziwe klasy")
        plt.title(f"Macierz pomyłek - {model.__class__.__name__}")
        plt.show()

        fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f"AUC = {auc_roc:.2f}")
        plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"Krzywa ROC - {model.__class__.__name__}")
        plt.legend()
        plt.show()

    def main_knn(self):
        file_path = input("Podaj ścieżkę do pliku CSV z danymi: ")
        data = self.load_data(file_path)

        features = input("Podaj listę cech (np. 'cukry, tluszcz, kalorie'): ").split(', ')
        target_column = input("Podaj nazwę kolumny docelowej: ")

        X, y = self.preprocess_data(data, features, target_column)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model_name = input("Wybierz model (KNN, SVM, DecisionTree): ")
        if model_name == 'KNN':
            k = int(input("Podaj liczbę sąsiadów (k): "))
        else:
            k = None
        model = self.select_model(model_name, k)

        model.fit(X_train, y_train)

        self.evaluate_model(model, X_train, X_test, y_train, y_test)

