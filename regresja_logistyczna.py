import numpy as np
import pandas as pd
from scipy.optimize import minimize
import matplotlib.pyplot as plt

file_path = 'liczba_bledow2.csv'  
df = pd.read_csv(file_path, sep=';')
df_train = df[df['Numer miesiaca'] <= 73]
df_train.loc[:, 'y'] = (df_train['Liczba bledow'] > 0).astype(int)

def logistic(x, beta_0, beta_1):
    return 1 / (1 + np.exp(-(beta_0 + beta_1 * x)))


def log_likelihood(params, x, y):
    beta_0, beta_1 = params
    p = logistic(x, beta_0, beta_1)
    ll = np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))
    return -ll  

initial_params = [0, 0] 

result = minimize(log_likelihood, initial_params, args=(df_train['Numer miesiaca'], df_train['y']), method='BFGS')

beta_0, beta_1 = result.x
print(f"------------Wyznaczone parametry: beta_0 = {beta_0:.4f}, beta_1 = {beta_1:.4f}")

df_train['logit(p)'] = -(beta_0 + beta_1 * df_train['Numer miesiaca'])
df_train['p'] = logistic(df_train['Numer miesiaca'], beta_0, beta_1)
df_train['1-p'] = 1 - df_train['p']
df_train['ln(p)'] = np.log(df_train['p'])
df_train['ln(1-p)'] = np.log(df_train['1-p'])
df_train['y*ln(p)'] = df_train['y'] * df_train['ln(p)']
df_train['(1-y)*ln(1-p)'] = (1 - df_train['y']) * df_train['ln(1-p)']
df_train['y*ln(p)+(1-y)*ln(1-p)'] = df_train['y*ln(p)'] + df_train['(1-y)*ln(1-p)']


print(df_train[['Numer miesiaca', 'Liczba bledow', 'y', 'logit(p)', 'p', '1-p', 'ln(p)', 'ln(1-p)', 'y*ln(p)', '(1-y)*ln(1-p)', 'y*ln(p)+(1-y)*ln(1-p)']])

blad_78 = 78
p_85 = logistic(blad_78, beta_0, beta_1)
print(f"Prawdopodobieństwo, ze błąd wystąpi w 78 miesiącu: {p_85:.4f}")


plt.figure(figsize=(10, 6))
plt.scatter(df_train['Numer miesiaca'], df_train['y'], color='blue', label='Rzeczywiste dane', zorder=5)
x_vals = np.linspace(0, 80, 100) 
y_vals = logistic(x_vals, beta_0, beta_1)
plt.plot(x_vals, y_vals, color='red', label='Model logistyczny', linewidth=2)

plt.title('Dopasowanie modelu logistycznego do wykrywania błędów')
plt.xlabel('Nr miesiąca')
plt.ylabel('Prawdopodobieństwo błędu')
plt.legend()
plt.grid(True)
plt.show()

