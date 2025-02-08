import pandas as pd
import numpy as np

class NaiveBayesClassifier_Laplace:
    def __init__(self):
        self.class_probs = {}
        self.conditional_probs = {}
        # Przykładowe dane
        self.data = pd.DataFrame({
            'pieniądz': ['nie', 'tak', 'nie', 'nie', 'tak', 'nie', 'nie', 'nie', 'nie', 'nie', 'tak', 'tak', 'nie'],
            'darmowy': ['nie', 'tak', 'nie', 'tak', 'nie', 'tak', 'tak', 'nie', 'tak', 'nie', 'tak', 'nie', 'tak'],
            'bogaty': ['tak', 'tak', 'nie', 'nie', 'nie', 'nie', 'nie', 'nie', 'nie', 'nie', 'tak', 'nie', 'tak'],
            'nieprzyzwoicie': ['nie', 'nie', 'nie', 'nie', 'nie', 'tak', 'tak', 'tak', 'nie', 'nie', 'nie', 'nie', 'nie'],
            'tajny': ['tak', 'nie', 'nie', 'nie', 'nie', 'tak', 'nie', 'nie', 'nie', 'tak', 'tak', 'tak', 'nie'],
            'spam': ['tak', 'tak', 'nie', 'tak', 'nie', 'tak', 'tak', 'tak', 'nie', 'nie', 'tak', 'tak', 'nie']
        })
        # chcemy przewidziec
        self.to_predict = {
            'pieniądz': 'tak',
            'darmowy': 'nie',
            'bogaty': 'tak',
            'nieprzyzwoicie': 'nie',
            'tajny': 'tak'
        }

    def fit(self):
        # Obliczanie prawdopodobieństw apriori P(C)
        total_count = len(self.data)
        for c in self.data['spam'].unique():
            class_count = len(self.data[self.data['spam'] == c])
            self.class_probs[c] = class_count / total_count
        
        # Obliczanie prawdopodobieństw warunkowych P(X|C)
        for c in self.data['spam'].unique():
            class_data = self.data[self.data['spam'] == c]
            class_prob = {}
            for feature in self.data.columns: 
                feature_count = class_data[feature].value_counts()
                class_prob[feature] = (feature_count +1) / (len(class_data)+len(self.data.columns))
            self.conditional_probs[c] = class_prob

    def predict(self):
        class_scores = {}
        total_score = 0
    
        for c, class_prob in self.class_probs.items():
            score = class_prob  
            
            for feature, value in self.to_predict.items():
                feature_prob = self.conditional_probs[c].get(feature, {}).get(value, 0)  # P(Xi | C)
                score *= feature_prob  
            class_scores[c] = score
            total_score += score * class_prob 

        print(class_scores)
        estimated_probabilities = {}

        for c, score in class_scores.items():

             estimated_probabilities[c] = score * self.class_probs[c] / total_score
        print("Estymowane prawdopodobieństwa dla klas:")
        print(estimated_probabilities)

        return max(class_scores, key=class_scores.get)
    
    def print_conditional_probs(self):
        for c, class_prob in self.conditional_probs.items():
            print(f'\nPrawdopodobieństwa warunkowe dla klasy "{c}":')
            for feature, probs in class_prob.items():
                print(f'  Cecha "{feature}":')
                for value, prob in probs.items():
                    print(f'    P("{feature}" = {value} | {c}) = {prob:.4f}')

    def print_class_probs(self):
        print('Prawdopodobieństwa klas:')
        for c, prob in self.class_probs.items():
            print(f'  P({c}) = {prob:.4f}')

    def bayes(self):
        self.fit()
        self.print_class_probs()
        self.print_conditional_probs()
        prediction = self.predict()
        print(f'Predykcja dla nowej wiadomości: {prediction}')



