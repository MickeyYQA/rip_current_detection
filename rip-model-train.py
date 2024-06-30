# rip-model-train.py

# import necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report
import pickle
import time
from colorama import Fore, Back, Style

# import data
train_data = pd.read_csv('splitted_csv/train-tsht.csv')
test_data = pd.read_csv('splitted_csv/test.csv')

# data convertion
Train_data = train_data.values
Test_data = test_data.values

# print data shapes
print("Train_data shape:", Train_data.shape)
print("Test_data shape:", Test_data.shape)

# Convert DataFrame to NumPy arrays
X_train = Train_data[:,1:]
y_train = Train_data[:,0]
X_test = Test_data[:,1:]
y_test = Test_data[:,0]

# Remove rows with NaN values
nan_rows_train = np.isnan(X_train).any(axis=1)
nan_rows_test = np.isnan(X_test).any(axis=1)
X_train = X_train[~nan_rows_train]
y_train = y_train[~nan_rows_train]
X_test = X_test[~nan_rows_test]
y_test = y_test[~nan_rows_test]

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)

train_begin = time.time()
# train model
model = MLPClassifier(solver='adam',
                      hidden_layer_sizes=(60, ),
                      random_state=1, max_iter=150)
model.fit(X_train, y_train)
train_end = time.time()

# use pickle to export model
with open('model_splitted.pkl', 'wb') as file:
    pickle.dump(model, file)

begin = time.time()
print("training time:",(train_end - train_begin) * 10**3 ,"ms")

# evaluate model
train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)
end = time.time()
probs = model.predict_proba(X_test)
'''
with open('probabilities.txt', 'w') as f:
    for sample_probs in probs:
        sample_probs_str = ','.join(map(str, sample_probs))
        f.write(sample_probs_str + '\n')
'''
# show results
intend = input("Intend for this test:")
print(Fore.RED + intend)
print('Accuracy on training set: %f'%train_score)
print('Accuracy on testing set: %f'%test_score)
print("run time:",(end - begin) * 10**3 ,"ms")

pos_predictions = model.predict(X_test)
print(classification_report(y_test, pos_predictions, zero_division = 0))
