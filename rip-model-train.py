# rip-model-train.py

# import necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
import pickle
import time

# import data
train_data = pd.read_csv('csv_data/all_pics.csv')
test_data = pd.read_csv('with_rips_100.csv')

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

# train model
model = MLPClassifier(solver='adam',
                      hidden_layer_sizes=(60, ),
                      random_state=1, max_iter=150)
model.fit(X_train, y_train)

# use pickle to export model
with open('model_100.pkl', 'wb') as file:
    pickle.dump(model, file)

# start timing
begin = time.time()

# evaluate model
train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)

# show results
print('训练集的准确率：%f'%train_score,"%")
print('测试集的准确率：%f'%test_score,"%")

# end timing
end = time.time()
print("run time:",(end - begin) * 10**3 ,"ms")