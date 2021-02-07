#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  6 20:03:00 2021

@author: kelliechong
"""

import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
#from tensorflow.keras.layers.experimental import preprocessing
from sklearn.model_selection import train_test_split

import json 
import numpy as np
from tensorflow import keras
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import model_from_json
import os


#retrieve data from github repo
diagnosis_url = "https://raw.githubusercontent.com/KellieChong/MacHacks2021/main/symptoms_diseases_data/dia_t.csv"
symptoms_url = "https://raw.githubusercontent.com/KellieChong/MacHacks2021/main/symptoms_diseases_data/sym_t.csv"
symptoms2_url = "https://raw.githubusercontent.com/KellieChong/MacHacks2021/main/symptoms_diseases_data/symptoms2.csv"
diffsydiw_url = "https://raw.githubusercontent.com/KellieChong/MacHacks2021/main/symptoms_diseases_data/diffsydiw.csv"

diagnosis_mapping = pd.read_csv(diagnosis_url)  #the diagnosis id is the first column of this array for all rows, think of this more as a mapping back after
csv_column_names = ['syd', 'did', 'wei']
x = pd.read_csv(diffsydiw_url)
x.dropna(how='any', inplace=True)

# Convert to numpy arrays
y = x.pop('did').to_numpy()
x = x.to_numpy()

# Map labels to class indices and make inverse mapping
'''class_to_idx = {cls: i for i, cls in enumerate(np.unique(y))}
idx_to_class = {i: cls for cls, i in class_to_idx.items()}
y = np.array([class_to_idx[yi] for yi in y])'''

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.25, random_state=100)

x_train = tf.keras.utils.normalize(x_train, axis = 1) #normalize input data
x_test = tf.keras.utils.normalize(x_test, axis = 1)

#y_train = y_train.dropna(how = 'any')

'''
def input_fn(features, labels, training=True, batch_size=256):
    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))

    # Shuffle and repeat if you are in training mode.
    if training:
        dataset = dataset.shuffle(1000).repeat()
    
    return dataset.batch(batch_size)'''

# Build a DNN with 2 hidden layers with 128 and 128 hidden nodes each.
diagnosis_model = tf.keras.Sequential() #input is 1D array
diagnosis_model.add(tf.keras.layers.Flatten())
diagnosis_model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
diagnosis_model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
#the number of nodes in the output layer is equal to the number of possible diagnosis ids in the diagnosis file
diagnosis_model.add(tf.keras.layers.Dense(1538, activation=tf.nn.softmax))

diagnosis_model.compile(optimizer='adam', 
    loss = 'sparse_categorical_crossentropy', 
    metrics=['accuracy'])

diagnosis_model.fit(x_train, y_train, batch_size = 400, epochs = 500)

val_loss, val_acc = diagnosis_model.evaluate(x_test, y_test)
print(val_loss, val_acc)

model_json = diagnosis_model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
diagnosis_model.save_weights("diagnosis_model.h5")

