# -*- coding: utf-8 -*-
"""
Created on Sat Jun  8 13:10:57 2019

@author: Matthew
"""
from __future__ import absolute_import, division, print_function

import pathlib

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

dataset = pd.read_excel('more_data.xlsx', index_col = 0)

def get_team(title):
    return title.partition('_')[0]

def get_year(title):
    return title.partition('_')[2]

for header in dataset.columns:
    if('%' not in header and header != 'Team' and header != 'Year' and header != 'G'):
        dataset[header] = dataset[header]/dataset['G']

dataset.pop('G');
dataset.pop('opp_G');

dataset.pop('FG');
dataset.pop('opp_FG');

team = dataset.pop('Team')
year = dataset.pop('Year')

dataset.pop('FT');

dataset.pop('2P');

#last_year = dataset[year == 2019]
#dataset = dataset.drop(last_year.index)

"""
if len(dataset.isna().sum().unique().nonzero()): 
    dataset = dataset.dropna()
"""

#Split into Training a Test Data
train_dataset = dataset.sample(frac=0.8,random_state=0)
test_dataset = dataset.drop(train_dataset.index)

#sns.pairplot(train_dataset[['Wins','FG%','3P%','ORB','AST']], diag_kind="kde")


train_stats = train_dataset.describe()
train_stats.pop("Wins")
train_stats = train_stats.transpose()
train_stats

train_labels = train_dataset.pop('Wins')
test_labels = test_dataset.pop('Wins')

def norm(x):
  return (x - train_stats['mean']) / train_stats['std']
normed_train_data = norm(train_dataset)
normed_test_data = norm(test_dataset)

def build_model():
  model = keras.Sequential([
    layers.Dense(64, activation=tf.nn.relu, input_shape=[len(train_dataset.keys())]),
    layers.Dense(64, activation=tf.nn.relu),
    layers.Dense(64, activation=tf.nn.relu),
    layers.Dense(1)
  ])

  optimizer = tf.keras.optimizers.RMSprop(0.001)

  model.compile(loss='mean_squared_error',
                optimizer=optimizer,
                metrics=['mean_absolute_error', 'mean_squared_error'])
  return model

model = build_model()
model.summary()

#example_batch = normed_train_data[:10]
#example_result = model.predict(example_batch)
#example_result

# Display training progress by printing a single dot for each completed epoch
class PrintDot(keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs):
    if epoch % 100 == 0: print('')
    print('.', end='')

EPOCHS = 1000

history = model.fit(
  normed_train_data, train_labels,
  epochs=EPOCHS, validation_split = 0.2, verbose=0,
  callbacks=[PrintDot()])



hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
hist.tail()

def plot_history(history):
  hist = pd.DataFrame(history.history)
  hist['epoch'] = history.epoch
  
  plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel('Mean Abs Error [Games]')
  plt.plot(hist['epoch'], hist['mean_absolute_error'],
           label='Train Error')
  plt.plot(hist['epoch'], hist['val_mean_absolute_error'],
           label = 'Val Error')
  plt.ylim([0,5])
  plt.legend()
  
  plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel('Mean Square Error [$Games^2$]')
  plt.plot(hist['epoch'], hist['mean_squared_error'],
           label='Train Error')
  plt.plot(hist['epoch'], hist['val_mean_squared_error'],
           label = 'Val Error')
  plt.ylim([0,20])
  plt.legend()
  plt.show()


plot_history(history)

model = build_model()

# The patience parameter is the amount of epochs to check for improvement
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

history = model.fit(normed_train_data, train_labels, epochs=EPOCHS,
                    validation_split = 0.05, verbose=0, callbacks=[early_stop, PrintDot()])

plot_history(history)

loss, mae, mse = model.evaluate(normed_test_data, test_labels, verbose=0)

print("Testing set Mean Abs Error: {:5.2f} Games".format(mae))

test_predictions = model.predict(normed_test_data).flatten()

plt.scatter(test_labels*82, test_predictions*82)
plt.xlabel('True Values [Games]')
plt.ylabel('Predictions [Games]')
plt.axis('equal')
plt.axis('square')
plt.xlim([0,plt.xlim()[1]])
plt.ylim([0,plt.ylim()[1]])
_ = plt.plot([-100, 100], [-100, 100])

error = 82*test_predictions - 82*test_labels
plt.hist(error, bins = 25)
plt.xlabel("Prediction Error [Games]")
_ = plt.ylabel("Count")







