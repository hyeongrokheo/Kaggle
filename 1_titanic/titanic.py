import tensorflow as tf
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

# train_data = pd.read_csv('data/train.csv').dropna()
train_data = pd.read_csv('data/train.csv')
test_data = pd.read_csv('data/test.csv')
gender_data = pd.read_csv('data/gender_submission.csv')
# data = [train_data, test_data]

print(train_data)
print(train_data.describe())
exit()
# print(train_data)
# print(test_data)
# print(gender_data)
# print(train_data.columns)
# print(test_data.columns)

train_data_x = train_data[['Pclass', 'Sex', 'Age']]
train_data_y = train_data['Survived']

train_data_x.loc[train_data_x['Sex'] == 'male', 'Sex'] = 1
train_data_x.loc[train_data_x['Sex'] == 'female', 'Sex'] = 0
train_data_x['Sex'] = train_data_x['Sex'].astype('int')

test_data_x = test_data[['Pclass', 'Sex', 'Age']]
test_data_y = gender_data['Survived']

test_data_x.loc[test_data_x['Sex'] == 'male', 'Sex'] = 1
test_data_x.loc[test_data_x['Sex'] == 'female', 'Sex'] = 0
test_data_x['Sex'] = test_data_x['Sex'].astype('int')

# train_data_x['Age'].dropna()
# train_data_x = train_data_x.dropna()
train_data_x.info()
print(train_data_x)
print(train_data_y)
print(test_data)
# train_data_x.loc[train_data_x['Age'] == None, 'Age'] =

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(8, activation='relu', input_dim=3),
    tf.keras.layers.Dense(4, activation='relu'),
    tf.keras.layers.Dense(1, activation='softmax')
])

# model.com
sgd = tf.keras.optimizers.SGD(learning_rate=0.01)
model.compile(optimizer=sgd, loss='mae', metrics=['acc'])
EPOCHS = 100

history = model.fit(train_data_x, train_data_y,
                    validation_data=(test_data_x, test_data_y),
                    epochs=EPOCHS)

# print(history)

plt.plot(np.arange(1, 1+EPOCHS), history.history['loss'])
plt.plot(np.arange(1, 1+EPOCHS), history.history['val_loss'])
plt.title('Loss / Val Loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend(['loss', 'val_loss'], fontsize=15)
plt.show()

plt.plot(np.arange(1, 1+EPOCHS), history.history['acc'])
plt.plot(np.arange(1, 1+EPOCHS), history.history['val_acc'])
plt.title('Acc / Val Acc')
plt.xlabel('epochs')
plt.ylabel('acc')
plt.legend(['acc', 'val_acc'], fontsize=15)
plt.show()
