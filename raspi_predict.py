#!/usr/bin/env python
# coding: utf-8

# In[ ]:
from keras import layers, models, optimizers
import numpy as np

X_train = np.load('./X_train.npy')
y_train = np.load('./y_train.npy')
X_test = np.load('./X_test.npy')
y_test = np.load('./y_test.npy')


# In[ ]:


#np.load('/Users/shibaharanaoki/Documents/raspi/cnn2_pic/X_train', X_train)
#np.load('/Users/shibaharanaoki/Documents/raspi/cnn2_pic/y_train', y_train)
#np.load('/Users/shibaharanaoki/Documents/raspi/cnn2_pic/X_test', X_test)
#np.load('/Users/shibaharanaoki/Documents/raspi/cnn2_pic/y_test', y_test)


# In[ ]:


model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(256, 256, 1)))
model.add(layers.Conv2D(32, (3, 3), activation='relu'))
model.add(layers.MaxPool2D(pool_size=(2, 2)))
model.add(layers.Dropout(0.2))

model.add(layers.Conv2D(32, (3, 3), activation='relu'))
model.add(layers.Conv2D(32, (3, 3), activation='relu'))
model.add(layers.Conv2D(32, (3, 3), activation='relu'))
model.add(layers.MaxPool2D(pool_size=(2, 2)))
model.add(layers.Dropout(0.2))

model.add(layers.GlobalAveragePooling2D())
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dropout(0.2))
model.add(layers.Dense(5, activation='softmax'))
model.summary()


# In[ ]:


model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


# In[ ]:


history = model.fit(X_train, y_train, epochs=20, batch_size=128, validation_split=0.2)


# In[ ]:


score = model.model.evaluate(X_test, y_test)


# In[ ]:


print('loss=', score[0])
print('accuracy=', score[1])


# In[ ]:


import matplotlib.pyplot as plt
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(len(acc))
plt.plot(epochs, acc, 'bo', label='training acc')
plt.plot(epochs, val_acc, 'b', label='validation acc')
plt.title('training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='training loss')
plt.plot(epochs, val_loss, 'b', label='validation loss')
plt.title('training and validation loss')
plt.legend()

model.save('./model1.h5')
